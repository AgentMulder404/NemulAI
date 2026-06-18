# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Experience logger for the self-learning optimization agent.

Logs every (context, action, outcome) tuple produced by the heuristic
recommendation engine.  The resulting corpus is used to warm-start the
contextual bandit in Phase 2.

Storage: flock-protected JSONL WAL in DATA_DIR/experience/, following
the same durability pattern as the metrics uploader (uploader.py).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

from learner.reward import compute_energy_reward

log = logging.getLogger("nemulai-learner")


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class WorkloadContext:
    gpu_name: str
    gpu_arch: str
    workload_class: str
    utilization_gpu_pct: float
    utilization_memory_pct: float
    memory_pressure: float
    power_draw_w: float
    power_limit_w: float
    temperature_c: float
    power_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.power_limit_w > 0 and self.power_ratio == 0.0:
            self.power_ratio = self.power_draw_w / self.power_limit_w


@dataclass
class ActionTaken:
    action_type: str
    source: str
    recommended_value: float
    current_value: float
    estimated_savings_pct: float


@dataclass
class ActionOutcome:
    energy_delta_j_before: float
    energy_delta_j_after: float
    throughput_before: float
    throughput_after: float
    recommendation_status: str
    actual_savings_pct: float
    observation_window_s: float


@dataclass
class ExperienceTuple:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    machine_id: str = ""
    gpu_index: int = 0
    context: Optional[WorkloadContext] = None
    action: Optional[ActionTaken] = None
    outcome: Optional[ActionOutcome] = None
    reward: Optional[float] = None

    def is_complete(self) -> bool:
        return self.outcome is not None and self.reward is not None


# ── Pending action tracker (for outcome correlation) ─────────────────────────

@dataclass
class _PendingAction:
    tuple_id: str
    logged_at: float
    gpu_index: int
    energy_snapshot: float
    throughput_snapshot: float


# ── Experience Logger ────────────────────────────────────────────────────────

class ExperienceLogger:
    """Logs (context, action, outcome) tuples to a local JSONL WAL."""

    def __init__(
        self,
        data_dir: Path,
        machine_id: str,
        outcome_window_s: float = 300.0,
        flush_threshold: int = 50,
    ) -> None:
        self._dir = data_dir / "experience"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._wal_path = self._dir / "experience.wal"
        self._machine_id = machine_id
        self._outcome_window_s = outcome_window_s
        self._flush_threshold = flush_threshold

        self._pending: dict[str, _PendingAction] = {}
        self._tuples: dict[str, ExperienceTuple] = {}
        self._completed_count = 0

    # ── Public API ───────────────────────────────────────────────────────

    def log_action(
        self,
        context: WorkloadContext,
        action: ActionTaken,
        gpu_index: int,
        energy_snapshot: float,
        throughput_snapshot: float,
    ) -> str:
        """Log a new (context, action) pair.  Returns the tuple ID."""
        t = ExperienceTuple(
            machine_id=self._machine_id,
            gpu_index=gpu_index,
            context=context,
            action=action,
        )
        self._tuples[t.id] = t
        self._pending[t.id] = _PendingAction(
            tuple_id=t.id,
            logged_at=t.timestamp,
            gpu_index=gpu_index,
            energy_snapshot=energy_snapshot,
            throughput_snapshot=throughput_snapshot,
        )
        self._wal_append(t)
        log.info(
            "Experience: logged action %s on GPU %d (%s, %s)",
            t.id[:8], gpu_index, action.action_type, action.source,
        )
        return t.id

    def check_pending_outcomes(
        self,
        current_energy_by_gpu: dict[int, float],
        current_throughput_by_gpu: dict[int, float],
    ) -> int:
        """Resolve any pending actions whose outcome window has elapsed.

        Call this periodically from the agent main loop.
        Returns the number of newly completed tuples.
        """
        now = time.time()
        resolved = 0

        expired_ids = [
            pid for pid, p in self._pending.items()
            if now - p.logged_at >= self._outcome_window_s
        ]

        for pid in expired_ids:
            pending = self._pending.pop(pid)
            t = self._tuples.get(pid)
            if t is None:
                continue

            energy_after = current_energy_by_gpu.get(pending.gpu_index, pending.energy_snapshot)
            throughput_after = current_throughput_by_gpu.get(pending.gpu_index, pending.throughput_snapshot)

            actual_savings = 0.0
            if pending.energy_snapshot > 0:
                actual_savings = (1.0 - energy_after / pending.energy_snapshot) * 100.0

            outcome = ActionOutcome(
                energy_delta_j_before=pending.energy_snapshot,
                energy_delta_j_after=energy_after,
                throughput_before=pending.throughput_snapshot,
                throughput_after=throughput_after,
                recommendation_status="applied" if t.action and t.action.source == "auto_tuner" else "pending",
                actual_savings_pct=actual_savings,
                observation_window_s=now - pending.logged_at,
            )

            reward = compute_energy_reward(
                energy_before_j=pending.energy_snapshot,
                energy_after_j=energy_after,
                throughput_before=pending.throughput_snapshot,
                throughput_after=throughput_after,
            )

            t.outcome = outcome
            t.reward = reward
            self._completed_count += 1
            resolved += 1

            self._wal_append(t)
            log.info(
                "Experience: outcome for %s — savings=%.1f%% reward=%.3f",
                t.id[:8], actual_savings, reward,
            )

        return resolved

    def record_completed(self, t: ExperienceTuple) -> None:
        """Record an externally observed, already-complete tuple.

        Used for actions whose outcome was measured outside the logger's own
        pending-window flow (e.g. autopilot commands with their own
        observation window).
        """
        self._tuples[t.id] = t
        self._completed_count += 1
        self._wal_append(t)

    def get_corpus_stats(self) -> dict:
        """Return summary statistics of the experience corpus."""
        stats: dict = {
            "total": len(self._tuples),
            "pending": len(self._pending),
            "completed": self._completed_count,
            "by_gpu_class": {},
            "by_workload_class": {},
            "by_action_type": {},
        }

        for t in self._tuples.values():
            if t.context:
                gpu = t.context.gpu_arch or "unknown"
                wl = t.context.workload_class or "unknown"
                stats["by_gpu_class"][gpu] = stats["by_gpu_class"].get(gpu, 0) + 1
                stats["by_workload_class"][wl] = stats["by_workload_class"].get(wl, 0) + 1
            if t.action:
                at = t.action.action_type
                stats["by_action_type"][at] = stats["by_action_type"].get(at, 0) + 1

        return stats

    def iter_completed(self, gpu_class: Optional[str] = None) -> Iterator[ExperienceTuple]:
        """Yield completed experience tuples, optionally filtered by GPU class."""
        for t in self._tuples.values():
            if not t.is_complete():
                continue
            if gpu_class and t.context and t.context.gpu_arch != gpu_class:
                continue
            yield t

    def flush_to_cloud(self, endpoint: str, api_key: str) -> int:
        """Upload completed experience tuples to the fleet aggregation API.

        Returns the number of tuples successfully uploaded.
        """
        import requests

        completed = [asdict(t) for t in self.iter_completed()]
        if not completed:
            return 0

        uploaded = 0
        batch_size = 100
        for i in range(0, len(completed), batch_size):
            batch = completed[i:i + batch_size]
            try:
                resp = requests.post(
                    f"{endpoint}/api/agent/experience",
                    json={"experiences": batch},
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": api_key,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    uploaded += len(batch)
                else:
                    log.warning("Experience upload failed: %d %s", resp.status_code, resp.text[:200])
            except Exception as exc:
                log.warning("Experience upload error: %s", exc)
                break

        if uploaded:
            log.info("Experience: uploaded %d/%d tuples to cloud", uploaded, len(completed))

        return uploaded

    def load_from_wal(self) -> int:
        """Load experience tuples from the WAL file on startup.

        Returns the number of tuples loaded.
        """
        if not self._wal_path.exists():
            return 0

        loaded = 0
        try:
            with open(self._wal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        data = entry.get("row", entry)
                        t = self._deserialize_tuple(data)
                        if t:
                            self._tuples[t.id] = t
                            if t.is_complete():
                                self._completed_count += 1
                            elif t.action:
                                self._pending[t.id] = _PendingAction(
                                    tuple_id=t.id,
                                    logged_at=t.timestamp,
                                    gpu_index=t.gpu_index,
                                    energy_snapshot=0.0,
                                    throughput_snapshot=0.0,
                                )
                            loaded += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError as exc:
            log.warning("Failed to load experience WAL: %s", exc)

        log.info("Experience: loaded %d tuples from WAL", loaded)
        return loaded

    # ── Internal ─────────────────────────────────────────────────────────

    def _wal_append(self, t: ExperienceTuple) -> None:
        """Append a tuple to the JSONL WAL with flock protection."""
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._wal_path, "a") as f:
                if _HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    entry = {"ts": time.time(), "row": asdict(t)}
                    f.write(json.dumps(entry) + "\n")
                finally:
                    if _HAS_FCNTL:
                        fcntl.flock(f, fcntl.LOCK_UN)
        except OSError as exc:
            log.error("Experience WAL write failed: %s", exc)

    @staticmethod
    def _deserialize_tuple(data: dict) -> Optional[ExperienceTuple]:
        """Reconstruct an ExperienceTuple from a dict."""
        try:
            ctx_data = data.get("context")
            ctx = WorkloadContext(**ctx_data) if ctx_data else None

            act_data = data.get("action")
            act = ActionTaken(**act_data) if act_data else None

            out_data = data.get("outcome")
            out = ActionOutcome(**out_data) if out_data else None

            return ExperienceTuple(
                id=data.get("id", str(uuid.uuid4())),
                timestamp=data.get("timestamp", 0.0),
                machine_id=data.get("machine_id", ""),
                gpu_index=data.get("gpu_index", 0),
                context=ctx,
                action=act,
                outcome=out,
                reward=data.get("reward"),
            )
        except (TypeError, KeyError):
            return None
