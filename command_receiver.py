"""
Command Receiver — polls the cloud for pending commands and executes them.

The agent polls GET /api/agent/commands every COMMAND_POLL_INTERVAL seconds.
When a command is received (e.g., apply_power_cap), it validates params,
executes via the appropriate module, and reports the result back.

Autopilot guardrail: every applied power cap opens an observation window
(default 300s, cloud policy can override per command). The agent loop feeds
per-GPU samples via record_sample(); when the window closes,
check_observations() compares GPU utilization against the pre-apply baseline
and auto-rolls-back the cap if it dropped more than the configured tolerance,
reporting the rollback (or the measured savings) to the cloud.

Safety: validates power cap between 100W and TDP, requires COMMAND_POLL_ENABLED.
"""
from __future__ import annotations

import json
import logging
import socket
import time
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_OBSERVATION_WINDOW_S = 300.0
DEFAULT_THROUGHPUT_TOLERANCE_PCT = 10.0

# Minimum samples inside the window required to judge a regression — fewer
# means we can't tell, so we leave the cap in place and report low confidence
MIN_OBSERVATION_SAMPLES = 3

# How much pre-apply history feeds the baseline
BASELINE_LOOKBACK_S = 120.0


@dataclass
class PendingObservation:
    """An applied power cap being watched for throughput regression."""

    command_id: str
    gpu_index: int
    watts: float              # action magnitude: W for power caps, max MHz for clock locks
    prev_limit_w: float
    applied_at: float
    window_s: float
    tolerance_pct: float
    baseline_util_pct: float
    baseline_power_w: float
    action_type: str = "power_cap"  # power_cap | clock_lock
    baseline_throughput: float = 0.0  # app-level tokens/s when available
    samples: list = field(default_factory=list)  # (ts, util_pct, power_w, throughput) after apply

    @property
    def deadline(self) -> float:
        return self.applied_at + self.window_s


@dataclass
class ObservationOutcome:
    """Result of a closed observation window — consumed by the agent loop
    for experience logging."""

    command_id: str
    gpu_index: int
    watts: float
    prev_limit_w: float
    rolled_back: bool
    baseline_util_pct: float
    observed_util_pct: float
    util_drop_pct: float
    baseline_power_w: float
    observed_power_w: float
    actual_savings_pct: float
    window_s: float
    sample_count: int
    # True-throughput signal (0.0 when no app-level source was configured)
    baseline_throughput: float = 0.0
    observed_throughput: float = 0.0
    throughput_drop_pct: float = 0.0
    regression_signal: str = "utilization"  # which signal judged the regression


class CommandReceiver:
    """Polls cloud for pending commands and executes them safely."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        machine_id: str,
        dry_run: bool = False,
        base_interval: float = 60.0,
        max_interval: float = 300.0,
    ):
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        self._base = f"{parsed.scheme}://{parsed.netloc}"
        self._api_key = api_key
        self._machine_id = machine_id
        self._dry_run = dry_run
        self._last_poll: float = 0.0
        self._base_interval = base_interval
        self._max_interval = max_interval
        self._current_interval = base_interval
        self._empty_polls: int = 0

        # Autopilot observation state
        self._recent: dict[int, deque] = {}  # gpu_index -> (ts, util_pct, power_w)
        self._observations: list[PendingObservation] = []

    @property
    def poll_interval(self) -> float:
        """Current adaptive poll interval in seconds."""
        return self._current_interval

    def poll_and_execute(self) -> int:
        """Poll for pending commands and execute them. Returns count executed."""
        commands = self._fetch_commands()
        if not commands:
            self._empty_polls += 1
            if self._empty_polls >= 3:
                self._current_interval = min(
                    self._max_interval,
                    self._current_interval * 2,
                )
            return 0

        self._empty_polls = 0
        self._current_interval = self._base_interval

        executed = 0
        for cmd in commands:
            success, message = self._execute(cmd)
            self._report_result(
                cmd["id"],
                success=success,
                message=message,
            )
            executed += 1

        self._last_poll = time.time()
        return executed

    def _fetch_commands(self) -> list[dict]:
        url = f"{self._base}/api/agent/commands?machine_id={self._machine_id}"
        req = urllib.request.Request(
            url,
            headers={"X-API-Key": self._api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                return body.get("commands", [])
        except Exception as exc:
            logger.debug("Command poll failed (non-fatal): %s", exc)
            return []

    def _execute(self, cmd: dict) -> tuple[bool, str]:
        cmd_type = cmd.get("command_type", "")
        params = cmd.get("params", {})

        handler = self._handlers.get(cmd_type)
        if not handler:
            return False, f"Unknown command type: {cmd_type}"

        # Snapshot the current limit before changing it, so a regression can
        # roll back to what was actually in effect (not just the GPU default)
        prev_limit_w = 0.0
        if cmd_type == "apply_power_cap" and not self._dry_run:
            prev_limit_w = self._read_power_limit(params.get("gpu_index"))

        success, message = handler(params)

        if success and not self._dry_run and cmd_type in ("apply_power_cap", "apply_clock_lock"):
            action_type = "clock_lock" if cmd_type == "apply_clock_lock" else "power_cap"
            self._open_observation(cmd.get("id", ""), params, prev_limit_w, action_type)

        return success, message

    @staticmethod
    def _read_power_limit(gpu_index) -> float:
        if not isinstance(gpu_index, int):
            return 0.0
        try:
            from efficiency.power_control import get_power_limit
            return float(get_power_limit(gpu_index))
        except Exception:
            try:
                from efficiency.power_control import get_default_power_limit
                return float(get_default_power_limit(gpu_index))
            except Exception:
                return 0.0

    @property
    def _handlers(self) -> dict:
        return {
            "apply_power_cap": self._apply_power_cap,
            "rollback_power_cap": self._rollback_power_cap,
            "apply_clock_lock": self._apply_clock_lock,
            "rollback_clock_lock": self._rollback_clock_lock,
            "set_precision": self._set_precision,
        }

    def _apply_power_cap(self, params: dict) -> tuple[bool, str]:
        gpu_index = params.get("gpu_index")
        watts = params.get("watts")

        if gpu_index is None or watts is None:
            return False, "Missing gpu_index or watts"

        if not isinstance(gpu_index, int) or gpu_index < 0 or gpu_index > 31:
            return False, f"Invalid gpu_index: {gpu_index}"

        if not isinstance(watts, (int, float)) or watts < 100 or watts > 1200:
            return False, f"Power cap {watts}W out of safe range (100-1200W)"

        if self._dry_run:
            logger.info("Command: would set GPU %d power cap to %dW (dry run)", gpu_index, watts)
            return True, f"Dry run: would set GPU {gpu_index} to {watts}W"

        try:
            from efficiency.power_control import set_power_limit
            ok = set_power_limit(gpu_index, int(watts), quiet=True)
            if ok:
                logger.info("Command: set GPU %d power cap to %dW", gpu_index, watts)
                return True, f"Power cap set to {watts}W on GPU {gpu_index}"
            else:
                return False, f"set_power_limit returned False for GPU {gpu_index}"
        except Exception as exc:
            logger.warning("Command: power cap failed GPU %d: %s", gpu_index, exc)
            return False, str(exc)

    def _rollback_power_cap(self, params: dict) -> tuple[bool, str]:
        gpu_index = params.get("gpu_index", 0)

        if not isinstance(gpu_index, int) or gpu_index < 0 or gpu_index > 31:
            return False, f"Invalid gpu_index: {gpu_index}"

        if self._dry_run:
            return True, f"Dry run: would reset GPU {gpu_index} to default"

        try:
            from efficiency.power_control import get_default_power_limit, set_power_limit
            default_w = get_default_power_limit(gpu_index)
            ok = set_power_limit(gpu_index, default_w, quiet=True)
            if ok:
                logger.info("Command: reset GPU %d to default %dW", gpu_index, default_w)
                return True, f"Power cap reset to default {default_w}W on GPU {gpu_index}"
            return False, f"Rollback failed for GPU {gpu_index}"
        except Exception as exc:
            return False, str(exc)

    def _apply_clock_lock(self, params: dict) -> tuple[bool, str]:
        gpu_index = params.get("gpu_index")
        min_mhz = params.get("min_mhz")
        max_mhz = params.get("max_mhz")

        # Fleet-portable form: a fraction of this GPU's max boost clock,
        # resolved locally so the same command works on any GPU model
        if max_mhz is None and params.get("sm_fraction") is not None:
            try:
                fraction = float(params["sm_fraction"])
            except (TypeError, ValueError):
                return False, f"Invalid sm_fraction: {params.get('sm_fraction')}"
            if not (0.3 <= fraction <= 1.0):
                return False, f"sm_fraction {fraction} out of safe range (0.3-1.0)"
            try:
                from efficiency.power_control import get_max_sm_clock
                boost = get_max_sm_clock(gpu_index if isinstance(gpu_index, int) else 0)
            except Exception:
                boost = 0
            if boost <= 0:
                return False, "Cannot resolve sm_fraction: max SM clock unavailable"
            max_mhz = int(boost * fraction)
            # Mutate in place so _open_observation sees the resolved magnitude
            params["max_mhz"] = max_mhz

        if gpu_index is None or max_mhz is None:
            return False, "Missing gpu_index or max_mhz"
        if min_mhz is None:
            min_mhz = max_mhz

        if not isinstance(gpu_index, int) or gpu_index < 0 or gpu_index > 31:
            return False, f"Invalid gpu_index: {gpu_index}"

        for label, mhz in (("min_mhz", min_mhz), ("max_mhz", max_mhz)):
            if not isinstance(mhz, (int, float)) or mhz < 210 or mhz > 3000:
                return False, f"{label} {mhz} out of safe range (210-3000 MHz)"
        if min_mhz > max_mhz:
            return False, f"min_mhz {min_mhz} > max_mhz {max_mhz}"

        if self._dry_run:
            return True, f"Dry run: would lock GPU {gpu_index} SM clocks to {min_mhz}-{max_mhz} MHz"

        try:
            from efficiency.power_control import set_gpu_clock_lock
            ok = set_gpu_clock_lock(gpu_index, int(min_mhz), int(max_mhz), quiet=True)
            if ok:
                logger.info("Command: locked GPU %d SM clocks to %d-%d MHz", gpu_index, min_mhz, max_mhz)
                return True, f"SM clocks locked to {min_mhz}-{max_mhz} MHz on GPU {gpu_index}"
            return False, f"set_gpu_clock_lock returned False for GPU {gpu_index}"
        except Exception as exc:
            logger.warning("Command: clock lock failed GPU %d: %s", gpu_index, exc)
            return False, str(exc)

    def _rollback_clock_lock(self, params: dict) -> tuple[bool, str]:
        gpu_index = params.get("gpu_index", 0)

        if not isinstance(gpu_index, int) or gpu_index < 0 or gpu_index > 31:
            return False, f"Invalid gpu_index: {gpu_index}"

        if self._dry_run:
            return True, f"Dry run: would remove SM clock lock on GPU {gpu_index}"

        try:
            from efficiency.power_control import reset_gpu_clock_lock
            ok = reset_gpu_clock_lock(gpu_index, quiet=True)
            if ok:
                logger.info("Command: removed SM clock lock on GPU %d", gpu_index)
                return True, f"SM clock lock removed on GPU {gpu_index}"
            return False, f"Clock lock reset failed for GPU {gpu_index}"
        except Exception as exc:
            return False, str(exc)

    def _set_precision(self, params: dict) -> tuple[bool, str]:
        gpu_index = params.get("gpu_index", 0)
        precision = params.get("precision", "")

        if not isinstance(gpu_index, int) or gpu_index < 0 or gpu_index > 31:
            return False, f"Invalid gpu_index: {gpu_index}"

        if precision not in ("fp16", "bf16", "fp32", "tf32"):
            return False, f"Unsupported precision: {precision}"

        if self._dry_run:
            return True, f"Dry run: would set GPU {gpu_index} to {precision}"

        # Precision switching is advisory — logged for the user to apply manually
        logger.info("Command: precision switch GPU %d → %s (advisory)", gpu_index, precision)
        return True, f"Precision switch to {precision} logged for GPU {gpu_index} (apply in training config)"

    # ── Autopilot observation window ─────────────────────────────────────

    def record_sample(
        self,
        gpu_index: int,
        util_pct: float,
        power_w: float,
        throughput: float = 0.0,
    ) -> None:
        """Feed one per-GPU metrics sample. Call every collection cycle.

        throughput is the app-level rate (tokens/s) when a ThroughputProbe
        source is configured; 0.0 means "no true-throughput signal" and the
        observation falls back to the utilization proxy.
        """
        now = time.time()
        buf = self._recent.setdefault(gpu_index, deque(maxlen=256))
        buf.append((now, float(util_pct), float(power_w), float(throughput)))

        for obs in self._observations:
            if obs.gpu_index == gpu_index:
                obs.samples.append((now, float(util_pct), float(power_w), float(throughput)))

    def check_observations(self) -> list[ObservationOutcome]:
        """Close any observation windows past their deadline.

        On regression (utilization dropped more than tolerance vs. the
        pre-apply baseline) the cap is rolled back to the previous limit and
        the rollback reported. Otherwise the measured power savings are
        reported. Returns outcomes for experience logging.
        """
        now = time.time()
        outcomes: list[ObservationOutcome] = []
        still_open: list[PendingObservation] = []

        for obs in self._observations:
            if now < obs.deadline:
                still_open.append(obs)
                continue
            outcomes.append(self._close_observation(obs))

        self._observations = still_open
        return outcomes

    @property
    def open_observations(self) -> int:
        return len(self._observations)

    def _open_observation(
        self,
        command_id: str,
        params: dict,
        prev_limit_w: float,
        action_type: str = "power_cap",
    ) -> None:
        gpu_index = params.get("gpu_index")
        if action_type == "clock_lock":
            magnitude = params.get("max_mhz", 0)
        else:
            magnitude = params.get("watts", 0)
        if not isinstance(gpu_index, int):
            return

        window_s = float(params.get("observation_window_s", DEFAULT_OBSERVATION_WINDOW_S))
        tolerance = float(params.get("throughput_tolerance_pct", DEFAULT_THROUGHPUT_TOLERANCE_PCT))
        window_s = max(30.0, min(3600.0, window_s))
        tolerance = max(1.0, min(50.0, tolerance))

        baseline_util, baseline_power, baseline_tps = self._baseline(gpu_index)

        self._observations.append(PendingObservation(
            command_id=command_id,
            gpu_index=gpu_index,
            watts=float(magnitude),
            prev_limit_w=prev_limit_w,
            applied_at=time.time(),
            window_s=window_s,
            tolerance_pct=tolerance,
            baseline_util_pct=baseline_util,
            baseline_power_w=baseline_power,
            action_type=action_type,
            baseline_throughput=baseline_tps,
        ))
        logger.info(
            "Autopilot: observing GPU %d for %.0fs after %s "
            "(baseline %s, tolerance %.0f%%)",
            gpu_index, window_s,
            (f"{magnitude:.0f} MHz clock lock" if action_type == "clock_lock"
             else f"{magnitude}W cap"),
            f"{baseline_tps:.0f} tok/s" if baseline_tps > 0 else f"util {baseline_util:.0f}%",
            tolerance,
        )

    def _baseline(self, gpu_index: int) -> tuple[float, float, float]:
        buf = self._recent.get(gpu_index)
        if not buf:
            return 0.0, 0.0, 0.0
        cutoff = time.time() - BASELINE_LOOKBACK_S
        recent = [(u, p, t) for ts, u, p, t in buf if ts >= cutoff]
        if not recent:
            recent = [(u, p, t) for _, u, p, t in list(buf)[-10:]]
        if not recent:
            return 0.0, 0.0, 0.0
        utils = [u for u, _, _ in recent]
        powers = [p for _, p, _ in recent]
        # Throughput baseline only counts samples where a signal was present
        tps_samples = [t for _, _, t in recent if t > 0]
        baseline_tps = sum(tps_samples) / len(tps_samples) if tps_samples else 0.0
        return sum(utils) / len(utils), sum(powers) / len(powers), baseline_tps

    def _close_observation(self, obs: PendingObservation) -> ObservationOutcome:
        sample_count = len(obs.samples)
        observed_util = 0.0
        observed_power = 0.0
        observed_tps = 0.0
        tps_count = 0
        if sample_count > 0:
            observed_util = sum(u for _, u, _, _ in obs.samples) / sample_count
            observed_power = sum(p for _, _, p, _ in obs.samples) / sample_count
            tps_samples = [t for _, _, _, t in obs.samples if t > 0]
            tps_count = len(tps_samples)
            if tps_samples:
                observed_tps = sum(tps_samples) / tps_count

        util_drop_pct = 0.0
        if obs.baseline_util_pct > 0 and sample_count >= MIN_OBSERVATION_SAMPLES:
            util_drop_pct = (
                (obs.baseline_util_pct - observed_util) / obs.baseline_util_pct * 100.0
            )

        throughput_drop_pct = 0.0
        have_tps = obs.baseline_throughput > 0 and tps_count >= MIN_OBSERVATION_SAMPLES
        if have_tps:
            throughput_drop_pct = (
                (obs.baseline_throughput - observed_tps) / obs.baseline_throughput * 100.0
            )

        # True throughput is authoritative when present — NVML utilization
        # can't see memory-stalled work and misses real regressions
        if have_tps:
            regression = throughput_drop_pct > obs.tolerance_pct
            signal = "throughput"
            drop_for_log = throughput_drop_pct
        else:
            regression = util_drop_pct > obs.tolerance_pct
            signal = "utilization"
            drop_for_log = util_drop_pct

        actual_savings_pct = 0.0
        if obs.baseline_power_w > 0 and observed_power > 0:
            actual_savings_pct = (
                (obs.baseline_power_w - observed_power) / obs.baseline_power_w * 100.0
            )

        rolled_back = False
        if regression:
            if obs.action_type == "clock_lock":
                rolled_back = self._rollback_clock(obs.gpu_index)
            else:
                rolled_back = self._rollback_to(obs.gpu_index, obs.prev_limit_w)
            logger.warning(
                "Autopilot: GPU %d %s dropped %.1f%% (> %.0f%% tolerance) "
                "after %s — %s",
                obs.gpu_index, signal, drop_for_log, obs.tolerance_pct,
                (f"{obs.watts:.0f} MHz clock lock" if obs.action_type == "clock_lock"
                 else f"{obs.watts:.0f}W cap"),
                "rolled back" if rolled_back else "ROLLBACK FAILED",
            )
        else:
            logger.info(
                "Autopilot: GPU %d cap %sW held — %s, power saving %.1f%% "
                "(%d samples over %.0fs)",
                obs.gpu_index, obs.watts,
                (f"throughput {obs.baseline_throughput:.0f}→{observed_tps:.0f} tok/s"
                 if have_tps else f"util {obs.baseline_util_pct:.0f}%→{observed_util:.0f}%"),
                actual_savings_pct, sample_count, obs.window_s,
            )

        observation_payload = {
            "baseline_util_pct": round(obs.baseline_util_pct, 1),
            "observed_util_pct": round(observed_util, 1),
            "util_drop_pct": round(util_drop_pct, 1),
            # Watts let the cloud compute *measured* reclaimed capacity exactly
            "baseline_power_w": round(obs.baseline_power_w, 1),
            "observed_power_w": round(observed_power, 1),
            "window_s": round(obs.window_s, 0),
            "sample_count": sample_count,
            "regression_signal": signal,
        }
        if have_tps:
            observation_payload["baseline_throughput"] = round(obs.baseline_throughput, 1)
            observation_payload["observed_throughput"] = round(observed_tps, 1)
            observation_payload["throughput_drop_pct"] = round(throughput_drop_pct, 1)
        if not regression:
            observation_payload["actual_savings_pct"] = round(actual_savings_pct, 1)

        if regression:
            action_desc = (
                f"{obs.watts:.0f} MHz clock lock" if obs.action_type == "clock_lock"
                else f"{obs.watts:.0f}W cap"
            )
            message = (
                f"Auto-rolled back {action_desc} on GPU {obs.gpu_index}: "
                f"{signal} dropped {drop_for_log:.1f}% (tolerance {obs.tolerance_pct:.0f}%)"
            )
        elif sample_count < MIN_OBSERVATION_SAMPLES:
            message = (
                f"Observation window closed with only {sample_count} samples — "
                f"cap left in place (low confidence)"
            )
        else:
            held = "Clock lock" if obs.action_type == "clock_lock" else "Cap"
            message = (
                f"{held} held on GPU {obs.gpu_index}: power saving {actual_savings_pct:.1f}%, "
                f"{signal} within tolerance"
            )

        self._report_result(
            obs.command_id,
            success=not regression,
            message=message,
            rolled_back=regression and rolled_back,
            observation=observation_payload,
        )

        return ObservationOutcome(
            command_id=obs.command_id,
            gpu_index=obs.gpu_index,
            watts=obs.watts,
            prev_limit_w=obs.prev_limit_w,
            rolled_back=regression and rolled_back,
            baseline_util_pct=obs.baseline_util_pct,
            observed_util_pct=observed_util,
            util_drop_pct=util_drop_pct,
            baseline_power_w=obs.baseline_power_w,
            observed_power_w=observed_power,
            actual_savings_pct=actual_savings_pct,
            window_s=obs.window_s,
            sample_count=sample_count,
            baseline_throughput=obs.baseline_throughput,
            observed_throughput=observed_tps,
            throughput_drop_pct=throughput_drop_pct,
            regression_signal=signal,
        )

    def _rollback_clock(self, gpu_index: int) -> bool:
        try:
            from efficiency.power_control import reset_gpu_clock_lock
            return bool(reset_gpu_clock_lock(gpu_index, quiet=True))
        except Exception as exc:
            logger.warning("Autopilot clock-lock rollback failed for GPU %d: %s", gpu_index, exc)
            return False

    def _rollback_to(self, gpu_index: int, prev_limit_w: float) -> bool:
        try:
            from efficiency.power_control import set_power_limit, get_default_power_limit
            target = int(prev_limit_w) if prev_limit_w > 0 else get_default_power_limit(gpu_index)
            return bool(set_power_limit(gpu_index, target, quiet=True))
        except Exception as exc:
            logger.warning("Autopilot rollback failed for GPU %d: %s", gpu_index, exc)
            return False

    def _report_result(
        self,
        command_id: str,
        success: bool,
        message: str,
        rolled_back: bool = False,
        observation: Optional[dict] = None,
    ) -> None:
        url = f"{self._base}/api/agent/commands/{command_id}/result"
        body: dict = {
            "success": success,
            "message": message,
            "hostname": socket.gethostname(),
            "machine_id": self._machine_id,
        }
        if rolled_back:
            body["rolled_back"] = True
        if observation:
            body["observation"] = observation
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", "X-API-Key": self._api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            logger.debug("Command result report failed: %s", exc)
