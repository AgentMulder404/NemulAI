# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI
"""
Phase-aware dynamic clock control.

Inference alternates prefill (compute-bound: wants full SM clocks) and
decode (memory-bound: SM clocks mostly wait on HBM) on a timescale of
seconds; training alternates step / eval / checkpoint. A single static cap
is the average of two wrong answers.

PhaseDetector classifies each GPU per sample from DCGM activity counters
(DRAM_ACTIVE vs SM activity — ground truth, unlike NVML utilization) with
hysteresis so transient blips don't flap the clocks.

DynamicClockTuner reacts to *stable* phase changes: memory-bound → lock SM
clocks to a fraction of boost (memory clocks untouched, throughput ~flat,
power −25-35%); compute-bound → release the lock. A minimum dwell time
prevents oscillation. Disabled by default; enable with
DYNAMIC_CLOCKS_ENABLED=1.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("nemulai-phase")

PHASE_COMPUTE = "compute_bound"
PHASE_MEMORY = "memory_bound"
PHASE_IDLE = "idle"

# Raw classification thresholds
IDLE_UTIL_PCT = 5.0
MEMORY_DRAM_FLOOR = 0.25       # DRAM_ACTIVE below this is never memory-bound
MEMORY_DRAM_TO_SM_RATIO = 0.7  # DRAM activity must be a large share of SM activity

# Consecutive identical raw classifications required to switch the stable phase
DEFAULT_HYSTERESIS_SAMPLES = 3


@dataclass
class PhaseChange:
    gpu_index: int
    previous: str
    current: str
    at: float


class PhaseDetector:
    """Per-GPU workload phase with hysteresis."""

    def __init__(self, hysteresis_samples: int = DEFAULT_HYSTERESIS_SAMPLES):
        self._hysteresis = max(1, hysteresis_samples)
        self._raw: dict[int, deque] = {}
        self._stable: dict[int, str] = {}

    @staticmethod
    def classify(activity: dict, util_pct: float) -> str:
        """Classify one sample from DCGM activity (or NVML fallback) values.

        activity keys (0.0-1.0): fp32_activity (SM proxy), tensor_activity,
        fp16_activity, memory_activity (DRAM).
        """
        if util_pct < IDLE_UTIL_PCT:
            return PHASE_IDLE

        dram = activity.get("memory_activity", 0.0)
        sm = max(
            activity.get("fp32_activity", 0.0),
            activity.get("tensor_activity", 0.0),
            activity.get("fp16_activity", 0.0),
        )

        if dram >= MEMORY_DRAM_FLOOR and dram >= sm * MEMORY_DRAM_TO_SM_RATIO:
            return PHASE_MEMORY
        return PHASE_COMPUTE

    def update(self, gpu_index: int, activity: dict, util_pct: float) -> Optional[PhaseChange]:
        """Feed one sample; returns a PhaseChange when the stable phase flips."""
        raw = self.classify(activity, util_pct)
        buf = self._raw.setdefault(gpu_index, deque(maxlen=self._hysteresis))
        buf.append(raw)

        if len(buf) < self._hysteresis or any(r != raw for r in buf):
            return None

        previous = self._stable.get(gpu_index)
        if previous == raw:
            return None

        self._stable[gpu_index] = raw
        return PhaseChange(
            gpu_index=gpu_index,
            previous=previous or "unknown",
            current=raw,
            at=time.time(),
        )

    def stable_phase(self, gpu_index: int) -> str:
        return self._stable.get(gpu_index, "unknown")


class DynamicClockTuner:
    """Applies/releases SM clock locks as the stable phase changes."""

    def __init__(
        self,
        memory_fraction: float = 0.65,
        min_dwell_s: float = 15.0,
        dry_run: bool = False,
    ):
        self._fraction = max(0.3, min(1.0, memory_fraction))
        self._min_dwell = max(5.0, min_dwell_s)
        self._dry_run = dry_run
        self._locked: dict[int, bool] = {}        # gpu -> lock currently applied by us
        self._last_switch: dict[int, float] = {}
        self.switch_count = 0

    def on_phase(self, gpu_index: int, phase: str) -> Optional[str]:
        """React to a stable phase. Returns the action taken ('lock'/'release')
        or None when nothing changed (already in the right state or dwelling)."""
        want_lock = phase == PHASE_MEMORY
        have_lock = self._locked.get(gpu_index, False)
        if want_lock == have_lock:
            return None

        now = time.time()
        if now - self._last_switch.get(gpu_index, 0.0) < self._min_dwell:
            return None

        if want_lock:
            ok = self._apply_lock(gpu_index)
            action = "lock"
        else:
            ok = self._release_lock(gpu_index)
            action = "release"

        if not ok:
            return None

        self._locked[gpu_index] = want_lock
        self._last_switch[gpu_index] = now
        self.switch_count += 1
        log.info(
            "Phase tuner: GPU %d %s (%s phase)%s",
            gpu_index,
            "SM clocks locked to %.0f%% of boost" % (self._fraction * 100)
            if want_lock else "SM clock lock released",
            phase,
            " [dry run]" if self._dry_run else "",
        )
        return action

    def shutdown(self) -> None:
        """Release every lock this tuner applied."""
        for gpu_index, locked in list(self._locked.items()):
            if locked:
                self._release_lock(gpu_index)
                self._locked[gpu_index] = False

    # ── Actuation ─────────────────────────────────────────────────────────

    def _apply_lock(self, gpu_index: int) -> bool:
        if self._dry_run:
            return True
        try:
            from efficiency.power_control import get_max_sm_clock, set_gpu_clock_lock
            boost = get_max_sm_clock(gpu_index)
            if boost <= 0:
                log.debug("Phase tuner: max SM clock unavailable for GPU %d", gpu_index)
                return False
            target = int(boost * self._fraction)
            return bool(set_gpu_clock_lock(gpu_index, target, target, quiet=True))
        except Exception as exc:
            log.debug("Phase tuner lock failed for GPU %d: %s", gpu_index, exc)
            return False

    def _release_lock(self, gpu_index: int) -> bool:
        if self._dry_run:
            return True
        try:
            from efficiency.power_control import reset_gpu_clock_lock
            return bool(reset_gpu_clock_lock(gpu_index, quiet=True))
        except Exception as exc:
            log.debug("Phase tuner release failed for GPU %d: %s", gpu_index, exc)
            return False
