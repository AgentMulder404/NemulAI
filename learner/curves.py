# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI
"""
Empirical power/throughput curves and knee picking.

The bandit explores seven fixed power-cap arms; this module fits what the
fleet has actually *measured* — throughput ratio and power ratio as a
function of cap fraction — and picks the knee analytically: the lowest cap
whose predicted throughput stays within tolerance. The bandit then only
fine-tunes around the knee instead of re-discovering physics one arm-pull
at a time.

Data source: completed ExperienceTuples (the corpus the experience logger
already maintains and the cloud already pools per GPU class).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("nemulai-curves")

# Bin width over cap fraction for aggregation
BIN_WIDTH = 0.05

# A bin needs this many observations to count toward a knee decision
MIN_BIN_SAMPLES = 3

# A curve needs this many distinct usable bins before we trust a knee
MIN_BINS_FOR_KNEE = 2

GLOBAL_WORKLOAD = "*"


@dataclass
class CurvePoint:
    """One aggregated observation bin on the cap-fraction axis."""

    cap_fraction: float       # bin center: applied cap / pre-action limit
    throughput_ratio: float   # median observed/baseline throughput
    power_ratio: float        # median observed/baseline power (energy proxy)
    sample_count: int


@dataclass
class FittedCurve:
    gpu_class: str
    workload_class: str
    points: list[CurvePoint] = field(default_factory=list)  # sorted by fraction asc

    @property
    def sample_count(self) -> int:
        return sum(p.sample_count for p in self.points)


@dataclass
class KneeResult:
    fraction: float
    predicted_throughput_ratio: float
    predicted_power_ratio: float
    sample_count: int
    confidence: float  # 0-1, grows with corpus size


def _median(values: list[float]) -> float:
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return 0.0
    mid = n // 2
    return vs[mid] if n % 2 else (vs[mid - 1] + vs[mid]) / 2.0


def _tuple_to_raw_point(t) -> Optional[tuple[float, float, float]]:
    """ExperienceTuple -> (cap_fraction, throughput_ratio, power_ratio).

    Only power-cap actions with a complete outcome and sane baselines count.
    """
    ctx, act, out = t.context, t.action, t.outcome
    if not (ctx and act and out):
        return None
    if act.action_type != "power_cap":
        return None
    limit = getattr(ctx, "power_limit_w", 0.0) or 0.0
    cap = getattr(act, "recommended_value", 0.0) or 0.0
    if limit <= 0 or cap <= 0:
        return None
    fraction = cap / limit
    if not (0.3 <= fraction <= 1.05):
        return None

    tp_before = getattr(out, "throughput_before", 0.0) or 0.0
    tp_after = getattr(out, "throughput_after", 0.0) or 0.0
    if tp_before <= 0:
        return None
    throughput_ratio = tp_after / tp_before

    e_before = getattr(out, "energy_delta_j_before", 0.0) or 0.0
    e_after = getattr(out, "energy_delta_j_after", 0.0) or 0.0
    if e_before <= 0:
        return None
    power_ratio = e_after / e_before

    # Discard physically implausible observations (noise, attribution errors)
    if not (0.0 < throughput_ratio <= 2.0 and 0.0 < power_ratio <= 2.0):
        return None

    return min(1.0, fraction), throughput_ratio, power_ratio


def fit_curve(tuples, gpu_class: str = "", workload_class: str = GLOBAL_WORKLOAD) -> FittedCurve:
    """Aggregate raw experience tuples into a binned curve."""
    bins: dict[int, list[tuple[float, float]]] = {}
    for t in tuples:
        raw = _tuple_to_raw_point(t)
        if raw is None:
            continue
        fraction, tp_ratio, pw_ratio = raw
        # epsilon guards float artifacts: 0.7/0.05 = 13.999... must bin as 14
        bins.setdefault(int((fraction + 1e-9) / BIN_WIDTH), []).append((tp_ratio, pw_ratio))

    points = []
    for bin_idx in sorted(bins):
        obs = bins[bin_idx]
        if len(obs) < MIN_BIN_SAMPLES:
            continue
        points.append(CurvePoint(
            cap_fraction=round((bin_idx + 0.5) * BIN_WIDTH, 3),
            throughput_ratio=round(_median([o[0] for o in obs]), 4),
            power_ratio=round(_median([o[1] for o in obs]), 4),
            sample_count=len(obs),
        ))

    return FittedCurve(gpu_class=gpu_class, workload_class=workload_class, points=points)


def knee_fraction(curve: FittedCurve, tolerance_pct: float) -> Optional[KneeResult]:
    """Pick the lowest cap fraction whose throughput holds within tolerance.

    Enforces monotone trust: scanning from the highest fraction downward,
    stop at the first bin that violates tolerance — bins below a violation
    are not trusted even if they look fine (likely workload mix noise).
    Returns None when the curve is too sparse to decide.
    """
    usable = [p for p in curve.points if p.sample_count >= MIN_BIN_SAMPLES]
    if len(usable) < MIN_BINS_FOR_KNEE:
        return None

    floor_ratio = 1.0 - tolerance_pct / 100.0
    best: Optional[CurvePoint] = None
    for p in sorted(usable, key=lambda p: p.cap_fraction, reverse=True):
        if p.throughput_ratio >= floor_ratio:
            best = p
        else:
            break

    if best is None or best.cap_fraction >= 0.975:
        # Either everything violates tolerance, or the knee is "no cap at all"
        return None

    total = curve.sample_count
    confidence = min(1.0, total / 50.0)

    return KneeResult(
        fraction=best.cap_fraction,
        predicted_throughput_ratio=best.throughput_ratio,
        predicted_power_ratio=best.power_ratio,
        sample_count=total,
        confidence=round(confidence, 3),
    )


class CurveLibrary:
    """Curves per (gpu_class, workload_class) with fallback to per-GPU global."""

    def __init__(self):
        self._curves: dict[tuple[str, str], FittedCurve] = {}

    def fit_from_corpus(self, tuples) -> int:
        """(Re)fit all curves from an iterable of completed experience tuples.

        Returns the number of curves with at least one usable bin.
        """
        grouped: dict[tuple[str, str], list] = {}
        all_tuples = list(tuples)
        for t in all_tuples:
            ctx = t.context
            if not ctx:
                continue
            gpu_class = getattr(ctx, "gpu_arch", "") or getattr(ctx, "gpu_name", "")
            workload = getattr(ctx, "workload_class", GLOBAL_WORKLOAD) or GLOBAL_WORKLOAD
            grouped.setdefault((gpu_class, workload), []).append(t)
            grouped.setdefault((gpu_class, GLOBAL_WORKLOAD), []).append(t)

        self._curves = {}
        for (gpu_class, workload), ts in grouped.items():
            curve = fit_curve(ts, gpu_class, workload)
            if curve.points:
                self._curves[(gpu_class, workload)] = curve

        return len(self._curves)

    def get(self, gpu_class: str, workload_class: str = GLOBAL_WORKLOAD) -> Optional[FittedCurve]:
        return (
            self._curves.get((gpu_class, workload_class))
            or self._curves.get((gpu_class, GLOBAL_WORKLOAD))
        )

    def recommend_fraction(
        self,
        gpu_class: str,
        workload_class: str,
        tolerance_pct: float = 10.0,
    ) -> Optional[KneeResult]:
        """Knee for this context, falling back to the GPU's global curve."""
        for wl in (workload_class, GLOBAL_WORKLOAD):
            curve = self._curves.get((gpu_class, wl))
            if curve:
                knee = knee_fraction(curve, tolerance_pct)
                if knee:
                    return knee
        return None
