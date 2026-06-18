# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Roofline-based GPU ranking for profiled models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from efficiency.gpu_specs import (
    ArchSpec,
    ModelProfile,
    GPU_ARCHITECTURES,
    MODEL_PROFILES,
)
from efficiency.hardware_match import HardwareMatchScorer

log = logging.getLogger("nemulai-intel")

try:
    from efficiency.cloud_detect import GPU_HOURLY_RATES
except ImportError:
    GPU_HOURLY_RATES = {}


@dataclass
class GPUEstimate:
    gpu_name: str
    family: str
    score: float
    joules_per_tflop: float
    effective_tflops: float
    cost_per_hr: float
    recommendation: str


@dataclass
class EstimationResult:
    model_tag: str
    model_profile: ModelProfile
    gpu_rankings: list[GPUEstimate] = field(default_factory=list)
    best_gpu: str = ""
    worst_gpu: str = ""
    efficiency_spread_pct: float = 0.0
    estimated_at: float = 0.0


class BenchmarkEstimator:
    def __init__(self):
        self._scorer = HardwareMatchScorer()

    def estimate(self, profile: ModelProfile, top_n: int = 0) -> EstimationResult:
        was_registered = profile.tag in MODEL_PROFILES
        if not was_registered:
            MODEL_PROFILES[profile.tag] = profile

        try:
            raw_rankings = self._scorer.rank_gpus_for_model(profile.tag)
        finally:
            if not was_registered:
                MODEL_PROFILES.pop(profile.tag, None)

        gpu_rankings = []
        for entry in raw_rankings:
            arch_name = entry["arch_name"]
            spec = GPU_ARCHITECTURES.get(arch_name)
            if not spec:
                continue

            effective_tflops = spec.roofline_tflops(
                profile.math_intensity,
                profile.typical_util_mid,
                profile.precision,
            )

            gpu_rankings.append(GPUEstimate(
                gpu_name=arch_name,
                family=entry.get("family", spec.family),
                score=entry["score"],
                joules_per_tflop=entry["joules_per_tflop"],
                effective_tflops=round(effective_tflops, 2),
                cost_per_hr=GPU_HOURLY_RATES.get(arch_name, 0.0),
                recommendation=entry.get("recommendation", ""),
            ))

        if top_n > 0:
            gpu_rankings = gpu_rankings[:top_n]

        best_gpu = gpu_rankings[0].gpu_name if gpu_rankings else ""
        worst_gpu = gpu_rankings[-1].gpu_name if gpu_rankings else ""

        spread = 0.0
        if len(gpu_rankings) >= 2:
            best_jpt = gpu_rankings[0].joules_per_tflop
            worst_jpt = gpu_rankings[-1].joules_per_tflop
            if worst_jpt > 0:
                spread = round((1.0 - best_jpt / worst_jpt) * 100, 1)

        return EstimationResult(
            model_tag=profile.tag,
            model_profile=profile,
            gpu_rankings=gpu_rankings,
            best_gpu=best_gpu,
            worst_gpu=worst_gpu,
            efficiency_spread_pct=spread,
            estimated_at=time.time(),
        )

    def estimate_pair(
        self, profile: ModelProfile, gpu_name: str
    ) -> Optional[GPUEstimate]:
        spec = GPU_ARCHITECTURES.get(gpu_name)
        if not spec:
            return None

        was_registered = profile.tag in MODEL_PROFILES
        if not was_registered:
            MODEL_PROFILES[profile.tag] = profile

        try:
            result = self._scorer.score(gpu_name, profile.tag)
        finally:
            if not was_registered:
                MODEL_PROFILES.pop(profile.tag, None)

        if not result:
            return None

        effective_tflops = spec.roofline_tflops(
            profile.math_intensity,
            profile.typical_util_mid,
            profile.precision,
        )

        return GPUEstimate(
            gpu_name=gpu_name,
            family=spec.family,
            score=result.score,
            joules_per_tflop=result.current_jpt,
            effective_tflops=round(effective_tflops, 2),
            cost_per_hr=GPU_HOURLY_RATES.get(gpu_name, 0.0),
            recommendation=result.recommendation,
        )
