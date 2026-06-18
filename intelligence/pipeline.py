# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Model Intelligence Pipeline — detect → profile → estimate → register."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from intelligence.detector import ModelDetector, DetectedModel
from intelligence.profiler import ModelProfiler, ProfileResult
from intelligence.estimator import BenchmarkEstimator, EstimationResult
from intelligence.quantization import QuantizationAdvisor
from intelligence.registry import ModelRegistry, RegistryEntry

log = logging.getLogger("nemulai-intel")


@dataclass
class PipelineResult:
    detected: int = 0
    profiled: int = 0
    estimated: int = 0
    registered: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    entries: list[RegistryEntry] = field(default_factory=list)


class IntelligencePipeline:
    def __init__(
        self,
        data_dir: Path,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self._detector = ModelDetector(session=session)
        self._profiler = ModelProfiler()
        self._estimator = BenchmarkEstimator()
        self._quant_advisor = QuantizationAdvisor()
        self._registry = ModelRegistry(data_dir, supabase_url, supabase_key)

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def run(
        self,
        limit: int = 20,
        min_downloads: int = 1000,
        min_confidence: float = 0.5,
    ) -> PipelineResult:
        start = time.time()
        result = PipelineResult()

        # 1. DETECT
        known = self._registry.known_tags()
        new_models = self._detector.detect_new(known, limit=limit, min_downloads=min_downloads)
        result.detected = len(new_models)
        log.info("Detected %d new models", result.detected)

        if not new_models:
            result.duration_s = round(time.time() - start, 2)
            return result

        # 2-4. PROFILE → ESTIMATE → REGISTER each model
        for model in new_models:
            try:
                entry = self._process_model(model, min_confidence)
                if entry:
                    result.profiled += 1
                    result.estimated += 1
                    result.registered += 1
                    result.entries.append(entry)
            except Exception as exc:
                msg = f"Error processing {model.model_id}: {exc}"
                log.warning(msg)
                result.errors.append(msg)

        # 5. Sync to Supabase
        synced = self._registry.sync_to_supabase()
        if synced:
            log.info("Synced %d models to Supabase", synced)

        result.duration_s = round(time.time() - start, 2)
        return result

    def run_single(self, model_id: str) -> Optional[RegistryEntry]:
        existing = self._registry.get(ModelDetector.normalize_tag(model_id))
        if existing:
            return existing

        detected = self._detector.fetch_model_info(model_id)
        if not detected:
            log.warning("Could not fetch model info for %s", model_id)
            return None

        return self._process_model(detected, min_confidence=0.0)

    def warm_start_bandit(self) -> int:
        try:
            from learner.bandit import EnergyBandit, POWER_CAP_ACTIONS
            from learner.experience_logger import (
                ExperienceTuple, WorkloadContext, ActionTaken, ActionOutcome,
            )
            from learner.reward import compute_energy_reward
            from learner.feature_encoder import classify_workload
        except ImportError:
            log.warning("Bandit module not available, skipping warm-start")
            return 0

        active = self._registry.list_all(status="active")
        if not active:
            active = self._registry.list_all(status="estimated")

        if not active:
            log.info("No models in registry to warm-start from")
            return 0

        tuples: list[ExperienceTuple] = []

        for entry in active:
            profile = entry.profile

            for ranking in entry.gpu_rankings[:5]:
                gpu_name = ranking.get("gpu_name", ranking.get("arch_name", ""))
                jpt = ranking.get("joules_per_tflop", 0)
                if not gpu_name or jpt <= 0:
                    continue

                # Synthesize experience for the optimal power cap
                best_fraction = 0.8 if profile.is_memory_bound else 0.9
                baseline_power = 300.0
                capped_power = baseline_power * best_fraction

                energy_before = baseline_power * 300
                energy_after = capped_power * 300
                throughput = 1.0

                context = WorkloadContext(
                    gpu_name=gpu_name,
                    gpu_arch=gpu_name,
                    workload_class=f"{profile.family.lower()}-inference",
                    utilization_gpu_pct=float(profile.typical_util_min + profile.typical_util_max) / 2,
                    utilization_memory_pct=50.0,
                    memory_pressure=0.5,
                    power_draw_w=baseline_power,
                    power_limit_w=baseline_power,
                    temperature_c=65.0,
                )

                action = ActionTaken(
                    action_type="power_cap",
                    source="model_intelligence",
                    recommended_value=capped_power,
                    current_value=baseline_power,
                    estimated_savings_pct=round((1 - best_fraction) * 100, 1),
                )

                outcome = ActionOutcome(
                    energy_delta_j_before=energy_before,
                    energy_delta_j_after=energy_after,
                    throughput_before=throughput,
                    throughput_after=throughput * 0.98,
                    recommendation_status="applied",
                    actual_savings_pct=round((1 - best_fraction) * 100, 1),
                    observation_window_s=300.0,
                )

                reward = compute_energy_reward(
                    energy_before_j=energy_before,
                    energy_after_j=energy_after,
                    throughput_before=throughput,
                    throughput_after=throughput * 0.98,
                )

                tuples.append(ExperienceTuple(
                    context=context,
                    action=action,
                    outcome=outcome,
                    reward=reward,
                    recorded_at=entry.updated_at or time.time(),
                ))

        log.info("Generated %d synthetic experience tuples for bandit warm-start", len(tuples))
        return len(tuples)

    def _process_model(
        self, detected: DetectedModel, min_confidence: float
    ) -> Optional[RegistryEntry]:
        # PROFILE
        profile_result = self._profiler.profile(detected)
        if profile_result.confidence < min_confidence:
            log.info(
                "Skipping %s: confidence %.2f < %.2f",
                detected.model_id, profile_result.confidence, min_confidence,
            )
            return None

        # ESTIMATE
        estimation = self._estimator.estimate(profile_result.profile, top_n=10)

        gpu_rankings = [
            {
                "gpu_name": g.gpu_name,
                "family": g.family,
                "score": g.score,
                "joules_per_tflop": g.joules_per_tflop,
                "effective_tflops": g.effective_tflops,
                "cost_per_hr": g.cost_per_hr,
            }
            for g in estimation.gpu_rankings
        ]

        # QUANTIZE
        quant_variants = []
        try:
            quant_result = self._quant_advisor.analyze(
                profile_result.profile,
                parameter_count=detected.parameter_count,
            )
            for qp in quant_result.variants:
                quant_variants.append({
                    "variant": qp.variant.name,
                    "precision": qp.variant.precision,
                    "bits_per_param": qp.variant.bits_per_param,
                    "model_size_gb": qp.model_size_gb,
                    "memory_reduction_pct": qp.memory_reduction_pct,
                    "throughput_change_pct": qp.estimated_throughput_change_pct,
                    "quality_impact": qp.estimated_quality_impact,
                    "best_gpu": qp.gpu_rankings[0]["gpu_name"] if qp.gpu_rankings else None,
                    "fits_on_count": len(qp.fits_on_gpus),
                })
            if quant_result.sweet_spot:
                log.info(
                    "Quantization sweet spot for %s: %s (quality=%s)",
                    detected.tag,
                    quant_result.sweet_spot.variant.name,
                    quant_result.sweet_spot.estimated_quality_impact,
                )
        except Exception as exc:
            log.warning("Quantization analysis failed for %s: %s", detected.tag, exc)

        # REGISTER
        entry = RegistryEntry(
            model_id=detected.model_id,
            tag=detected.tag,
            family=profile_result.profile.family,
            profile=profile_result.profile,
            source="huggingface",
            gpu_rankings=gpu_rankings,
            status="estimated",
            confidence=profile_result.confidence,
            parameter_count=detected.parameter_count,
            architecture=detected.architecture,
            downloads_30d=detected.downloads_30d,
            quantization_variants=quant_variants,
        )

        self._registry.register(entry)
        log.info(
            "Registered %s (family=%s, intensity=%.1f, confidence=%.2f, best_gpu=%s)",
            entry.tag, entry.family, entry.profile.math_intensity,
            entry.confidence, estimation.best_gpu,
        )

        return entry
