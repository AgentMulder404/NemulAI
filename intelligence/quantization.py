# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Quantization Advisor — analyze precision variants for GPU-model pairings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from efficiency.gpu_specs import (
    ArchSpec,
    ModelProfile,
    GPU_ARCHITECTURES,
    MODEL_PROFILES,
)
from intelligence.estimator import BenchmarkEstimator

log = logging.getLogger("nemulai-intel")


@dataclass(frozen=True)
class QuantizationVariant:
    name: str
    precision: str
    bits_per_param: float
    memory_multiplier: float
    throughput_multiplier: float
    quality_retention: float
    requires_transformer_engine: bool
    format_name: str


QUANTIZATION_VARIANTS: list[QuantizationVariant] = [
    QuantizationVariant("fp16",      "fp16", 16.0, 1.0,  1.0, 1.00, False, "native"),
    QuantizationVariant("bf16",      "bf16", 16.0, 1.0,  1.0, 1.00, False, "native"),
    QuantizationVariant("fp8",       "fp8",   8.0, 0.5,  2.0, 0.99, True,  "native"),
    QuantizationVariant("int8",      "int8",  8.0, 0.5,  2.0, 0.97, False, "native"),
    QuantizationVariant("int4-gptq", "int8",  4.0, 0.25, 1.8, 0.93, False, "GPTQ"),
    QuantizationVariant("int4-awq",  "int8",  4.0, 0.25, 1.8, 0.94, False, "AWQ"),
    QuantizationVariant("q4_k_m",    "int8",  4.5, 0.28, 1.6, 0.95, False, "GGUF"),
]

_GPU_PRECISION_SUPPORT: dict[str, set[str]] = {
    "Hopper":         {"fp16", "bf16", "fp32", "fp8", "int8"},
    "Ampere":         {"fp16", "bf16", "fp32", "int8"},
    "Ada Lovelace":   {"fp16", "bf16", "fp32", "fp8", "int8"},
    "Turing":         {"fp16", "fp32"},
    "Volta":          {"fp16", "fp32"},
    "Apple Silicon":  {"fp16", "bf16", "fp32"},
    "Gaudi":          {"fp16", "bf16", "fp32", "fp8", "int8"},
    "Arc":            {"fp16", "bf16", "fp32"},
    "Ponte Vecchio":  {"fp16", "bf16", "fp32", "fp8", "int8"},
    "CDNA2":          {"fp16", "bf16", "fp32", "int8"},
    "CDNA3":          {"fp16", "bf16", "fp32", "fp8", "int8"},
    "RDNA3":          {"fp16", "bf16", "fp32"},
}

_NATIVE_BITS = 16.0
_VRAM_OVERHEAD = 0.85


@dataclass
class QuantizedModelProfile:
    base_profile: ModelProfile
    variant: QuantizationVariant
    adjusted_profile: ModelProfile
    model_size_gb: float
    base_size_gb: float
    memory_reduction_pct: float
    estimated_throughput_change_pct: float
    estimated_quality_impact: str
    gpu_rankings: list[dict] = field(default_factory=list)
    fits_on_gpus: list[str] = field(default_factory=list)
    recommended_for: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class QuantizationRecommendation:
    model_tag: str
    parameter_count: Optional[int]
    sweet_spot: Optional[QuantizedModelProfile]
    variants: list[QuantizedModelProfile] = field(default_factory=list)
    per_gpu_recommendations: dict[str, str] = field(default_factory=dict)


class QuantizationAdvisor:
    def __init__(self):
        self._estimator = BenchmarkEstimator()

    def analyze(
        self,
        profile: ModelProfile,
        parameter_count: Optional[int] = None,
    ) -> QuantizationRecommendation:
        variants: list[QuantizedModelProfile] = []

        for variant in QUANTIZATION_VARIANTS:
            qp = self._generate_variant_profile(profile, variant, parameter_count)
            variants.append(qp)

        sweet_spot = self._pick_sweet_spot(variants)
        per_gpu = self._compute_per_gpu_recommendations(variants)

        return QuantizationRecommendation(
            model_tag=profile.tag,
            parameter_count=parameter_count,
            sweet_spot=sweet_spot,
            variants=variants,
            per_gpu_recommendations=per_gpu,
        )

    def recommend_per_gpu(
        self,
        profile: ModelProfile,
        gpu_name: str,
        parameter_count: Optional[int] = None,
    ) -> Optional[QuantizedModelProfile]:
        spec = GPU_ARCHITECTURES.get(gpu_name)
        if not spec:
            return None

        supported = _GPU_PRECISION_SUPPORT.get(spec.family, {"fp16", "fp32"})
        best: Optional[QuantizedModelProfile] = None
        best_score = 0.0

        for variant in QUANTIZATION_VARIANTS:
            if variant.precision not in supported:
                continue
            if variant.requires_transformer_engine and not spec.has_transformer_engine:
                continue

            qp = self._generate_variant_profile(profile, variant, parameter_count)

            if qp.model_size_gb > spec.memory_gb * _VRAM_OVERHEAD:
                continue

            score = variant.quality_retention * variant.throughput_multiplier
            if score > best_score:
                best_score = score
                best = qp

        return best

    def _generate_variant_profile(
        self,
        base: ModelProfile,
        variant: QuantizationVariant,
        param_count: Optional[int],
    ) -> QuantizedModelProfile:
        # Estimate model size
        if param_count and param_count > 0:
            base_size_gb = param_count * _NATIVE_BITS / 8 / 1e9
            variant_size_gb = param_count * variant.bits_per_param / 8 / 1e9
        else:
            base_size_gb = self._estimate_size_from_family(base.family)
            variant_size_gb = base_size_gb * variant.memory_multiplier

        memory_reduction = round((1 - variant.memory_multiplier) * 100, 1)

        # Adjust math intensity: fewer bytes transferred per operation
        # means higher arithmetic intensity (same FLOPs, less memory traffic)
        intensity_scale = _NATIVE_BITS / variant.bits_per_param
        adjusted_intensity = min(base.math_intensity * intensity_scale, 500.0)

        # Quantized models tend toward compute-bound
        is_memory_bound = adjusted_intensity < 50.0

        adjusted_profile = ModelProfile(
            tag=f"{base.tag}-{variant.name}",
            family=base.family,
            math_intensity=round(adjusted_intensity, 1),
            precision=variant.precision,
            is_memory_bound=is_memory_bound,
            typical_util_min=base.typical_util_min,
            typical_util_max=base.typical_util_max,
        )

        # Estimate GPU rankings
        gpu_rankings = self._estimate_gpu_rankings(adjusted_profile, variant)

        # Check memory fit
        fits_on = self._check_memory_fit(variant_size_gb)

        # Quality impact
        quality_impact = self._assess_quality_impact(
            variant, param_count, base.family
        )

        # Throughput change
        throughput_change = round((variant.throughput_multiplier - 1.0) * 100, 1)

        # Warnings
        warnings = self._generate_warnings(variant, param_count, base.family)

        # Recommended GPU families
        recommended_for = self._get_recommended_families(variant)

        return QuantizedModelProfile(
            base_profile=base,
            variant=variant,
            adjusted_profile=adjusted_profile,
            model_size_gb=round(variant_size_gb, 2),
            base_size_gb=round(base_size_gb, 2),
            memory_reduction_pct=memory_reduction,
            estimated_throughput_change_pct=throughput_change,
            estimated_quality_impact=quality_impact,
            gpu_rankings=gpu_rankings,
            fits_on_gpus=fits_on,
            recommended_for=recommended_for,
            warnings=warnings,
        )

    def _estimate_gpu_rankings(
        self,
        adjusted_profile: ModelProfile,
        variant: QuantizationVariant,
    ) -> list[dict]:
        result = self._estimator.estimate(adjusted_profile, top_n=10)

        supported_gpus = set()
        for family, precisions in _GPU_PRECISION_SUPPORT.items():
            if variant.precision in precisions:
                for name, spec in GPU_ARCHITECTURES.items():
                    if spec.family == family:
                        if variant.requires_transformer_engine and not spec.has_transformer_engine:
                            continue
                        supported_gpus.add(name)

        filtered = []
        for g in result.gpu_rankings:
            if g.gpu_name in supported_gpus:
                filtered.append({
                    "gpu_name": g.gpu_name,
                    "family": g.family,
                    "score": g.score,
                    "joules_per_tflop": g.joules_per_tflop,
                    "effective_tflops": g.effective_tflops,
                    "cost_per_hr": g.cost_per_hr,
                })

        return filtered[:10]

    def _check_memory_fit(self, model_size_gb: float) -> list[str]:
        fits = []
        for name, spec in GPU_ARCHITECTURES.items():
            if model_size_gb <= spec.memory_gb * _VRAM_OVERHEAD:
                fits.append(name)
        fits.sort()
        return fits

    def _assess_quality_impact(
        self,
        variant: QuantizationVariant,
        param_count: Optional[int],
        family: str,
    ) -> str:
        retention = variant.quality_retention

        # Larger models lose less from quantization
        if param_count:
            if param_count > 70e9:
                retention = min(retention + 0.02, 1.0)
            elif param_count > 30e9:
                retention = min(retention + 0.01, 1.0)
            elif param_count < 1e9:
                retention = max(retention - 0.05, 0.5)

        # Family adjustments
        if family == "Diffusion" and variant.bits_per_param <= 4:
            retention = max(retention - 0.03, 0.5)
        elif family in ("BERT", "T5"):
            retention = min(retention + 0.01, 1.0)

        if retention >= 0.98:
            return "negligible"
        elif retention >= 0.95:
            return "minimal"
        elif retention >= 0.90:
            return "moderate"
        return "significant"

    def _generate_warnings(
        self,
        variant: QuantizationVariant,
        param_count: Optional[int],
        family: str,
    ) -> list[str]:
        warnings = []

        if variant.requires_transformer_engine:
            warnings.append("Requires Transformer Engine (H100/H200/Ada only)")

        if param_count and param_count < 1e9 and variant.bits_per_param <= 4:
            warnings.append(
                f"Model has <1B params — INT4 quantization may cause significant quality loss"
            )

        if family == "Diffusion" and variant.bits_per_param <= 4:
            warnings.append(
                "Diffusion models are sensitive to aggressive quantization"
            )

        return warnings

    def _get_recommended_families(self, variant: QuantizationVariant) -> list[str]:
        families = []
        for family, precisions in _GPU_PRECISION_SUPPORT.items():
            if variant.precision in precisions:
                families.append(family)
        return sorted(families)

    def _pick_sweet_spot(
        self, variants: list[QuantizedModelProfile]
    ) -> Optional[QuantizedModelProfile]:
        best: Optional[QuantizedModelProfile] = None
        best_score = 0.0

        for qp in variants:
            # Skip native precision — sweet spot should be a quantized variant
            if qp.variant.bits_per_param >= 16:
                continue

            score = qp.variant.quality_retention * qp.variant.throughput_multiplier

            # Penalize if quality impact is significant
            if qp.estimated_quality_impact == "significant":
                score *= 0.7

            if score > best_score:
                best_score = score
                best = qp

        return best

    def _compute_per_gpu_recommendations(
        self, variants: list[QuantizedModelProfile]
    ) -> dict[str, str]:
        recommendations: dict[str, str] = {}

        for gpu_name, spec in GPU_ARCHITECTURES.items():
            supported = _GPU_PRECISION_SUPPORT.get(spec.family, {"fp16", "fp32"})
            best_variant = "fp16"
            best_score = 0.0

            for qp in variants:
                v = qp.variant
                if v.precision not in supported:
                    continue
                if v.requires_transformer_engine and not spec.has_transformer_engine:
                    continue
                if qp.model_size_gb > spec.memory_gb * _VRAM_OVERHEAD:
                    continue

                score = v.quality_retention * v.throughput_multiplier
                if score > best_score:
                    best_score = score
                    best_variant = v.name

            recommendations[gpu_name] = best_variant

        return recommendations

    @staticmethod
    def _estimate_size_from_family(family: str) -> float:
        defaults = {
            "Llama": 16.0,
            "Mistral": 14.0,
            "Qwen": 14.0,
            "Phi": 5.5,
            "Gemma": 5.0,
            "GPT-NeoX": 40.0,
            "Falcon": 80.0,
            "DeepSeek": 100.0,
            "BERT": 0.5,
            "T5": 22.0,
            "Whisper": 3.0,
            "ViT": 0.6,
            "Diffusion": 4.0,
        }
        return defaults.get(family, 14.0)
