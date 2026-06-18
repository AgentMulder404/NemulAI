# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Quantization Advisor."""

from __future__ import annotations

import unittest

from efficiency.gpu_specs import MODEL_PROFILES, ModelProfile, GPU_ARCHITECTURES
from intelligence.quantization import (
    QuantizationAdvisor,
    QuantizationVariant,
    QUANTIZATION_VARIANTS,
    _GPU_PRECISION_SUPPORT,
)


class TestQuantizationVariants(unittest.TestCase):
    def test_variant_definitions_valid(self):
        for v in QUANTIZATION_VARIANTS:
            self.assertGreater(v.bits_per_param, 0, f"{v.name}: bits must be > 0")
            self.assertGreater(v.memory_multiplier, 0, f"{v.name}: memory_multiplier must be > 0")
            self.assertLessEqual(v.memory_multiplier, 1.0, f"{v.name}: memory_multiplier must be <= 1")
            self.assertGreater(v.quality_retention, 0, f"{v.name}: quality must be > 0")
            self.assertLessEqual(v.quality_retention, 1.0, f"{v.name}: quality must be <= 1")
            self.assertGreater(v.throughput_multiplier, 0)
            self.assertTrue(v.name)
            self.assertTrue(v.precision)

    def test_gpu_precision_support_covers_all_families(self):
        families_in_gpus = {spec.family for spec in GPU_ARCHITECTURES.values()}
        families_in_support = set(_GPU_PRECISION_SUPPORT.keys())
        for family in families_in_gpus:
            self.assertIn(family, families_in_support, f"Missing precision support for {family}")


class TestQuantizationAdvisor(unittest.TestCase):
    def setUp(self):
        self.advisor = QuantizationAdvisor()

    def test_analyze_llama_8b(self):
        profile = ModelProfile(
            tag="test-llama-8b", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.advisor.analyze(profile, parameter_count=8_000_000_000)

        self.assertEqual(len(result.variants), len(QUANTIZATION_VARIANTS))
        self.assertIsNotNone(result.sweet_spot)

        # int4 should show ~75% memory reduction
        int4_variants = [v for v in result.variants if "int4" in v.variant.name]
        self.assertTrue(int4_variants)
        for v in int4_variants:
            self.assertAlmostEqual(v.memory_reduction_pct, 75.0, delta=1.0)

    def test_analyze_generates_correct_sizes(self):
        profile = ModelProfile(
            tag="test-size", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.advisor.analyze(profile, parameter_count=7_000_000_000)

        fp16_variant = [v for v in result.variants if v.variant.name == "fp16"][0]
        int8_variant = [v for v in result.variants if v.variant.name == "int8"][0]

        # 7B params * 16 bits / 8 bits/byte / 1e9 = 14 GB
        self.assertAlmostEqual(fp16_variant.model_size_gb, 14.0, delta=0.1)
        # 7B params * 8 bits / 8 bits/byte / 1e9 = 7 GB
        self.assertAlmostEqual(int8_variant.model_size_gb, 7.0, delta=0.1)

    def test_large_model_better_quality_retention(self):
        small = ModelProfile(
            tag="test-small", family="Llama",
            math_intensity=90.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        large = ModelProfile(
            tag="test-large", family="Llama",
            math_intensity=185.0, precision="bf16",
            is_memory_bound=False, typical_util_min=75, typical_util_max=95,
        )

        r_small = self.advisor.analyze(small, parameter_count=500_000_000)
        r_large = self.advisor.analyze(large, parameter_count=70_000_000_000)

        # Find int4-gptq quality for both
        small_int4 = [v for v in r_small.variants if v.variant.name == "int4-gptq"][0]
        large_int4 = [v for v in r_large.variants if v.variant.name == "int4-gptq"][0]

        # Small model with <1B params should have worse quality impact
        quality_order = {"negligible": 4, "minimal": 3, "moderate": 2, "significant": 1}
        small_q = quality_order.get(small_int4.estimated_quality_impact, 0)
        large_q = quality_order.get(large_int4.estimated_quality_impact, 0)
        self.assertGreaterEqual(large_q, small_q)

    def test_fp8_only_ranks_supported_gpus(self):
        profile = ModelProfile(
            tag="test-fp8", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.advisor.analyze(profile, parameter_count=8_000_000_000)

        fp8_variant = [v for v in result.variants if v.variant.name == "fp8"][0]

        # gpu_rankings (filtered by precision) should only include fp8-capable GPUs
        for entry in fp8_variant.gpu_rankings:
            gpu_name = entry["gpu_name"]
            if gpu_name in GPU_ARCHITECTURES:
                spec = GPU_ARCHITECTURES[gpu_name]
                supported = _GPU_PRECISION_SUPPORT.get(spec.family, set())
                self.assertIn("fp8", supported, f"{gpu_name} ({spec.family}) shouldn't be ranked for fp8")

    def test_v100_excluded_from_int8(self):
        profile = ModelProfile(
            tag="test-v100", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.advisor.analyze(profile, parameter_count=8_000_000_000)

        int8_variant = [v for v in result.variants if v.variant.name == "int8"][0]
        ranked_gpus = {r["gpu_name"] for r in int8_variant.gpu_rankings}

        # V100 is Volta family — no int8 support
        for gpu_name in ranked_gpus:
            spec = GPU_ARCHITECTURES.get(gpu_name)
            if spec:
                self.assertNotEqual(spec.family, "Volta", f"V100 should not be ranked for int8")

    def test_memory_fit_large_model(self):
        profile = ModelProfile(
            tag="test-large-fit", family="Llama",
            math_intensity=185.0, precision="bf16",
            is_memory_bound=False, typical_util_min=75, typical_util_max=95,
        )
        result = self.advisor.analyze(profile, parameter_count=70_000_000_000)

        fp16_variant = [v for v in result.variants if v.variant.name == "fp16"][0]
        # 70B * 16 bits / 8 / 1e9 = 140 GB — very few GPUs have 165+ GB VRAM
        self.assertLess(len(fp16_variant.fits_on_gpus), 10)

        # int4 should fit on more GPUs: 70B * 4 / 8 / 1e9 = 35 GB
        int4_variant = [v for v in result.variants if v.variant.name == "int4-gptq"][0]
        self.assertGreater(len(int4_variant.fits_on_gpus), len(fp16_variant.fits_on_gpus))

    def test_adjusted_intensity_increases(self):
        profile = ModelProfile(
            tag="test-intensity", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.advisor.analyze(profile, parameter_count=8_000_000_000)

        fp16_intensity = [v for v in result.variants if v.variant.name == "fp16"][0].adjusted_profile.math_intensity
        int4_intensity = [v for v in result.variants if v.variant.name == "int4-gptq"][0].adjusted_profile.math_intensity

        # int4 should have ~4x the math intensity (fewer bytes, same FLOPs)
        self.assertGreater(int4_intensity, fp16_intensity * 2)

    def test_sweet_spot_is_quantized(self):
        profile = ModelProfile(
            tag="test-sweet", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.advisor.analyze(profile, parameter_count=8_000_000_000)

        self.assertIsNotNone(result.sweet_spot)
        # Sweet spot should NOT be native precision
        self.assertLess(result.sweet_spot.variant.bits_per_param, 16)

    def test_per_gpu_recommendations_populated(self):
        profile = ModelProfile(
            tag="test-per-gpu", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.advisor.analyze(profile, parameter_count=8_000_000_000)

        self.assertGreater(len(result.per_gpu_recommendations), 0)
        # H100 should recommend fp8
        h100_rec = result.per_gpu_recommendations.get("H100-SXM5-80GB")
        self.assertIsNotNone(h100_rec)
        self.assertEqual(h100_rec, "fp8")

    def test_does_not_mutate_model_profiles(self):
        original_keys = set(MODEL_PROFILES.keys())
        profile = ModelProfile(
            tag="test-mutate", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        self.advisor.analyze(profile, parameter_count=8_000_000_000)
        self.assertEqual(set(MODEL_PROFILES.keys()), original_keys)

    def test_recommend_per_gpu(self):
        profile = ModelProfile(
            tag="test-rec-gpu", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        rec = self.advisor.recommend_per_gpu(profile, "H100-SXM5-80GB", 8_000_000_000)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.variant.name, "fp8")

    def test_recommend_per_gpu_unknown(self):
        profile = ModelProfile(
            tag="test-unknown-gpu", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        rec = self.advisor.recommend_per_gpu(profile, "FakeGPU-9000", 8_000_000_000)
        self.assertIsNone(rec)

    def test_diffusion_model_warnings(self):
        profile = ModelProfile(
            tag="test-diffusion", family="Diffusion",
            math_intensity=48.0, precision="fp16",
            is_memory_bound=True, typical_util_min=50, typical_util_max=80,
        )
        result = self.advisor.analyze(profile, parameter_count=3_000_000_000)

        int4_variants = [v for v in result.variants if "int4" in v.variant.name]
        for v in int4_variants:
            self.assertTrue(
                any("Diffusion" in w for w in v.warnings),
                f"int4 variant {v.variant.name} should warn about diffusion sensitivity"
            )


if __name__ == "__main__":
    unittest.main()
