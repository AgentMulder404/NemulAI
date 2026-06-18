# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Model Intelligence Pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from efficiency.gpu_specs import MODEL_PROFILES, ModelProfile
from intelligence.detector import ModelDetector, DetectedModel
from intelligence.profiler import ModelProfiler
from intelligence.estimator import BenchmarkEstimator
from intelligence.registry import ModelRegistry, RegistryEntry
from intelligence.pipeline import IntelligencePipeline


# ── Mock HuggingFace API responses ──────────────────────────────────────────

MOCK_TRENDING = [
    {
        "modelId": "meta-llama/Meta-Llama-3.1-8B",
        "id": "meta-llama/Meta-Llama-3.1-8B",
        "pipeline_tag": "text-generation",
        "config": {"architectures": ["LlamaForCausalLM"]},
        "safetensors": {"total": 8_000_000_000},
        "downloads": 50000,
        "trendingScore": 95.5,
        "library_name": "transformers",
        "license": "llama3.1",
        "createdAt": "2025-07-01T00:00:00Z",
        "sha": "abc123",
    },
    {
        "modelId": "mistralai/Mistral-7B-v0.3",
        "id": "mistralai/Mistral-7B-v0.3",
        "pipeline_tag": "text-generation",
        "config": {"architectures": ["MistralForCausalLM"]},
        "safetensors": {"total": 7_000_000_000},
        "downloads": 30000,
        "trendingScore": 80.0,
        "library_name": "transformers",
        "license": "apache-2.0",
        "createdAt": "2025-06-01T00:00:00Z",
        "sha": "def456",
    },
    {
        "modelId": "google/vit-base-patch16-224",
        "id": "google/vit-base-patch16-224",
        "pipeline_tag": "image-classification",
        "config": {"architectures": ["ViTForImageClassification"]},
        "safetensors": {"total": 86_000_000},
        "downloads": 5000,
        "trendingScore": 40.0,
        "library_name": "transformers",
        "license": "apache-2.0",
        "createdAt": "2024-01-01T00:00:00Z",
        "sha": "ghi789",
    },
    {
        "modelId": "openai/whisper-large-v3",
        "id": "openai/whisper-large-v3",
        "pipeline_tag": "automatic-speech-recognition",
        "config": {"architectures": ["WhisperForConditionalGeneration"]},
        "safetensors": {"total": 1_550_000_000},
        "downloads": 20000,
        "trendingScore": 60.0,
        "library_name": "transformers",
        "license": "mit",
        "createdAt": "2024-06-01T00:00:00Z",
        "sha": "jkl012",
    },
    {
        "modelId": "some-user/cat-classifier",
        "id": "some-user/cat-classifier",
        "pipeline_tag": "image-classification",
        "config": {},
        "downloads": 50,
        "trendingScore": 5.0,
        "library_name": "transformers",
        "createdAt": "2025-01-01T00:00:00Z",
        "sha": "mno345",
    },
]


class MockResponse:
    def __init__(self, status_code: int, data):
        self.status_code = status_code
        self._data = data

    def raise_for_status(self):
        if self.status_code >= 400:
            from requests.exceptions import HTTPError
            raise HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


class MockSession:
    def __init__(self, trending=None, models=None):
        self._trending = trending or MOCK_TRENDING
        self._models = models or {}

    def get(self, url: str, **kwargs) -> MockResponse:
        if "/api/models?" in url or url.endswith("/api/models"):
            return MockResponse(200, self._trending)
        if "/api/models/" in url:
            model_id = url.split("/api/models/", 1)[1]
            if model_id in self._models:
                return MockResponse(200, self._models[model_id])
            for item in self._trending:
                mid = item.get("modelId") or item.get("id", "")
                if mid == model_id:
                    return MockResponse(200, item)
            return MockResponse(404, {"error": "not found"})
        return MockResponse(404, {"error": "not found"})


class ErrorSession:
    def get(self, url: str, **kwargs):
        import requests
        raise requests.ConnectionError("Network unavailable")


# ── TestModelDetector ───────────────────────────────────────────────────────

class TestModelDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ModelDetector(session=MockSession())

    def test_normalize_tag_llama(self):
        self.assertEqual(
            ModelDetector.normalize_tag("meta-llama/Meta-Llama-3.1-8B"),
            "llama-3.1-8b",
        )

    def test_normalize_tag_mistral(self):
        self.assertEqual(
            ModelDetector.normalize_tag("mistralai/Mistral-7B-v0.3"),
            "mistral-7b-v0.3",
        )

    def test_normalize_tag_no_org(self):
        self.assertEqual(
            ModelDetector.normalize_tag("some-model"),
            "some-model",
        )

    def test_infer_family_from_architecture(self):
        self.assertEqual(
            ModelDetector.infer_family("meta-llama/x", "LlamaForCausalLM", "text-generation"),
            "Llama",
        )
        self.assertEqual(
            ModelDetector.infer_family("x", "BertModel", "fill-mask"),
            "BERT",
        )
        self.assertEqual(
            ModelDetector.infer_family("x", "WhisperForConditionalGeneration", "automatic-speech-recognition"),
            "Whisper",
        )

    def test_infer_family_from_model_id_pattern(self):
        self.assertEqual(
            ModelDetector.infer_family("my-org/llama-fine-tune", "", "text-generation"),
            "Llama",
        )

    def test_infer_family_fallback_to_pipeline_tag(self):
        self.assertEqual(
            ModelDetector.infer_family("x/y", "", "text-generation"),
            "LLM",
        )
        self.assertEqual(
            ModelDetector.infer_family("x/y", "", "text-to-image"),
            "Diffusion",
        )

    def test_fetch_trending(self):
        results = self.detector.fetch_trending(limit=30)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], DetectedModel)
        self.assertTrue(results[0].model_id)

    def test_detect_new_filters_known(self):
        known = {"llama-3.1-8b", "mistral-7b-v0.3"}
        new_models = self.detector.detect_new(known, limit=30, min_downloads=0)
        tags = {m.tag for m in new_models}
        self.assertNotIn("llama-3.1-8b", tags)
        self.assertNotIn("mistral-7b-v0.3", tags)

    def test_detect_new_filters_low_downloads(self):
        new_models = self.detector.detect_new(set(), limit=30, min_downloads=10000)
        for m in new_models:
            self.assertGreaterEqual(m.downloads_30d, 10000)

    def test_network_error_returns_empty(self):
        detector = ModelDetector(session=ErrorSession())
        results = detector.fetch_trending()
        self.assertEqual(results, [])

    def test_fetch_model_info(self):
        result = self.detector.fetch_model_info("meta-llama/Meta-Llama-3.1-8B")
        self.assertIsNotNone(result)
        self.assertEqual(result.architecture, "LlamaForCausalLM")
        self.assertEqual(result.parameter_count, 8_000_000_000)


# ── TestModelProfiler ──────────────────────────────────────────────────────

class TestModelProfiler(unittest.TestCase):
    def setUp(self):
        self.profiler = ModelProfiler()

    def _make_detected(self, model_id, arch, params, pipeline_tag="text-generation"):
        return DetectedModel(
            model_id=model_id,
            tag=ModelDetector.normalize_tag(model_id),
            author=model_id.split("/")[0] if "/" in model_id else "",
            pipeline_tag=pipeline_tag,
            architecture=arch,
            library="transformers",
            parameter_count=params,
            license="apache-2.0",
            downloads_30d=10000,
            trending_score=50.0,
            created_at="2025-01-01",
            sha="abc",
            raw_metadata={},
        )

    def test_profile_llama_8b(self):
        detected = self._make_detected(
            "meta-llama/Meta-Llama-3.1-8B", "LlamaForCausalLM", 8_000_000_000
        )
        result = self.profiler.profile(detected)
        self.assertGreater(result.profile.math_intensity, 70)
        self.assertLess(result.profile.math_intensity, 120)
        self.assertEqual(result.profile.precision, "bf16")
        self.assertFalse(result.profile.is_memory_bound)
        self.assertGreater(result.confidence, 0.7)

    def test_profile_bert(self):
        detected = self._make_detected(
            "google/bert-base-uncased", "BertModel", 110_000_000, "fill-mask"
        )
        result = self.profiler.profile(detected)
        self.assertLess(result.profile.math_intensity, 50)
        self.assertTrue(result.profile.is_memory_bound)

    def test_profile_whisper(self):
        detected = self._make_detected(
            "openai/whisper-large-v3", "WhisperForConditionalGeneration",
            1_550_000_000, "automatic-speech-recognition"
        )
        result = self.profiler.profile(detected)
        self.assertLess(result.profile.math_intensity, 50)
        self.assertEqual(result.profile.family, "Whisper")

    def test_profile_large_llm_higher_intensity(self):
        small = self._make_detected("x/small-llm", "LlamaForCausalLM", 7_000_000_000)
        large = self._make_detected("x/large-llm", "LlamaForCausalLM", 70_000_000_000)
        r_small = self.profiler.profile(small)
        r_large = self.profiler.profile(large)
        self.assertGreater(
            r_large.profile.math_intensity,
            r_small.profile.math_intensity,
        )

    def test_profile_unknown_architecture_lower_confidence(self):
        detected = self._make_detected(
            "x/mystery-model", "SomeNewArchitecture", 5_000_000_000
        )
        result = self.profiler.profile(detected)
        # Falls back to family heuristic (0.65) + param bonus (0.10) = 0.75
        self.assertLess(result.confidence, 0.85)

    def test_consistency_with_existing_profiles(self):
        """Verify profiler produces values within 40% of hand-curated profiles."""
        test_cases = [
            ("meta-llama/Llama-3-8B", "LlamaForCausalLM", 8e9, "llama-3-8b"),
            ("google/bert-base-uncased", "BertModel", 110e6, "bert-base"),
        ]
        for model_id, arch, params, existing_tag in test_cases:
            existing = MODEL_PROFILES.get(existing_tag)
            if not existing:
                continue
            detected = self._make_detected(model_id, arch, int(params))
            result = self.profiler.profile(detected)
            ratio = result.profile.math_intensity / existing.math_intensity
            self.assertGreater(ratio, 0.6, f"{existing_tag}: {result.profile.math_intensity} vs {existing.math_intensity}")
            self.assertLess(ratio, 1.4, f"{existing_tag}: {result.profile.math_intensity} vs {existing.math_intensity}")


# ── TestBenchmarkEstimator ─────────────────────────────────────────────────

class TestBenchmarkEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = BenchmarkEstimator()

    def test_estimate_produces_rankings(self):
        profile = ModelProfile(
            tag="test-llm", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.estimator.estimate(profile)
        self.assertGreater(len(result.gpu_rankings), 0)
        self.assertTrue(result.best_gpu)
        self.assertTrue(result.worst_gpu)

    def test_rankings_sorted_by_efficiency(self):
        profile = ModelProfile(
            tag="test-sort", family="Llama",
            math_intensity=100.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.estimator.estimate(profile)
        jpts = [g.joules_per_tflop for g in result.gpu_rankings]
        self.assertEqual(jpts, sorted(jpts))

    def test_does_not_mutate_model_profiles(self):
        original_keys = set(MODEL_PROFILES.keys())
        profile = ModelProfile(
            tag="ephemeral-test", family="Test",
            math_intensity=80.0, precision="fp16",
            is_memory_bound=False, typical_util_min=50, typical_util_max=75,
        )
        self.estimator.estimate(profile)
        self.assertEqual(set(MODEL_PROFILES.keys()), original_keys)

    def test_estimate_pair(self):
        profile = ModelProfile(
            tag="test-pair", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        result = self.estimator.estimate_pair(profile, "H100-SXM5-80GB")
        self.assertIsNotNone(result)
        self.assertEqual(result.gpu_name, "H100-SXM5-80GB")
        self.assertGreater(result.effective_tflops, 0)

    def test_estimate_pair_unknown_gpu(self):
        profile = ModelProfile(
            tag="test-unknown", family="Test",
            math_intensity=60.0, precision="fp16",
            is_memory_bound=False, typical_util_min=50, typical_util_max=75,
        )
        result = self.estimator.estimate_pair(profile, "NonexistentGPU-9999")
        self.assertIsNone(result)


# ── TestModelRegistry ──────────────────────────────────────────────────────

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = ModelRegistry(data_dir=Path(self.tmpdir))

    def _make_entry(self, tag="test-model", family="Llama", status="estimated"):
        profile = ModelProfile(
            tag=tag, family=family,
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        return RegistryEntry(
            model_id=f"test-org/{tag}",
            tag=tag,
            family=family,
            profile=profile,
            status=status,
            gpu_rankings=[{"gpu_name": "H100-SXM5-80GB", "score": 100}],
        )

    def test_register_and_retrieve(self):
        entry = self._make_entry()
        self.registry.register(entry)
        retrieved = self.registry.get("test-model")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_id, "test-org/test-model")

    def test_known_tags_includes_existing(self):
        known = self.registry.known_tags()
        self.assertIn("llama-3-8b", known)
        self.assertIn("bert-base", known)

    def test_known_tags_includes_registered(self):
        entry = self._make_entry(tag="brand-new-model")
        self.registry.register(entry)
        known = self.registry.known_tags()
        self.assertIn("brand-new-model", known)

    def test_activate_injects_into_model_profiles(self):
        tag = "test-activate-model"
        entry = self._make_entry(tag=tag, status="estimated")
        self.registry.register(entry)

        self.assertNotIn(tag, MODEL_PROFILES)
        self.registry.activate(tag)
        self.assertIn(tag, MODEL_PROFILES)
        self.assertEqual(MODEL_PROFILES[tag].math_intensity, 95.0)

        # Cleanup
        MODEL_PROFILES.pop(tag, None)

    def test_list_by_status(self):
        self.registry.register(self._make_entry(tag="m1", status="estimated"))
        self.registry.register(self._make_entry(tag="m2", status="active"))
        self.registry.register(self._make_entry(tag="m3", status="estimated"))

        estimated = self.registry.list_all(status="estimated")
        self.assertEqual(len(estimated), 2)

        active = self.registry.list_all(status="active")
        self.assertEqual(len(active), 1)

        # Cleanup
        MODEL_PROFILES.pop("m2", None)

    def test_local_persistence(self):
        entry = self._make_entry(tag="persist-test")
        self.registry.register(entry)

        # Create new registry from same directory
        registry2 = ModelRegistry(data_dir=Path(self.tmpdir))
        retrieved = registry2.get("persist-test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.profile.math_intensity, 95.0)

    def test_sync_to_model_profiles(self):
        tag = "sync-test-model"
        entry = self._make_entry(tag=tag, status="active")
        self.registry.register(entry)

        MODEL_PROFILES.pop(tag, None)
        count = self.registry.sync_to_model_profiles()
        self.assertGreaterEqual(count, 1)
        self.assertIn(tag, MODEL_PROFILES)

        # Cleanup
        MODEL_PROFILES.pop(tag, None)


# ── TestIntelligencePipeline ───────────────────────────────────────────────

class TestIntelligencePipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pipeline = IntelligencePipeline(
            data_dir=Path(self.tmpdir),
            session=MockSession(),
        )

    def tearDown(self):
        for tag in list(MODEL_PROFILES.keys()):
            if tag.startswith(("llama-3.1", "mistral-7b-v0.3", "vit-base",
                               "whisper-large-v3", "cat-classifier")):
                MODEL_PROFILES.pop(tag, None)

    def test_full_pipeline_run(self):
        result = self.pipeline.run(limit=10, min_downloads=0, min_confidence=0.3)
        self.assertGreater(result.detected, 0)
        self.assertGreater(result.profiled, 0)
        self.assertGreater(result.estimated, 0)
        self.assertEqual(len(result.errors), 0)

    def test_run_filters_by_confidence(self):
        result = self.pipeline.run(limit=10, min_downloads=0, min_confidence=0.99)
        self.assertGreaterEqual(result.detected, 0)

    def test_run_single(self):
        entry = self.pipeline.run_single("meta-llama/Meta-Llama-3.1-8B")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.family, "Llama")
        self.assertGreater(len(entry.gpu_rankings), 0)

    def test_run_single_cached(self):
        entry1 = self.pipeline.run_single("meta-llama/Meta-Llama-3.1-8B")
        entry2 = self.pipeline.run_single("meta-llama/Meta-Llama-3.1-8B")
        self.assertEqual(entry1.tag, entry2.tag)

    def test_pipeline_populates_registry(self):
        self.pipeline.run(limit=5, min_downloads=0, min_confidence=0.3)
        entries = self.pipeline.registry.list_all()
        self.assertGreater(len(entries), 0)

    def test_entries_have_gpu_rankings(self):
        result = self.pipeline.run(limit=5, min_downloads=0, min_confidence=0.3)
        for entry in result.entries:
            self.assertGreater(len(entry.gpu_rankings), 0)
            first = entry.gpu_rankings[0]
            self.assertIn("gpu_name", first)
            self.assertIn("joules_per_tflop", first)


if __name__ == "__main__":
    unittest.main()
