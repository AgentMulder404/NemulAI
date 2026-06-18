# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GPU Price-Performance Tracker."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from efficiency.gpu_specs import ModelProfile
from efficiency.cloud_detect import GPU_HOURLY_RATES
from intelligence.pricing import GPUPricingTracker, PricingSource


class TestGPUPricingTracker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tracker = GPUPricingTracker(data_dir=Path(self.tmpdir))

    def test_load_static_returns_all_rates(self):
        rates = self.tracker.load_static()
        self.assertEqual(len(rates), len(GPU_HOURLY_RATES))
        for gpu, rate in GPU_HOURLY_RATES.items():
            self.assertEqual(rates[gpu], rate)

    def test_get_rate_fallback_to_static(self):
        rate = self.tracker.get_rate("H100-SXM5-80GB")
        self.assertEqual(rate, GPU_HOURLY_RATES["H100-SXM5-80GB"])

    def test_get_rate_unknown_gpu(self):
        rate = self.tracker.get_rate("FakeGPU-9999")
        self.assertEqual(rate, 0.0)

    def test_get_all_rates(self):
        rates = self.tracker.get_all_rates()
        self.assertGreaterEqual(len(rates), len(GPU_HOURLY_RATES))

    def test_update_from_json(self):
        pricing_data = [
            {"gpu_model": "H100-SXM5-80GB", "provider": "lambda", "on_demand_rate": 2.99, "spot_rate": 1.50},
            {"gpu_model": "NewGPU-X100", "provider": "test", "on_demand_rate": 5.00},
        ]
        json_path = Path(self.tmpdir) / "prices.json"
        json_path.write_text(json.dumps(pricing_data))

        count = self.tracker.update_from_json(json_path)
        self.assertEqual(count, 2)

        # Cached rate should override static
        rate = self.tracker.get_rate("H100-SXM5-80GB")
        self.assertEqual(rate, 2.99)

        # New GPU should be available
        rate = self.tracker.get_rate("NewGPU-X100")
        self.assertEqual(rate, 5.00)

    def test_update_from_invalid_json(self):
        json_path = Path(self.tmpdir) / "bad.json"
        json_path.write_text("not valid json")
        count = self.tracker.update_from_json(json_path)
        self.assertEqual(count, 0)

    def test_update_from_missing_file(self):
        count = self.tracker.update_from_json(Path("/nonexistent/file.json"))
        self.assertEqual(count, 0)

    def test_compute_price_performance_sorted(self):
        profile = ModelProfile(
            tag="test-pp", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        results = self.tracker.compute_price_performance(profile, top_n=20)
        self.assertGreater(len(results), 0)

        # Should be sorted by $/TFLOP ascending
        dpts = [r.dollars_per_tflop_hr for r in results]
        self.assertEqual(dpts, sorted(dpts))

        # First result should be best value
        self.assertTrue(results[0].is_best_value)

    def test_compute_price_performance_skips_no_rate(self):
        profile = ModelProfile(
            tag="test-skip", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        results = self.tracker.compute_price_performance(profile)

        gpu_names = {r.gpu_name for r in results}
        # Apple Silicon GPUs have no pricing and should not appear
        for name in gpu_names:
            rate = self.tracker.get_rate(name)
            self.assertGreater(rate, 0, f"{name} should have pricing > 0")

    def test_value_scores_in_range(self):
        profile = ModelProfile(
            tag="test-scores", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        results = self.tracker.compute_price_performance(profile)
        for r in results:
            self.assertGreaterEqual(r.value_score, 0)
            self.assertLessEqual(r.value_score, 100)

    def test_record_history_persistence(self):
        self.tracker.record_history()
        self.tracker.record_history()

        # Create new tracker from same directory
        tracker2 = GPUPricingTracker(data_dir=Path(self.tmpdir))
        history = tracker2._load_history()
        self.assertEqual(len(history), 2)

    def test_detect_alerts_price_drop(self):
        # Record initial history
        self.tracker.record_history()

        # Simulate price drop by injecting cheaper cached rate
        pricing_data = [
            {"gpu_model": "H100-SXM5-80GB", "provider": "test", "on_demand_rate": 2.00},
        ]
        json_path = Path(self.tmpdir) / "drop.json"
        json_path.write_text(json.dumps(pricing_data))
        self.tracker.update_from_json(json_path)

        alerts = self.tracker.detect_alerts()
        price_drops = [a for a in alerts if a.alert_type == "price_drop"]
        self.assertGreater(len(price_drops), 0)
        self.assertEqual(price_drops[0].gpu_name, "H100-SXM5-80GB")

    def test_detect_alerts_spot_opportunity(self):
        # Record history first so detect_alerts doesn't early-return
        self.tracker.record_history()

        pricing_data = [
            {"gpu_model": "A100-SXM4-80GB", "provider": "test", "on_demand_rate": 1.89, "spot_rate": 0.50},
        ]
        json_path = Path(self.tmpdir) / "spot.json"
        json_path.write_text(json.dumps(pricing_data))
        self.tracker.update_from_json(json_path)

        alerts = self.tracker.detect_alerts()
        spot_alerts = [a for a in alerts if a.alert_type == "spot_opportunity"]
        self.assertGreater(len(spot_alerts), 0)

    def test_update_gpu_hourly_rates(self):
        original = dict(GPU_HOURLY_RATES)
        pricing_data = [
            {"gpu_model": "TestGPU-Pricing", "provider": "test", "on_demand_rate": 99.99},
        ]
        json_path = Path(self.tmpdir) / "inject.json"
        json_path.write_text(json.dumps(pricing_data))
        self.tracker.update_from_json(json_path)
        self.tracker.update_gpu_hourly_rates()

        self.assertIn("TestGPU-Pricing", GPU_HOURLY_RATES)
        self.assertEqual(GPU_HOURLY_RATES["TestGPU-Pricing"], 99.99)

        # Cleanup
        GPU_HOURLY_RATES.pop("TestGPU-Pricing", None)

    def test_cache_persistence(self):
        pricing_data = [
            {"gpu_model": "CacheTest-GPU", "provider": "test", "on_demand_rate": 1.23},
        ]
        json_path = Path(self.tmpdir) / "cache.json"
        json_path.write_text(json.dumps(pricing_data))
        self.tracker.update_from_json(json_path)

        # New tracker should load from cache
        tracker2 = GPUPricingTracker(data_dir=Path(self.tmpdir))
        rate = tracker2.get_rate("CacheTest-GPU")
        self.assertEqual(rate, 1.23)

    def test_offline_mode(self):
        # No Supabase configured — everything should work
        tracker = GPUPricingTracker(data_dir=Path(self.tmpdir))
        rates = tracker.get_all_rates()
        self.assertGreater(len(rates), 0)

        profile = ModelProfile(
            tag="test-offline", family="Llama",
            math_intensity=95.0, precision="bf16",
            is_memory_bound=False, typical_util_min=60, typical_util_max=85,
        )
        results = tracker.compute_price_performance(profile)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
