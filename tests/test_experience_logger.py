# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the self-learning experience logger (Phase 1)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learner.reward import compute_energy_reward, normalize_j_per_flop
from learner.feature_encoder import classify_workload, gpu_class, encode_context
from learner.experience_logger import (
    WorkloadContext,
    ActionTaken,
    ActionOutcome,
    ExperienceTuple,
    ExperienceLogger,
)


# ── Reward tests ─────────────────────────────────────────────────────────────

class TestReward(unittest.TestCase):

    def test_perfect_savings(self):
        r = compute_energy_reward(100.0, 50.0, 80.0, 80.0)
        self.assertAlmostEqual(r, 0.5, places=2)

    def test_no_improvement(self):
        r = compute_energy_reward(100.0, 100.0, 80.0, 80.0)
        self.assertAlmostEqual(r, 0.0)

    def test_energy_increase(self):
        r = compute_energy_reward(100.0, 150.0, 80.0, 80.0)
        self.assertAlmostEqual(r, 0.0)

    def test_throughput_regression_penalty(self):
        r_no_reg = compute_energy_reward(100.0, 60.0, 80.0, 80.0)
        r_with_reg = compute_energy_reward(100.0, 60.0, 80.0, 40.0)
        self.assertGreater(r_no_reg, r_with_reg)

    def test_zero_energy_before(self):
        r = compute_energy_reward(0.0, 50.0, 80.0, 80.0)
        self.assertAlmostEqual(r, 0.0)

    def test_throughput_collapse(self):
        r = compute_energy_reward(100.0, 50.0, 80.0, 0.0)
        self.assertGreaterEqual(r, 0.0)

    def test_reward_clamped_to_unit(self):
        r = compute_energy_reward(1000.0, 1.0, 80.0, 80.0)
        self.assertLessEqual(r, 1.0)
        self.assertGreaterEqual(r, 0.0)

    def test_normalize_j_per_flop_zero_throughput(self):
        j = normalize_j_per_flop(100.0, 0.0, 1.0)
        self.assertEqual(j, float("inf"))

    def test_normalize_j_per_flop_normal(self):
        j = normalize_j_per_flop(100.0, 10.0, 1.0)
        self.assertAlmostEqual(j, 10.0)


# ── Feature encoder tests ───────────────────────────────────────────────────

class TestFeatureEncoder(unittest.TestCase):

    def test_classify_idle(self):
        wl = classify_workload("llama3", "bf16", 2.0, 10.0)
        self.assertEqual(wl, "idle")

    def test_classify_inference(self):
        wl = classify_workload("llama3-8b", "bf16", 50.0, 30.0)
        self.assertEqual(wl, "llm-inference-bf16")

    def test_classify_training(self):
        wl = classify_workload("llama3-8b", "fp16", 80.0, 85.0)
        self.assertEqual(wl, "llm-training-fp16")

    def test_classify_unknown_model(self):
        wl = classify_workload("my-custom-model", "fp32", 60.0, 20.0)
        self.assertIn("unknown", wl)

    def test_classify_no_precision(self):
        wl = classify_workload("llama3", None, 50.0, 30.0)
        self.assertIn("mixed", wl)

    def test_gpu_class_normalizes(self):
        g = gpu_class("NVIDIA A100-SXM4-80GB")
        self.assertEqual(g, "a100_sxm4_80gb")

    def test_gpu_class_rtx(self):
        g = gpu_class("NVIDIA GeForce RTX 4090")
        self.assertEqual(g, "geforce_rtx_4090")

    def test_encode_context_keys(self):
        feats = encode_context(
            gpu_name="NVIDIA A100-SXM4-80GB",
            gpu_arch="a100_sxm4_80gb",
            workload_class="llm-inference-bf16",
            utilization_gpu_pct=75.0,
            utilization_memory_pct=45.0,
            memory_pressure=0.56,
            power_draw_w=200.0,
            power_limit_w=300.0,
            temperature_c=65.0,
        )
        self.assertIn("util_gpu_bucket", feats)
        self.assertIn("power_ratio", feats)
        self.assertEqual(feats["util_gpu_bucket"], 75.0)
        self.assertAlmostEqual(feats["power_ratio"], 0.67, places=2)


# ── Experience logger tests ──────────────────────────────────────────────────

class TestExperienceLogger(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = ExperienceLogger(
            data_dir=Path(self.tmpdir),
            machine_id="test-machine-001",
            outcome_window_s=1.0,
        )

    def _make_context(self, gpu="NVIDIA A100-SXM4-80GB", util=60.0):
        return WorkloadContext(
            gpu_name=gpu,
            gpu_arch="a100_sxm4_80gb",
            workload_class="llm-inference-bf16",
            utilization_gpu_pct=util,
            utilization_memory_pct=40.0,
            memory_pressure=0.5,
            power_draw_w=200.0,
            power_limit_w=300.0,
            temperature_c=65.0,
        )

    def _make_action(self, cap=240.0):
        return ActionTaken(
            action_type="power_cap",
            source="auto_tuner",
            recommended_value=cap,
            current_value=300.0,
            estimated_savings_pct=20.0,
        )

    def test_log_action_returns_id(self):
        ctx = self._make_context()
        act = self._make_action()
        tid = self.logger.log_action(ctx, act, gpu_index=0,
                                     energy_snapshot=50.0, throughput_snapshot=60.0)
        self.assertTrue(len(tid) > 0)

    def test_log_action_writes_wal(self):
        ctx = self._make_context()
        act = self._make_action()
        self.logger.log_action(ctx, act, gpu_index=0,
                               energy_snapshot=50.0, throughput_snapshot=60.0)

        wal_path = Path(self.tmpdir) / "experience" / "experience.wal"
        self.assertTrue(wal_path.exists())

        with open(wal_path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

        entry = json.loads(lines[0])
        self.assertIn("row", entry)
        self.assertEqual(entry["row"]["machine_id"], "test-machine-001")

    def test_corpus_stats(self):
        ctx = self._make_context()
        act = self._make_action()
        self.logger.log_action(ctx, act, gpu_index=0,
                               energy_snapshot=50.0, throughput_snapshot=60.0)

        stats = self.logger.get_corpus_stats()
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["pending"], 1)
        self.assertEqual(stats["completed"], 0)
        self.assertIn("a100_sxm4_80gb", stats["by_gpu_class"])

    def test_outcome_resolution(self):
        import time as t

        ctx = self._make_context()
        act = self._make_action()
        tid = self.logger.log_action(ctx, act, gpu_index=0,
                                     energy_snapshot=100.0, throughput_snapshot=60.0)

        t.sleep(1.1)

        resolved = self.logger.check_pending_outcomes(
            current_energy_by_gpu={0: 80.0},
            current_throughput_by_gpu={0: 58.0},
        )
        self.assertEqual(resolved, 1)

        tup = self.logger._tuples[tid]
        self.assertTrue(tup.is_complete())
        self.assertIsNotNone(tup.reward)
        self.assertGreater(tup.reward, 0.0)
        self.assertEqual(tup.outcome.energy_delta_j_before, 100.0)
        self.assertEqual(tup.outcome.energy_delta_j_after, 80.0)

    def test_iter_completed_filters(self):
        ctx = self._make_context()
        act = self._make_action()
        self.logger.log_action(ctx, act, gpu_index=0,
                               energy_snapshot=50.0, throughput_snapshot=60.0)

        completed = list(self.logger.iter_completed())
        self.assertEqual(len(completed), 0)

    def test_load_from_wal_roundtrip(self):
        ctx = self._make_context()
        act = self._make_action()
        self.logger.log_action(ctx, act, gpu_index=0,
                               energy_snapshot=50.0, throughput_snapshot=60.0)

        new_logger = ExperienceLogger(
            data_dir=Path(self.tmpdir),
            machine_id="test-machine-001",
        )
        loaded = new_logger.load_from_wal()
        self.assertEqual(loaded, 1)
        self.assertEqual(new_logger.get_corpus_stats()["total"], 1)

    def test_workload_context_auto_power_ratio(self):
        ctx = WorkloadContext(
            gpu_name="A100", gpu_arch="a100", workload_class="llm-inference-bf16",
            utilization_gpu_pct=80, utilization_memory_pct=50,
            memory_pressure=0.5, power_draw_w=200, power_limit_w=400,
            temperature_c=70,
        )
        self.assertAlmostEqual(ctx.power_ratio, 0.5)

    def test_experience_tuple_complete_check(self):
        t = ExperienceTuple()
        self.assertFalse(t.is_complete())

        t.outcome = ActionOutcome(
            energy_delta_j_before=100, energy_delta_j_after=80,
            throughput_before=60, throughput_after=58,
            recommendation_status="applied", actual_savings_pct=20,
            observation_window_s=300,
        )
        self.assertFalse(t.is_complete())

        t.reward = 0.2
        self.assertTrue(t.is_complete())


if __name__ == "__main__":
    unittest.main()
