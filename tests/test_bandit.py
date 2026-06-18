# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contextual bandit (Phase 2)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learner.bandit import (
    POWER_CAP_ACTIONS,
    BanditSuggestion,
    EnergyBandit,
    _SimpleBackend,
)
from learner.experience_logger import (
    ActionOutcome,
    ActionTaken,
    ExperienceTuple,
    WorkloadContext,
)
from learner.feature_encoder import encode_context


def _make_features(util: float = 60.0, power: float = 200.0, limit: float = 300.0):
    return encode_context(
        gpu_name="NVIDIA A100-SXM4-80GB",
        gpu_arch="a100_sxm4_80gb",
        workload_class="llm-inference-bf16",
        utilization_gpu_pct=util,
        utilization_memory_pct=40.0,
        memory_pressure=0.5,
        power_draw_w=power,
        power_limit_w=limit,
        temperature_c=65.0,
    )


def _make_experience_tuple(reward: float = 0.3, cap_watts: float = 240.0) -> ExperienceTuple:
    return ExperienceTuple(
        machine_id="test-001",
        gpu_index=0,
        context=WorkloadContext(
            gpu_name="NVIDIA A100-SXM4-80GB",
            gpu_arch="a100_sxm4_80gb",
            workload_class="llm-inference-bf16",
            utilization_gpu_pct=60.0,
            utilization_memory_pct=40.0,
            memory_pressure=0.5,
            power_draw_w=200.0,
            power_limit_w=300.0,
            temperature_c=65.0,
        ),
        action=ActionTaken(
            action_type="power_cap",
            source="auto_tuner",
            recommended_value=cap_watts,
            current_value=300.0,
            estimated_savings_pct=20.0,
        ),
        outcome=ActionOutcome(
            energy_delta_j_before=100.0,
            energy_delta_j_after=80.0,
            throughput_before=60.0,
            throughput_after=58.0,
            recommendation_status="applied",
            actual_savings_pct=20.0,
            observation_window_s=300.0,
        ),
        reward=reward,
    )


# ── Simple Backend Tests ─────────────────────────────────────────────────────

class TestSimpleBackend(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.model_path = Path(self.tmpdir) / "model.json"
        self.backend = _SimpleBackend(self.model_path, epsilon=0.1)

    def test_predict_returns_probabilities(self):
        features = _make_features()
        probs = self.backend.predict(features)
        self.assertEqual(len(probs), len(POWER_CAP_ACTIONS))
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_probabilities_sum_to_one(self):
        features = _make_features()
        probs = self.backend.predict(features)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_learn_updates_estimates(self):
        features = _make_features()
        initial_probs = self.backend.predict(features)

        for _ in range(50):
            self.backend.learn(features, action_idx=2, cost=0.1, probability=0.5)

        updated_probs = self.backend.predict(features)
        self.assertGreater(updated_probs[2], initial_probs[2])

    def test_save_and_load(self):
        features = _make_features()
        for _ in range(20):
            self.backend.learn(features, action_idx=3, cost=0.2, probability=0.5)

        self.backend.save()

        new_backend = _SimpleBackend(self.model_path, epsilon=0.1)
        self.assertEqual(new_backend._counts[3], 20)


# ── EnergyBandit Tests ───────────────────────────────────────────────────────

class TestEnergyBandit(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.bandit = EnergyBandit(
            data_dir=Path(self.tmpdir),
            epsilon=0.1,
            retrain_every=10,
            min_corpus=5,
        )

    def test_suggest_returns_valid_action(self):
        features = _make_features()
        suggestion = self.bandit.suggest(features, tdp_w=300.0)

        self.assertIsInstance(suggestion, BanditSuggestion)
        self.assertIn(suggestion.action_name, [a["name"] for a in POWER_CAP_ACTIONS])
        self.assertGreaterEqual(suggestion.cap_watts, 300.0 * 0.40)
        self.assertLessEqual(suggestion.cap_watts, 300.0)

    def test_safety_bounds(self):
        features = _make_features()
        for _ in range(20):
            suggestion = self.bandit.suggest(features, tdp_w=250.0)
            self.assertGreaterEqual(suggestion.cap_watts, 100.0)
            self.assertLessEqual(suggestion.cap_watts, 250.0)

    def test_not_ready_before_min_corpus(self):
        self.assertFalse(self.bandit.is_ready())

    def test_ready_after_updates(self):
        features = _make_features()
        for i in range(5):
            self.bandit.update(features, action_index=i % len(POWER_CAP_ACTIONS),
                               reward=0.5, probability=0.14)
        self.assertTrue(self.bandit.is_ready())

    def test_warm_start(self):
        tuples = [_make_experience_tuple(reward=0.3 + i * 0.05) for i in range(10)]
        count = self.bandit.warm_start(tuples)
        self.assertEqual(count, 10)
        self.assertTrue(self.bandit.is_ready())

    def test_warm_start_skips_non_power_cap(self):
        t = _make_experience_tuple()
        t.action.action_type = "precision"
        count = self.bandit.warm_start([t])
        self.assertEqual(count, 0)

    def test_warm_start_skips_incomplete(self):
        t = ExperienceTuple(context=WorkloadContext(
            gpu_name="A100", gpu_arch="a100", workload_class="test",
            utilization_gpu_pct=50, utilization_memory_pct=30,
            memory_pressure=0.5, power_draw_w=200, power_limit_w=300,
            temperature_c=65,
        ))
        count = self.bandit.warm_start([t])
        self.assertEqual(count, 0)

    def test_stats_persist(self):
        features = _make_features()
        for i in range(5):
            self.bandit.update(features, action_index=0, reward=0.5, probability=0.14)

        stats = self.bandit.get_stats()
        self.assertEqual(stats["corpus_size"], 5)

        new_bandit = EnergyBandit(
            data_dir=Path(self.tmpdir),
            epsilon=0.1,
            min_corpus=5,
        )
        new_stats = new_bandit.get_stats()
        self.assertEqual(new_stats["corpus_size"], 5)

    def test_match_action(self):
        idx = EnergyBandit._match_action(120.0, 300.0)
        self.assertEqual(POWER_CAP_ACTIONS[idx]["fraction"], 0.40)

        idx = EnergyBandit._match_action(300.0, 300.0)
        self.assertEqual(POWER_CAP_ACTIONS[idx]["fraction"], 1.00)

        idx = EnergyBandit._match_action(240.0, 300.0)
        self.assertEqual(POWER_CAP_ACTIONS[idx]["fraction"], 0.80)

    def test_evaluate_offline(self):
        tuples = [_make_experience_tuple(reward=0.3) for _ in range(20)]
        self.bandit.warm_start(tuples)
        reward = self.bandit.evaluate_offline(tuples, sample_size=10)
        self.assertGreaterEqual(reward, 0.0)

    def test_model_checkpoint_on_retrain(self):
        features = _make_features()
        for i in range(10):
            self.bandit.update(features, action_index=i % 7, reward=0.4, probability=0.14)

        self.assertEqual(self.bandit._stats.model_version, 1)


if __name__ == "__main__":
    unittest.main()
