# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Contextual bandit for GPU power cap optimization (Phase 2).

Uses Vowpal Wabbit's --cb_explore_adf for action-dependent features
with epsilon-greedy exploration.  Falls back to a lightweight
pure-Python epsilon-greedy when VW is not installed.

Actions are discrete power cap levels expressed as fractions of TDP.
Safety: never below 40% TDP, never above 100% TDP.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from learner.feature_encoder import encode_context, gpu_class

log = logging.getLogger("nemulai-bandit")

try:
    import vowpalwabbit
    _HAS_VW = True
except ImportError:
    _HAS_VW = False


# ── Action space ─────────────────────────────────────────────────────────────

POWER_CAP_ACTIONS = [
    {"name": "cap_40pct", "fraction": 0.40},
    {"name": "cap_50pct", "fraction": 0.50},
    {"name": "cap_60pct", "fraction": 0.60},
    {"name": "cap_70pct", "fraction": 0.70},
    {"name": "cap_80pct", "fraction": 0.80},
    {"name": "cap_90pct", "fraction": 0.90},
    {"name": "cap_100pct", "fraction": 1.00},
]


@dataclass
class BanditSuggestion:
    action_name: str
    action_index: int
    cap_fraction: float
    cap_watts: float
    confidence: float
    is_exploration: bool


@dataclass
class BanditStats:
    corpus_size: int = 0
    updates_since_retrain: int = 0
    model_version: int = 0
    estimated_reward: float = 0.0
    last_retrain_at: float = 0.0


# ── VW Backend ───────────────────────────────────────────────────────────────

class _VWBackend:
    """Vowpal Wabbit contextual bandit backend."""

    def __init__(self, model_path: Path, epsilon: float = 0.1):
        self._model_path = model_path
        self._epsilon = epsilon
        self._workspace = None
        self._init_workspace()

    def _init_workspace(self) -> None:
        args = (
            f"--cb_explore_adf --epsilon {self._epsilon} "
            f"--quiet --coin --interactions ax"
        )
        if self._model_path.exists():
            args += f" -i {self._model_path}"
        self._workspace = vowpalwabbit.Workspace(args)

    def predict(self, context_features: dict[str, float]) -> list[float]:
        shared = self._format_shared(context_features)
        examples = [self._workspace.parse(shared, vowpalwabbit.LabelType.CONTEXTUAL_BANDIT)]
        for i, action in enumerate(POWER_CAP_ACTIONS):
            ex_str = f"| action_{action['name']} frac:{action['fraction']}"
            examples.append(self._workspace.parse(ex_str, vowpalwabbit.LabelType.CONTEXTUAL_BANDIT))
        self._workspace.predict(examples)
        probs = examples[0].get_action_probs() if hasattr(examples[0], 'get_action_probs') else []
        for ex in examples:
            self._workspace.finish_example(ex)
        if not probs:
            n = len(POWER_CAP_ACTIONS)
            return [1.0 / n] * n
        return [p for _, p in sorted(probs)]

    def learn(self, context_features: dict[str, float], action_idx: int,
              cost: float, probability: float) -> None:
        shared = self._format_shared(context_features)
        examples = [self._workspace.parse(shared, vowpalwabbit.LabelType.CONTEXTUAL_BANDIT)]
        for i, action in enumerate(POWER_CAP_ACTIONS):
            label = ""
            if i == action_idx:
                label = f"{action_idx}:{cost}:{probability} "
            ex_str = f"{label}| action_{action['name']} frac:{action['fraction']}"
            examples.append(self._workspace.parse(ex_str, vowpalwabbit.LabelType.CONTEXTUAL_BANDIT))
        self._workspace.learn(examples)
        for ex in examples:
            self._workspace.finish_example(ex)

    def save(self) -> None:
        if self._workspace:
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            self._workspace.save(str(self._model_path))

    @staticmethod
    def _format_shared(features: dict[str, float]) -> str:
        parts = ["shared |"]
        for k, v in features.items():
            parts.append(f"{k}:{v}")
        return " ".join(parts)


# ── Pure-Python Fallback ─────────────────────────────────────────────────────

class _SimpleBackend:
    """Lightweight epsilon-greedy bandit without VW dependency.

    Maintains per-action reward estimates using incremental mean updates.
    No feature interaction — treats context features as independent.
    """

    def __init__(self, model_path: Path, epsilon: float = 0.1):
        self._model_path = model_path
        self._epsilon = epsilon
        self._n_actions = len(POWER_CAP_ACTIONS)
        self._counts: list[int] = [0] * self._n_actions
        self._rewards: list[float] = [0.5] * self._n_actions
        self._load()

    def predict(self, context_features: dict[str, float]) -> list[float]:
        probs = [self._epsilon / self._n_actions] * self._n_actions
        best = max(range(self._n_actions), key=lambda i: self._rewards[i])
        probs[best] += 1.0 - self._epsilon
        return probs

    def learn(self, context_features: dict[str, float], action_idx: int,
              cost: float, probability: float) -> None:
        reward = 1.0 - cost
        self._counts[action_idx] += 1
        n = self._counts[action_idx]
        self._rewards[action_idx] += (reward - self._rewards[action_idx]) / n

    def save(self) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"counts": self._counts, "rewards": self._rewards}
        with open(self._model_path, "w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        if self._model_path.exists():
            try:
                with open(self._model_path) as f:
                    data = json.load(f)
                self._counts = data.get("counts", self._counts)
                self._rewards = data.get("rewards", self._rewards)
            except (json.JSONDecodeError, KeyError):
                pass


# ── Energy Bandit ────────────────────────────────────────────────────────────

class EnergyBandit:
    """Contextual bandit for GPU power cap optimization.

    Selects power cap levels (40-100% of TDP) based on workload context.
    Uses VW when available, falls back to simple epsilon-greedy.
    """

    def __init__(
        self,
        data_dir: Path,
        epsilon: float = 0.1,
        retrain_every: int = 500,
        min_corpus: int = 1000,
    ) -> None:
        self._dir = data_dir / "learner"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._epsilon = epsilon
        self._retrain_every = retrain_every
        self._min_corpus = min_corpus

        model_path = self._dir / ("bandit_model.vw" if _HAS_VW else "bandit_model.json")
        if _HAS_VW:
            self._backend = _VWBackend(model_path, epsilon)
            log.info("EnergyBandit: using Vowpal Wabbit backend (epsilon=%.2f)", epsilon)
        else:
            self._backend = _SimpleBackend(model_path, epsilon)
            log.info("EnergyBandit: using simple epsilon-greedy fallback (epsilon=%.2f)", epsilon)

        self._stats = BanditStats()
        self._stats_path = self._dir / "bandit_stats.json"
        self._load_stats()

    # ── Public API ───────────────────────────────────────────────────────

    def suggest(
        self,
        context_features: dict[str, float],
        tdp_w: float,
        anchor_fraction: Optional[float] = None,
    ) -> BanditSuggestion:
        """Suggest a power cap action given the current workload context.

        anchor_fraction: empirically fitted knee from the curve library.
        When provided, exploration is restricted to arms within ±0.10 of
        the knee — the curves already learned the physics; the bandit only
        fine-tunes around it.

        Returns a BanditSuggestion with the cap in watts, clamped to
        [40% TDP, 100% TDP] for safety.
        """
        probs = self._backend.predict(context_features)

        if anchor_fraction is not None:
            masked = [
                p if abs(POWER_CAP_ACTIONS[i]["fraction"] - anchor_fraction) <= 0.10 else 0.0
                for i, p in enumerate(probs)
            ]
            total = sum(masked)
            if total > 0:
                probs = [p / total for p in masked]

        r = random.random()
        cumulative = 0.0
        chosen_idx = len(POWER_CAP_ACTIONS) - 1
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen_idx = i
                break

        action = POWER_CAP_ACTIONS[chosen_idx]
        cap_watts = max(tdp_w * 0.40, min(tdp_w, tdp_w * action["fraction"]))

        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        is_exploration = chosen_idx != best_idx

        return BanditSuggestion(
            action_name=action["name"],
            action_index=chosen_idx,
            cap_fraction=action["fraction"],
            cap_watts=round(cap_watts, 0),
            confidence=probs[chosen_idx],
            is_exploration=is_exploration,
        )

    def update(
        self,
        context_features: dict[str, float],
        action_index: int,
        reward: float,
        probability: float,
    ) -> None:
        """Update the bandit with an observed (context, action, reward) triple."""
        cost = 1.0 - reward
        self._backend.learn(context_features, action_index, cost, probability)
        self._stats.corpus_size += 1
        self._stats.updates_since_retrain += 1

        if self._stats.updates_since_retrain >= self._retrain_every:
            self._save_model()
            self._stats.updates_since_retrain = 0
            self._stats.model_version += 1
            self._stats.last_retrain_at = time.time()
            log.info(
                "EnergyBandit: model checkpoint v%d (corpus=%d)",
                self._stats.model_version, self._stats.corpus_size,
            )

        self._save_stats()

    def warm_start(self, experience_tuples) -> int:
        """Train on historical experience tuples from Phase 1.

        Returns the number of tuples consumed.
        """
        count = 0
        for t in experience_tuples:
            if not t.is_complete() or t.context is None or t.action is None:
                continue
            if t.action.action_type != "power_cap":
                continue

            features = encode_context(
                gpu_name=t.context.gpu_name,
                gpu_arch=t.context.gpu_arch,
                workload_class=t.context.workload_class,
                utilization_gpu_pct=t.context.utilization_gpu_pct,
                utilization_memory_pct=t.context.utilization_memory_pct,
                memory_pressure=t.context.memory_pressure,
                power_draw_w=t.context.power_draw_w,
                power_limit_w=t.context.power_limit_w,
                temperature_c=t.context.temperature_c,
            )

            action_idx = self._match_action(
                t.action.recommended_value, t.context.power_limit_w
            )
            probability = max(0.01, self._epsilon / len(POWER_CAP_ACTIONS))
            self.update(features, action_idx, t.reward or 0.0, probability)
            count += 1

        if count > 0:
            self._save_model()
            log.info("EnergyBandit: warm-started on %d historical tuples", count)

        return count

    def is_ready(self) -> bool:
        """Whether the bandit has enough data to make useful suggestions."""
        return self._stats.corpus_size >= self._min_corpus

    def get_stats(self) -> dict:
        return {
            "backend": "vowpalwabbit" if _HAS_VW else "simple",
            "corpus_size": self._stats.corpus_size,
            "model_version": self._stats.model_version,
            "updates_since_retrain": self._stats.updates_since_retrain,
            "estimated_reward": self._stats.estimated_reward,
            "min_corpus": self._min_corpus,
            "ready": self.is_ready(),
        }

    def evaluate_offline(self, experience_tuples, sample_size: int = 500) -> float:
        """Doubly-robust off-policy evaluation.

        Estimates the expected reward of the current policy using
        logged data from a different policy (the heuristic engine).
        """
        estimates = []
        count = 0

        for t in experience_tuples:
            if count >= sample_size:
                break
            if not t.is_complete() or t.context is None or t.action is None:
                continue
            if t.action.action_type != "power_cap":
                continue

            features = encode_context(
                gpu_name=t.context.gpu_name,
                gpu_arch=t.context.gpu_arch,
                workload_class=t.context.workload_class,
                utilization_gpu_pct=t.context.utilization_gpu_pct,
                utilization_memory_pct=t.context.utilization_memory_pct,
                memory_pressure=t.context.memory_pressure,
                power_draw_w=t.context.power_draw_w,
                power_limit_w=t.context.power_limit_w,
                temperature_c=t.context.temperature_c,
            )

            logged_action_idx = self._match_action(
                t.action.recommended_value, t.context.power_limit_w
            )
            logged_reward = t.reward or 0.0
            logged_prob = max(0.01, self._epsilon / len(POWER_CAP_ACTIONS))

            probs = self._backend.predict(features)
            pi_prob = probs[logged_action_idx] if logged_action_idx < len(probs) else 0.0

            # Inverse propensity scoring with clipping
            iw = min(10.0, pi_prob / logged_prob)
            dr_estimate = iw * logged_reward
            estimates.append(dr_estimate)
            count += 1

        if not estimates:
            return 0.0

        result = sum(estimates) / len(estimates)
        self._stats.estimated_reward = result
        self._save_stats()
        return result

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _match_action(recommended_watts: float, power_limit_w: float) -> int:
        """Map a recommended wattage to the closest discrete action index."""
        if power_limit_w <= 0:
            return len(POWER_CAP_ACTIONS) - 1
        fraction = recommended_watts / power_limit_w
        best_idx = 0
        best_dist = float("inf")
        for i, action in enumerate(POWER_CAP_ACTIONS):
            dist = abs(action["fraction"] - fraction)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _save_model(self) -> None:
        try:
            self._backend.save()
        except Exception as exc:
            log.warning("Failed to save bandit model: %s", exc)

    def _save_stats(self) -> None:
        try:
            with open(self._stats_path, "w") as f:
                json.dump({
                    "corpus_size": self._stats.corpus_size,
                    "model_version": self._stats.model_version,
                    "updates_since_retrain": self._stats.updates_since_retrain,
                    "estimated_reward": self._stats.estimated_reward,
                    "last_retrain_at": self._stats.last_retrain_at,
                }, f)
        except OSError:
            pass

    def _load_stats(self) -> None:
        if self._stats_path.exists():
            try:
                with open(self._stats_path) as f:
                    data = json.load(f)
                self._stats.corpus_size = data.get("corpus_size", 0)
                self._stats.model_version = data.get("model_version", 0)
                self._stats.updates_since_retrain = data.get("updates_since_retrain", 0)
                self._stats.estimated_reward = data.get("estimated_reward", 0.0)
                self._stats.last_retrain_at = data.get("last_retrain_at", 0.0)
            except (json.JSONDecodeError, OSError):
                pass
