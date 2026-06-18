# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Reward computation for the self-learning optimization agent.

Isolated from the logger so it can be tested and tuned independently.
"""

from __future__ import annotations


def compute_energy_reward(
    energy_before_j: float,
    energy_after_j: float,
    throughput_before: float,
    throughput_after: float,
    throughput_penalty_weight: float = 0.3,
) -> float:
    """Compute a [0, 1] reward for an energy-optimization action.

    Primary signal: fraction of energy saved (higher is better).
    Penalty: multiplicative throughput regression penalty.

    Returns 0.0 when no improvement or when throughput collapses.
    """
    if energy_before_j <= 0:
        return 0.0

    energy_ratio = energy_after_j / energy_before_j
    energy_improvement = max(0.0, 1.0 - energy_ratio)

    throughput_factor = 1.0
    if throughput_before > 0 and throughput_after < throughput_before:
        regression = (throughput_before - throughput_after) / throughput_before
        throughput_factor = max(0.0, 1.0 - throughput_penalty_weight * regression)

    reward = energy_improvement * throughput_factor
    return max(0.0, min(1.0, reward))


def normalize_j_per_flop(
    energy_j: float,
    achieved_tflops: float,
    duration_s: float,
) -> float:
    """Convert raw energy + throughput into Joules per TFLOP.

    Lower values indicate more efficient operation. Returns inf
    when throughput is zero (GPU is idle).
    """
    if achieved_tflops <= 0 or duration_s <= 0:
        return float("inf")
    total_flops = achieved_tflops * duration_s
    return energy_j / total_flops if total_flops > 0 else float("inf")
