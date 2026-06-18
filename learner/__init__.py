# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

from learner.experience_logger import (
    ActionOutcome,
    ActionTaken,
    ExperienceLogger,
    ExperienceTuple,
    WorkloadContext,
)
from learner.feature_encoder import classify_workload, encode_context, gpu_class
from learner.reward import compute_energy_reward

try:
    from learner.bandit import BanditSuggestion, EnergyBandit
    _HAS_BANDIT = True
except ImportError:
    _HAS_BANDIT = False

__all__ = [
    "ActionOutcome",
    "ActionTaken",
    "ExperienceLogger",
    "ExperienceTuple",
    "WorkloadContext",
    "classify_workload",
    "compute_energy_reward",
    "encode_context",
    "gpu_class",
]

if _HAS_BANDIT:
    __all__ += ["BanditSuggestion", "EnergyBandit"]
