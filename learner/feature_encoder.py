# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Feature encoding for the self-learning optimization agent.

Converts raw GPU metrics into structured context features for the
contextual bandit (Phase 2) and consistent workload classification
for experience logging (Phase 1).
"""

from __future__ import annotations

import math
import re
from typing import Optional

_PRECISION_HINTS = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "int8": "int8",
    "int4": "int4",
    "fp32": "fp32",
    "float32": "fp32",
    "tf32": "tf32",
    "fp8": "fp8",
}

_MODEL_FAMILIES = {
    "llama": "llm",
    "mistral": "llm",
    "qwen": "llm",
    "phi": "llm",
    "gemma": "llm",
    "gpt": "llm",
    "falcon": "llm",
    "deepseek": "llm",
    "codellama": "llm",
    "starcoder": "llm",
    "stable-diffusion": "diffusion",
    "sdxl": "diffusion",
    "flux": "diffusion",
    "dalle": "diffusion",
    "whisper": "audio",
    "wav2vec": "audio",
    "resnet": "vision",
    "vit": "vision",
    "clip": "vision",
    "yolo": "vision",
    "sam": "vision",
    "bert": "nlp",
    "t5": "nlp",
    "roberta": "nlp",
}


def classify_workload(
    model_tag: Optional[str],
    detected_precision: Optional[str],
    utilization_gpu_pct: float,
    utilization_memory_pct: float,
) -> str:
    """Classify a workload into a human-readable class string.

    Returns strings like "llm-inference-bf16", "training-fp16", "idle".
    """
    if utilization_gpu_pct < 5:
        return "idle"

    family = "unknown"
    if model_tag:
        tag_lower = model_tag.lower()
        for pattern, fam in _MODEL_FAMILIES.items():
            if pattern in tag_lower:
                family = fam
                break

    precision = "mixed"
    if detected_precision:
        precision = _PRECISION_HINTS.get(detected_precision.lower(), detected_precision.lower())

    is_training = utilization_memory_pct > 70 and utilization_gpu_pct > 60
    phase = "training" if is_training else "inference"

    return f"{family}-{phase}-{precision}"


def gpu_class(gpu_name: str) -> str:
    """Normalize a GPU name to a canonical class key.

    "NVIDIA A100-SXM4-80GB" -> "a100_sxm4_80gb"
    """
    name = gpu_name.lower()
    name = re.sub(r"^nvidia\s+", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


def encode_context(
    gpu_name: str,
    gpu_arch: str,
    workload_class: str,
    utilization_gpu_pct: float,
    utilization_memory_pct: float,
    memory_pressure: float,
    power_draw_w: float,
    power_limit_w: float,
    temperature_c: float,
) -> dict[str, float]:
    """Encode a WorkloadContext into flat numerical features.

    Continuous values are bucketized to 5% increments (matching
    the efficiency curve builder's bucketing scheme).
    """

    def _bucket5(val: float) -> float:
        return math.floor(val / 5.0) * 5.0

    features: dict[str, float] = {
        "gpu_class": hash(gpu_class(gpu_name)) % 10000,
        "workload_class": hash(workload_class) % 10000,
        "util_gpu_bucket": _bucket5(max(0, min(100, utilization_gpu_pct))),
        "util_mem_bucket": _bucket5(max(0, min(100, utilization_memory_pct))),
        "mem_pressure": round(max(0, min(1, memory_pressure)), 2),
        "power_ratio": round(power_draw_w / power_limit_w, 2) if power_limit_w > 0 else 0.0,
        "temp_bucket": _bucket5(max(0, min(120, temperature_c))),
        "power_draw_w": round(power_draw_w, 1),
        "power_limit_w": round(power_limit_w, 1),
    }
    return features
