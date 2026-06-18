# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Auto-generate ModelProfile from HuggingFace model metadata.

Heuristic maps are validated against the 16 hand-curated profiles in
gpu_specs.MODEL_PROFILES to ensure consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from efficiency.gpu_specs import ModelProfile
from intelligence.detector import DetectedModel, ModelDetector

log = logging.getLogger("nemulai-intel")

# architecture → (base_math_intensity, default_precision)
# Values calibrated against existing MODEL_PROFILES ground truth:
#   llama-3-8b:  intensity=95,  arch=LlamaForCausalLM
#   llama-3-70b: intensity=185, arch=LlamaForCausalLM (scaled by params)
#   bert-base:   intensity=10,  arch=BertModel
#   whisper-large: intensity=28, arch=WhisperForConditionalGeneration
_ARCHITECTURE_INTENSITY: dict[str, tuple[float, str]] = {
    "LlamaForCausalLM": (90.0, "bf16"),
    "MistralForCausalLM": (85.0, "bf16"),
    "MixtralForCausalLM": (105.0, "bf16"),
    "Qwen2ForCausalLM": (85.0, "bf16"),
    "Qwen2MoeForCausalLM": (100.0, "bf16"),
    "PhiForCausalLM": (80.0, "bf16"),
    "Phi3ForCausalLM": (80.0, "bf16"),
    "GemmaForCausalLM": (85.0, "bf16"),
    "Gemma2ForCausalLM": (85.0, "bf16"),
    "GPTNeoXForCausalLM": (100.0, "fp16"),
    "GPTJForCausalLM": (95.0, "fp16"),
    "FalconForCausalLM": (95.0, "bf16"),
    "DeepseekV2ForCausalLM": (110.0, "bf16"),
    "DeepseekV3ForCausalLM": (120.0, "bf16"),
    "BertModel": (10.0, "fp16"),
    "BertForMaskedLM": (10.0, "fp16"),
    "RobertaModel": (12.0, "fp16"),
    "T5ForConditionalGeneration": (95.0, "bf16"),
    "WhisperForConditionalGeneration": (25.0, "fp16"),
    "ViTModel": (35.0, "fp16"),
    "ViTForImageClassification": (35.0, "fp16"),
    "CLIPModel": (30.0, "fp16"),
    "StableDiffusionPipeline": (48.0, "fp16"),
    "StableDiffusionXLPipeline": (50.0, "fp16"),
}

# Family-level fallbacks when architecture is not in the map
_FAMILY_INTENSITY: dict[str, tuple[float, str]] = {
    "Llama": (90.0, "bf16"),
    "Mistral": (85.0, "bf16"),
    "Qwen": (85.0, "bf16"),
    "Phi": (80.0, "bf16"),
    "Gemma": (85.0, "bf16"),
    "GPT-NeoX": (100.0, "fp16"),
    "Falcon": (95.0, "bf16"),
    "DeepSeek": (115.0, "bf16"),
    "StarCoder": (90.0, "bf16"),
    "BERT": (10.0, "fp16"),
    "T5": (95.0, "bf16"),
    "Whisper": (25.0, "fp16"),
    "ViT": (35.0, "fp16"),
    "Diffusion": (48.0, "fp16"),
    "LLM": (100.0, "bf16"),
    "Unknown": (60.0, "fp16"),
}

# Larger models have higher effective math intensity because
# matmul dimensions grow (O(n^3) compute vs O(n^2) memory)
_PARAM_SCALING: list[tuple[float, float]] = [
    (1e9, 0.85),
    (7e9, 1.0),
    (30e9, 1.15),
    (70e9, 1.35),
    (200e9, 1.55),
    (float("inf"), 1.70),
]

# Family → (typical_util_min, typical_util_max)
_FAMILY_UTIL_RANGES: dict[str, tuple[int, int]] = {
    "Llama": (60, 85),
    "Mistral": (55, 80),
    "Qwen": (55, 80),
    "Phi": (50, 75),
    "Gemma": (55, 80),
    "GPT-NeoX": (65, 90),
    "Falcon": (65, 90),
    "DeepSeek": (70, 90),
    "StarCoder": (55, 80),
    "BERT": (35, 65),
    "T5": (55, 85),
    "Whisper": (30, 60),
    "ViT": (40, 70),
    "Diffusion": (50, 80),
    "LLM": (55, 80),
    "Unknown": (40, 70),
}

MEMORY_BOUND_THRESHOLD = 50.0


@dataclass
class ProfileResult:
    profile: ModelProfile
    confidence: float
    reasoning: str
    inferred_from: str


class ModelProfiler:
    def profile(self, detected: DetectedModel) -> ProfileResult:
        return self.profile_from_metadata(
            model_id=detected.model_id,
            tag=detected.tag,
            architecture=detected.architecture,
            parameter_count=detected.parameter_count,
            pipeline_tag=detected.pipeline_tag,
            detected_family=ModelDetector.infer_family(
                detected.model_id, detected.architecture, detected.pipeline_tag
            ),
        )

    def profile_from_metadata(
        self,
        model_id: str,
        tag: str,
        architecture: str,
        parameter_count: Optional[int],
        pipeline_tag: str,
        family: Optional[str] = None,
        detected_family: Optional[str] = None,
    ) -> ProfileResult:
        resolved_family = detected_family or family or "Unknown"
        reasoning_parts = []

        # Step 1: Resolve base intensity + precision
        if architecture and architecture in _ARCHITECTURE_INTENSITY:
            base_intensity, precision = _ARCHITECTURE_INTENSITY[architecture]
            inferred_from = "architecture"
            reasoning_parts.append(
                f"Architecture {architecture} → base intensity {base_intensity}"
            )
            confidence = 0.85
        elif resolved_family in _FAMILY_INTENSITY:
            base_intensity, precision = _FAMILY_INTENSITY[resolved_family]
            inferred_from = "family"
            reasoning_parts.append(
                f"Family {resolved_family} → base intensity {base_intensity}"
            )
            confidence = 0.65
        else:
            base_intensity, precision = _FAMILY_INTENSITY["Unknown"]
            inferred_from = "fallback"
            reasoning_parts.append(
                f"Unknown family/architecture → fallback intensity {base_intensity}"
            )
            confidence = 0.35

        # Step 2: Scale by parameter count
        if parameter_count and parameter_count > 0:
            scale = self._scale_intensity_by_params(parameter_count)
            intensity = base_intensity * scale
            reasoning_parts.append(
                f"Parameter count {parameter_count/1e9:.1f}B → scale {scale:.2f} → intensity {intensity:.1f}"
            )
            confidence = min(confidence + 0.10, 0.95)
        else:
            intensity = base_intensity
            reasoning_parts.append("No parameter count available, using base intensity")
            confidence = max(confidence - 0.10, 0.20)

        # Step 3: Memory-bound determination
        is_memory_bound = intensity < MEMORY_BOUND_THRESHOLD
        reasoning_parts.append(
            f"{'Memory' if is_memory_bound else 'Compute'}-bound "
            f"(intensity {intensity:.1f} {'<' if is_memory_bound else '>='} {MEMORY_BOUND_THRESHOLD})"
        )

        # Step 4: Utilization range
        util_min, util_max = self._estimate_util_range(resolved_family, parameter_count)
        reasoning_parts.append(f"Utilization range: {util_min}-{util_max}%")

        profile = ModelProfile(
            tag=tag,
            family=resolved_family,
            math_intensity=round(intensity, 1),
            precision=precision,
            is_memory_bound=is_memory_bound,
            typical_util_min=util_min,
            typical_util_max=util_max,
        )

        return ProfileResult(
            profile=profile,
            confidence=round(confidence, 2),
            reasoning="; ".join(reasoning_parts),
            inferred_from=inferred_from,
        )

    @staticmethod
    def _scale_intensity_by_params(param_count: int) -> float:
        for threshold, scale in _PARAM_SCALING:
            if param_count < threshold:
                return scale
        return _PARAM_SCALING[-1][1]

    @staticmethod
    def _estimate_util_range(
        family: str, param_count: Optional[int]
    ) -> tuple[int, int]:
        base_min, base_max = _FAMILY_UTIL_RANGES.get(
            family, _FAMILY_UTIL_RANGES["Unknown"]
        )
        # Larger models tend to have higher utilization (better saturation)
        if param_count and param_count > 30e9:
            base_min = min(base_min + 10, 90)
            base_max = min(base_max + 5, 95)
        elif param_count and param_count < 1e9:
            base_min = max(base_min - 10, 20)
            base_max = max(base_max - 5, 50)

        return base_min, base_max
