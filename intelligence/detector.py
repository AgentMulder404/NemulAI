# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""HuggingFace Hub API client for detecting new model releases."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests

log = logging.getLogger("nemulai-intel")

HF_API_BASE = "https://huggingface.co/api"

_SUPPORTED_PIPELINE_TAGS = frozenset({
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    "image-to-image",
    "automatic-speech-recognition",
    "audio-classification",
    "feature-extraction",
})

_FAMILY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"llama", re.I), "Llama"),
    (re.compile(r"mistral|mixtral", re.I), "Mistral"),
    (re.compile(r"qwen", re.I), "Qwen"),
    (re.compile(r"phi-?\d", re.I), "Phi"),
    (re.compile(r"gemma", re.I), "Gemma"),
    (re.compile(r"gpt[-_]?neo|gpt[-_]?j", re.I), "GPT-NeoX"),
    (re.compile(r"falcon", re.I), "Falcon"),
    (re.compile(r"deepseek", re.I), "DeepSeek"),
    (re.compile(r"starcoder", re.I), "StarCoder"),
    (re.compile(r"codellama", re.I), "Llama"),
    (re.compile(r"stable[-_]?diffusion|sdxl|sd[-_]?\d", re.I), "Diffusion"),
    (re.compile(r"flux", re.I), "Diffusion"),
    (re.compile(r"whisper", re.I), "Whisper"),
    (re.compile(r"wav2vec", re.I), "Whisper"),
    (re.compile(r"bert|roberta|deberta", re.I), "BERT"),
    (re.compile(r"\bt5\b|flan-t5", re.I), "T5"),
    (re.compile(r"vit\b|vision[-_]?transformer", re.I), "ViT"),
    (re.compile(r"clip", re.I), "ViT"),
    (re.compile(r"yolo", re.I), "ViT"),
    (re.compile(r"sam\b", re.I), "ViT"),
    (re.compile(r"resnet|convnext", re.I), "ViT"),
]

_PIPELINE_TAG_TO_FAMILY: dict[str, str] = {
    "text-generation": "LLM",
    "text2text-generation": "T5",
    "fill-mask": "BERT",
    "token-classification": "BERT",
    "question-answering": "BERT",
    "summarization": "T5",
    "translation": "T5",
    "image-classification": "ViT",
    "object-detection": "ViT",
    "image-segmentation": "ViT",
    "text-to-image": "Diffusion",
    "image-to-image": "Diffusion",
    "automatic-speech-recognition": "Whisper",
    "audio-classification": "Whisper",
    "feature-extraction": "BERT",
}

_ARCH_TO_FAMILY: dict[str, str] = {
    "LlamaForCausalLM": "Llama",
    "MistralForCausalLM": "Mistral",
    "MixtralForCausalLM": "Mistral",
    "Qwen2ForCausalLM": "Qwen",
    "Qwen2MoeForCausalLM": "Qwen",
    "PhiForCausalLM": "Phi",
    "Phi3ForCausalLM": "Phi",
    "GemmaForCausalLM": "Gemma",
    "Gemma2ForCausalLM": "Gemma",
    "GPTNeoXForCausalLM": "GPT-NeoX",
    "GPTJForCausalLM": "GPT-NeoX",
    "FalconForCausalLM": "Falcon",
    "DeepseekV2ForCausalLM": "DeepSeek",
    "DeepseekV3ForCausalLM": "DeepSeek",
    "StableDiffusionPipeline": "Diffusion",
    "StableDiffusionXLPipeline": "Diffusion",
    "WhisperForConditionalGeneration": "Whisper",
    "BertModel": "BERT",
    "BertForMaskedLM": "BERT",
    "RobertaModel": "BERT",
    "T5ForConditionalGeneration": "T5",
    "ViTModel": "ViT",
    "ViTForImageClassification": "ViT",
    "CLIPModel": "ViT",
}


@dataclass(frozen=True)
class DetectedModel:
    model_id: str
    tag: str
    author: str
    pipeline_tag: str
    architecture: str
    library: str
    parameter_count: Optional[int]
    license: str
    downloads_30d: int
    trending_score: float
    created_at: str
    sha: str
    raw_metadata: dict


class ModelDetector:
    def __init__(self, session: Optional[requests.Session] = None):
        self._session = session or requests.Session()

    def fetch_trending(self, limit: int = 30) -> list[DetectedModel]:
        try:
            resp = self._session.get(
                f"{HF_API_BASE}/models",
                params={"sort": "trending", "direction": -1, "limit": limit},
                timeout=30,
            )
            resp.raise_for_status()
        except (requests.RequestException, Exception) as exc:
            log.warning("Failed to fetch trending models: %s", exc)
            return []

        results = []
        for item in resp.json():
            pipeline_tag = item.get("pipeline_tag", "")
            if pipeline_tag not in _SUPPORTED_PIPELINE_TAGS:
                continue

            detected = self._parse_model_item(item)
            if detected:
                results.append(detected)

        return results

    def fetch_model_info(self, model_id: str) -> Optional[DetectedModel]:
        try:
            resp = self._session.get(
                f"{HF_API_BASE}/models/{model_id}",
                timeout=30,
            )
            resp.raise_for_status()
        except (requests.RequestException, Exception) as exc:
            log.warning("Failed to fetch model info for %s: %s", model_id, exc)
            return None

        return self._parse_model_item(resp.json())

    def detect_new(
        self,
        known_tags: set[str],
        limit: int = 30,
        min_downloads: int = 1000,
    ) -> list[DetectedModel]:
        trending = self.fetch_trending(limit=limit)
        return [
            m for m in trending
            if m.tag not in known_tags
            and m.downloads_30d >= min_downloads
        ]

    @staticmethod
    def normalize_tag(model_id: str) -> str:
        # "meta-llama/Meta-Llama-3.1-8B" -> "llama-3.1-8b"
        # Take the repo name (after /)
        parts = model_id.split("/")
        name = parts[-1] if len(parts) > 1 else parts[0]

        name = name.lower()
        # Strip common prefixes
        for prefix in ("meta-", "meta_"):
            if name.startswith(prefix):
                name = name[len(prefix):]

        # Collapse repeated hyphens
        name = re.sub(r"-+", "-", name).strip("-")
        return name

    @staticmethod
    def infer_family(
        model_id: str,
        architecture: str,
        pipeline_tag: str,
    ) -> str:
        if architecture and architecture in _ARCH_TO_FAMILY:
            return _ARCH_TO_FAMILY[architecture]

        combined = f"{model_id} {architecture}"
        for pattern, family in _FAMILY_PATTERNS:
            if pattern.search(combined):
                return family

        return _PIPELINE_TAG_TO_FAMILY.get(pipeline_tag, "Unknown")

    def _parse_model_item(self, item: dict) -> Optional[DetectedModel]:
        model_id = item.get("modelId") or item.get("id", "")
        if not model_id:
            return None

        pipeline_tag = item.get("pipeline_tag", "")

        # Extract architecture from config
        config = item.get("config", {})
        architectures = config.get("architectures", [])
        architecture = architectures[0] if architectures else ""

        # Extract parameter count from safetensors metadata
        param_count = None
        safetensors = item.get("safetensors", {})
        if isinstance(safetensors, dict):
            params = safetensors.get("total", 0)
            if params:
                param_count = int(params)

        tag = self.normalize_tag(model_id)
        author = model_id.split("/")[0] if "/" in model_id else ""
        family = self.infer_family(model_id, architecture, pipeline_tag)

        return DetectedModel(
            model_id=model_id,
            tag=tag,
            author=author,
            pipeline_tag=pipeline_tag,
            architecture=architecture,
            library=item.get("library_name", ""),
            parameter_count=param_count,
            license=item.get("license", "") if isinstance(item.get("license"), str) else "",
            downloads_30d=item.get("downloads", 0),
            trending_score=item.get("trendingScore", 0.0) if isinstance(item.get("trendingScore"), (int, float)) else 0.0,
            created_at=item.get("createdAt", ""),
            sha=item.get("sha", ""),
            raw_metadata=item,
        )
