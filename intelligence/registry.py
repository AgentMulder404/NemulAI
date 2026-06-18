# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Local + cloud model registry for discovered models."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from efficiency.gpu_specs import ModelProfile, MODEL_PROFILES

log = logging.getLogger("nemulai-intel")


@dataclass
class RegistryEntry:
    model_id: str
    tag: str
    family: str
    profile: ModelProfile
    source: str = "huggingface"
    gpu_rankings: list[dict] = field(default_factory=list)
    status: str = "detected"
    confidence: float = 0.0
    parameter_count: Optional[int] = None
    architecture: str = ""
    downloads_30d: int = 0
    quantization_variants: list[dict] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "model_id": self.model_id,
            "tag": self.tag,
            "family": self.family,
            "source": self.source,
            "gpu_rankings": self.gpu_rankings[:10],
            "status": self.status,
            "confidence": self.confidence,
            "parameter_count": self.parameter_count,
            "architecture": self.architecture,
            "downloads_30d": self.downloads_30d,
            "quantization_variants": self.quantization_variants,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "profile": {
                "tag": self.profile.tag,
                "family": self.profile.family,
                "math_intensity": self.profile.math_intensity,
                "precision": self.profile.precision,
                "is_memory_bound": self.profile.is_memory_bound,
                "typical_util_min": self.profile.typical_util_min,
                "typical_util_max": self.profile.typical_util_max,
            },
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RegistryEntry:
        p = d.get("profile", {})
        profile = ModelProfile(
            tag=p.get("tag", d.get("tag", "")),
            family=p.get("family", d.get("family", "")),
            math_intensity=p.get("math_intensity", 60.0),
            precision=p.get("precision", "fp16"),
            is_memory_bound=p.get("is_memory_bound", False),
            typical_util_min=p.get("typical_util_min", 50),
            typical_util_max=p.get("typical_util_max", 80),
        )
        return cls(
            model_id=d.get("model_id", ""),
            tag=d.get("tag", ""),
            family=d.get("family", ""),
            profile=profile,
            source=d.get("source", "huggingface"),
            gpu_rankings=d.get("gpu_rankings", []),
            status=d.get("status", "detected"),
            confidence=d.get("confidence", 0.0),
            parameter_count=d.get("parameter_count"),
            architecture=d.get("architecture", ""),
            downloads_30d=d.get("downloads_30d", 0),
            quantization_variants=d.get("quantization_variants", []),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )


class ModelRegistry:
    def __init__(
        self,
        data_dir: Path,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self._dir = data_dir / "intelligence"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._local_path = self._dir / "model_registry.json"
        self._entries: dict[str, RegistryEntry] = {}
        self._supabase_url = supabase_url
        self._supabase_key = supabase_key
        self._load_local()

    def register(self, entry: RegistryEntry) -> None:
        now = time.time()
        if entry.created_at == 0.0:
            entry.created_at = now
        entry.updated_at = now

        self._entries[entry.tag] = entry

        if entry.status == "active":
            MODEL_PROFILES[entry.tag] = entry.profile

        self._save_local()

    def get(self, tag: str) -> Optional[RegistryEntry]:
        return self._entries.get(tag)

    def list_all(self, status: Optional[str] = None) -> list[RegistryEntry]:
        entries = list(self._entries.values())
        if status and status != "all":
            entries = [e for e in entries if e.status == status]
        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return entries

    def known_tags(self) -> set[str]:
        existing = set(MODEL_PROFILES.keys())
        discovered = set(self._entries.keys())
        return existing | discovered

    def activate(self, tag: str) -> bool:
        entry = self._entries.get(tag)
        if not entry:
            return False
        entry.status = "active"
        entry.updated_at = time.time()
        MODEL_PROFILES[entry.tag] = entry.profile
        self._save_local()
        return True

    def sync_to_model_profiles(self) -> int:
        count = 0
        for entry in self._entries.values():
            if entry.status == "active":
                MODEL_PROFILES[entry.tag] = entry.profile
                count += 1
        return count

    def sync_to_supabase(self) -> int:
        if not self._supabase_url or not self._supabase_key:
            log.debug("Supabase not configured, skipping sync")
            return 0

        try:
            import requests
        except ImportError:
            return 0

        count = 0
        for entry in self._entries.values():
            row = {
                "model_id": entry.model_id,
                "tag": entry.tag,
                "family": entry.family,
                "source": entry.source,
                "math_intensity": entry.profile.math_intensity,
                "precision": entry.profile.precision,
                "is_memory_bound": entry.profile.is_memory_bound,
                "typical_util_min": entry.profile.typical_util_min,
                "typical_util_max": entry.profile.typical_util_max,
                "parameter_count": entry.parameter_count,
                "architecture": entry.architecture,
                "downloads_30d": entry.downloads_30d,
                "gpu_rankings": entry.gpu_rankings[:10],
                "status": entry.status,
            }

            try:
                resp = requests.post(
                    f"{self._supabase_url}/rest/v1/model_registry",
                    json=row,
                    headers={
                        "apikey": self._supabase_key,
                        "Authorization": f"Bearer {self._supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "resolution=merge-duplicates",
                    },
                    timeout=15,
                )
                if resp.status_code < 300:
                    count += 1
                else:
                    log.warning("Supabase upsert failed for %s: %s", entry.tag, resp.text)
            except Exception as exc:
                log.warning("Supabase upsert error for %s: %s", entry.tag, exc)

        return count

    def _load_local(self) -> None:
        if not self._local_path.exists():
            return

        try:
            data = json.loads(self._local_path.read_text())
            for tag, d in data.items():
                self._entries[tag] = RegistryEntry.from_dict(d)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("Failed to load model registry: %s", exc)

    def _save_local(self) -> None:
        data = {tag: entry.to_dict() for tag, entry in self._entries.items()}
        tmp = self._local_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self._local_path)
