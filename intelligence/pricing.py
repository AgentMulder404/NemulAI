# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""GPU Price-Performance Tracker — live pricing, $/TFLOP rankings, alerts."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from efficiency.gpu_specs import (
    ArchSpec,
    ModelProfile,
    GPU_ARCHITECTURES,
)
from efficiency.cloud_detect import GPU_HOURLY_RATES

log = logging.getLogger("nemulai-intel")


@dataclass(frozen=True)
class PricingSource:
    gpu_model: str
    provider: str
    instance_type: Optional[str]
    on_demand_rate: float
    spot_rate: Optional[float]
    region: str
    fetched_at: float


@dataclass
class PricePerformanceMetrics:
    gpu_name: str
    on_demand_rate: float
    spot_rate: Optional[float]
    effective_tflops: float
    dollars_per_tflop_hr: float
    value_score: float
    provider: str
    is_best_value: bool = False


@dataclass
class PriceAlert:
    gpu_name: str
    provider: str
    alert_type: str
    message: str
    previous_rate: Optional[float]
    current_rate: float
    model_family: str
    created_at: float = 0.0


class GPUPricingTracker:
    def __init__(
        self,
        data_dir: Path,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self._dir = data_dir / "intelligence"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._dir / "gpu_pricing.json"
        self._history_path = self._dir / "gpu_price_history.json"
        self._supabase_url = supabase_url
        self._supabase_key = supabase_key
        self._cached_rates: dict[str, PricingSource] = {}
        self._load_cache()

    def load_static(self) -> dict[str, float]:
        return dict(GPU_HOURLY_RATES)

    def get_rate(self, gpu_name: str, provider: Optional[str] = None) -> float:
        if gpu_name in self._cached_rates:
            src = self._cached_rates[gpu_name]
            if provider is None or src.provider == provider:
                return src.on_demand_rate

        return GPU_HOURLY_RATES.get(gpu_name, 0.0)

    def get_all_rates(self) -> dict[str, float]:
        rates = dict(GPU_HOURLY_RATES)
        for gpu_name, src in self._cached_rates.items():
            rates[gpu_name] = src.on_demand_rate
        return rates

    def get_all_sources(self) -> dict[str, PricingSource]:
        now = time.time()
        result: dict[str, PricingSource] = {}
        for gpu_name, rate in GPU_HOURLY_RATES.items():
            result[gpu_name] = PricingSource(
                gpu_model=gpu_name,
                provider="static",
                instance_type=None,
                on_demand_rate=rate,
                spot_rate=None,
                region="",
                fetched_at=0.0,
            )
        result.update(self._cached_rates)
        return result

    def update_from_json(self, path: Path) -> int:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError, OSError) as exc:
            log.warning("Failed to read pricing JSON %s: %s", path, exc)
            return 0

        if not isinstance(data, list):
            log.warning("Pricing JSON must be a list of objects")
            return 0

        count = 0
        now = time.time()
        for item in data:
            gpu_model = item.get("gpu_model", "")
            if not gpu_model:
                continue

            self._cached_rates[gpu_model] = PricingSource(
                gpu_model=gpu_model,
                provider=item.get("provider", "community"),
                instance_type=item.get("instance_type"),
                on_demand_rate=float(item.get("on_demand_rate", 0)),
                spot_rate=float(item["spot_rate"]) if item.get("spot_rate") else None,
                region=item.get("region", ""),
                fetched_at=now,
            )
            count += 1

        self._save_cache()
        log.info("Updated %d GPU prices from JSON", count)
        return count

    def update_from_supabase(self) -> int:
        if not self._supabase_url or not self._supabase_key:
            log.debug("Supabase not configured, skipping price fetch")
            return 0

        try:
            import requests
        except ImportError:
            return 0

        try:
            resp = requests.get(
                f"{self._supabase_url}/rest/v1/gpu_reference_pricing",
                params={"select": "*"},
                headers={
                    "apikey": self._supabase_key,
                    "Authorization": f"Bearer {self._supabase_key}",
                },
                timeout=15,
            )
            if resp.status_code >= 300:
                log.warning("Supabase pricing fetch failed: %s", resp.text)
                return 0
            rows = resp.json()
        except Exception as exc:
            log.warning("Supabase pricing fetch error: %s", exc)
            return 0

        now = time.time()
        count = 0
        for row in rows:
            gpu_model = row.get("gpu_model", "")
            if not gpu_model:
                continue

            self._cached_rates[gpu_model] = PricingSource(
                gpu_model=gpu_model,
                provider=row.get("provider", "cloud"),
                instance_type=row.get("instance_type"),
                on_demand_rate=float(row.get("rate_usd_per_gpu_hour", 0)),
                spot_rate=float(row["spot_rate_usd_per_gpu_hour"]) if row.get("spot_rate_usd_per_gpu_hour") else None,
                region=row.get("region", ""),
                fetched_at=now,
            )
            count += 1

        self._save_cache()
        log.info("Updated %d GPU prices from Supabase", count)
        return count

    def compute_price_performance(
        self,
        profile: ModelProfile,
        top_n: int = 10,
    ) -> list[PricePerformanceMetrics]:
        rates = self.get_all_rates()
        results: list[PricePerformanceMetrics] = []

        for gpu_name, spec in GPU_ARCHITECTURES.items():
            rate = rates.get(gpu_name, 0.0)
            if rate <= 0:
                continue

            effective_tflops = spec.roofline_tflops(
                profile.math_intensity,
                profile.typical_util_mid,
                profile.precision,
            )

            if effective_tflops <= 0:
                continue

            dollars_per_tflop_hr = rate / effective_tflops

            src = self._cached_rates.get(gpu_name)
            spot = src.spot_rate if src else None
            provider = src.provider if src else "static"

            results.append(PricePerformanceMetrics(
                gpu_name=gpu_name,
                on_demand_rate=rate,
                spot_rate=spot,
                effective_tflops=round(effective_tflops, 2),
                dollars_per_tflop_hr=round(dollars_per_tflop_hr, 4),
                value_score=0.0,
                provider=provider,
            ))

        results.sort(key=lambda x: x.dollars_per_tflop_hr)

        if results:
            best_dpt = results[0].dollars_per_tflop_hr
            worst_dpt = results[-1].dollars_per_tflop_hr if len(results) > 1 else best_dpt
            spread = worst_dpt - best_dpt if worst_dpt > best_dpt else 1.0

            for i, r in enumerate(results):
                r.value_score = round(
                    max(0, (1.0 - (r.dollars_per_tflop_hr - best_dpt) / spread)) * 100,
                    1,
                )
                r.is_best_value = (i == 0)

        if top_n > 0:
            results = results[:top_n]

        return results

    def detect_alerts(self, model_family: str = "all") -> list[PriceAlert]:
        history = self._load_history()
        if not history:
            return []

        last_snapshot = history[-1] if history else {}
        current_rates = self.get_all_rates()
        alerts: list[PriceAlert] = []
        now = time.time()

        for gpu_name, current_rate in current_rates.items():
            prev_rate = last_snapshot.get(gpu_name)
            if prev_rate is None:
                continue

            # Price drop > 10%
            if current_rate < prev_rate * 0.9:
                drop_pct = round((1 - current_rate / prev_rate) * 100, 1)
                alerts.append(PriceAlert(
                    gpu_name=gpu_name,
                    provider="",
                    alert_type="price_drop",
                    message=f"{gpu_name} price dropped {drop_pct}%: ${prev_rate:.2f} → ${current_rate:.2f}/hr",
                    previous_rate=prev_rate,
                    current_rate=current_rate,
                    model_family=model_family,
                    created_at=now,
                ))

        # Spot opportunities
        for gpu_name, src in self._cached_rates.items():
            if src.spot_rate and src.on_demand_rate > 0:
                if src.spot_rate < src.on_demand_rate * 0.5:
                    discount = round((1 - src.spot_rate / src.on_demand_rate) * 100, 1)
                    alerts.append(PriceAlert(
                        gpu_name=gpu_name,
                        provider=src.provider,
                        alert_type="spot_opportunity",
                        message=f"{gpu_name} spot price {discount}% below on-demand: ${src.spot_rate:.2f}/hr",
                        previous_rate=src.on_demand_rate,
                        current_rate=src.spot_rate,
                        model_family=model_family,
                        created_at=now,
                    ))

        return alerts

    def record_history(self) -> None:
        history = self._load_history()
        snapshot = {gpu: rate for gpu, rate in self.get_all_rates().items()}
        snapshot["_timestamp"] = time.time()
        history.append(snapshot)

        # Keep last 90 entries
        if len(history) > 90:
            history = history[-90:]

        tmp = self._history_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(history, indent=2))
        tmp.rename(self._history_path)

    def update_gpu_hourly_rates(self) -> None:
        GPU_HOURLY_RATES.update(self.get_all_rates())

    def sync_to_supabase(self) -> int:
        if not self._supabase_url or not self._supabase_key:
            return 0

        try:
            import requests
        except ImportError:
            return 0

        count = 0
        for gpu_name, src in self._cached_rates.items():
            row = {
                "gpu_model": src.gpu_model,
                "provider": src.provider,
                "instance_type": src.instance_type,
                "rate_usd_per_gpu_hour": src.on_demand_rate,
                "spot_rate_usd_per_gpu_hour": src.spot_rate,
            }
            try:
                resp = requests.post(
                    f"{self._supabase_url}/rest/v1/gpu_reference_pricing",
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
            except Exception as exc:
                log.warning("Supabase price upsert error for %s: %s", gpu_name, exc)

        return count

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return

        try:
            data = json.loads(self._cache_path.read_text())
            for gpu_name, item in data.items():
                self._cached_rates[gpu_name] = PricingSource(
                    gpu_model=item.get("gpu_model", gpu_name),
                    provider=item.get("provider", "cached"),
                    instance_type=item.get("instance_type"),
                    on_demand_rate=float(item.get("on_demand_rate", 0)),
                    spot_rate=float(item["spot_rate"]) if item.get("spot_rate") else None,
                    region=item.get("region", ""),
                    fetched_at=float(item.get("fetched_at", 0)),
                )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("Failed to load pricing cache: %s", exc)

    def _save_cache(self) -> None:
        data = {}
        for gpu_name, src in self._cached_rates.items():
            data[gpu_name] = {
                "gpu_model": src.gpu_model,
                "provider": src.provider,
                "instance_type": src.instance_type,
                "on_demand_rate": src.on_demand_rate,
                "spot_rate": src.spot_rate,
                "region": src.region,
                "fetched_at": src.fetched_at,
            }
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self._cache_path)

    def _load_history(self) -> list[dict]:
        if not self._history_path.exists():
            return []
        try:
            return json.loads(self._history_path.read_text())
        except (json.JSONDecodeError, OSError):
            return []


def get_current_rates() -> dict[str, float]:
    """Module-level helper for estimator/recommend.py integration."""
    # Returns whatever is in GPU_HOURLY_RATES (may have been updated by tracker)
    return dict(GPU_HOURLY_RATES)
