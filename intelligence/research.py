# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Research Agent — continuous model research with self-calibrating benchmarks.

Sits on top of the IntelligencePipeline and closes the loop between
roofline *predictions* and *measured* runs:

  1. Each research cycle discovers new models (HuggingFace trending),
     profiles them, and sets a benchmark target (expected tokens/s,
     J/token, $/1M tokens) for every model x GPU pairing.
  2. Measured runs from `nemulai test --output results.json` dropped in
     DATA_DIR/benchmarks/ are ingested automatically; the ratio of
     measured to predicted throughput updates per-(GPU, model-family)
     calibration factors (EWMA), so every future prediction gets closer
     to reality.
  3. suggest() ranks model+GPU pairings by calibrated $/1M tokens and
     attaches the quantization sweet spot for the winning GPU.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from efficiency.gpu_specs import GPU_ARCHITECTURES, resolve_arch
from intelligence.detector import ModelDetector
from intelligence.pipeline import IntelligencePipeline
from intelligence.registry import RegistryEntry

log = logging.getLogger("nemulai-research")

# Decode FLOPs per parameter per generated token (standard 2N approximation)
FLOPS_PER_PARAM_PER_TOKEN = 2.0

# EWMA smoothing for calibration factor updates
CALIBRATION_ALPHA = 0.3

# Sanity bounds on a single measured/predicted ratio — outside this range
# the run is treated as anomalous and skipped
RATIO_MIN, RATIO_MAX = 0.05, 20.0

# Quantization variants considered safe enough to recommend by default
ACCEPTABLE_QUALITY = frozenset({"negligible", "minimal"})

GLOBAL_KEY = "*|*"


@dataclass
class BenchmarkTarget:
    """Expected performance for one model x GPU pairing.

    expected_* fields are RAW roofline predictions; calibrated values are
    derived by multiplying with calibration_factor at read time so the raw
    prediction is never lost as factors evolve.
    """

    model_tag: str
    model_family: str
    gpu_name: str
    gpu_family: str
    precision: str
    expected_tokens_per_sec: float
    expected_joules_per_token: float
    expected_cost_per_1m_tokens_usd: float
    effective_tflops: float
    power_w: float
    cost_per_hr: float
    calibration_factor: float = 1.0
    calibration_samples: int = 0
    measured_tokens_per_sec: float = 0.0
    measured_at: float = 0.0
    updated_at: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.model_tag}|{self.gpu_name}"

    @property
    def calibrated_tokens_per_sec(self) -> float:
        return round(self.expected_tokens_per_sec * self.calibration_factor, 2)

    @property
    def calibrated_joules_per_token(self) -> float:
        if self.calibrated_tokens_per_sec <= 0:
            return 0.0
        return round(self.power_w / self.calibrated_tokens_per_sec, 4)

    @property
    def calibrated_cost_per_1m_tokens_usd(self) -> float:
        if self.calibrated_tokens_per_sec <= 0 or self.cost_per_hr <= 0:
            return 0.0
        return round(self.cost_per_hr / 3600.0 * 1e6 / self.calibrated_tokens_per_sec, 4)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkTarget:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class CalibrationUpdate:
    gpu_name: str
    model_tag: str
    model_family: str
    predicted_tokens_per_sec: float
    measured_tokens_per_sec: float
    ratio: float
    new_factor: float
    samples: int


@dataclass
class PairingSuggestion:
    model_tag: str
    model_family: str
    gpu_name: str
    tokens_per_sec: float
    joules_per_token: float
    cost_per_1m_tokens_usd: float
    cost_per_hr: float
    quantization: str
    quantization_note: str
    calibrated: bool
    calibration_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearchCycleResult:
    new_models: int = 0
    targets_total: int = 0
    targets_new: int = 0
    measurements_ingested: int = 0
    calibration_updates: list[CalibrationUpdate] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


class CalibrationStore:
    """Persistent EWMA correction factors keyed by (GPU, model family).

    Lookups fall back from most to least specific:
      "<gpu_name>|<family>" -> "<gpu_family>|<family>" -> "*|*"
    so a measurement on one GPU improves predictions for its whole
    architecture family, and any measurement nudges the global prior.
    """

    def __init__(self, path: Path):
        self._path = path
        self._factors: dict[str, dict] = {}
        self._ingested: dict[str, float] = {}
        self._load()

    def lookup(self, gpu_name: str, gpu_family: str, model_family: str) -> tuple[float, int]:
        for key in (
            f"{gpu_name}|{model_family}",
            f"{gpu_family}|{model_family}",
            GLOBAL_KEY,
        ):
            entry = self._factors.get(key)
            if entry and entry.get("samples", 0) > 0:
                return entry["factor"], entry["samples"]
        return 1.0, 0

    def update(self, gpu_name: str, gpu_family: str, model_family: str, ratio: float) -> tuple[float, int]:
        ratio = max(RATIO_MIN, min(RATIO_MAX, ratio))
        result: tuple[float, int] = (1.0, 0)
        for i, key in enumerate((
            f"{gpu_name}|{model_family}",
            f"{gpu_family}|{model_family}",
            GLOBAL_KEY,
        )):
            entry = self._factors.setdefault(key, {"factor": 1.0, "samples": 0})
            entry["factor"] = round(
                (1 - CALIBRATION_ALPHA) * entry["factor"] + CALIBRATION_ALPHA * ratio, 4
            )
            entry["samples"] += 1
            if i == 0:
                result = (entry["factor"], entry["samples"])
        self._save()
        return result

    def is_ingested(self, path: Path) -> bool:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return True
        return self._ingested.get(str(path)) == mtime

    def mark_ingested(self, path: Path) -> None:
        try:
            self._ingested[str(path)] = path.stat().st_mtime
        except OSError:
            return
        self._save()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._factors = data.get("factors", {})
            self._ingested = data.get("ingested_files", {})
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("Failed to load calibration store: %s", exc)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"factors": self._factors, "ingested_files": self._ingested}, indent=2
        ))
        tmp.rename(self._path)


class ResearchAgent:
    def __init__(
        self,
        data_dir: Path,
        pipeline: Optional[IntelligencePipeline] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self._data_dir = data_dir
        self._pipeline = pipeline or IntelligencePipeline(
            data_dir, supabase_url=supabase_url, supabase_key=supabase_key
        )
        intel_dir = data_dir / "intelligence"
        intel_dir.mkdir(parents=True, exist_ok=True)
        self._targets_path = intel_dir / "benchmark_targets.json"
        self._calibration = CalibrationStore(intel_dir / "calibration.json")
        self._targets: dict[str, BenchmarkTarget] = {}
        self._load_targets()

    @property
    def pipeline(self) -> IntelligencePipeline:
        return self._pipeline

    @property
    def watch_dir(self) -> Path:
        return self._data_dir / "benchmarks"

    @property
    def targets(self) -> list[BenchmarkTarget]:
        return sorted(self._targets.values(), key=lambda t: t.key)

    # ── Research cycle ──────────────────────────────────────────────────

    def run_cycle(
        self,
        limit: int = 20,
        min_downloads: int = 1000,
        min_confidence: float = 0.5,
        scan: bool = True,
    ) -> ResearchCycleResult:
        start = time.time()
        result = ResearchCycleResult()

        if scan:
            try:
                pipeline_result = self._pipeline.run(
                    limit=limit,
                    min_downloads=min_downloads,
                    min_confidence=min_confidence,
                )
                result.new_models = pipeline_result.registered
                result.errors.extend(pipeline_result.errors)
                log.info("Research scan registered %d new models", result.new_models)
            except Exception as exc:
                msg = f"Model scan failed: {exc}"
                log.warning(msg)
                result.errors.append(msg)

        updates = self.ingest_results()
        result.measurements_ingested = len(updates)
        result.calibration_updates = updates

        result.targets_new = self.rebuild_targets()
        result.targets_total = len(self._targets)

        result.duration_s = round(time.time() - start, 2)
        return result

    def run_forever(self, interval_s: float = 3600.0, **cycle_kwargs) -> None:
        log.info("Research agent watching (interval=%ss)", interval_s)
        while True:
            try:
                result = self.run_cycle(**cycle_kwargs)
                log.info(
                    "Cycle done: %d new models, %d targets (%d new), %d measurements",
                    result.new_models, result.targets_total,
                    result.targets_new, result.measurements_ingested,
                )
            except Exception as exc:
                log.warning("Research cycle failed: %s", exc)
            time.sleep(interval_s)

    # ── Benchmark targets ───────────────────────────────────────────────

    def rebuild_targets(self, top_n_gpus: int = 5) -> int:
        new_count = 0
        for entry in self._pipeline.registry.list_all():
            if entry.status not in ("estimated", "active"):
                continue
            for ranking in entry.gpu_rankings[:top_n_gpus]:
                target = self._compute_target(entry, ranking)
                if not target:
                    continue
                existing = self._targets.get(target.key)
                if existing:
                    # Preserve learned state; refresh raw predictions
                    target.calibration_factor = existing.calibration_factor
                    target.calibration_samples = existing.calibration_samples
                    target.measured_tokens_per_sec = existing.measured_tokens_per_sec
                    target.measured_at = existing.measured_at
                else:
                    factor, samples = self._calibration.lookup(
                        target.gpu_name, target.gpu_family, target.model_family
                    )
                    target.calibration_factor = factor
                    target.calibration_samples = samples
                    new_count += 1
                self._targets[target.key] = target
        self._save_targets()
        return new_count

    def _compute_target(self, entry: RegistryEntry, ranking: dict) -> Optional[BenchmarkTarget]:
        gpu_name = ranking.get("gpu_name", "")
        spec = GPU_ARCHITECTURES.get(gpu_name) or resolve_arch(gpu_name)
        if not spec:
            return None

        profile = entry.profile
        effective_tflops = ranking.get("effective_tflops") or spec.roofline_tflops(
            profile.math_intensity, profile.typical_util_mid, profile.precision
        )
        power_w = spec.estimated_power_at_utilization(profile.typical_util_mid)
        cost_per_hr = ranking.get("cost_per_hr", 0.0) or 0.0

        tokens_per_sec = 0.0
        joules_per_token = 0.0
        cost_per_1m = 0.0
        if entry.parameter_count and entry.parameter_count > 0:
            tokens_per_sec = (effective_tflops * 1e12) / (
                FLOPS_PER_PARAM_PER_TOKEN * entry.parameter_count
            )
            if tokens_per_sec > 0:
                joules_per_token = power_w / tokens_per_sec
                if cost_per_hr > 0:
                    cost_per_1m = cost_per_hr / 3600.0 * 1e6 / tokens_per_sec

        return BenchmarkTarget(
            model_tag=entry.tag,
            model_family=entry.family,
            gpu_name=spec.name,
            gpu_family=spec.family,
            precision=profile.precision,
            expected_tokens_per_sec=round(tokens_per_sec, 2),
            expected_joules_per_token=round(joules_per_token, 4),
            expected_cost_per_1m_tokens_usd=round(cost_per_1m, 4),
            effective_tflops=round(effective_tflops, 2),
            power_w=round(power_w, 1),
            cost_per_hr=cost_per_hr,
            updated_at=time.time(),
        )

    # ── Calibration (the learning loop) ─────────────────────────────────

    def ingest_results(self) -> list[CalibrationUpdate]:
        """Ingest measured runs dropped in DATA_DIR/benchmarks/*.json."""
        if not self.watch_dir.is_dir():
            return []

        updates: list[CalibrationUpdate] = []
        for path in sorted(self.watch_dir.glob("*.json")):
            if self._calibration.is_ingested(path):
                continue
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Skipping unreadable result file %s: %s", path, exc)
                self._calibration.mark_ingested(path)
                continue

            self._calibration.mark_ingested(path)

            if not data.get("nemulai_test") or not data.get("model"):
                continue
            tok_per_sec = (data.get("throughput") or {}).get("tok_per_sec", 0.0)
            if tok_per_sec <= 0:
                continue

            update = self.record_measurement(
                gpu_name=data.get("gpu", ""),
                model_id=data["model"],
                tokens_per_sec=tok_per_sec,
            )
            if update:
                updates.append(update)
        return updates

    def record_measurement(
        self,
        gpu_name: str,
        model_id: str,
        tokens_per_sec: float,
    ) -> Optional[CalibrationUpdate]:
        if tokens_per_sec <= 0:
            return None

        spec = resolve_arch(gpu_name)
        if not spec:
            log.warning("Unknown GPU '%s', cannot calibrate", gpu_name)
            return None

        tag = ModelDetector.normalize_tag(model_id)
        entry = self._pipeline.registry.get(tag)
        if not entry:
            # A measured run of an unknown model is itself a research signal:
            # profile it on the fly so it enters the registry
            try:
                entry = self._pipeline.run_single(model_id)
            except Exception as exc:
                log.warning("Could not profile %s for calibration: %s", model_id, exc)
            if not entry:
                return None
            self.rebuild_targets()

        target = self._targets.get(f"{tag}|{spec.name}")
        if not target or target.expected_tokens_per_sec <= 0:
            log.info("No benchmark target for %s on %s, skipping calibration", tag, spec.name)
            return None

        ratio = tokens_per_sec / target.expected_tokens_per_sec
        if not (RATIO_MIN <= ratio <= RATIO_MAX):
            log.warning(
                "Anomalous measurement for %s on %s (ratio %.2f), skipping",
                tag, spec.name, ratio,
            )
            return None

        factor, samples = self._calibration.update(
            spec.name, spec.family, entry.family, ratio
        )
        target.calibration_factor = factor
        target.calibration_samples = samples
        target.measured_tokens_per_sec = round(tokens_per_sec, 2)
        target.measured_at = time.time()

        # Propagate the improved factor to uncalibrated siblings
        for sibling in self._targets.values():
            if sibling.calibration_samples == 0:
                f, s = self._calibration.lookup(
                    sibling.gpu_name, sibling.gpu_family, sibling.model_family
                )
                sibling.calibration_factor = f

        self._save_targets()

        return CalibrationUpdate(
            gpu_name=spec.name,
            model_tag=tag,
            model_family=entry.family,
            predicted_tokens_per_sec=target.expected_tokens_per_sec,
            measured_tokens_per_sec=round(tokens_per_sec, 2),
            ratio=round(ratio, 4),
            new_factor=factor,
            samples=samples,
        )

    # ── Suggestions ─────────────────────────────────────────────────────

    def suggest(
        self,
        query: Optional[str] = None,
        budget_per_hr: Optional[float] = None,
        top_n: int = 5,
    ) -> list[PairingSuggestion]:
        """Rank model+GPU pairings by calibrated $/1M tokens (cheapest first).

        Pairings without pricing fall back to J/token ordering after all
        priced pairings.
        """
        candidates: list[tuple[tuple, PairingSuggestion]] = []
        q = query.lower() if query else None

        for target in self._targets.values():
            if target.expected_tokens_per_sec <= 0:
                continue
            if q and q not in target.model_tag.lower() and q not in target.model_family.lower():
                continue
            if budget_per_hr is not None and (
                target.cost_per_hr <= 0 or target.cost_per_hr > budget_per_hr
            ):
                continue

            cost_1m = target.calibrated_cost_per_1m_tokens_usd
            quant, quant_note = self._pick_quantization(target)
            suggestion = PairingSuggestion(
                model_tag=target.model_tag,
                model_family=target.model_family,
                gpu_name=target.gpu_name,
                tokens_per_sec=target.calibrated_tokens_per_sec,
                joules_per_token=target.calibrated_joules_per_token,
                cost_per_1m_tokens_usd=cost_1m,
                cost_per_hr=target.cost_per_hr,
                quantization=quant,
                quantization_note=quant_note,
                calibrated=target.calibration_samples > 0,
                calibration_samples=target.calibration_samples,
            )
            sort_key = (
                cost_1m if cost_1m > 0 else float("inf"),
                target.calibrated_joules_per_token or float("inf"),
            )
            candidates.append((sort_key, suggestion))

        candidates.sort(key=lambda c: c[0])
        return [s for _, s in candidates[:top_n]]

    def _pick_quantization(self, target: BenchmarkTarget) -> tuple[str, str]:
        entry = self._pipeline.registry.get(target.model_tag)
        if not entry or not entry.quantization_variants:
            return target.precision, ""

        best = None
        for variant in entry.quantization_variants:
            if variant.get("quality_impact") not in ACCEPTABLE_QUALITY:
                continue
            if best is None or variant.get("throughput_change_pct", 0) > best.get(
                "throughput_change_pct", 0
            ):
                best = variant

        if not best:
            return target.precision, ""

        note = (
            f"{best.get('memory_reduction_pct', 0):.0f}% less VRAM, "
            f"{best.get('throughput_change_pct', 0):+.0f}% throughput, "
            f"{best.get('quality_impact', '?')} quality impact"
        )
        return best.get("variant", target.precision), note

    # ── Persistence ─────────────────────────────────────────────────────

    def _load_targets(self) -> None:
        if not self._targets_path.exists():
            return
        try:
            data = json.loads(self._targets_path.read_text())
            for key, d in data.items():
                self._targets[key] = BenchmarkTarget.from_dict(d)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            log.warning("Failed to load benchmark targets: %s", exc)

    def _save_targets(self) -> None:
        data = {key: t.to_dict() for key, t in self._targets.items()}
        tmp = self._targets_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self._targets_path)
