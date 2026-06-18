# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI
"""
Quantization eval harness — measured variants with a quality gate.

The QuantizationAdvisor's tables are priors, not measurements. This harness
actually runs each variant on the local GPU and records:

  - tokens/s and J/token (the economics)
  - perplexity delta vs the baseline precision (the quality gate)

A variant is only recommendable when its perplexity increase stays under the
gate (default 2%) AND it actually improves J/token. Results are persisted and
merged into the model registry, replacing heuristic numbers with measured
ones — and the baseline run feeds the research agent's calibration loop.

Heavy dependencies (torch/transformers/bitsandbytes) load lazily inside
TransformersRunner; the gating/recommendation logic is pure and testable.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("nemulai-quant-eval")

DEFAULT_VARIANTS = ("fp16", "int8", "int4")
DEFAULT_QUALITY_GATE_PCT = 2.0   # max perplexity increase vs baseline
DEFAULT_EVAL_TOKENS = 256        # generation length per prompt
DEFAULT_PPL_WINDOW = 512         # tokens of eval text for perplexity

DEFAULT_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "Write a short story about a robot who learns to garden.",
    "Summarize the causes of the French Revolution in three sentences.",
    "What are the trade-offs between microservices and monoliths?",
]

DEFAULT_PPL_TEXT = (
    "The transformer architecture replaced recurrence with attention, letting "
    "models process whole sequences in parallel. Scaling laws showed that loss "
    "falls predictably with parameters, data, and compute, which motivated the "
    "training of ever larger language models on ever larger corpora. Inference "
    "cost then became the dominant concern: generating a token requires reading "
    "every weight once, so memory bandwidth, not arithmetic, bounds decode "
    "throughput on modern accelerators. Quantization reduces the bytes each "
    "weight occupies, trading a small amount of fidelity for proportionally "
    "higher decode speed and lower energy per token."
)


@dataclass
class MeasuredVariant:
    variant: str
    load_ok: bool = False
    tokens_per_sec: float = 0.0
    avg_power_w: float = 0.0
    joules_per_token: float = 0.0
    vram_gb: float = 0.0
    perplexity: float = 0.0
    ppl_delta_pct: float = 0.0       # vs baseline (set by the harness)
    passes_quality_gate: bool = False
    j_per_token_change_pct: float = 0.0  # vs baseline, negative = better
    error: str = ""


@dataclass
class QuantEvalResult:
    model_id: str
    gpu_name: str
    baseline_variant: str
    quality_gate_pct: float
    variants: list[MeasuredVariant] = field(default_factory=list)
    recommended: Optional[str] = None
    evaluated_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "gpu_name": self.gpu_name,
            "baseline_variant": self.baseline_variant,
            "quality_gate_pct": self.quality_gate_pct,
            "recommended": self.recommended,
            "evaluated_at": self.evaluated_at,
            "variants": [asdict(v) for v in self.variants],
        }


# RunnerFn(model_id, variant, gpu_index) -> MeasuredVariant
RunnerFn = Callable[[str, str, int], MeasuredVariant]


class QuantEvalHarness:
    def __init__(
        self,
        data_dir: Path,
        runner: Optional[RunnerFn] = None,
        quality_gate_pct: float = DEFAULT_QUALITY_GATE_PCT,
    ):
        self._dir = data_dir / "intelligence"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._results_path = self._dir / "quant_eval.json"
        self._runner = runner or TransformersRunner().run
        self._gate = quality_gate_pct

    def evaluate(
        self,
        model_id: str,
        variants=DEFAULT_VARIANTS,
        gpu_index: int = 0,
        gpu_name: str = "",
    ) -> QuantEvalResult:
        baseline_variant = variants[0]
        result = QuantEvalResult(
            model_id=model_id,
            gpu_name=gpu_name,
            baseline_variant=baseline_variant,
            quality_gate_pct=self._gate,
        )

        for variant in variants:
            log.info("Quant eval: running %s as %s", model_id, variant)
            try:
                measured = self._runner(model_id, variant, gpu_index)
            except Exception as exc:
                measured = MeasuredVariant(variant=variant, error=str(exc))
            result.variants.append(measured)
            # A failed baseline invalidates the whole eval
            if variant == baseline_variant and not measured.load_ok:
                log.warning("Quant eval: baseline %s failed (%s)", variant, measured.error)
                break

        self.apply_gates(result)
        result.evaluated_at = time.time()
        self._persist(result)
        return result

    def apply_gates(self, result: QuantEvalResult) -> None:
        """Pure gating + recommendation logic (testable without a GPU)."""
        baseline = next(
            (v for v in result.variants
             if v.variant == result.baseline_variant and v.load_ok),
            None,
        )
        if baseline is None:
            result.recommended = None
            return

        baseline.ppl_delta_pct = 0.0
        baseline.passes_quality_gate = True
        baseline.j_per_token_change_pct = 0.0

        candidates: list[MeasuredVariant] = []
        for v in result.variants:
            if v.variant == result.baseline_variant or not v.load_ok:
                continue

            if baseline.perplexity > 0 and v.perplexity > 0:
                v.ppl_delta_pct = round(
                    (v.perplexity - baseline.perplexity) / baseline.perplexity * 100.0, 3
                )
                v.passes_quality_gate = v.ppl_delta_pct <= self._gate
            else:
                # No quality measurement -> cannot pass the gate
                v.passes_quality_gate = False

            if baseline.joules_per_token > 0 and v.joules_per_token > 0:
                v.j_per_token_change_pct = round(
                    (v.joules_per_token - baseline.joules_per_token)
                    / baseline.joules_per_token * 100.0, 2
                )

            if v.passes_quality_gate and v.j_per_token_change_pct < 0:
                candidates.append(v)

        if candidates:
            best = min(candidates, key=lambda v: v.joules_per_token)
            result.recommended = best.variant
        else:
            result.recommended = result.baseline_variant

    # ── Persistence + registry merge ──────────────────────────────────────

    def _persist(self, result: QuantEvalResult) -> None:
        data = {}
        if self._results_path.exists():
            try:
                data = json.loads(self._results_path.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
        data[result.model_id] = result.to_dict()
        tmp = self._results_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self._results_path)

    def merge_into_registry(self, registry, result: QuantEvalResult) -> bool:
        """Replace heuristic numbers on the registry entry with measured ones."""
        from intelligence.detector import ModelDetector

        tag = ModelDetector.normalize_tag(result.model_id)
        entry = registry.get(tag)
        if not entry:
            return False

        measured_by_name = {v.variant: v for v in result.variants if v.load_ok}
        for qv in entry.quantization_variants:
            m = measured_by_name.get(qv.get("variant", ""))
            if not m:
                continue
            qv["measured_tokens_per_sec"] = round(m.tokens_per_sec, 1)
            qv["measured_j_per_token"] = round(m.joules_per_token, 4)
            qv["measured_ppl_delta_pct"] = m.ppl_delta_pct
            qv["passes_quality_gate"] = m.passes_quality_gate
            qv["recommended"] = (m.variant == result.recommended)

        registry.register(entry)
        return True


class TransformersRunner:
    """Real measurement runner: loads the model at the requested precision,
    benchmarks generation tokens/s + power, and computes perplexity.

    Requires torch + transformers (+ bitsandbytes for int8/int4). All imports
    are lazy so the harness stays importable on machines without them.
    """

    def __init__(
        self,
        prompts: Optional[list[str]] = None,
        max_new_tokens: int = DEFAULT_EVAL_TOKENS,
        ppl_text: str = DEFAULT_PPL_TEXT,
    ):
        self._prompts = prompts or DEFAULT_PROMPTS
        self._max_new_tokens = max_new_tokens
        self._ppl_text = ppl_text

    def run(self, model_id: str, variant: str, gpu_index: int) -> MeasuredVariant:
        out = MeasuredVariant(variant=variant)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            out.error = f"transformers/torch not installed: {exc}"
            return out

        load_kwargs: dict = {"device_map": {"": gpu_index}}
        if variant in ("fp16", "bf16"):
            load_kwargs["torch_dtype"] = torch.float16 if variant == "fp16" else torch.bfloat16
        elif variant == "int8":
            load_kwargs["load_in_8bit"] = True
        elif variant in ("int4", "int4-gptq", "int4-awq"):
            load_kwargs["load_in_4bit"] = True
        else:
            out.error = f"Unsupported variant: {variant}"
            return out

        monitor = self._start_power_monitor(gpu_index)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            model.eval()
            out.load_ok = True

            # Throughput: greedy generation over the prompt set
            total_tokens = 0
            start = time.time()
            with torch.no_grad():
                for prompt in self._prompts:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=self._max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    total_tokens += generated.shape[-1] - inputs["input_ids"].shape[-1]
            duration = max(1e-6, time.time() - start)
            out.tokens_per_sec = total_tokens / duration

            # Quality: perplexity over a fixed text window
            enc = tokenizer(
                self._ppl_text, return_tensors="pt",
                truncation=True, max_length=DEFAULT_PPL_WINDOW,
            ).to(model.device)
            with torch.no_grad():
                loss = model(**enc, labels=enc["input_ids"]).loss
            out.perplexity = float(torch.exp(loss))

            # VRAM footprint
            try:
                out.vram_gb = torch.cuda.memory_allocated(gpu_index) / 1e9
            except Exception:
                pass

            del model
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as exc:
            out.error = str(exc)
            out.load_ok = out.load_ok and out.tokens_per_sec > 0
        finally:
            avg_power = self._stop_power_monitor(monitor)
            out.avg_power_w = avg_power
            if out.tokens_per_sec > 0 and avg_power > 0:
                out.joules_per_token = avg_power / out.tokens_per_sec

        return out

    # ── Power sampling (reuses the nemulai-test monitor) ──────────────────

    @staticmethod
    def _start_power_monitor(gpu_index: int):
        try:
            from test_runner import GPUMonitor
            monitor = GPUMonitor(gpu_index=gpu_index)
            monitor.start()
            return monitor
        except Exception:
            return None

    @staticmethod
    def _stop_power_monitor(monitor) -> float:
        if monitor is None:
            return 0.0
        try:
            samples = monitor.stop()
            if not samples:
                return 0.0
            return sum(s.power_w for s in samples) / len(samples)
        except Exception:
            return 0.0
