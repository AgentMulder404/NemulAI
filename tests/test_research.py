# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Research Agent (benchmark targets, calibration, suggestions)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from efficiency.gpu_specs import GPU_ARCHITECTURES, ModelProfile
from intelligence.registry import ModelRegistry, RegistryEntry
from intelligence.research import (
    CALIBRATION_ALPHA,
    FLOPS_PER_PARAM_PER_TOKEN,
    BenchmarkTarget,
    CalibrationStore,
    ResearchAgent,
)

GPU_A100 = "A100-SXM4-80GB"
A100_SPEC = GPU_ARCHITECTURES[GPU_A100]


def make_entry(
    tag: str = "llama-3.1-8b",
    family: str = "Llama",
    param_count: int = 8_000_000_000,
    status: str = "estimated",
    gpu_names: list[str] | None = None,
    quant_variants: list[dict] | None = None,
) -> RegistryEntry:
    profile = ModelProfile(
        tag=tag,
        family=family,
        math_intensity=120.0,
        precision="fp16",
        is_memory_bound=False,
        typical_util_min=60,
        typical_util_max=90,
    )
    rankings = []
    for name in gpu_names or [GPU_A100]:
        spec = GPU_ARCHITECTURES[name]
        rankings.append({
            "gpu_name": name,
            "family": spec.family,
            "score": 90.0,
            "joules_per_tflop": 1.2,
            "effective_tflops": round(
                spec.roofline_tflops(
                    profile.math_intensity, profile.typical_util_mid, profile.precision
                ), 2,
            ),
            "cost_per_hr": 3.0,
        })
    return RegistryEntry(
        model_id=f"meta-llama/{tag}",
        tag=tag,
        family=family,
        profile=profile,
        gpu_rankings=rankings,
        status=status,
        confidence=0.9,
        parameter_count=param_count,
        quantization_variants=quant_variants or [],
    )


def make_agent(tmpdir: str, entries: list[RegistryEntry] | None = None) -> ResearchAgent:
    data_dir = Path(tmpdir)
    registry = ModelRegistry(data_dir)
    for entry in entries or []:
        registry.register(entry)

    pipeline = MagicMock()
    pipeline.registry = registry
    pipeline.run_single.return_value = None
    return ResearchAgent(data_dir=data_dir, pipeline=pipeline)


class TestBenchmarkTargets(unittest.TestCase):
    def test_rebuild_creates_targets_with_roofline_math(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = make_entry()
            agent = make_agent(tmp, [entry])

            new = agent.rebuild_targets()

            self.assertEqual(new, 1)
            target = agent.targets[0]
            self.assertEqual(target.model_tag, "llama-3.1-8b")
            self.assertEqual(target.gpu_name, GPU_A100)

            eff_tflops = entry.gpu_rankings[0]["effective_tflops"]
            expected_tps = (eff_tflops * 1e12) / (
                FLOPS_PER_PARAM_PER_TOKEN * entry.parameter_count
            )
            self.assertAlmostEqual(target.expected_tokens_per_sec, expected_tps, places=1)

            power = A100_SPEC.estimated_power_at_utilization(entry.profile.typical_util_mid)
            self.assertAlmostEqual(target.power_w, power, places=1)
            self.assertAlmostEqual(
                target.expected_joules_per_token, power / expected_tps, places=3
            )
            self.assertAlmostEqual(
                target.expected_cost_per_1m_tokens_usd,
                3.0 / 3600.0 * 1e6 / expected_tps,
                places=3,
            )

    def test_unknown_param_count_yields_zero_token_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = make_entry(param_count=0)
            agent = make_agent(tmp, [entry])
            agent.rebuild_targets()

            target = agent.targets[0]
            self.assertEqual(target.expected_tokens_per_sec, 0.0)
            self.assertEqual(target.expected_joules_per_token, 0.0)

    def test_detected_status_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry(status="detected")])
            agent.rebuild_targets()
            self.assertEqual(len(agent.targets), 0)

    def test_targets_persist_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()

            reloaded = make_agent(tmp, [make_entry()])
            self.assertEqual(len(reloaded.targets), 1)
            self.assertEqual(reloaded.targets[0].model_tag, "llama-3.1-8b")

    def test_rebuild_preserves_calibration_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()
            target = agent.targets[0]
            agent.record_measurement(GPU_A100, "meta-llama/llama-3.1-8b",
                                     target.expected_tokens_per_sec * 0.5)

            agent.rebuild_targets()
            target = agent.targets[0]
            self.assertEqual(target.calibration_samples, 1)
            self.assertLess(target.calibration_factor, 1.0)


class TestCalibration(unittest.TestCase):
    def test_measurement_moves_factor_toward_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()
            predicted = agent.targets[0].expected_tokens_per_sec

            update = agent.record_measurement(
                f"NVIDIA {GPU_A100}", "meta-llama/llama-3.1-8b", predicted * 0.6
            )

            self.assertIsNotNone(update)
            self.assertAlmostEqual(update.ratio, 0.6, places=2)
            expected_factor = (1 - CALIBRATION_ALPHA) * 1.0 + CALIBRATION_ALPHA * 0.6
            self.assertAlmostEqual(update.new_factor, expected_factor, places=3)
            self.assertEqual(update.samples, 1)

            target = agent.targets[0]
            self.assertEqual(target.calibration_samples, 1)
            self.assertAlmostEqual(
                target.calibrated_tokens_per_sec,
                round(predicted * expected_factor, 2),
                places=1,
            )

    def test_second_measurement_uses_ewma(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()
            predicted = agent.targets[0].expected_tokens_per_sec

            first = agent.record_measurement(GPU_A100, "meta-llama/llama-3.1-8b", predicted * 0.6)
            second = agent.record_measurement(GPU_A100, "meta-llama/llama-3.1-8b", predicted * 0.6)

            expected = (1 - CALIBRATION_ALPHA) * first.new_factor + CALIBRATION_ALPHA * 0.6
            self.assertAlmostEqual(second.new_factor, expected, places=3)
            self.assertEqual(second.samples, 2)

    def test_anomalous_ratio_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()
            predicted = agent.targets[0].expected_tokens_per_sec

            update = agent.record_measurement(
                GPU_A100, "meta-llama/llama-3.1-8b", predicted * 100
            )
            self.assertIsNone(update)
            self.assertEqual(agent.targets[0].calibration_samples, 0)

    def test_unknown_gpu_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()
            update = agent.record_measurement(
                "TotallyFakeGPU 9000", "meta-llama/llama-3.1-8b", 100.0
            )
            self.assertIsNone(update)

    def test_unknown_model_triggers_on_the_fly_profiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [])
            new_entry = make_entry(tag="qwen3-7b", family="Qwen", param_count=7_000_000_000)

            def fake_run_single(model_id):
                agent.pipeline.registry.register(new_entry)
                return new_entry

            agent.pipeline.run_single.side_effect = fake_run_single

            # No targets yet for this model; record_measurement should profile it,
            # rebuild targets, then calibrate
            target_tps = (new_entry.gpu_rankings[0]["effective_tflops"] * 1e12) / (
                FLOPS_PER_PARAM_PER_TOKEN * new_entry.parameter_count
            )
            update = agent.record_measurement(GPU_A100, "qwen/qwen3-7b", target_tps * 0.8)

            agent.pipeline.run_single.assert_called_once_with("qwen/qwen3-7b")
            self.assertIsNotNone(update)
            self.assertEqual(update.model_tag, "qwen3-7b")

    def test_calibration_generalizes_to_sibling_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            llama_a = make_entry(tag="llama-3.1-8b")
            llama_b = make_entry(tag="llama-3.1-70b", param_count=70_000_000_000)
            agent = make_agent(tmp, [llama_a, llama_b])
            agent.rebuild_targets()

            predicted = next(
                t for t in agent.targets if t.model_tag == "llama-3.1-8b"
            ).expected_tokens_per_sec
            agent.record_measurement(GPU_A100, "meta-llama/llama-3.1-8b", predicted * 0.5)

            sibling = next(t for t in agent.targets if t.model_tag == "llama-3.1-70b")
            # Sibling has no direct measurements but inherits the family factor
            self.assertEqual(sibling.calibration_samples, 0)
            self.assertLess(sibling.calibration_factor, 1.0)

    def test_calibration_store_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            store = CalibrationStore(path)
            store.update(GPU_A100, "Ampere", "Llama", 0.5)

            reloaded = CalibrationStore(path)
            factor, samples = reloaded.lookup(GPU_A100, "Ampere", "Llama")
            self.assertLess(factor, 1.0)
            self.assertEqual(samples, 1)


class TestResultIngestion(unittest.TestCase):
    def _write_result(self, watch_dir: Path, name: str, tok_per_sec: float) -> Path:
        watch_dir.mkdir(parents=True, exist_ok=True)
        path = watch_dir / name
        path.write_text(json.dumps({
            "nemulai_test": True,
            "mode": "model",
            "gpu": f"NVIDIA {GPU_A100}",
            "model": "meta-llama/llama-3.1-8b",
            "throughput": {"tokens": 10000, "tok_per_sec": tok_per_sec, "duration_s": 60.0},
            "efficiency": {"j_per_token": 0.3},
        }))
        return path

    def test_ingests_dropped_result_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()
            predicted = agent.targets[0].expected_tokens_per_sec
            self._write_result(agent.watch_dir, "run1.json", predicted * 0.7)

            updates = agent.ingest_results()

            self.assertEqual(len(updates), 1)
            self.assertAlmostEqual(updates[0].ratio, 0.7, places=2)

    def test_does_not_reingest_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()
            predicted = agent.targets[0].expected_tokens_per_sec
            self._write_result(agent.watch_dir, "run1.json", predicted * 0.7)

            first = agent.ingest_results()
            second = agent.ingest_results()

            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 0)

    def test_skips_non_nemulai_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()
            agent.watch_dir.mkdir(parents=True, exist_ok=True)
            (agent.watch_dir / "other.json").write_text(json.dumps({"foo": "bar"}))
            (agent.watch_dir / "broken.json").write_text("{not json")

            updates = agent.ingest_results()
            self.assertEqual(len(updates), 0)


class TestSuggestions(unittest.TestCase):
    QUANT_VARIANTS = [
        {
            "variant": "int8", "quality_impact": "minimal",
            "throughput_change_pct": 40.0, "memory_reduction_pct": 50.0,
        },
        {
            "variant": "int4-gptq", "quality_impact": "significant",
            "throughput_change_pct": 120.0, "memory_reduction_pct": 75.0,
        },
        {
            "variant": "bf16", "quality_impact": "negligible",
            "throughput_change_pct": 0.0, "memory_reduction_pct": 0.0,
        },
    ]

    def test_ranks_by_cost_per_million_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = make_entry(tag="llama-3.1-8b", param_count=8_000_000_000)
            big = make_entry(tag="llama-3.1-70b", param_count=70_000_000_000)
            agent = make_agent(tmp, [small, big])
            agent.rebuild_targets()

            suggestions = agent.suggest(top_n=5)

            self.assertEqual(len(suggestions), 2)
            # Same GPU + cost, smaller model → more tokens/s → cheaper per token
            self.assertEqual(suggestions[0].model_tag, "llama-3.1-8b")
            self.assertLess(
                suggestions[0].cost_per_1m_tokens_usd,
                suggestions[1].cost_per_1m_tokens_usd,
            )

    def test_budget_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()

            self.assertEqual(len(agent.suggest(budget_per_hr=5.0)), 1)
            self.assertEqual(len(agent.suggest(budget_per_hr=1.0)), 0)

    def test_query_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [
                make_entry(tag="llama-3.1-8b", family="Llama"),
                make_entry(tag="qwen3-7b", family="Qwen", param_count=7_000_000_000),
            ])
            agent.rebuild_targets()

            results = agent.suggest(query="qwen")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].model_tag, "qwen3-7b")

    def test_quantization_pick_respects_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry(quant_variants=self.QUANT_VARIANTS)])
            agent.rebuild_targets()

            suggestion = agent.suggest()[0]
            # int4-gptq is faster but "significant" quality loss — int8 wins
            self.assertEqual(suggestion.quantization, "int8")
            self.assertIn("quality impact", suggestion.quantization_note)

    def test_suggestion_marks_calibrated_pairings(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.rebuild_targets()
            predicted = agent.targets[0].expected_tokens_per_sec
            agent.record_measurement(GPU_A100, "meta-llama/llama-3.1-8b", predicted * 0.8)

            suggestion = agent.suggest()[0]
            self.assertTrue(suggestion.calibrated)
            self.assertEqual(suggestion.calibration_samples, 1)


class TestResearchCycle(unittest.TestCase):
    def test_cycle_without_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])

            result = agent.run_cycle(scan=False)

            self.assertEqual(result.new_models, 0)
            self.assertEqual(result.targets_total, 1)
            self.assertEqual(result.targets_new, 1)
            agent.pipeline.run.assert_not_called()

    def test_cycle_with_scan_calls_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            pipeline_result = MagicMock()
            pipeline_result.registered = 3
            pipeline_result.errors = []
            agent.pipeline.run.return_value = pipeline_result

            result = agent.run_cycle(limit=10, min_downloads=500, min_confidence=0.6)

            self.assertEqual(result.new_models, 3)
            agent.pipeline.run.assert_called_once_with(
                limit=10, min_downloads=500, min_confidence=0.6
            )

    def test_cycle_survives_scan_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp, [make_entry()])
            agent.pipeline.run.side_effect = RuntimeError("HF API down")

            result = agent.run_cycle()

            self.assertEqual(result.new_models, 0)
            self.assertEqual(len(result.errors), 1)
            # Targets still rebuilt despite scan failure
            self.assertEqual(result.targets_total, 1)


if __name__ == "__main__":
    unittest.main()
