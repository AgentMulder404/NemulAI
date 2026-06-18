# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optimization engineering stack:

  1. ThroughputProbe — true tokens/s from inference-server metrics
  2. Throughput-aware autopilot observation + clock-lock commands
  3. Empirical cap curves + knee picking + bandit anchor
  4. Phase detection + dynamic clock tuner
  5. Quantization eval harness with quality gates
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from command_receiver import CommandReceiver
from efficiency.phase_control import (
    PHASE_COMPUTE,
    PHASE_IDLE,
    PHASE_MEMORY,
    DynamicClockTuner,
    PhaseDetector,
)
from intelligence.quant_eval import (
    MeasuredVariant,
    QuantEvalHarness,
    QuantEvalResult,
)
from learner.bandit import POWER_CAP_ACTIONS, EnergyBandit
from learner.curves import (
    CurveLibrary,
    fit_curve,
    knee_fraction,
)
from learner.experience_logger import (
    ActionOutcome,
    ActionTaken,
    ExperienceTuple,
    WorkloadContext,
)
from throughput_probe import (
    ThroughputProbe,
    parse_prometheus_total,
    parse_sources,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. ThroughputProbe
# ═══════════════════════════════════════════════════════════════════════

VLLM_METRICS = """\
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="llama-3.1-8b"} 120000.0
vllm:prompt_tokens_total{model_name="llama-3.1-8b"} 50000.0
some_other_metric 42
"""


class TestThroughputProbe(unittest.TestCase):
    def test_parse_sources_with_and_without_gpus(self):
        sources = parse_sources("http://a:8000/metrics=0,1;http://b:8001/metrics")
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].gpu_indices, [0, 1])
        self.assertIsNone(sources[1].gpu_indices)

    def test_parse_sources_empty(self):
        self.assertEqual(parse_sources(""), [])
        self.assertEqual(parse_sources(" ; "), [])

    def test_parse_prometheus_finds_vllm_counter(self):
        self.assertEqual(parse_prometheus_total(VLLM_METRICS), 120000.0)

    def test_parse_prometheus_no_known_counter(self):
        self.assertIsNone(parse_prometheus_total("foo_total 5\nbar 7\n"))

    def test_rate_from_two_scrapes(self):
        probe = ThroughputProbe("http://x/metrics", scrape_interval_s=10)
        src = probe.sources[0]

        bodies = iter([
            VLLM_METRICS,
            VLLM_METRICS.replace("120000.0", "121500.0"),
        ])

        def fake_urlopen(req, timeout=0):
            mock = MagicMock()
            mock.read.return_value = next(bodies).encode()
            mock.__enter__ = lambda s: s
            mock.__exit__ = lambda s, *a: False
            return mock

        with patch("throughput_probe.urllib.request.urlopen", side_effect=fake_urlopen):
            first = probe.sample()
            self.assertEqual(first, {})  # first scrape: no delta yet

            src.last_scrape_ts -= 15  # age the scrape so the next one fires
            ts_before = src.last_scrape_ts
            second = probe.sample()

        # 1500 tokens over ~15s ≈ 100 tok/s, applied to all GPUs (-1)
        self.assertIn(-1, second)
        self.assertAlmostEqual(second[-1], 1500 / (time.time() - ts_before), delta=15)
        self.assertAlmostEqual(probe.rate_for_gpu(3, second), second[-1])

    def test_counter_reset_yields_no_rate(self):
        probe = ThroughputProbe("http://x/metrics")
        src = probe.sources[0]
        src.last_total = 999999.0
        src.last_scrape_ts = time.time() - 20

        def fake_urlopen(req, timeout=0):
            mock = MagicMock()
            mock.read.return_value = VLLM_METRICS.encode()
            mock.__enter__ = lambda s: s
            mock.__exit__ = lambda s, *a: False
            return mock

        with patch("throughput_probe.urllib.request.urlopen", side_effect=fake_urlopen):
            rates = probe.sample()
        self.assertEqual(rates, {})

    def test_scrape_failure_is_quiet(self):
        probe = ThroughputProbe("http://nope:1/metrics")
        with patch("throughput_probe.urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertEqual(probe.sample(), {})


# ═══════════════════════════════════════════════════════════════════════
# 2. Throughput-aware observation + clock locks
# ═══════════════════════════════════════════════════════════════════════

def make_receiver(**kwargs) -> CommandReceiver:
    rx = CommandReceiver(
        endpoint="https://www.nemulai.com/v1/metrics/ingest",
        api_key="alum_test", machine_id="m-test", **kwargs,
    )
    rx._report_result = MagicMock()
    return rx


class TestThroughputObservation(unittest.TestCase):
    def _capped(self, baseline_tps: float) -> CommandReceiver:
        rx = make_receiver()
        for _ in range(10):
            rx.record_sample(0, 90.0, 300.0, throughput=baseline_tps)
        with patch("efficiency.power_control.set_power_limit", return_value=True), \
             patch("efficiency.power_control.get_power_limit", return_value=350):
            rx._execute({"id": "cmd-1", "command_type": "apply_power_cap",
                         "params": {"gpu_index": 0, "watts": 250,
                                    "observation_window_s": 60,
                                    "throughput_tolerance_pct": 10}})
        return rx

    def test_throughput_is_authoritative_over_util(self):
        rx = self._capped(baseline_tps=1000.0)
        # Utilization HOLDS (the NVML lie) but true throughput collapses 40%
        for _ in range(10):
            rx.record_sample(0, 95.0, 240.0, throughput=600.0)
        rx._observations[0].applied_at = time.time() - 120

        with patch("efficiency.power_control.set_power_limit", return_value=True) as set_mock:
            outcomes = rx.check_observations()
            set_mock.assert_called_once_with(0, 350, quiet=True)

        out = outcomes[0]
        self.assertTrue(out.rolled_back)
        self.assertEqual(out.regression_signal, "throughput")
        self.assertAlmostEqual(out.throughput_drop_pct, 40.0, delta=0.1)

    def test_throughput_within_tolerance_holds(self):
        rx = self._capped(baseline_tps=1000.0)
        for _ in range(10):
            rx.record_sample(0, 50.0, 240.0, throughput=950.0)  # util halves; tps fine
        rx._observations[0].applied_at = time.time() - 120

        with patch("efficiency.power_control.set_power_limit") as set_mock:
            outcomes = rx.check_observations()
            set_mock.assert_not_called()

        self.assertFalse(outcomes[0].rolled_back)
        self.assertEqual(outcomes[0].regression_signal, "throughput")

    def test_falls_back_to_util_without_tps(self):
        rx = make_receiver()
        for _ in range(10):
            rx.record_sample(0, 80.0, 300.0)  # no throughput signal
        with patch("efficiency.power_control.set_power_limit", return_value=True), \
             patch("efficiency.power_control.get_power_limit", return_value=350):
            rx._execute({"id": "cmd-1", "command_type": "apply_power_cap",
                         "params": {"gpu_index": 0, "watts": 250}})
        for _ in range(10):
            rx.record_sample(0, 75.0, 240.0)
        rx._observations[0].applied_at = time.time() - 400

        outcomes = rx.check_observations()
        self.assertEqual(outcomes[0].regression_signal, "utilization")


class TestClockLockCommands(unittest.TestCase):
    def test_apply_clock_lock_opens_observation(self):
        rx = make_receiver()
        with patch("efficiency.power_control.set_gpu_clock_lock", return_value=True) as lock:
            success, msg = rx._execute({
                "id": "cmd-1", "command_type": "apply_clock_lock",
                "params": {"gpu_index": 0, "max_mhz": 900, "min_mhz": 900},
            })
            lock.assert_called_once_with(0, 900, 900, quiet=True)
        self.assertTrue(success)
        self.assertEqual(rx.open_observations, 1)
        self.assertEqual(rx._observations[0].action_type, "clock_lock")
        self.assertEqual(rx._observations[0].watts, 900.0)

    def test_sm_fraction_resolves_to_mhz(self):
        rx = make_receiver()
        with patch("efficiency.power_control.set_gpu_clock_lock", return_value=True) as lock, \
             patch("efficiency.power_control.get_max_sm_clock", return_value=1980):
            success, msg = rx._execute({
                "id": "cmd-1", "command_type": "apply_clock_lock",
                "params": {"gpu_index": 0, "sm_fraction": 0.65},
            })
        self.assertTrue(success, msg)
        lock.assert_called_once_with(0, 1287, 1287, quiet=True)
        self.assertEqual(rx._observations[0].watts, 1287.0)

    def test_sm_fraction_without_boost_clock_fails(self):
        rx = make_receiver()
        with patch("efficiency.power_control.get_max_sm_clock", return_value=0):
            success, msg = rx._execute({
                "id": "cmd-1", "command_type": "apply_clock_lock",
                "params": {"gpu_index": 0, "sm_fraction": 0.65},
            })
        self.assertFalse(success)
        self.assertIn("max SM clock unavailable", msg)

    def test_clock_regression_resets_lock(self):
        rx = make_receiver()
        for _ in range(10):
            rx.record_sample(0, 90.0, 300.0, throughput=1000.0)
        with patch("efficiency.power_control.set_gpu_clock_lock", return_value=True):
            rx._execute({"id": "cmd-1", "command_type": "apply_clock_lock",
                         "params": {"gpu_index": 0, "max_mhz": 700,
                                    "throughput_tolerance_pct": 10}})
        for _ in range(10):
            rx.record_sample(0, 90.0, 200.0, throughput=500.0)
        rx._observations[0].applied_at = time.time() - 400

        with patch("efficiency.power_control.reset_gpu_clock_lock", return_value=True) as reset:
            outcomes = rx.check_observations()
            reset.assert_called_once_with(0, quiet=True)
        self.assertTrue(outcomes[0].rolled_back)

    def test_clock_lock_validation(self):
        rx = make_receiver()
        bad = [
            {"gpu_index": 0},                                  # missing mhz
            {"gpu_index": 0, "max_mhz": 100},                  # below floor
            {"gpu_index": 0, "max_mhz": 5000},                 # above ceiling
            {"gpu_index": 0, "min_mhz": 1500, "max_mhz": 900}, # inverted
            {"gpu_index": 0, "sm_fraction": 0.1},              # fraction floor
        ]
        for params in bad:
            success, _ = rx._execute(
                {"id": "x", "command_type": "apply_clock_lock", "params": params}
            )
            self.assertFalse(success, f"should reject {params}")

    def test_rollback_clock_lock_command(self):
        rx = make_receiver()
        with patch("efficiency.power_control.reset_gpu_clock_lock", return_value=True):
            success, msg = rx._execute({
                "id": "cmd-1", "command_type": "rollback_clock_lock",
                "params": {"gpu_index": 0},
            })
        self.assertTrue(success)
        self.assertEqual(rx.open_observations, 0)  # rollbacks aren't observed


class TestAnalyzerClockLockRecommendation(unittest.TestCase):
    def test_memory_bound_busy_gpu_recommends_clock_lock(self):
        from optimize import WorkloadAnalyzer
        from efficiency.gpu_specs import GPU_ARCHITECTURES

        analyzer = WorkloadAnalyzer(arch_spec=GPU_ARCHITECTURES["A100-SXM4-80GB"])
        samples = [
            {"power_draw_w": 300, "utilization_gpu_pct": 55,
             "utilization_memory_pct": 85, "temperature_c": 65,
             "power_limit_w": 400}
            for _ in range(20)
        ]
        result = analyzer.analyze(samples, "A100-SXM4-80GB", gpu_index=2, duration_s=100)

        clock_recs = [
            r for r in result.recommendations
            if r.action_payload.get("command") == "apply_clock_lock"
        ]
        self.assertEqual(len(clock_recs), 1)
        payload = clock_recs[0].action_payload
        self.assertEqual(payload["gpu_index"], 2)
        self.assertAlmostEqual(payload["sm_fraction"], 0.65)

    def test_compute_bound_gpu_gets_no_clock_lock(self):
        from optimize import WorkloadAnalyzer
        from efficiency.gpu_specs import GPU_ARCHITECTURES

        analyzer = WorkloadAnalyzer(arch_spec=GPU_ARCHITECTURES["A100-SXM4-80GB"])
        samples = [
            {"power_draw_w": 380, "utilization_gpu_pct": 95,
             "utilization_memory_pct": 40, "temperature_c": 70,
             "power_limit_w": 400}
            for _ in range(20)
        ]
        result = analyzer.analyze(samples, "A100-SXM4-80GB", gpu_index=0, duration_s=100)
        self.assertFalse(any(
            r.action_payload.get("command") == "apply_clock_lock"
            for r in result.recommendations
        ))


# ═══════════════════════════════════════════════════════════════════════
# 3. Curves + knee + bandit anchor
# ═══════════════════════════════════════════════════════════════════════

def make_tuple(fraction: float, tp_ratio: float, pw_ratio: float,
               gpu_arch: str = "ampere", workload: str = "llm-inference") -> ExperienceTuple:
    limit = 400.0
    return ExperienceTuple(
        machine_id="m", gpu_index=0,
        context=WorkloadContext(
            gpu_name="A100", gpu_arch=gpu_arch, workload_class=workload,
            utilization_gpu_pct=80.0, utilization_memory_pct=60.0,
            memory_pressure=0.5, power_draw_w=300.0,
            power_limit_w=limit, temperature_c=65.0,
        ),
        action=ActionTaken(
            action_type="power_cap", source="autopilot",
            recommended_value=limit * fraction, current_value=limit,
            estimated_savings_pct=0.0,
        ),
        outcome=ActionOutcome(
            energy_delta_j_before=1000.0, energy_delta_j_after=1000.0 * pw_ratio,
            throughput_before=100.0, throughput_after=100.0 * tp_ratio,
            recommendation_status="applied", actual_savings_pct=0.0,
            observation_window_s=300.0,
        ),
        reward=0.1,
    )


def corpus_with_knee_at_70() -> list[ExperienceTuple]:
    """0.9/0.8/0.7 hold throughput; 0.6/0.5 collapse."""
    tuples = []
    profile = {0.9: (0.99, 0.91), 0.8: (0.97, 0.82), 0.7: (0.95, 0.72),
               0.6: (0.80, 0.62), 0.5: (0.60, 0.55)}
    for fraction, (tp, pw) in profile.items():
        for _ in range(5):
            tuples.append(make_tuple(fraction, tp, pw))
    return tuples


class TestCurvesAndKnee(unittest.TestCase):
    def test_fit_curve_bins_and_medians(self):
        curve = fit_curve(corpus_with_knee_at_70(), "ampere", "llm-inference")
        self.assertEqual(len(curve.points), 5)
        p70 = next(p for p in curve.points if abs(p.cap_fraction - 0.725) < 0.03)
        self.assertAlmostEqual(p70.throughput_ratio, 0.95, places=2)
        self.assertEqual(p70.sample_count, 5)

    def test_sparse_bins_excluded(self):
        tuples = [make_tuple(0.7, 0.95, 0.7)] * 2  # below MIN_BIN_SAMPLES
        curve = fit_curve(tuples)
        self.assertEqual(curve.points, [])

    def test_knee_picked_at_lowest_safe_fraction(self):
        curve = fit_curve(corpus_with_knee_at_70())
        knee = knee_fraction(curve, tolerance_pct=10.0)
        self.assertIsNotNone(knee)
        self.assertAlmostEqual(knee.fraction, 0.725, delta=0.03)
        self.assertGreaterEqual(knee.predicted_throughput_ratio, 0.90)

    def test_monotone_trust_stops_at_violation(self):
        # 0.6 violates; 0.5 looks fine (noise) — must NOT be trusted
        tuples = []
        profile = {0.9: 0.99, 0.8: 0.97, 0.7: 0.95, 0.6: 0.80, 0.5: 0.96}
        for fraction, tp in profile.items():
            for _ in range(5):
                tuples.append(make_tuple(fraction, tp, fraction))
        knee = knee_fraction(fit_curve(tuples), tolerance_pct=10.0)
        self.assertAlmostEqual(knee.fraction, 0.725, delta=0.03)

    def test_no_knee_when_everything_violates(self):
        tuples = [make_tuple(f, 0.5, f) for f in (0.6, 0.7, 0.8) for _ in range(5)]
        self.assertIsNone(knee_fraction(fit_curve(tuples), tolerance_pct=10.0))

    def test_no_knee_from_sparse_curve(self):
        tuples = [make_tuple(0.7, 0.95, 0.7)] * 5  # single bin
        self.assertIsNone(knee_fraction(fit_curve(tuples), tolerance_pct=10.0))

    def test_non_power_cap_actions_ignored(self):
        t = make_tuple(0.7, 0.95, 0.7)
        t.action.action_type = "carbon_schedule"
        self.assertEqual(fit_curve([t] * 10).points, [])

    def test_library_grouping_and_fallback(self):
        lib = CurveLibrary()
        n = lib.fit_from_corpus(corpus_with_knee_at_70())
        self.assertGreaterEqual(n, 1)

        # Exact workload match
        knee = lib.recommend_fraction("ampere", "llm-inference", tolerance_pct=10.0)
        self.assertIsNotNone(knee)

        # Unknown workload falls back to the GPU's global curve
        knee_global = lib.recommend_fraction("ampere", "training", tolerance_pct=10.0)
        self.assertIsNotNone(knee_global)

        # Unknown GPU has no curve at all
        self.assertIsNone(lib.recommend_fraction("hopper", "llm-inference"))

    def test_bandit_anchor_restricts_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            bandit = EnergyBandit(data_dir=Path(tmp))
            stub = MagicMock()
            stub.predict.return_value = [1.0 / len(POWER_CAP_ACTIONS)] * len(POWER_CAP_ACTIONS)
            bandit._backend = stub

            for _ in range(50):
                s = bandit.suggest({"f": 1.0}, tdp_w=400.0, anchor_fraction=0.7)
                self.assertLessEqual(abs(s.cap_fraction - 0.7), 0.101)

    def test_bandit_without_anchor_unrestricted(self):
        with tempfile.TemporaryDirectory() as tmp:
            bandit = EnergyBandit(data_dir=Path(tmp))
            stub = MagicMock()
            n = len(POWER_CAP_ACTIONS)
            stub.predict.return_value = [1.0 / n] * n
            bandit._backend = stub

            seen = {bandit.suggest({"f": 1.0}, tdp_w=400.0).cap_fraction for _ in range(200)}
            self.assertGreater(len(seen), 3)


# ═══════════════════════════════════════════════════════════════════════
# 4. Phase detection + dynamic clocks
# ═══════════════════════════════════════════════════════════════════════

MEM_ACT = {"fp32_activity": 0.4, "tensor_activity": 0.1,
           "fp16_activity": 0.0, "memory_activity": 0.6}
COMPUTE_ACT = {"fp32_activity": 0.3, "tensor_activity": 0.9,
               "fp16_activity": 0.0, "memory_activity": 0.2}


class TestPhaseDetector(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(PhaseDetector.classify(MEM_ACT, 80.0), PHASE_MEMORY)
        self.assertEqual(PhaseDetector.classify(COMPUTE_ACT, 95.0), PHASE_COMPUTE)
        self.assertEqual(PhaseDetector.classify(MEM_ACT, 2.0), PHASE_IDLE)

    def test_low_dram_never_memory_bound(self):
        act = {"fp32_activity": 0.1, "tensor_activity": 0.0,
               "fp16_activity": 0.0, "memory_activity": 0.1}
        self.assertEqual(PhaseDetector.classify(act, 50.0), PHASE_COMPUTE)

    def test_hysteresis_requires_consecutive_samples(self):
        det = PhaseDetector(hysteresis_samples=3)
        self.assertIsNone(det.update(0, MEM_ACT, 80))
        self.assertIsNone(det.update(0, MEM_ACT, 80))
        change = det.update(0, MEM_ACT, 80)
        self.assertIsNotNone(change)
        self.assertEqual(change.current, PHASE_MEMORY)

        # A single compute blip does not flip the stable phase
        self.assertIsNone(det.update(0, COMPUTE_ACT, 95))
        self.assertEqual(det.stable_phase(0), PHASE_MEMORY)

        # No duplicate change events while the phase persists
        det.update(0, MEM_ACT, 80)
        det.update(0, MEM_ACT, 80)
        self.assertIsNone(det.update(0, MEM_ACT, 80))


class TestDynamicClockTuner(unittest.TestCase):
    def test_memory_phase_locks_compute_releases(self):
        tuner = DynamicClockTuner(memory_fraction=0.65, min_dwell_s=5.0)
        with patch("efficiency.power_control.get_max_sm_clock", return_value=1980), \
             patch("efficiency.power_control.set_gpu_clock_lock", return_value=True) as lock, \
             patch("efficiency.power_control.reset_gpu_clock_lock", return_value=True) as reset:
            self.assertEqual(tuner.on_phase(0, PHASE_MEMORY), "lock")
            lock.assert_called_once_with(0, 1287, 1287, quiet=True)

            tuner._last_switch[0] = time.time() - 10  # dwell elapsed
            self.assertEqual(tuner.on_phase(0, PHASE_COMPUTE), "release")
            reset.assert_called_once_with(0, quiet=True)

    def test_dwell_prevents_flapping(self):
        tuner = DynamicClockTuner(min_dwell_s=300.0)
        with patch("efficiency.power_control.get_max_sm_clock", return_value=1980), \
             patch("efficiency.power_control.set_gpu_clock_lock", return_value=True), \
             patch("efficiency.power_control.reset_gpu_clock_lock", return_value=True) as reset:
            tuner.on_phase(0, PHASE_MEMORY)
            self.assertIsNone(tuner.on_phase(0, PHASE_COMPUTE))  # inside dwell
            reset.assert_not_called()

    def test_noop_when_already_in_state(self):
        tuner = DynamicClockTuner()
        self.assertIsNone(tuner.on_phase(0, PHASE_COMPUTE))  # no lock held

    def test_shutdown_releases_held_locks(self):
        tuner = DynamicClockTuner(min_dwell_s=5.0)
        with patch("efficiency.power_control.get_max_sm_clock", return_value=1980), \
             patch("efficiency.power_control.set_gpu_clock_lock", return_value=True), \
             patch("efficiency.power_control.reset_gpu_clock_lock", return_value=True) as reset:
            tuner.on_phase(0, PHASE_MEMORY)
            tuner.shutdown()
            reset.assert_called_once_with(0, quiet=True)

    def test_dry_run_touches_nothing(self):
        tuner = DynamicClockTuner(dry_run=True)
        with patch("efficiency.power_control.set_gpu_clock_lock") as lock:
            self.assertEqual(tuner.on_phase(0, PHASE_MEMORY), "lock")
            lock.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# 5. Quantization eval harness
# ═══════════════════════════════════════════════════════════════════════

def fake_runner_factory(results: dict[str, MeasuredVariant]):
    def runner(model_id: str, variant: str, gpu_index: int) -> MeasuredVariant:
        return results.get(variant, MeasuredVariant(variant=variant, error="not found"))
    return runner


def mv(variant: str, tps: float, jpt: float, ppl: float, ok: bool = True) -> MeasuredVariant:
    return MeasuredVariant(
        variant=variant, load_ok=ok, tokens_per_sec=tps,
        joules_per_token=jpt, avg_power_w=jpt * tps, perplexity=ppl,
    )


class TestQuantEvalHarness(unittest.TestCase):
    def _harness(self, tmp: str, results: dict) -> QuantEvalHarness:
        return QuantEvalHarness(
            data_dir=Path(tmp),
            runner=fake_runner_factory(results),
            quality_gate_pct=2.0,
        )

    def test_better_variant_within_gate_recommended(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._harness(tmp, {
                "fp16": mv("fp16", tps=100, jpt=3.0, ppl=8.00),
                "int8": mv("int8", tps=140, jpt=2.0, ppl=8.10),   # +1.25% ppl: pass
                "int4": mv("int4", tps=200, jpt=1.4, ppl=8.60),   # +7.5% ppl: fail
            })
            result = harness.evaluate("meta-llama/llama-3.1-8b")

        self.assertEqual(result.recommended, "int8")
        int8 = next(v for v in result.variants if v.variant == "int8")
        self.assertTrue(int8.passes_quality_gate)
        int4 = next(v for v in result.variants if v.variant == "int4")
        self.assertFalse(int4.passes_quality_gate)

    def test_no_improvement_stays_on_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._harness(tmp, {
                "fp16": mv("fp16", tps=100, jpt=2.0, ppl=8.0),
                "int8": mv("int8", tps=90, jpt=2.5, ppl=8.05),  # passes gate, worse J/token
            })
            result = harness.evaluate("m", variants=("fp16", "int8"))
        self.assertEqual(result.recommended, "fp16")

    def test_failed_baseline_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._harness(tmp, {
                "fp16": MeasuredVariant(variant="fp16", error="OOM"),
                "int8": mv("int8", tps=140, jpt=2.0, ppl=8.1),
            })
            result = harness.evaluate("m")
        self.assertIsNone(result.recommended)
        self.assertEqual(len(result.variants), 1)  # stopped after baseline failure

    def test_variant_without_perplexity_cannot_pass_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._harness(tmp, {
                "fp16": mv("fp16", tps=100, jpt=3.0, ppl=8.0),
                "int8": mv("int8", tps=140, jpt=2.0, ppl=0.0),  # no quality measurement
            })
            result = harness.evaluate("m", variants=("fp16", "int8"))
        self.assertEqual(result.recommended, "fp16")

    def test_results_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = self._harness(tmp, {
                "fp16": mv("fp16", tps=100, jpt=3.0, ppl=8.0),
            })
            harness.evaluate("my/model", variants=("fp16",))
            path = Path(tmp) / "intelligence" / "quant_eval.json"
            self.assertTrue(path.exists())
            import json as _json
            data = _json.loads(path.read_text())
            self.assertIn("my/model", data)

    def test_merge_into_registry_updates_measured_fields(self):
        from intelligence.registry import ModelRegistry, RegistryEntry
        from efficiency.gpu_specs import ModelProfile

        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(Path(tmp))
            registry.register(RegistryEntry(
                model_id="meta-llama/Meta-Llama-3.1-8B", tag="llama-3.1-8b",
                family="Llama",
                profile=ModelProfile(
                    tag="llama-3.1-8b", family="Llama", math_intensity=120.0,
                    precision="fp16", is_memory_bound=False,
                    typical_util_min=60, typical_util_max=90,
                ),
                status="estimated",
                quantization_variants=[
                    {"variant": "int8", "quality_impact": "minimal"},
                ],
            ))

            harness = self._harness(tmp, {
                "fp16": mv("fp16", tps=100, jpt=3.0, ppl=8.0),
                "int8": mv("int8", tps=140, jpt=2.0, ppl=8.1),
            })
            result = harness.evaluate(
                "meta-llama/Meta-Llama-3.1-8B", variants=("fp16", "int8"),
            )
            self.assertTrue(harness.merge_into_registry(registry, result))

            entry = registry.get("llama-3.1-8b")
            qv = entry.quantization_variants[0]
            self.assertEqual(qv["measured_tokens_per_sec"], 140.0)
            self.assertTrue(qv["recommended"])
            self.assertTrue(qv["passes_quality_gate"])


if __name__ == "__main__":
    unittest.main()
