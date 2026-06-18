# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CommandReceiver autopilot executor (observation + rollback)."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from command_receiver import (
    DEFAULT_OBSERVATION_WINDOW_S,
    MIN_OBSERVATION_SAMPLES,
    CommandReceiver,
)


def make_receiver(**kwargs) -> CommandReceiver:
    rx = CommandReceiver(
        endpoint="https://www.nemulai.com/v1/metrics/ingest",
        api_key="alum_test",
        machine_id="m-test",
        **kwargs,
    )
    rx._report_result = MagicMock()
    return rx


def feed_samples(rx: CommandReceiver, gpu_index: int, util: float, power: float, n: int = 10):
    for _ in range(n):
        rx.record_sample(gpu_index, util, power)


class TestApplyPowerCap(unittest.TestCase):
    def test_apply_opens_observation_with_baseline(self):
        rx = make_receiver()
        feed_samples(rx, 0, util=80.0, power=300.0)

        with patch("efficiency.power_control.set_power_limit", return_value=True), \
             patch("efficiency.power_control.get_power_limit", return_value=350):
            success, msg = rx._execute({
                "id": "cmd-1",
                "command_type": "apply_power_cap",
                "params": {"gpu_index": 0, "watts": 250,
                           "observation_window_s": 120, "throughput_tolerance_pct": 15},
            })

        self.assertTrue(success)
        self.assertEqual(rx.open_observations, 1)
        obs = rx._observations[0]
        self.assertEqual(obs.command_id, "cmd-1")
        self.assertEqual(obs.prev_limit_w, 350.0)
        self.assertAlmostEqual(obs.baseline_util_pct, 80.0)
        self.assertAlmostEqual(obs.baseline_power_w, 300.0)
        self.assertEqual(obs.window_s, 120.0)
        self.assertEqual(obs.tolerance_pct, 15.0)

    def test_window_and_tolerance_clamped(self):
        rx = make_receiver()
        with patch("efficiency.power_control.set_power_limit", return_value=True), \
             patch("efficiency.power_control.get_power_limit", return_value=350):
            rx._execute({
                "id": "cmd-1",
                "command_type": "apply_power_cap",
                "params": {"gpu_index": 0, "watts": 250,
                           "observation_window_s": 999999, "throughput_tolerance_pct": 0.1},
            })
        obs = rx._observations[0]
        self.assertEqual(obs.window_s, 3600.0)
        self.assertEqual(obs.tolerance_pct, 1.0)

    def test_default_window_when_params_missing(self):
        rx = make_receiver()
        with patch("efficiency.power_control.set_power_limit", return_value=True), \
             patch("efficiency.power_control.get_power_limit", return_value=350):
            rx._execute({
                "id": "cmd-1",
                "command_type": "apply_power_cap",
                "params": {"gpu_index": 0, "watts": 250},
            })
        self.assertEqual(rx._observations[0].window_s, DEFAULT_OBSERVATION_WINDOW_S)

    def test_dry_run_does_not_observe(self):
        rx = make_receiver(dry_run=True)
        success, msg = rx._execute({
            "id": "cmd-1",
            "command_type": "apply_power_cap",
            "params": {"gpu_index": 0, "watts": 250},
        })
        self.assertTrue(success)
        self.assertIn("Dry run", msg)
        self.assertEqual(rx.open_observations, 0)

    def test_failed_apply_does_not_observe(self):
        rx = make_receiver()
        with patch("efficiency.power_control.set_power_limit", return_value=False), \
             patch("efficiency.power_control.get_power_limit", return_value=350):
            success, _ = rx._execute({
                "id": "cmd-1",
                "command_type": "apply_power_cap",
                "params": {"gpu_index": 0, "watts": 250},
            })
        self.assertFalse(success)
        self.assertEqual(rx.open_observations, 0)

    def test_watts_out_of_range_rejected(self):
        rx = make_receiver()
        success, msg = rx._execute({
            "id": "cmd-1",
            "command_type": "apply_power_cap",
            "params": {"gpu_index": 0, "watts": 5000},
        })
        self.assertFalse(success)
        self.assertIn("out of safe range", msg)

    def test_unknown_command_type(self):
        rx = make_receiver()
        success, msg = rx._execute({"id": "x", "command_type": "rm_rf_slash", "params": {}})
        self.assertFalse(success)
        self.assertIn("Unknown command type", msg)


class TestObservationWindow(unittest.TestCase):
    def _applied_receiver(self, tolerance_pct: float = 10.0) -> CommandReceiver:
        rx = make_receiver()
        feed_samples(rx, 0, util=80.0, power=300.0)
        with patch("efficiency.power_control.set_power_limit", return_value=True), \
             patch("efficiency.power_control.get_power_limit", return_value=350):
            rx._execute({
                "id": "cmd-1",
                "command_type": "apply_power_cap",
                "params": {"gpu_index": 0, "watts": 250,
                           "observation_window_s": 60,
                           "throughput_tolerance_pct": tolerance_pct},
            })
        return rx

    def test_open_window_not_closed_early(self):
        rx = self._applied_receiver()
        self.assertEqual(rx.check_observations(), [])
        self.assertEqual(rx.open_observations, 1)

    def test_healthy_window_reports_savings_no_rollback(self):
        rx = self._applied_receiver()
        # Utilization holds, power drops — the cap is saving energy
        feed_samples(rx, 0, util=78.0, power=240.0)
        rx._observations[0].applied_at = time.time() - 120  # force deadline past

        with patch("efficiency.power_control.set_power_limit") as set_mock:
            outcomes = rx.check_observations()
            set_mock.assert_not_called()

        self.assertEqual(len(outcomes), 1)
        out = outcomes[0]
        self.assertFalse(out.rolled_back)
        self.assertAlmostEqual(out.observed_util_pct, 78.0)
        self.assertAlmostEqual(out.actual_savings_pct, 20.0)
        self.assertEqual(rx.open_observations, 0)

        # Reported with observation payload incl. savings
        _, kwargs = rx._report_result.call_args
        self.assertTrue(kwargs["success"])
        self.assertFalse(kwargs.get("rolled_back", False))
        self.assertAlmostEqual(kwargs["observation"]["actual_savings_pct"], 20.0)

    def test_regression_rolls_back_to_prev_limit(self):
        rx = self._applied_receiver(tolerance_pct=10.0)
        # Utilization collapses 80 -> 40 (50% drop > 10% tolerance)
        feed_samples(rx, 0, util=40.0, power=200.0)
        rx._observations[0].applied_at = time.time() - 120

        with patch("efficiency.power_control.set_power_limit", return_value=True) as set_mock:
            outcomes = rx.check_observations()
            set_mock.assert_called_once_with(0, 350, quiet=True)

        out = outcomes[0]
        self.assertTrue(out.rolled_back)
        self.assertAlmostEqual(out.util_drop_pct, 50.0)

        _, kwargs = rx._report_result.call_args
        self.assertFalse(kwargs["success"])
        self.assertTrue(kwargs["rolled_back"])
        self.assertNotIn("actual_savings_pct", kwargs["observation"])

    def test_drop_within_tolerance_holds_cap(self):
        rx = self._applied_receiver(tolerance_pct=10.0)
        # 80 -> 75 is a 6.25% drop, inside the 10% tolerance
        feed_samples(rx, 0, util=75.0, power=250.0)
        rx._observations[0].applied_at = time.time() - 120

        with patch("efficiency.power_control.set_power_limit") as set_mock:
            outcomes = rx.check_observations()
            set_mock.assert_not_called()

        self.assertFalse(outcomes[0].rolled_back)

    def test_insufficient_samples_never_rolls_back(self):
        rx = self._applied_receiver(tolerance_pct=10.0)
        # Fewer than MIN_OBSERVATION_SAMPLES post-apply samples
        for _ in range(MIN_OBSERVATION_SAMPLES - 1):
            rx.record_sample(0, 10.0, 100.0)
        rx._observations[0].applied_at = time.time() - 120

        with patch("efficiency.power_control.set_power_limit") as set_mock:
            outcomes = rx.check_observations()
            set_mock.assert_not_called()

        out = outcomes[0]
        self.assertFalse(out.rolled_back)
        self.assertEqual(out.util_drop_pct, 0.0)
        _, kwargs = rx._report_result.call_args
        self.assertTrue(kwargs["success"])

    def test_samples_only_attach_to_matching_gpu(self):
        rx = self._applied_receiver()
        rx.record_sample(1, 5.0, 50.0)   # different GPU
        rx.record_sample(0, 70.0, 240.0)
        self.assertEqual(len(rx._observations[0].samples), 1)

    def test_rollback_failure_reported(self):
        rx = self._applied_receiver(tolerance_pct=10.0)
        feed_samples(rx, 0, util=40.0, power=200.0)
        rx._observations[0].applied_at = time.time() - 120

        with patch("efficiency.power_control.set_power_limit", return_value=False):
            outcomes = rx.check_observations()

        # Regression detected but rollback failed -> rolled_back False in
        # outcome, reported as unsuccessful
        self.assertFalse(outcomes[0].rolled_back)
        _, kwargs = rx._report_result.call_args
        self.assertFalse(kwargs["success"])


class TestPollAndAdaptiveInterval(unittest.TestCase):
    def test_poll_executes_and_reports(self):
        rx = make_receiver()
        rx._fetch_commands = MagicMock(return_value=[
            {"id": "cmd-1", "command_type": "apply_power_cap",
             "params": {"gpu_index": 0, "watts": 250}},
        ])
        with patch("efficiency.power_control.set_power_limit", return_value=True), \
             patch("efficiency.power_control.get_power_limit", return_value=350):
            n = rx.poll_and_execute()

        self.assertEqual(n, 1)
        args, kwargs = rx._report_result.call_args
        self.assertEqual(args[0], "cmd-1")
        self.assertTrue(kwargs["success"])

    def test_empty_polls_back_off_then_reset(self):
        rx = make_receiver(base_interval=60.0, max_interval=300.0)
        rx._fetch_commands = MagicMock(return_value=[])
        for _ in range(3):
            rx.poll_and_execute()
        self.assertGreater(rx.poll_interval, 60.0)

        rx._fetch_commands = MagicMock(return_value=[
            {"id": "cmd-1", "command_type": "apply_power_cap",
             "params": {"gpu_index": 0, "watts": 250}},
        ])
        with patch("efficiency.power_control.set_power_limit", return_value=True), \
             patch("efficiency.power_control.get_power_limit", return_value=350):
            rx.poll_and_execute()
        self.assertEqual(rx.poll_interval, 60.0)


class TestExperienceRecordCompleted(unittest.TestCase):
    def test_record_completed_appends_to_wal(self):
        import tempfile
        from pathlib import Path
        from learner.experience_logger import (
            ExperienceLogger, ExperienceTuple, WorkloadContext,
            ActionTaken, ActionOutcome,
        )

        with tempfile.TemporaryDirectory() as tmp:
            logger = ExperienceLogger(Path(tmp), machine_id="m-test")
            t = ExperienceTuple(
                machine_id="m-test",
                gpu_index=0,
                context=WorkloadContext(
                    gpu_name="A100-SXM4-80GB", gpu_arch="ampere",
                    workload_class="llm-inference",
                    utilization_gpu_pct=80.0, utilization_memory_pct=50.0,
                    memory_pressure=0.5, power_draw_w=300.0,
                    power_limit_w=350.0, temperature_c=65.0,
                ),
                action=ActionTaken(
                    action_type="power_cap", source="autopilot",
                    recommended_value=250.0, current_value=350.0,
                    estimated_savings_pct=0.0,
                ),
                outcome=ActionOutcome(
                    energy_delta_j_before=90000.0, energy_delta_j_after=72000.0,
                    throughput_before=80.0, throughput_after=78.0,
                    recommendation_status="applied", actual_savings_pct=20.0,
                    observation_window_s=300.0,
                ),
                reward=0.18,
            )
            logger.record_completed(t)

            completed = list(logger.iter_completed())
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0].action.source, "autopilot")
            self.assertTrue(completed[0].is_complete())


if __name__ == "__main__":
    unittest.main()
