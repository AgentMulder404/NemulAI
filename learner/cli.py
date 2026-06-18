# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""CLI handler for `nemulai learn` subcommand."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from config import DATA_DIR
from machine_id import get_machine_id


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemulai learn",
        description="Manage the self-learning experience corpus.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="Show corpus statistics")
    sub.add_parser("corpus-size", help="Show tuple counts per GPU class vs target")

    export_p = sub.add_parser("export", help="Export experience corpus")
    export_p.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    export_p.add_argument("--gpu-class", default=None, help="Filter by GPU class")
    export_p.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")

    sub.add_parser("bandit-status", help="Show bandit model status")

    train_p = sub.add_parser("bandit-train", help="Warm-start bandit on logged experience")
    train_p.add_argument("--gpu-class", default=None, help="Filter corpus by GPU class")

    eval_p = sub.add_parser("bandit-eval", help="Offline evaluation of current bandit policy")
    eval_p.add_argument("--sample-size", type=int, default=500, help="Number of tuples to evaluate")

    return parser


def run_learn(args: argparse.Namespace) -> int:
    from learner.experience_logger import ExperienceLogger

    logger = ExperienceLogger(
        data_dir=DATA_DIR,
        machine_id=get_machine_id(),
    )
    logger.load_from_wal()

    if args.action == "status":
        return _cmd_status(logger)
    elif args.action == "corpus-size":
        return _cmd_corpus_size(logger)
    elif args.action == "export":
        return _cmd_export(logger, args)
    elif args.action == "bandit-status":
        return _cmd_bandit_status()
    elif args.action == "bandit-train":
        return _cmd_bandit_train(logger, args)
    elif args.action == "bandit-eval":
        return _cmd_bandit_eval(logger, args)
    return 1


def _cmd_status(logger) -> int:
    stats = logger.get_corpus_stats()
    print(f"Experience Corpus Status")
    print(f"{'─' * 40}")
    print(f"  Total tuples:     {stats['total']}")
    print(f"  Completed:        {stats['completed']}")
    print(f"  Pending outcome:  {stats['pending']}")
    print()

    if stats["by_gpu_class"]:
        print("  By GPU class:")
        for gpu, count in sorted(stats["by_gpu_class"].items(), key=lambda x: -x[1]):
            print(f"    {gpu:30s} {count:>6d}")
        print()

    if stats["by_workload_class"]:
        print("  By workload class:")
        for wl, count in sorted(stats["by_workload_class"].items(), key=lambda x: -x[1]):
            print(f"    {wl:30s} {count:>6d}")
        print()

    if stats["by_action_type"]:
        print("  By action type:")
        for at, count in sorted(stats["by_action_type"].items(), key=lambda x: -x[1]):
            print(f"    {at:30s} {count:>6d}")

    return 0


TARGET_PER_GPU_CLASS = 10_000


def _cmd_corpus_size(logger) -> int:
    stats = logger.get_corpus_stats()
    by_gpu = stats.get("by_gpu_class", {})
    if not by_gpu:
        print("No experience data collected yet.")
        print(f"Target: {TARGET_PER_GPU_CLASS} tuples per GPU class.")
        return 0

    print(f"{'GPU Class':30s} {'Current':>8s} {'Target':>8s} {'Progress':>10s}")
    print(f"{'─' * 60}")
    for gpu, count in sorted(by_gpu.items(), key=lambda x: -x[1]):
        pct = min(100.0, count / TARGET_PER_GPU_CLASS * 100.0)
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {gpu:28s} {count:>8d} {TARGET_PER_GPU_CLASS:>8d}  {bar} {pct:.0f}%")

    return 0


def _cmd_export(logger, args: argparse.Namespace) -> int:
    tuples = list(logger.iter_completed(gpu_class=args.gpu_class))
    if not tuples:
        print("No completed experience tuples to export.", file=sys.stderr)
        return 1

    out = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        if args.format == "jsonl":
            from dataclasses import asdict
            for t in tuples:
                out.write(json.dumps(asdict(t)) + "\n")
        elif args.format == "csv":
            writer = csv.writer(out)
            writer.writerow([
                "id", "timestamp", "machine_id", "gpu_index",
                "gpu_name", "gpu_arch", "workload_class",
                "utilization_gpu_pct", "utilization_memory_pct", "memory_pressure",
                "power_draw_w", "power_limit_w", "temperature_c",
                "action_type", "action_source", "recommended_value", "current_value",
                "estimated_savings_pct",
                "energy_before_j", "energy_after_j",
                "throughput_before", "throughput_after",
                "actual_savings_pct", "reward",
            ])
            for t in tuples:
                ctx = t.context
                act = t.action
                out_ = t.outcome
                writer.writerow([
                    t.id, t.timestamp, t.machine_id, t.gpu_index,
                    ctx.gpu_name if ctx else "", ctx.gpu_arch if ctx else "",
                    ctx.workload_class if ctx else "",
                    ctx.utilization_gpu_pct if ctx else "",
                    ctx.utilization_memory_pct if ctx else "",
                    ctx.memory_pressure if ctx else "",
                    ctx.power_draw_w if ctx else "", ctx.power_limit_w if ctx else "",
                    ctx.temperature_c if ctx else "",
                    act.action_type if act else "", act.source if act else "",
                    act.recommended_value if act else "", act.current_value if act else "",
                    act.estimated_savings_pct if act else "",
                    out_.energy_delta_j_before if out_ else "",
                    out_.energy_delta_j_after if out_ else "",
                    out_.throughput_before if out_ else "",
                    out_.throughput_after if out_ else "",
                    out_.actual_savings_pct if out_ else "",
                    t.reward if t.reward is not None else "",
                ])

        print(f"Exported {len(tuples)} tuples.", file=sys.stderr)
    finally:
        if out is not sys.stdout:
            out.close()

    return 0


# ── Bandit CLI commands ──────────────────────────────────────────────────────

def _cmd_bandit_status() -> int:
    from learner.bandit import EnergyBandit, _HAS_VW

    bandit = EnergyBandit(data_dir=DATA_DIR)
    stats = bandit.get_stats()

    print("Contextual Bandit Status")
    print(f"{'─' * 40}")
    print(f"  Backend:              {stats['backend']}")
    print(f"  VW available:         {'yes' if _HAS_VW else 'no (using simple fallback)'}")
    print(f"  Ready:                {'yes' if stats['ready'] else f'no (need {stats[\"min_corpus\"]} tuples, have {stats[\"corpus_size\"]})'}")
    print(f"  Corpus size:          {stats['corpus_size']}")
    print(f"  Model version:        {stats['model_version']}")
    print(f"  Updates since save:   {stats['updates_since_retrain']}")
    print(f"  Estimated reward:     {stats['estimated_reward']:.4f}")
    return 0


def _cmd_bandit_train(logger, args: argparse.Namespace) -> int:
    from learner.bandit import EnergyBandit

    bandit = EnergyBandit(data_dir=DATA_DIR)
    tuples = logger.iter_completed(gpu_class=args.gpu_class)
    count = bandit.warm_start(tuples)
    print(f"Warm-started bandit on {count} experience tuples.")

    stats = bandit.get_stats()
    print(f"  Corpus size:     {stats['corpus_size']}")
    print(f"  Model version:   {stats['model_version']}")
    print(f"  Ready:           {'yes' if stats['ready'] else 'no'}")
    return 0


def _cmd_bandit_eval(logger, args: argparse.Namespace) -> int:
    from learner.bandit import EnergyBandit

    bandit = EnergyBandit(data_dir=DATA_DIR)
    if not bandit.is_ready():
        print(f"Bandit not ready — need {bandit._min_corpus} tuples.", file=sys.stderr)
        return 1

    tuples = list(logger.iter_completed())
    reward = bandit.evaluate_offline(tuples, sample_size=args.sample_size)
    print(f"Offline Evaluation (doubly-robust, n={min(len(tuples), args.sample_size)})")
    print(f"{'─' * 40}")
    print(f"  Estimated reward:  {reward:.4f}")
    print(f"  Interpretation:    {'good' if reward > 0.3 else 'moderate' if reward > 0.1 else 'needs more data'}")
    return 0
