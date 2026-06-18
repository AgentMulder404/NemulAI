# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""CLI handlers for `nemulai model-intel`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemulai model-intel",
        description="Model intelligence pipeline — discover and profile AI models.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    scan_p = sub.add_parser("scan", help="Scan for new model releases")
    scan_p.add_argument("--limit", type=int, default=20)
    scan_p.add_argument("--min-downloads", type=int, default=1000)
    scan_p.add_argument("--min-confidence", type=float, default=0.5)
    scan_p.add_argument("--json", action="store_true", dest="json_output")

    list_p = sub.add_parser("list", help="List discovered models")
    list_p.add_argument(
        "--status",
        choices=["detected", "profiled", "estimated", "active", "all"],
        default="all",
    )
    list_p.add_argument("--json", action="store_true", dest="json_output")

    pair_p = sub.add_parser("pair", help="Profile and rank GPUs for a model")
    pair_p.add_argument("model", help="HuggingFace model ID or local tag")
    pair_p.add_argument("--top", type=int, default=10)
    pair_p.add_argument("--json", action="store_true", dest="json_output")

    profile_p = sub.add_parser("profile", help="Show profiling details for a model")
    profile_p.add_argument("model", help="HuggingFace model ID")
    profile_p.add_argument("--json", action="store_true", dest="json_output")

    sub.add_parser("warm-start", help="Warm-start bandit with discovered model profiles")

    quant_p = sub.add_parser("quantize", help="Analyze quantization variants for a model")
    quant_p.add_argument("model", help="HuggingFace model ID or local tag")
    quant_p.add_argument("--gpu", default=None, help="Show recommendation for specific GPU")
    quant_p.add_argument("--json", action="store_true", dest="json_output")

    prices_p = sub.add_parser("prices", help="Show current GPU pricing")
    prices_p.add_argument("--update-from", default=None, help="Ingest pricing from JSON file")
    prices_p.add_argument("--json", action="store_true", dest="json_output")

    bv_p = sub.add_parser("best-value", help="Find best $/TFLOP GPU for a model")
    bv_p.add_argument("model", help="HuggingFace model ID or local tag")
    bv_p.add_argument("--budget", type=float, default=None, help="Max $/hr budget")
    bv_p.add_argument("--top", type=int, default=10)
    bv_p.add_argument("--json", action="store_true", dest="json_output")

    research_p = sub.add_parser(
        "research",
        help="Run a research cycle: discover models, set benchmark targets, ingest measured runs",
    )
    research_p.add_argument("--limit", type=int, default=20)
    research_p.add_argument("--min-downloads", type=int, default=1000)
    research_p.add_argument("--min-confidence", type=float, default=0.5)
    research_p.add_argument("--no-scan", action="store_true", help="Skip HuggingFace scan, only recalibrate")
    research_p.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                            help="Keep running, repeating the cycle on this interval")
    research_p.add_argument("--json", action="store_true", dest="json_output")

    targets_p = sub.add_parser("targets", help="List benchmark targets (predicted vs measured)")
    targets_p.add_argument("--model", default=None, help="Filter by model tag substring")
    targets_p.add_argument("--gpu", default=None, help="Filter by GPU name substring")
    targets_p.add_argument("--json", action="store_true", dest="json_output")

    cal_p = sub.add_parser("calibrate", help="Feed measured runs back into the research agent")
    cal_p.add_argument("results", nargs="*", help="JSON files from `nemulai test --output`")
    cal_p.add_argument("--gpu", default=None, help="GPU name for a manual measurement")
    cal_p.add_argument("--model", default=None, help="Model ID for a manual measurement")
    cal_p.add_argument("--tokens-per-sec", type=float, default=None, help="Measured tokens/s")
    cal_p.add_argument("--json", action="store_true", dest="json_output")

    qe_p = sub.add_parser("quant-eval", help="Measure quantization variants with a quality gate")
    qe_p.add_argument("model", help="HuggingFace model ID")
    qe_p.add_argument("--variants", default="fp16,int8,int4",
                      help="Comma-separated variants, baseline first (default fp16,int8,int4)")
    qe_p.add_argument("--gpu", type=int, default=0)
    qe_p.add_argument("--quality-gate", type=float, default=2.0,
                      help="Max perplexity increase vs baseline, percent (default 2.0)")
    qe_p.add_argument("--json", action="store_true", dest="json_output")

    suggest_p = sub.add_parser("suggest", help="Best model + GPU pairings by calibrated $/1M tokens")
    suggest_p.add_argument("query", nargs="?", default=None,
                           help="Filter by model tag or family substring (e.g. 'llama')")
    suggest_p.add_argument("--budget", type=float, default=None, help="Max $/hr per GPU")
    suggest_p.add_argument("--top", type=int, default=5)
    suggest_p.add_argument("--json", action="store_true", dest="json_output")

    return parser


def run_model_intel(args: argparse.Namespace) -> int:
    import config
    from intelligence.pipeline import IntelligencePipeline

    data_dir = Path(getattr(config, "DATA_DIR", "") or Path.home() / ".nemulai")
    pipeline = IntelligencePipeline(
        data_dir=data_dir,
        supabase_url=getattr(config, "SUPABASE_URL", None),
        supabase_key=getattr(config, "SUPABASE_SERVICE_KEY", None),
    )

    if args.action == "scan":
        return _cmd_scan(pipeline, args)
    elif args.action == "list":
        return _cmd_list(pipeline, args)
    elif args.action == "pair":
        return _cmd_pair(pipeline, args)
    elif args.action == "profile":
        return _cmd_profile(pipeline, args)
    elif args.action == "warm-start":
        return _cmd_warm_start(pipeline, args)
    elif args.action == "quantize":
        return _cmd_quantize(pipeline, args)
    elif args.action == "prices":
        return _cmd_prices(data_dir, args)
    elif args.action == "best-value":
        return _cmd_best_value(pipeline, data_dir, args)
    elif args.action == "research":
        return _cmd_research(pipeline, data_dir, args)
    elif args.action == "targets":
        return _cmd_targets(pipeline, data_dir, args)
    elif args.action == "calibrate":
        return _cmd_calibrate(pipeline, data_dir, args)
    elif args.action == "suggest":
        return _cmd_suggest(pipeline, data_dir, args)
    elif args.action == "quant-eval":
        return _cmd_quant_eval(pipeline, data_dir, args)

    return 1


def _make_research_agent(pipeline, data_dir):
    from intelligence.research import ResearchAgent
    return ResearchAgent(data_dir=data_dir, pipeline=pipeline)


def _cmd_scan(pipeline, args) -> int:
    console = Console()
    console.print("[bold]Scanning HuggingFace for new models...[/bold]\n")

    result = pipeline.run(
        limit=args.limit,
        min_downloads=args.min_downloads,
        min_confidence=args.min_confidence,
    )

    if args.json_output:
        print(json.dumps({
            "detected": result.detected,
            "profiled": result.profiled,
            "estimated": result.estimated,
            "registered": result.registered,
            "duration_s": result.duration_s,
            "errors": result.errors,
            "models": [e.to_dict() for e in result.entries],
        }, indent=2))
        return 0

    if not result.entries:
        console.print(f"No new models found ({result.detected} detected, filtered by confidence/downloads)")
        return 0

    table = Table(title=f"Discovered {len(result.entries)} New Models ({result.duration_s}s)")
    table.add_column("Tag", style="cyan")
    table.add_column("Family", style="green")
    table.add_column("Params")
    table.add_column("Intensity", justify="right")
    table.add_column("Precision")
    table.add_column("Best GPU", style="yellow")
    table.add_column("Confidence", justify="right")

    for entry in result.entries:
        params = f"{entry.parameter_count / 1e9:.1f}B" if entry.parameter_count else "?"
        best = entry.gpu_rankings[0]["gpu_name"] if entry.gpu_rankings else "?"
        table.add_row(
            entry.tag,
            entry.family,
            params,
            f"{entry.profile.math_intensity:.0f}",
            entry.profile.precision,
            best,
            f"{entry.confidence:.0%}",
        )

    console.print(table)

    if result.errors:
        console.print(f"\n[yellow]{len(result.errors)} errors:[/yellow]")
        for err in result.errors:
            console.print(f"  - {err}")

    return 0


def _cmd_list(pipeline, args) -> int:
    console = Console()
    entries = pipeline.registry.list_all(status=args.status)

    if args.json_output:
        print(json.dumps([e.to_dict() for e in entries], indent=2))
        return 0

    if not entries:
        console.print("No models in registry.")
        return 0

    table = Table(title=f"Model Registry ({len(entries)} models)")
    table.add_column("Tag", style="cyan")
    table.add_column("Family", style="green")
    table.add_column("Intensity", justify="right")
    table.add_column("Precision")
    table.add_column("Status", style="magenta")
    table.add_column("Best GPU", style="yellow")
    table.add_column("Downloads", justify="right")

    for entry in entries:
        best = entry.gpu_rankings[0]["gpu_name"] if entry.gpu_rankings else "?"
        downloads = f"{entry.downloads_30d:,}" if entry.downloads_30d else "?"
        table.add_row(
            entry.tag,
            entry.family,
            f"{entry.profile.math_intensity:.0f}",
            entry.profile.precision,
            entry.status,
            best,
            downloads,
        )

    console.print(table)
    return 0


def _cmd_pair(pipeline, args) -> int:
    console = Console()
    model_id = args.model

    console.print(f"[bold]Profiling and ranking GPUs for {model_id}...[/bold]\n")

    entry = pipeline.run_single(model_id)
    if not entry:
        console.print(f"[red]Could not profile model: {model_id}[/red]")
        return 1

    if args.json_output:
        print(json.dumps(entry.to_dict(), indent=2))
        return 0

    console.print(f"[bold cyan]{entry.tag}[/bold cyan] ({entry.family})")
    console.print(f"  Intensity: {entry.profile.math_intensity:.1f} FLOP/byte")
    console.print(f"  Precision: {entry.profile.precision}")
    console.print(f"  {'Memory' if entry.profile.is_memory_bound else 'Compute'}-bound")
    console.print(f"  Utilization range: {entry.profile.typical_util_min}-{entry.profile.typical_util_max}%")
    console.print()

    table = Table(title=f"Top {min(args.top, len(entry.gpu_rankings))} GPUs for {entry.tag}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("GPU", style="cyan")
    table.add_column("Family")
    table.add_column("Score", justify="right", style="green")
    table.add_column("J/TFLOP", justify="right")
    table.add_column("Eff. TFLOPS", justify="right")
    table.add_column("$/hr", justify="right", style="yellow")

    for i, r in enumerate(entry.gpu_rankings[:args.top], 1):
        cost = f"${r['cost_per_hr']:.2f}" if r.get("cost_per_hr") else "?"
        table.add_row(
            str(i),
            r["gpu_name"],
            r.get("family", ""),
            f"{r['score']:.1f}",
            f"{r['joules_per_tflop']:.2f}",
            f"{r.get('effective_tflops', 0):.1f}",
            cost,
        )

    console.print(table)
    return 0


def _cmd_profile(pipeline, args) -> int:
    console = Console()
    model_id = args.model

    from intelligence.detector import ModelDetector
    from intelligence.profiler import ModelProfiler

    detector = ModelDetector()
    detected = detector.fetch_model_info(model_id)
    if not detected:
        console.print(f"[red]Could not fetch model info for {model_id}[/red]")
        return 1

    profiler = ModelProfiler()
    result = profiler.profile(detected)

    if args.json_output:
        print(json.dumps({
            "model_id": model_id,
            "tag": result.profile.tag,
            "family": result.profile.family,
            "math_intensity": result.profile.math_intensity,
            "precision": result.profile.precision,
            "is_memory_bound": result.profile.is_memory_bound,
            "typical_util_min": result.profile.typical_util_min,
            "typical_util_max": result.profile.typical_util_max,
            "confidence": result.confidence,
            "inferred_from": result.inferred_from,
            "reasoning": result.reasoning,
        }, indent=2))
        return 0

    console.print(f"[bold]Profile: {model_id}[/bold]\n")
    console.print(f"  Tag:            [cyan]{result.profile.tag}[/cyan]")
    console.print(f"  Family:         [green]{result.profile.family}[/green]")
    console.print(f"  Math intensity: {result.profile.math_intensity:.1f} FLOP/byte")
    console.print(f"  Precision:      {result.profile.precision}")
    console.print(f"  Memory-bound:   {'Yes' if result.profile.is_memory_bound else 'No'}")
    console.print(f"  Utilization:    {result.profile.typical_util_min}-{result.profile.typical_util_max}%")
    console.print(f"  Confidence:     {result.confidence:.0%}")
    console.print(f"  Inferred from:  {result.inferred_from}")
    console.print(f"\n  [dim]Reasoning:[/dim]")
    for part in result.reasoning.split("; "):
        console.print(f"    - {part}")

    return 0


def _cmd_warm_start(pipeline, args) -> int:
    console = Console()
    console.print("[bold]Warming up bandit with model intelligence data...[/bold]\n")

    count = pipeline.warm_start_bandit()
    if count > 0:
        console.print(f"[green]Generated {count} synthetic experience tuples[/green]")
    else:
        console.print("[yellow]No models available for warm-start. Run 'scan' first.[/yellow]")

    return 0


def _cmd_quantize(pipeline, args) -> int:
    console = Console()
    model_id = args.model

    console.print(f"[bold]Analyzing quantization variants for {model_id}...[/bold]\n")

    entry = pipeline.run_single(model_id)
    if not entry:
        console.print(f"[red]Could not profile model: {model_id}[/red]")
        return 1

    from intelligence.quantization import QuantizationAdvisor

    advisor = QuantizationAdvisor()

    if args.gpu:
        rec = advisor.recommend_per_gpu(entry.profile, args.gpu, entry.parameter_count)
        if not rec:
            console.print(f"[yellow]No quantization recommendation for {args.gpu}[/yellow]")
            return 0

        if args.json_output:
            print(json.dumps({
                "model": model_id,
                "gpu": args.gpu,
                "recommended_variant": rec.variant.name,
                "model_size_gb": rec.model_size_gb,
                "memory_reduction_pct": rec.memory_reduction_pct,
                "throughput_change_pct": rec.estimated_throughput_change_pct,
                "quality_impact": rec.estimated_quality_impact,
            }, indent=2))
            return 0

        console.print(f"  GPU: [cyan]{args.gpu}[/cyan]")
        console.print(f"  Recommended: [green]{rec.variant.name}[/green] ({rec.variant.format_name})")
        console.print(f"  Size: {rec.model_size_gb:.1f} GB ({rec.memory_reduction_pct:.0f}% reduction)")
        console.print(f"  Throughput: {rec.estimated_throughput_change_pct:+.0f}%")
        console.print(f"  Quality: {rec.estimated_quality_impact}")
        if rec.warnings:
            for w in rec.warnings:
                console.print(f"  [yellow]Warning: {w}[/yellow]")
        return 0

    result = advisor.analyze(entry.profile, entry.parameter_count)

    if args.json_output:
        print(json.dumps({
            "model": model_id,
            "sweet_spot": result.sweet_spot.variant.name if result.sweet_spot else None,
            "variants": [
                {
                    "variant": v.variant.name,
                    "format": v.variant.format_name,
                    "precision": v.variant.precision,
                    "size_gb": v.model_size_gb,
                    "memory_reduction_pct": v.memory_reduction_pct,
                    "throughput_change_pct": v.estimated_throughput_change_pct,
                    "quality_impact": v.estimated_quality_impact,
                    "best_gpu": v.gpu_rankings[0]["gpu_name"] if v.gpu_rankings else None,
                    "fits_on_count": len(v.fits_on_gpus),
                    "warnings": v.warnings,
                }
                for v in result.variants
            ],
        }, indent=2))
        return 0

    sweet = result.sweet_spot.variant.name if result.sweet_spot else "N/A"
    table = Table(title=f"Quantization Variants for {entry.tag} (sweet spot: {sweet})")
    table.add_column("Variant", style="cyan")
    table.add_column("Format")
    table.add_column("Size (GB)", justify="right")
    table.add_column("Mem Reduction", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Quality", style="green")
    table.add_column("Best GPU", style="yellow")
    table.add_column("Fits On", justify="right")

    for v in result.variants:
        best_gpu = v.gpu_rankings[0]["gpu_name"] if v.gpu_rankings else "—"
        style = "bold" if result.sweet_spot and v.variant.name == result.sweet_spot.variant.name else ""
        table.add_row(
            v.variant.name,
            v.variant.format_name,
            f"{v.model_size_gb:.1f}",
            f"{v.memory_reduction_pct:.0f}%",
            f"{v.estimated_throughput_change_pct:+.0f}%",
            v.estimated_quality_impact,
            best_gpu,
            str(len(v.fits_on_gpus)),
            style=style,
        )

    console.print(table)

    if result.sweet_spot and result.sweet_spot.warnings:
        console.print()
        for w in result.sweet_spot.warnings:
            console.print(f"[yellow]  Warning: {w}[/yellow]")

    return 0


def _cmd_prices(data_dir, args) -> int:
    console = Console()
    from intelligence.pricing import GPUPricingTracker

    tracker = GPUPricingTracker(data_dir=data_dir)

    if args.update_from:
        count = tracker.update_from_json(Path(args.update_from))
        console.print(f"[green]Updated {count} GPU prices from {args.update_from}[/green]\n")

    sources = tracker.get_all_sources()

    if args.json_output:
        print(json.dumps({
            gpu: {
                "on_demand_rate": src.on_demand_rate,
                "spot_rate": src.spot_rate,
                "provider": src.provider,
            }
            for gpu, src in sources.items()
        }, indent=2))
        return 0

    table = Table(title=f"GPU Pricing ({len(sources)} GPUs)")
    table.add_column("GPU", style="cyan")
    table.add_column("$/hr", justify="right", style="green")
    table.add_column("Spot $/hr", justify="right", style="yellow")
    table.add_column("Provider")

    for gpu_name in sorted(sources.keys()):
        src = sources[gpu_name]
        spot = f"${src.spot_rate:.2f}" if src.spot_rate else "—"
        table.add_row(
            gpu_name,
            f"${src.on_demand_rate:.2f}",
            spot,
            src.provider,
        )

    console.print(table)
    return 0


def _cmd_best_value(pipeline, data_dir, args) -> int:
    console = Console()
    model_id = args.model

    from intelligence.pricing import GPUPricingTracker
    from efficiency.gpu_specs import MODEL_PROFILES

    tracker = GPUPricingTracker(data_dir=data_dir)

    # Try local profile first, then fetch
    profile = MODEL_PROFILES.get(model_id)
    if not profile:
        entry = pipeline.run_single(model_id)
        if not entry:
            console.print(f"[red]Could not profile model: {model_id}[/red]")
            return 1
        profile = entry.profile

    console.print(f"[bold]Best $/TFLOP GPUs for {profile.tag}[/bold]\n")

    results = tracker.compute_price_performance(profile, top_n=args.top)

    if args.budget:
        results = [r for r in results if r.on_demand_rate <= args.budget]

    if args.json_output:
        print(json.dumps([
            {
                "gpu": r.gpu_name,
                "on_demand_rate": r.on_demand_rate,
                "spot_rate": r.spot_rate,
                "effective_tflops": r.effective_tflops,
                "dollars_per_tflop_hr": r.dollars_per_tflop_hr,
                "value_score": r.value_score,
                "is_best_value": r.is_best_value,
            }
            for r in results
        ], indent=2))
        return 0

    if not results:
        console.print("[yellow]No GPUs found within budget[/yellow]")
        return 0

    table = Table(title=f"Price-Performance Ranking ({profile.tag})")
    table.add_column("#", justify="right", style="dim")
    table.add_column("GPU", style="cyan")
    table.add_column("$/hr", justify="right", style="yellow")
    table.add_column("Eff. TFLOPS", justify="right")
    table.add_column("$/TFLOP-hr", justify="right", style="green")
    table.add_column("Value Score", justify="right")

    for i, r in enumerate(results, 1):
        style = "bold" if r.is_best_value else ""
        table.add_row(
            str(i),
            r.gpu_name,
            f"${r.on_demand_rate:.2f}",
            f"{r.effective_tflops:.1f}",
            f"${r.dollars_per_tflop_hr:.4f}",
            f"{r.value_score:.0f}",
            style=style,
        )

    console.print(table)
    return 0


def _cmd_research(pipeline, data_dir, args) -> int:
    console = Console()
    agent = _make_research_agent(pipeline, data_dir)

    def one_cycle() -> int:
        result = agent.run_cycle(
            limit=args.limit,
            min_downloads=args.min_downloads,
            min_confidence=args.min_confidence,
            scan=not args.no_scan,
        )

        if args.json_output:
            print(json.dumps({
                "new_models": result.new_models,
                "targets_total": result.targets_total,
                "targets_new": result.targets_new,
                "measurements_ingested": result.measurements_ingested,
                "calibration_updates": [
                    {
                        "gpu": u.gpu_name, "model": u.model_tag,
                        "predicted_tok_s": u.predicted_tokens_per_sec,
                        "measured_tok_s": u.measured_tokens_per_sec,
                        "ratio": u.ratio, "new_factor": u.new_factor,
                        "samples": u.samples,
                    }
                    for u in result.calibration_updates
                ],
                "errors": result.errors,
                "duration_s": result.duration_s,
            }, indent=2))
            return 0

        console.print(
            f"[bold]Research cycle complete[/bold] ({result.duration_s}s): "
            f"{result.new_models} new models, "
            f"{result.targets_total} benchmark targets ({result.targets_new} new), "
            f"{result.measurements_ingested} measured runs ingested"
        )
        for u in result.calibration_updates:
            console.print(
                f"  [green]calibrated[/green] {u.model_tag} on {u.gpu_name}: "
                f"predicted {u.predicted_tokens_per_sec:.0f} tok/s, "
                f"measured {u.measured_tokens_per_sec:.0f} tok/s "
                f"(factor → {u.new_factor:.2f}, n={u.samples})"
            )
        if result.errors:
            console.print(f"[yellow]{len(result.errors)} errors[/yellow]")
            for err in result.errors:
                console.print(f"  - {err}")
        console.print(
            f"\n[dim]Drop `nemulai test --output` JSON files in {agent.watch_dir} "
            f"to keep calibrating predictions.[/dim]"
        )
        return 0

    if args.watch:
        import time as _time
        console.print(f"[bold]Research agent watching (every {args.watch:.0f}s, Ctrl-C to stop)[/bold]\n")
        try:
            while True:
                one_cycle()
                _time.sleep(args.watch)
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/dim]")
            return 0

    return one_cycle()


def _cmd_targets(pipeline, data_dir, args) -> int:
    console = Console()
    agent = _make_research_agent(pipeline, data_dir)

    targets = agent.targets
    if args.model:
        targets = [t for t in targets if args.model.lower() in t.model_tag.lower()]
    if args.gpu:
        targets = [t for t in targets if args.gpu.lower() in t.gpu_name.lower()]

    if args.json_output:
        print(json.dumps([t.to_dict() for t in targets], indent=2))
        return 0

    if not targets:
        console.print("No benchmark targets. Run [bold]nemulai model-intel research[/bold] first.")
        return 0

    table = Table(title=f"Benchmark Targets ({len(targets)})")
    table.add_column("Model", style="cyan")
    table.add_column("GPU", style="yellow")
    table.add_column("Predicted tok/s", justify="right")
    table.add_column("J/token", justify="right")
    table.add_column("$/1M tok", justify="right", style="green")
    table.add_column("Measured tok/s", justify="right")
    table.add_column("Calibration", justify="right")

    for t in targets:
        measured = f"{t.measured_tokens_per_sec:.0f}" if t.measured_tokens_per_sec else "—"
        cal = (
            f"{t.calibration_factor:.2f} (n={t.calibration_samples})"
            if t.calibration_samples else "—"
        )
        cost = f"${t.calibrated_cost_per_1m_tokens_usd:.2f}" if t.calibrated_cost_per_1m_tokens_usd else "?"
        table.add_row(
            t.model_tag,
            t.gpu_name,
            f"{t.calibrated_tokens_per_sec:.0f}",
            f"{t.calibrated_joules_per_token:.3f}",
            cost,
            measured,
            cal,
        )

    console.print(table)
    return 0


def _cmd_calibrate(pipeline, data_dir, args) -> int:
    console = Console()
    agent = _make_research_agent(pipeline, data_dir)

    updates = []

    if args.gpu and args.model and args.tokens_per_sec:
        update = agent.record_measurement(args.gpu, args.model, args.tokens_per_sec)
        if update:
            updates.append(update)
    elif args.results:
        for raw_path in args.results:
            path = Path(raw_path)
            if not path.exists():
                console.print(f"[red]File not found: {path}[/red]")
                continue
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                console.print(f"[red]Invalid JSON in {path}: {exc}[/red]")
                continue
            if not data.get("nemulai_test") or not data.get("model"):
                console.print(f"[yellow]{path} is not a `nemulai test` model-mode result, skipping[/yellow]")
                continue
            tok_per_sec = (data.get("throughput") or {}).get("tok_per_sec", 0.0)
            update = agent.record_measurement(data.get("gpu", ""), data["model"], tok_per_sec)
            if update:
                updates.append(update)
    else:
        console.print("Provide result files, or --gpu/--model/--tokens-per-sec for a manual entry.")
        return 1

    if args.json_output:
        print(json.dumps([
            {
                "gpu": u.gpu_name, "model": u.model_tag,
                "predicted_tok_s": u.predicted_tokens_per_sec,
                "measured_tok_s": u.measured_tokens_per_sec,
                "ratio": u.ratio, "new_factor": u.new_factor, "samples": u.samples,
            }
            for u in updates
        ], indent=2))
        return 0

    if not updates:
        console.print("[yellow]No measurements could be matched to benchmark targets.[/yellow]")
        return 1

    for u in updates:
        console.print(
            f"[green]Calibrated[/green] {u.model_tag} on {u.gpu_name}: "
            f"predicted {u.predicted_tokens_per_sec:.0f} → measured {u.measured_tokens_per_sec:.0f} tok/s "
            f"(ratio {u.ratio:.2f}, factor → {u.new_factor:.2f}, n={u.samples})"
        )
    return 0


def _cmd_suggest(pipeline, data_dir, args) -> int:
    console = Console()
    agent = _make_research_agent(pipeline, data_dir)

    suggestions = agent.suggest(
        query=args.query,
        budget_per_hr=args.budget,
        top_n=args.top,
    )

    if args.json_output:
        print(json.dumps([s.to_dict() for s in suggestions], indent=2))
        return 0

    if not suggestions:
        console.print(
            "No pairings available. Run [bold]nemulai model-intel research[/bold] first"
            + (" or relax --budget." if args.budget else ".")
        )
        return 0

    title = "Best Model + GPU Pairings"
    if args.query:
        title += f" — '{args.query}'"
    if args.budget:
        title += f" (≤ ${args.budget:.2f}/hr)"

    table = Table(title=title)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("GPU", style="yellow")
    table.add_column("tok/s", justify="right")
    table.add_column("J/token", justify="right")
    table.add_column("$/1M tok", justify="right", style="green")
    table.add_column("$/hr", justify="right")
    table.add_column("Quantize", style="magenta")
    table.add_column("Source", style="dim")

    for i, s in enumerate(suggestions, 1):
        cost_1m = f"${s.cost_per_1m_tokens_usd:.2f}" if s.cost_per_1m_tokens_usd else "?"
        cost_hr = f"${s.cost_per_hr:.2f}" if s.cost_per_hr else "?"
        source = f"calibrated (n={s.calibration_samples})" if s.calibrated else "estimated"
        table.add_row(
            str(i),
            s.model_tag,
            s.gpu_name,
            f"{s.tokens_per_sec:.0f}",
            f"{s.joules_per_token:.3f}",
            cost_1m,
            cost_hr,
            s.quantization,
            source,
            style="bold" if i == 1 else "",
        )

    console.print(table)

    best = suggestions[0]
    console.print(
        f"\n[bold]Best pairing:[/bold] [cyan]{best.model_tag}[/cyan] on "
        f"[yellow]{best.gpu_name}[/yellow] with [magenta]{best.quantization}[/magenta]"
        + (f" — {best.quantization_note}" if best.quantization_note else "")
    )
    return 0


def _cmd_quant_eval(pipeline, data_dir, args) -> int:
    console = Console()
    from intelligence.quant_eval import QuantEvalHarness

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    if not variants:
        console.print("[red]No variants given[/red]")
        return 1

    gpu_name = ""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        raw = pynvml.nvmlDeviceGetName(handle)
        gpu_name = raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass

    console.print(
        f"[bold]Measuring {len(variants)} variants of {args.model}"
        f"{' on ' + gpu_name if gpu_name else ''}...[/bold]\n"
        f"[dim]Baseline: {variants[0]} · quality gate: +{args.quality_gate}% perplexity[/dim]\n"
    )

    harness = QuantEvalHarness(data_dir=data_dir, quality_gate_pct=args.quality_gate)
    result = harness.evaluate(
        args.model, variants=variants, gpu_index=args.gpu, gpu_name=gpu_name,
    )
    harness.merge_into_registry(pipeline.registry, result)

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    table = Table(title=f"Measured Quantization Variants — {args.model}")
    table.add_column("Variant", style="cyan")
    table.add_column("tok/s", justify="right")
    table.add_column("J/token", justify="right")
    table.add_column("VRAM (GB)", justify="right")
    table.add_column("PPL Δ", justify="right")
    table.add_column("Gate", justify="center")
    table.add_column("Note", style="dim")

    for v in result.variants:
        if not v.load_ok:
            table.add_row(v.variant, "—", "—", "—", "—", "✗", v.error[:50])
            continue
        gate = "[green]pass[/green]" if v.passes_quality_gate else "[red]fail[/red]"
        note = "recommended" if v.variant == result.recommended else ""
        table.add_row(
            v.variant,
            f"{v.tokens_per_sec:.1f}",
            f"{v.joules_per_token:.3f}" if v.joules_per_token else "?",
            f"{v.vram_gb:.1f}" if v.vram_gb else "?",
            f"{v.ppl_delta_pct:+.2f}%",
            gate,
            note,
            style="bold" if v.variant == result.recommended else "",
        )

    console.print(table)
    if result.recommended and result.recommended != result.baseline_variant:
        console.print(
            f"\n[bold green]Recommended: {result.recommended}[/bold green] — "
            f"better J/token within the quality gate. Registry updated with measured numbers."
        )
    else:
        console.print(
            f"\n[yellow]Staying on {result.baseline_variant}[/yellow] — "
            f"no variant beat it within the quality gate."
        )
    return 0
