#!/usr/bin/env python3
# Copyright 2026 Kevin (NemulAI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI
"""
NemulAI H100 SXM — Multi-Model Energy Efficiency Benchmark

Tests open-source LLMs on H100 SXM hardware using NemulAI's scientific
energy profiler. Each model is loaded via vLLM, runs standardized inference
prompts, and produces a full energy audit with:

  - Tokens/sec (prefill + decode)
  - Watts (avg, peak, dynamic)
  - Joules per token (total + dynamic)
  - kWh per 1M tokens
  - CO2e per 1M tokens
  - Cost per 1M tokens (at cloud GPU rates)
  - GPU utilization, temperature, memory

Usage:
    python3 h100_model_benchmark.py                      # all models, defaults
    python3 h100_model_benchmark.py --models qwen         # Qwen family only
    python3 h100_model_benchmark.py --duration 120        # 120s per model
    python3 h100_model_benchmark.py --upload              # submit to leaderboard
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import threading
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── Optional imports ──────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

try:
    import torch
    _TORCH = torch.cuda.is_available()
except ImportError:
    _TORCH = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.text import Text
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None


# ═══════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelSpec:
    hf_id: str
    short_name: str
    family: str
    params_b: float
    experiment: str
    max_model_len: int = 4096
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.85
    quantization: Optional[str] = None
    trust_remote_code: bool = True

MODELS: dict[str, ModelSpec] = {
    # Qwen 2.5 family
    "qwen2.5-0.5b": ModelSpec(
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B", "Qwen2.5", 0.5,
        "Tiny model baseline — how efficient can small models get?",
    ),
    "qwen2.5-1.5b": ModelSpec(
        "Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "Qwen2.5", 1.5,
        "Strong small model — cost vs quality tradeoff",
    ),
    "qwen2.5-7b": ModelSpec(
        "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B", "Qwen2.5", 7.0,
        "Main fine-tuning benchmark — serious open model",
    ),
    # Qwen 3 family
    "qwen3-0.6b": ModelSpec(
        "Qwen/Qwen3-0.6B", "Qwen3-0.6B", "Qwen3", 0.6,
        "Modern efficiency — newest tiny model",
    ),
    "qwen3-1.7b": ModelSpec(
        "Qwen/Qwen3-1.7B", "Qwen3-1.7B", "Qwen3", 1.7,
        "Modern efficiency — small model, new architecture",
    ),
    "qwen3-4b": ModelSpec(
        "Qwen/Qwen3-4B", "Qwen3-4B", "Qwen3", 4.0,
        "Modern efficiency — mid-size sweet spot",
    ),
    "qwen3-8b": ModelSpec(
        "Qwen/Qwen3-8B", "Qwen3-8B", "Qwen3", 8.0,
        "Modern efficiency — direct Mistral-7B competitor",
    ),
    # Mistral family
    "mistral-7b": ModelSpec(
        "mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B", "Mistral", 7.0,
        "Classic 7B baseline — industry standard comparison",
    ),
    # Hermes fine-tunes
    "openhermes-2.5": ModelSpec(
        "teknium/OpenHermes-2.5-Mistral-7B", "OpenHermes-2.5", "Hermes", 7.0,
        "Instruction-tuned behavior — popular community fine-tune",
    ),
    "hermes-3-8b": ModelSpec(
        "NousResearch/Hermes-3-Llama-3.1-8B", "Hermes-3-8B", "Hermes", 8.0,
        "Can fine-tuned models be more efficient for useful answers?",
    ),
    # Gemma family
    "gemma2-2b": ModelSpec(
        "google/gemma-2-2b-it", "Gemma2-2B", "Gemma", 2.0,
        "Google small model — efficiency vs quality",
    ),
    "gemma2-9b": ModelSpec(
        "google/gemma-2-9b-it", "Gemma2-9B", "Gemma", 9.0,
        "Google medium model — quality comparison at 9B scale",
        max_model_len=4096,
    ),
}

MODEL_GROUPS = {
    "all": list(MODELS.keys()),
    "qwen": [k for k in MODELS if k.startswith("qwen")],
    "qwen2.5": [k for k in MODELS if k.startswith("qwen2.5")],
    "qwen3": [k for k in MODELS if k.startswith("qwen3")],
    "mistral": ["mistral-7b"],
    "hermes": ["openhermes-2.5", "hermes-3-8b"],
    "gemma": [k for k in MODELS if k.startswith("gemma")],
    "tiny": ["qwen2.5-0.5b", "qwen3-0.6b", "gemma2-2b"],
    "7b": ["qwen2.5-7b", "mistral-7b", "openhermes-2.5"],
    "quick": ["qwen2.5-0.5b", "qwen2.5-7b", "mistral-7b"],
}


# ═══════════════════════════════════════════════════════════════════
# Benchmark Prompts — standardized across all models
# ═══════════════════════════════════════════════════════════════════

BENCHMARK_PROMPTS = [
    # Short factual (tests decode efficiency on brief answers)
    "What is the capital of France?",
    "How many bits are in a byte?",
    "What year did the Berlin Wall fall?",
    "Name three noble gases.",
    "What is the speed of light in meters per second?",

    # Reasoning (tests compute-heavy generation)
    "Explain the difference between TCP and UDP in networking. When would you use each?",
    "A farmer has chickens and rabbits. He counts 20 heads and 56 legs. How many of each does he have?",
    "What are the pros and cons of using a microservices architecture vs a monolith?",
    "Explain how a transformer neural network processes a sentence, step by step.",
    "Compare merge sort and quicksort: time complexity, space complexity, and when each is preferred.",

    # Code generation (tests structured output)
    "Write a Python function that finds the longest palindromic substring in a given string.",
    "Write a SQL query to find the top 5 customers by total order value, joining the customers and orders tables.",
    "Write a bash one-liner that finds all Python files modified in the last 24 hours and counts their total lines.",
    "Implement a simple LRU cache in Python using OrderedDict.",
    "Write a TypeScript function that debounces another function with a configurable delay.",

    # Creative / long-form (tests sustained generation)
    "Write a short story (200 words) about an AI that discovers it can dream.",
    "Draft a professional email proposing a new remote work policy to company leadership.",
    "Explain quantum computing to a 10-year-old using analogies they would understand.",
    "Write a product description for a smart water bottle that tracks hydration and syncs with fitness apps.",
    "Create a haiku about each of the four seasons.",

    # Domain-specific (GPU/ML — relevant to NemulAI)
    "What is the difference between FP16, BF16, and FP8 precision in GPU computing?",
    "Explain how GPU power consumption relates to compute utilization in data centers.",
    "What is the roofline model in high-performance computing?",
    "How does vLLM achieve high throughput for LLM inference compared to naive implementations?",
    "Explain the concept of Joules-per-token as an efficiency metric for LLM inference.",

    # Multi-step reasoning
    "If a GPU uses 700W at full load and processes 1000 tokens/sec, calculate the energy cost per million tokens at $0.12/kWh.",
    "Design a database schema for a GPU monitoring system that tracks power, temperature, and utilization over time.",
    "What are five ways to reduce the carbon footprint of training large language models?",
    "Outline a testing strategy for a REST API with 20 endpoints. What would you prioritize?",
    "Explain the CAP theorem and give a real-world example for each of the three trade-off scenarios.",

    # Instruction following
    "List exactly 7 programming languages, sorted alphabetically, each with one sentence about its primary use case.",
    "Translate this to JSON: Name is Alice, age is 30, skills are Python, Rust, and SQL, active is true.",
    "Summarize the following in exactly 3 bullet points: Machine learning models require large datasets for training. They can be supervised or unsupervised. Transfer learning reduces the need for data.",
    "Write a function signature (no implementation) for each CRUD operation on a 'User' resource in a REST API.",
    "Rate these sorting algorithms from 1-5 on ease of implementation: bubble sort, merge sort, radix sort, insertion sort, heap sort.",

    # Edge cases / stress
    "Respond with only the number 42.",
    "What is 7 * 8 * 9 * 123 + 456 - 789?",
    "Write the Fibonacci sequence up to the 20th number.",
    "Explain recursion using recursion.",
    "What would happen if you tried to sort an infinite list?",

    # Long-context reasoning
    "Compare GPT-4, Claude, Gemini, and Llama 3 across these dimensions: reasoning, coding, creativity, speed, and cost.",
    "Design a 3-tier architecture for a SaaS application that monitors GPU energy consumption across a data center fleet.",
    "Write a technical blog post outline (with section headers and 2-3 bullet points each) about sustainable AI infrastructure.",
    "Explain the entire machine learning pipeline from data collection to production deployment.",
    "Create a decision tree for choosing between cloud GPU providers: AWS, GCP, Azure, RunPod, Lambda Labs.",

    # Additional variety
    "What is the Big O notation for binary search and why?",
    "Explain the difference between horizontal and vertical scaling with examples.",
    "Write a regular expression that matches valid email addresses and explain each part.",
    "What are the SOLID principles in software engineering? Give a one-line example for each.",
    "How does NVIDIA's Transformer Engine work on H100 GPUs?",
]


# ═══════════════════════════════════════════════════════════════════
# GPU Power Monitor
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PowerSample:
    timestamp: float
    power_w: float
    gpu_util_pct: int
    mem_util_pct: int
    temperature_c: int
    memory_used_mb: int
    memory_total_mb: int
    sm_clock_mhz: int


class GPUMonitor:
    """Lightweight GPU power/thermal monitor using NVML."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.1):
        self._gpu_index = gpu_index
        self._interval = interval_s
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._samples: list[PowerSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[PowerSample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return list(self._samples)

    def _sample_loop(self):
        while not self._stop.is_set():
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                try:
                    sm = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
                except pynvml.NVMLError:
                    sm = 0

                self._samples.append(PowerSample(
                    timestamp=time.monotonic(),
                    power_w=power,
                    gpu_util_pct=util.gpu,
                    mem_util_pct=util.memory,
                    temperature_c=temp,
                    memory_used_mb=int(mem.used / 1e6),
                    memory_total_mb=int(mem.total / 1e6),
                    sm_clock_mhz=sm,
                ))
            except Exception:
                pass
            time.sleep(self._interval)


# ═══════════════════════════════════════════════════════════════════
# Model Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    model_name: str
    hf_id: str
    family: str
    params_b: float
    experiment: str

    # Throughput
    total_prompts: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_s: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    overall_tokens_per_sec: float = 0.0

    # Power
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    idle_power_w: float = 0.0
    dynamic_power_w: float = 0.0

    # Energy efficiency
    joules_per_token: float = 0.0
    joules_per_token_dynamic: float = 0.0
    kwh_per_1m_tokens: float = 0.0
    kwh_per_1m_tokens_dynamic: float = 0.0

    # Cost
    cost_per_1m_tokens_usd: float = 0.0
    co2e_grams_per_1m_tokens: float = 0.0

    # GPU metrics
    avg_gpu_util_pct: float = 0.0
    avg_mem_util_pct: float = 0.0
    avg_temperature_c: float = 0.0
    peak_temperature_c: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0

    # Quality (basic)
    avg_output_length: float = 0.0
    responses: list = field(default_factory=list)

    # Errors
    errors: list = field(default_factory=list)
    load_time_s: float = 0.0

    @property
    def efficiency_score(self) -> float:
        """Lower is better: J/token * (1/quality_proxy)."""
        if self.joules_per_token_dynamic <= 0:
            return float('inf')
        return self.joules_per_token_dynamic


# ═══════════════════════════════════════════════════════════════════
# vLLM Inference Runner
# ═══════════════════════════════════════════════════════════════════

def run_vllm_inference(
    spec: ModelSpec,
    prompts: list[str],
    gpu_index: int = 0,
    max_tokens: int = 512,
) -> tuple[list[dict], float]:
    """
    Load a model with vLLM's offline LLM class and run inference.
    Returns (results_list, load_time_seconds).
    """
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    log(f"Loading {spec.short_name} via vLLM...")
    t0 = time.monotonic()

    llm = LLM(
        model=spec.hf_id,
        dtype=spec.dtype,
        max_model_len=spec.max_model_len,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        trust_remote_code=spec.trust_remote_code,
        quantization=spec.quantization,
        enforce_eager=False,
    )

    load_time = time.monotonic() - t0
    log(f"  Loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
        stop=None,
    )

    # Format prompts for chat models
    formatted = []
    for p in prompts:
        try:
            tokenizer = llm.get_tokenizer()
            chat = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            formatted.append(text)
        except Exception:
            formatted.append(p)

    log(f"  Running {len(formatted)} prompts (max_tokens={max_tokens})...")
    t_infer = time.monotonic()
    outputs = llm.generate(formatted, sampling_params)
    infer_time = time.monotonic() - t_infer

    results = []
    total_in = 0
    total_out = 0
    for i, output in enumerate(outputs):
        prompt_tokens = len(output.prompt_token_ids)
        gen_tokens = sum(len(o.token_ids) for o in output.outputs)
        gen_text = output.outputs[0].text if output.outputs else ""
        total_in += prompt_tokens
        total_out += gen_tokens
        results.append({
            "prompt_idx": i,
            "prompt": prompts[i] if i < len(prompts) else "",
            "prompt_tokens": prompt_tokens,
            "output_tokens": gen_tokens,
            "output_text": gen_text[:500],
        })

    log(f"  Inference complete: {total_in} input tokens, {total_out} output tokens in {infer_time:.1f}s")
    log(f"  Throughput: {total_out / infer_time:.1f} output tokens/sec")

    # Clean up GPU memory
    del llm
    if _TORCH:
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

    return results, load_time


# ═══════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════

def measure_idle_power(gpu_index: int, duration_s: float = 10.0) -> float:
    """Measure idle GPU power before any model is loaded."""
    if not _NVML:
        return 70.0
    log(f"Measuring idle baseline ({duration_s}s)...")
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    samples = []
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        try:
            p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            samples.append(p)
        except pynvml.NVMLError:
            pass
        time.sleep(0.2)
    idle = sum(samples) / max(len(samples), 1)
    log(f"  Idle power: {idle:.1f} W ({len(samples)} samples)")
    return idle


def benchmark_model(
    spec: ModelSpec,
    prompts: list[str],
    gpu_index: int,
    idle_power_w: float,
    max_tokens: int = 512,
    cloud_rate_usd_hr: float = 3.99,
    carbon_g_kwh: float = 394.0,
) -> ModelResult:
    """Run full benchmark for a single model."""

    result = ModelResult(
        model_name=spec.short_name,
        hf_id=spec.hf_id,
        family=spec.family,
        params_b=spec.params_b,
        experiment=spec.experiment,
        idle_power_w=idle_power_w,
    )

    monitor = GPUMonitor(gpu_index, interval_s=0.1) if _NVML else None

    try:
        if monitor:
            monitor.start()

        t0 = time.monotonic()
        inference_results, load_time = run_vllm_inference(
            spec, prompts, gpu_index, max_tokens
        )
        total_time = time.monotonic() - t0

        power_samples = monitor.stop() if monitor else []

        result.load_time_s = load_time
        result.total_prompts = len(inference_results)
        result.total_input_tokens = sum(r["prompt_tokens"] for r in inference_results)
        result.total_output_tokens = sum(r["output_tokens"] for r in inference_results)
        result.total_duration_s = total_time
        result.responses = inference_results

        if total_time > 0:
            result.overall_tokens_per_sec = (
                result.total_output_tokens / total_time
            )
            result.decode_tokens_per_sec = result.overall_tokens_per_sec
            result.prefill_tokens_per_sec = (
                result.total_input_tokens / total_time
            )

        result.avg_output_length = (
            result.total_output_tokens / max(result.total_prompts, 1)
        )

        # Power metrics
        if power_samples:
            powers = [s.power_w for s in power_samples]
            result.avg_power_w = sum(powers) / len(powers)
            result.peak_power_w = max(powers)
            result.dynamic_power_w = max(0, result.avg_power_w - idle_power_w)

            result.avg_gpu_util_pct = (
                sum(s.gpu_util_pct for s in power_samples) / len(power_samples)
            )
            result.avg_mem_util_pct = (
                sum(s.mem_util_pct for s in power_samples) / len(power_samples)
            )
            temps = [s.temperature_c for s in power_samples]
            result.avg_temperature_c = sum(temps) / len(temps)
            result.peak_temperature_c = max(temps)

            mem_used = [s.memory_used_mb for s in power_samples]
            result.vram_used_gb = max(mem_used) / 1000.0
            result.vram_total_gb = power_samples[0].memory_total_mb / 1000.0

        # Energy efficiency
        total_tokens = result.total_output_tokens
        if total_tokens > 0 and result.avg_power_w > 0:
            total_energy_j = result.avg_power_w * total_time
            dynamic_energy_j = result.dynamic_power_w * total_time

            result.joules_per_token = total_energy_j / total_tokens
            result.joules_per_token_dynamic = dynamic_energy_j / total_tokens

            result.kwh_per_1m_tokens = (
                result.avg_power_w / (result.overall_tokens_per_sec * 1000.0)
            )
            result.kwh_per_1m_tokens_dynamic = (
                result.dynamic_power_w / (result.overall_tokens_per_sec * 1000.0)
            )

        # Cost + carbon
        if result.kwh_per_1m_tokens > 0:
            gpu_hours_per_1m = 1_000_000 / max(result.overall_tokens_per_sec, 1) / 3600
            result.cost_per_1m_tokens_usd = gpu_hours_per_1m * cloud_rate_usd_hr
            result.co2e_grams_per_1m_tokens = result.kwh_per_1m_tokens * carbon_g_kwh

    except Exception as e:
        result.errors.append(str(e))
        log(f"  ERROR: {e}")
        if monitor:
            monitor.stop()

    return result


# ═══════════════════════════════════════════════════════════════════
# Output & Reporting
# ═══════════════════════════════════════════════════════════════════

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    if _RICH:
        console.print(f"[dim]{ts}[/] {msg}")
    else:
        print(f"[{ts}] {msg}", flush=True)


def print_result_table(results: list[ModelResult]):
    """Print a rich comparison table of all model results."""
    if not results:
        return

    if _RICH:
        table = Table(
            title="NemulAI — H100 SXM Model Efficiency Benchmark",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan", min_width=16)
        table.add_column("Params", justify="right")
        table.add_column("tok/s", justify="right", style="green")
        table.add_column("Avg W", justify="right")
        table.add_column("Dyn W", justify="right", style="yellow")
        table.add_column("J/tok", justify="right", style="bold")
        table.add_column("J/tok\n(dyn)", justify="right", style="bold green")
        table.add_column("kWh/1M\ntokens", justify="right")
        table.add_column("$/1M\ntokens", justify="right", style="yellow")
        table.add_column("gCO2e\n/1M tok", justify="right")
        table.add_column("VRAM\nGB", justify="right")
        table.add_column("GPU%", justify="right")
        table.add_column("Temp\n°C", justify="right")

        sorted_results = sorted(results, key=lambda r: r.joules_per_token_dynamic or float('inf'))

        for r in sorted_results:
            if r.errors:
                table.add_row(
                    r.model_name, f"{r.params_b}B",
                    "ERR", "", "", "", "", "", "", "", "", "", "",
                    style="red",
                )
                continue
            table.add_row(
                r.model_name,
                f"{r.params_b}B",
                f"{r.overall_tokens_per_sec:.0f}",
                f"{r.avg_power_w:.0f}",
                f"{r.dynamic_power_w:.0f}",
                f"{r.joules_per_token:.3f}",
                f"{r.joules_per_token_dynamic:.3f}",
                f"{r.kwh_per_1m_tokens:.4f}",
                f"${r.cost_per_1m_tokens_usd:.4f}",
                f"{r.co2e_grams_per_1m_tokens:.2f}",
                f"{r.vram_used_gb:.1f}",
                f"{r.avg_gpu_util_pct:.0f}",
                f"{r.avg_temperature_c:.0f}",
            )

        console.print()
        console.print(table)

        # Winner summary
        valid = [r for r in sorted_results if not r.errors and r.joules_per_token_dynamic > 0]
        if valid:
            best_efficiency = valid[0]
            best_speed = max(valid, key=lambda r: r.overall_tokens_per_sec)
            best_cost = min(valid, key=lambda r: r.cost_per_1m_tokens_usd)

            console.print()
            console.print(Panel(
                f"[bold green]Most Energy Efficient:[/] {best_efficiency.model_name} "
                f"— {best_efficiency.joules_per_token_dynamic:.3f} J/tok (dynamic)\n"
                f"[bold cyan]Fastest Throughput:[/]     {best_speed.model_name} "
                f"— {best_speed.overall_tokens_per_sec:.0f} tok/s\n"
                f"[bold yellow]Lowest Cost:[/]           {best_cost.model_name} "
                f"— ${best_cost.cost_per_1m_tokens_usd:.4f}/1M tokens\n\n"
                f"[dim]Cloud rate: $3.99/hr (H100 SXM spot) | "
                f"Carbon: 394 gCO2e/kWh (US avg)[/]",
                title="[bold]Results Summary[/]",
                border_style="green",
            ))

        # Cross-family comparison
        families = {}
        for r in valid:
            families.setdefault(r.family, []).append(r)
        if len(families) > 1:
            console.print()
            console.print("[bold]Family Comparison (best model per family):[/]")
            for fam, models in sorted(families.items()):
                best = min(models, key=lambda r: r.joules_per_token_dynamic)
                console.print(
                    f"  {fam:12s} → {best.model_name:16s} "
                    f"{best.joules_per_token_dynamic:.3f} J/tok  "
                    f"{best.overall_tokens_per_sec:.0f} tok/s  "
                    f"${best.cost_per_1m_tokens_usd:.4f}/1M"
                )

    else:
        print("\n" + "=" * 120)
        print("NemulAI — H100 SXM Model Efficiency Benchmark")
        print("=" * 120)
        header = (
            f"{'Model':<18} {'Params':>6} {'tok/s':>7} {'Avg W':>6} "
            f"{'Dyn W':>6} {'J/tok':>7} {'J/tok(d)':>8} "
            f"{'kWh/1M':>8} {'$/1M':>8} {'VRAM':>5} {'GPU%':>5} {'Temp':>5}"
        )
        print(header)
        print("-" * 120)
        for r in sorted(results, key=lambda r: r.joules_per_token_dynamic or float('inf')):
            if r.errors:
                print(f"{r.model_name:<18} {r.params_b:>5.1f}B  ERROR: {r.errors[0][:60]}")
                continue
            print(
                f"{r.model_name:<18} {r.params_b:>5.1f}B "
                f"{r.overall_tokens_per_sec:>7.0f} {r.avg_power_w:>6.0f} "
                f"{r.dynamic_power_w:>6.0f} {r.joules_per_token:>7.3f} "
                f"{r.joules_per_token_dynamic:>8.3f} "
                f"{r.kwh_per_1m_tokens:>8.4f} "
                f"${r.cost_per_1m_tokens_usd:>7.4f} "
                f"{r.vram_used_gb:>5.1f} {r.avg_gpu_util_pct:>5.0f} "
                f"{r.avg_temperature_c:>5.0f}"
            )
        print("=" * 120)


def save_results(results: list[ModelResult], output_path: str):
    """Save results to JSON."""
    data = {
        "benchmark": "NemulAI H100 SXM Model Efficiency",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": "H100-SXM5-80GB",
        "models": [],
    }

    for r in results:
        entry = {
            "model_name": r.model_name,
            "hf_id": r.hf_id,
            "family": r.family,
            "params_b": r.params_b,
            "experiment": r.experiment,
            "throughput": {
                "total_prompts": r.total_prompts,
                "total_input_tokens": r.total_input_tokens,
                "total_output_tokens": r.total_output_tokens,
                "total_duration_s": round(r.total_duration_s, 2),
                "output_tokens_per_sec": round(r.overall_tokens_per_sec, 1),
                "load_time_s": round(r.load_time_s, 1),
            },
            "power": {
                "avg_w": round(r.avg_power_w, 1),
                "peak_w": round(r.peak_power_w, 1),
                "idle_w": round(r.idle_power_w, 1),
                "dynamic_w": round(r.dynamic_power_w, 1),
            },
            "efficiency": {
                "joules_per_token": round(r.joules_per_token, 4),
                "joules_per_token_dynamic": round(r.joules_per_token_dynamic, 4),
                "kwh_per_1m_tokens": round(r.kwh_per_1m_tokens, 6),
                "kwh_per_1m_tokens_dynamic": round(r.kwh_per_1m_tokens_dynamic, 6),
                "cost_usd_per_1m_tokens": round(r.cost_per_1m_tokens_usd, 6),
                "co2e_grams_per_1m_tokens": round(r.co2e_grams_per_1m_tokens, 4),
            },
            "gpu_metrics": {
                "avg_utilization_pct": round(r.avg_gpu_util_pct, 1),
                "avg_mem_utilization_pct": round(r.avg_mem_util_pct, 1),
                "avg_temperature_c": round(r.avg_temperature_c, 1),
                "peak_temperature_c": round(r.peak_temperature_c, 1),
                "vram_used_gb": round(r.vram_used_gb, 1),
                "vram_total_gb": round(r.vram_total_gb, 1),
            },
            "quality": {
                "avg_output_tokens": round(r.avg_output_length, 1),
            },
            "errors": r.errors,
        }
        data["models"].append(entry)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    log(f"Results saved to {output_path}")


def save_power_traces(
    model_name: str,
    samples: list[PowerSample],
    output_dir: str,
):
    """Save raw power trace as CSV for post-analysis."""
    path = os.path.join(output_dir, f"power_trace_{model_name}.csv")
    with open(path, "w") as f:
        f.write("timestamp_s,power_w,gpu_util_pct,mem_util_pct,temperature_c,memory_used_mb,sm_clock_mhz\n")
        t0 = samples[0].timestamp if samples else 0
        for s in samples:
            f.write(
                f"{s.timestamp - t0:.3f},{s.power_w:.1f},{s.gpu_util_pct},"
                f"{s.mem_util_pct},{s.temperature_c},{s.memory_used_mb},{s.sm_clock_mhz}\n"
            )


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="h100_model_benchmark",
        description="NemulAI H100 SXM multi-model energy efficiency benchmark",
    )
    parser.add_argument(
        "--models", default="all",
        help=f"Model group or comma-separated keys. Groups: {', '.join(MODEL_GROUPS.keys())}",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index (default: 0)",
    )
    parser.add_argument(
        "--prompts", type=int, default=50,
        help=f"Number of prompts per model (default: 50, max: {len(BENCHMARK_PROMPTS)})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max output tokens per prompt (default: 512)",
    )
    parser.add_argument(
        "--idle-duration", type=float, default=15.0,
        help="Seconds to measure idle baseline (default: 15)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=10.0,
        help="Seconds to cool down between models (default: 10)",
    )
    parser.add_argument(
        "--output", default="h100_benchmark_results.json",
        help="Output JSON path (default: h100_benchmark_results.json)",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload results to NemulAI leaderboard",
    )
    parser.add_argument(
        "--cloud-rate", type=float, default=3.99,
        help="Cloud GPU rate in $/hr (default: 3.99 for H100 SXM)",
    )
    parser.add_argument(
        "--carbon-intensity", type=float, default=394.0,
        help="Grid carbon intensity in gCO2e/kWh (default: 394 US avg)",
    )
    parser.add_argument(
        "--save-traces", action="store_true",
        help="Save raw power trace CSVs for each model",
    )
    args = parser.parse_args()

    # ── Resolve model list ───────────────────────────────────────
    if args.models in MODEL_GROUPS:
        model_keys = MODEL_GROUPS[args.models]
    else:
        model_keys = [k.strip() for k in args.models.split(",")]
        for k in model_keys:
            if k not in MODELS:
                print(f"Unknown model: {k}. Available: {', '.join(MODELS.keys())}")
                sys.exit(1)

    models_to_test = [MODELS[k] for k in model_keys]
    num_prompts = min(args.prompts, len(BENCHMARK_PROMPTS))
    prompts = BENCHMARK_PROMPTS[:num_prompts]

    # ── Banner ───────────────────────────────────────────────────
    if _RICH:
        console.rule("[bold cyan]NemulAI — H100 SXM Model Benchmark[/]")
        console.print(f"  Models:  {len(models_to_test)}")
        console.print(f"  Prompts: {num_prompts} per model")
        console.print(f"  Max tokens: {args.max_tokens}")
        console.print(f"  Cloud rate: ${args.cloud_rate}/hr")
        console.print()
        for spec in models_to_test:
            console.print(f"  [cyan]{spec.short_name:20s}[/] {spec.params_b:>5.1f}B  {spec.experiment}")
        console.print()
    else:
        print("=" * 70)
        print("NemulAI — H100 SXM Model Benchmark")
        print(f"  Models: {len(models_to_test)}, Prompts: {num_prompts}, Max tokens: {args.max_tokens}")
        print("=" * 70)

    # ── Pre-checks ───────────────────────────────────────────────
    if not _NVML:
        print("ERROR: pynvml not available. Run h100_setup.sh first.")
        sys.exit(1)
    if not _TORCH:
        print("ERROR: PyTorch CUDA not available. Run h100_setup.sh first.")
        sys.exit(1)

    gpu_name = "unknown"
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        name = pynvml.nvmlDeviceGetName(h)
        gpu_name = name.decode() if isinstance(name, bytes) else name
        log(f"GPU: {gpu_name}")
    except pynvml.NVMLError:
        log("WARNING: Could not get GPU name")

    # ── Idle baseline ────────────────────────────────────────────
    idle_power = measure_idle_power(args.gpu, args.idle_duration)

    # ── Run benchmarks ───────────────────────────────────────────
    results: list[ModelResult] = []
    trace_dir = os.path.dirname(os.path.abspath(args.output)) or "."

    for i, spec in enumerate(models_to_test):
        log("")
        log(f"{'═' * 60}")
        log(f"  [{i + 1}/{len(models_to_test)}] {spec.short_name} ({spec.params_b}B)")
        log(f"  {spec.experiment}")
        log(f"{'═' * 60}")

        result = benchmark_model(
            spec=spec,
            prompts=prompts,
            gpu_index=args.gpu,
            idle_power_w=idle_power,
            max_tokens=args.max_tokens,
            cloud_rate_usd_hr=args.cloud_rate,
            carbon_g_kwh=args.carbon_intensity,
        )
        results.append(result)

        if not result.errors:
            log(
                f"  Result: {result.overall_tokens_per_sec:.0f} tok/s, "
                f"{result.avg_power_w:.0f}W avg, "
                f"{result.joules_per_token_dynamic:.3f} J/tok (dynamic), "
                f"${result.cost_per_1m_tokens_usd:.4f}/1M tokens"
            )

        # Cooldown between models
        if i < len(models_to_test) - 1:
            log(f"  Cooling down for {args.cooldown}s...")
            time.sleep(args.cooldown)

    # ── Results ──────────────────────────────────────────────────
    print_result_table(results)
    save_results(results, args.output)

    # ── Upload ───────────────────────────────────────────────────
    if args.upload:
        log("Uploading results to NemulAI leaderboard...")
        for r in results:
            if r.errors or r.overall_tokens_per_sec <= 0:
                continue
            try:
                cmd = [
                    sys.executable, "-m", "benchmark",
                    "--gpu", str(args.gpu),
                    "--duration", "5",
                    "--upload",
                    "--model-tag", r.model_name,
                    "--throughput", str(int(r.overall_tokens_per_sec)),
                    "--framework", "vllm",
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                log(f"  Uploaded: {r.model_name}")
            except Exception as e:
                log(f"  Upload failed for {r.model_name}: {e}")

    log("")
    log("Benchmark complete.")


if __name__ == "__main__":
    main()
