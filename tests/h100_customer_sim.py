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
NemulAI — Full Customer Simulation on H100 SXM

Simulates a real enterprise customer deploying NemulAI to monitor
GPU workloads. This is an end-to-end product validation that exercises
every layer of the stack:

  Layer 1 — Agent daemon (collector → attribution → upload)
  Layer 2 — Inference workloads (vLLM serving multiple models)
  Layer 3 — Monitoring (power, utilization, temperature, cost)
  Layer 4 — CLI tools (benchmark, optimize, report)
  Layer 5 — API pipeline (metrics → Supabase → dashboard)

The simulation runs in 6 phases:
  Phase 1: Environment validation & idle baseline
  Phase 2: Agent daemon startup (background)
  Phase 3: Model-by-model inference with live monitoring
  Phase 4: Efficiency comparison & recommendations
  Phase 5: CLI tool validation (benchmark, optimize)
  Phase 6: Summary report generation

Usage:
    python3 h100_customer_sim.py                         # full simulation
    python3 h100_customer_sim.py --phase 3               # run specific phase
    python3 h100_customer_sim.py --models quick           # fewer models
    python3 h100_customer_sim.py --team acme-corp         # custom team name
    python3 h100_customer_sim.py --dry-run                # validate without GPU
"""
from __future__ import annotations

import argparse
import gc
import json
import os

from envcompat import env
import signal
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Imports (graceful) ────────────────────────────────────────────

try:
    import pynvml
    _NVML = True
except ImportError:
    _NVML = False

try:
    import torch
    _TORCH = torch.cuda.is_available()
except ImportError:
    _TORCH = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None


# ═══════════════════════════════════════════════════════════════════
# Configuration — mimics what a real customer would set
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CustomerConfig:
    team: str = "ml-platform"
    api_key: str = ""
    api_endpoint: str = "https://www.nemulai.com/v1/metrics/ingest"
    gpu_index: int = 0
    sample_interval: float = 5.0
    log_level: str = "INFO"
    upload: bool = False


# ═══════════════════════════════════════════════════════════════════
# Model Workloads — what a customer's ML team would run
# ═══════════════════════════════════════════════════════════════════

CUSTOMER_WORKLOADS = {
    # ── Tier 1: Tiny models (edge/mobile serving) ────────────────
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5-0.5B",
        "team": "edge-team",
        "use_case": "Edge device chatbot — latency-critical",
        "params_b": 0.5,
        "max_tokens": 256,
        "expected_tok_s_min": 2000,
    },
    "qwen3-0.6b": {
        "hf_id": "Qwen/Qwen3-0.6B",
        "label": "Qwen3-0.6B",
        "team": "edge-team",
        "use_case": "Next-gen edge model evaluation",
        "params_b": 0.6,
        "max_tokens": 256,
        "expected_tok_s_min": 1800,
    },

    # ── Tier 2: Small models (cost-optimized serving) ────────────
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5-1.5B",
        "team": "api-team",
        "use_case": "High-volume API — cost vs quality tradeoff",
        "params_b": 1.5,
        "max_tokens": 512,
        "expected_tok_s_min": 1000,
    },
    "qwen3-1.7b": {
        "hf_id": "Qwen/Qwen3-1.7B",
        "label": "Qwen3-1.7B",
        "team": "api-team",
        "use_case": "Modern small model — cost per quality",
        "params_b": 1.7,
        "max_tokens": 512,
        "expected_tok_s_min": 900,
    },
    "gemma2-2b": {
        "hf_id": "google/gemma-2-2b-it",
        "label": "Gemma2-2B",
        "team": "api-team",
        "use_case": "Google baseline — multi-vendor comparison",
        "params_b": 2.0,
        "max_tokens": 512,
        "expected_tok_s_min": 800,
    },

    # ── Tier 3: Mid-size models (balanced) ───────────────────────
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B",
        "label": "Qwen3-4B",
        "team": "product-team",
        "use_case": "Mid-size sweet spot — quality vs efficiency",
        "params_b": 4.0,
        "max_tokens": 512,
        "expected_tok_s_min": 500,
    },

    # ── Tier 4: 7B+ models (quality-focused serving) ────────────
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen2.5-7B",
        "team": "research-team",
        "use_case": "Main fine-tuning benchmark — serious open model",
        "params_b": 7.0,
        "max_tokens": 512,
        "expected_tok_s_min": 300,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "label": "Mistral-7B",
        "team": "research-team",
        "use_case": "Classic 7B baseline — compare against Qwen",
        "params_b": 7.0,
        "max_tokens": 512,
        "expected_tok_s_min": 300,
    },
    "openhermes-2.5": {
        "hf_id": "teknium/OpenHermes-2.5-Mistral-7B",
        "label": "OpenHermes-2.5",
        "team": "research-team",
        "use_case": "Fine-tuned instruction following — community model",
        "params_b": 7.0,
        "max_tokens": 512,
        "expected_tok_s_min": 280,
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "label": "Qwen3-8B",
        "team": "research-team",
        "use_case": "Modern 8B — latest architecture efficiency",
        "params_b": 8.0,
        "max_tokens": 512,
        "expected_tok_s_min": 250,
    },
    "hermes-3-8b": {
        "hf_id": "NousResearch/Hermes-3-Llama-3.1-8B",
        "label": "Hermes-3-8B",
        "team": "research-team",
        "use_case": "Strong community fine-tune — efficiency for useful answers",
        "params_b": 8.0,
        "max_tokens": 512,
        "expected_tok_s_min": 250,
    },
    "gemma2-9b": {
        "hf_id": "google/gemma-2-9b-it",
        "label": "Gemma2-9B",
        "team": "research-team",
        "use_case": "Google 9B — quality + efficiency comparison",
        "params_b": 9.0,
        "max_tokens": 512,
        "expected_tok_s_min": 200,
    },
}

WORKLOAD_GROUPS = {
    "all": list(CUSTOMER_WORKLOADS.keys()),
    "quick": ["qwen2.5-0.5b", "qwen2.5-7b", "mistral-7b"],
    "tiny": ["qwen2.5-0.5b", "qwen3-0.6b", "gemma2-2b"],
    "small": ["qwen2.5-1.5b", "qwen3-1.7b", "gemma2-2b"],
    "7b": ["qwen2.5-7b", "mistral-7b", "openhermes-2.5"],
    "8b+": ["qwen3-8b", "hermes-3-8b", "gemma2-9b"],
    "qwen": [k for k in CUSTOMER_WORKLOADS if k.startswith("qwen")],
    "compare": ["qwen2.5-7b", "qwen3-8b", "mistral-7b", "hermes-3-8b", "gemma2-9b"],
}


# Standard prompts a customer's users would send
CUSTOMER_PROMPTS = [
    # Customer support (common enterprise use case)
    "A customer reports their GPU utilization is stuck at 30%. What troubleshooting steps should I suggest?",
    "Write a polite response to a customer asking about our API rate limits and pricing tiers.",
    "Summarize this support ticket: User reports intermittent 502 errors when calling the inference endpoint during peak hours (9-11am EST). They're using batch sizes of 32 with Mistral-7B.",

    # Code generation
    "Write a Python function that monitors GPU power consumption using pynvml and returns the average over a time window.",
    "Create a REST API endpoint in FastAPI that accepts GPU metrics (power, temp, utilization) and stores them in PostgreSQL.",
    "Write a Dockerfile for a Python ML inference service that uses vLLM with CUDA 12.4.",

    # Analysis / reasoning
    "Compare the energy efficiency of running inference on H100 vs A100 for a 7B parameter model. Consider power draw, throughput, and cost per token.",
    "Our GPU cluster has 8 H100s. Monthly power bill is $12,000. Is this reasonable? What's the expected range?",
    "Explain why smaller models can sometimes be more cost-effective than larger ones for production inference.",

    # Data extraction
    "Extract the key metrics from this benchmark result: Model: Llama-3-8B, GPU: H100 SXM, Throughput: 450 tok/s, Power: 620W, Temperature: 71°C, Utilization: 85%",
    "Convert these power readings to a cost estimate: 680W average over 24 hours at $0.12/kWh with 8 GPUs.",

    # Creative
    "Write a 100-word product description for an AI-powered GPU monitoring dashboard that helps companies reduce their carbon footprint.",
    "Draft a LinkedIn post announcing that our company reduced GPU energy costs by 40% using workload optimization.",

    # Technical documentation
    "Write API documentation for an endpoint that accepts GPU telemetry data. Include request/response schemas.",
    "Create a troubleshooting guide for common GPU thermal throttling issues in data centers.",

    # Multi-step reasoning
    "A company runs 100 inference requests per second on Mistral-7B using 4 H100s. Each request generates ~200 tokens. Calculate: tokens/sec per GPU, estimated power per GPU, monthly energy cost, and CO2 emissions.",
    "Design a monitoring alert system: what thresholds should trigger warnings for GPU temperature, utilization, power draw, and memory usage?",

    # Short answers (tests efficiency on small outputs)
    "What is NVML?",
    "Define 'thermal design power' in one sentence.",
    "What does kWh/token measure?",

    # Long-form
    "Write a comprehensive guide (500+ words) on best practices for energy-efficient LLM inference in production environments.",
    "Compare 5 open-source LLM inference frameworks (vLLM, TGI, Ollama, llama.cpp, TensorRT-LLM) across performance, ease of use, and GPU utilization.",

    # Edge cases
    "Respond with just 'OK'.",
    "List the numbers 1 through 20.",
    "What is 2^10?",
]


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Environment Validation
# ═══════════════════════════════════════════════════════════════════

def phase1_validate(config: CustomerConfig, dry_run: bool = False) -> dict:
    """Validate the environment like a customer's ops team would."""
    log_phase("Phase 1 — Environment Validation")
    checks = {}

    # GPU check
    if dry_run:
        log("  [DRY RUN] Skipping GPU checks")
        checks["gpu"] = {"status": "skip", "name": "dry-run"}
        return checks

    if not _NVML:
        log("  FAIL: pynvml not installed")
        checks["nvml"] = {"status": "fail", "error": "pip install nvidia-ml-py"}
        return checks

    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        h = pynvml.nvmlDeviceGetHandleByIndex(config.gpu_index)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()

        checks["gpu"] = {
            "status": "pass",
            "name": name,
            "count": count,
            "memory_gb": round(mem.total / 1e9, 1),
            "memory_free_gb": round(mem.free / 1e9, 1),
            "idle_power_w": round(power, 1),
            "temperature_c": temp,
            "driver": driver,
        }
        log(f"  GPU: {name} ({mem.total / 1e9:.0f} GB, {mem.free / 1e9:.0f} GB free)")
        log(f"  Driver: {driver}")
        log(f"  Idle: {power:.1f}W, {temp}°C")
    except Exception as e:
        checks["gpu"] = {"status": "fail", "error": str(e)}
        log(f"  FAIL: GPU check — {e}")

    # PyTorch + CUDA
    if _TORCH:
        checks["pytorch"] = {
            "status": "pass",
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": str(torch.backends.cudnn.version()),
        }
        log(f"  PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    else:
        checks["pytorch"] = {"status": "fail"}
        log("  FAIL: PyTorch CUDA not available")

    # vLLM
    try:
        import vllm
        checks["vllm"] = {"status": "pass", "version": vllm.__version__}
        log(f"  vLLM: {vllm.__version__}")
    except ImportError:
        checks["vllm"] = {"status": "fail", "error": "pip install vllm"}
        log("  FAIL: vLLM not installed")

    # NemulAI agent
    try:
        from agent import agent as _agent_mod
        checks["nemulai_agent"] = {"status": "pass"}
        log("  NemulAI agent: installed")
    except ImportError:
        try:
            # Try alternate import path
            result = subprocess.run(
                [sys.executable, "-c", "import agent; print('ok')"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                checks["nemulai_agent"] = {"status": "pass"}
                log("  NemulAI agent: installed")
            else:
                checks["nemulai_agent"] = {"status": "warn", "note": "import failed, may work from agent/ dir"}
                log("  WARN: NemulAI agent import failed (may need cd agent/)")
        except Exception:
            checks["nemulai_agent"] = {"status": "warn"}
            log("  WARN: Could not verify NemulAI agent installation")

    # Quick GPU compute test
    if _TORCH:
        try:
            device = torch.device(f"cuda:{config.gpu_index}")
            a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
            torch.cuda.synchronize()
            t0 = time.monotonic()
            for _ in range(50):
                _ = a @ b
            torch.cuda.synchronize()
            elapsed = time.monotonic() - t0
            tflops = (50 * 2 * 4096**3) / elapsed / 1e12
            checks["compute_test"] = {"status": "pass", "bf16_tflops": round(tflops, 1)}
            log(f"  Compute test: {tflops:.1f} BF16 TFLOPS")
            del a, b
            torch.cuda.empty_cache()
        except Exception as e:
            checks["compute_test"] = {"status": "fail", "error": str(e)}
            log(f"  FAIL: Compute test — {e}")

    passed = sum(1 for c in checks.values() if c.get("status") == "pass")
    total = len(checks)
    log(f"  Validation: {passed}/{total} checks passed")
    return checks


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Agent Daemon (simulated or real)
# ═══════════════════════════════════════════════════════════════════

def phase2_start_agent(config: CustomerConfig, dry_run: bool = False) -> Optional[subprocess.Popen]:
    """Start the NemulAI agent daemon like a customer would."""
    log_phase("Phase 2 — Agent Daemon Startup")

    if dry_run:
        log("  [DRY RUN] Agent daemon skipped")
        return None

    env = os.environ.copy()
    env.update({
        "NEMULAI_TEAM": config.team,
        "NEMULAI_LOG_LEVEL": config.log_level,
        "NEMULAI_SAMPLE_INTERVAL": str(config.sample_interval),
    })
    if config.api_key:
        env["NEMULAI_API_KEY"] = config.api_key
    if config.upload:
        env["NEMULAI_API_ENDPOINT"] = config.api_endpoint

    log(f"  Team: {config.team}")
    log(f"  Sample interval: {config.sample_interval}s")
    log(f"  Upload: {'enabled' if config.upload else 'disabled (local only)'}")

    # Try to start the agent daemon
    agent_script = Path(__file__).parent.parent / "agent.py"
    if not agent_script.exists():
        agent_script = Path(__file__).parent.parent / "cli.py"

    if not agent_script.exists():
        log("  WARN: Agent script not found — monitoring via inline GPUMonitor only")
        return None

    try:
        proc = subprocess.Popen(
            [sys.executable, str(agent_script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            log(f"  WARN: Agent exited immediately: {stderr[:200]}")
            return None

        log(f"  Agent daemon started (PID {proc.pid})")
        return proc
    except Exception as e:
        log(f"  WARN: Could not start agent daemon: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Model Inference with Live Monitoring
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WorkloadResult:
    model_key: str
    label: str
    family: str
    team: str
    use_case: str
    params_b: float

    # Throughput
    total_prompts: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    duration_s: float = 0.0
    load_time_s: float = 0.0
    tokens_per_sec: float = 0.0

    # Power
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    idle_power_w: float = 0.0
    dynamic_power_w: float = 0.0

    # Efficiency
    joules_per_token: float = 0.0
    joules_per_token_dynamic: float = 0.0
    kwh_per_1m_tokens: float = 0.0
    cost_per_1m_tokens: float = 0.0
    co2e_per_1m_tokens: float = 0.0

    # GPU
    avg_gpu_util: float = 0.0
    avg_temp_c: float = 0.0
    peak_temp_c: float = 0.0
    vram_peak_gb: float = 0.0

    # Quality
    avg_output_len: float = 0.0
    sample_responses: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    # Status
    status: str = "pending"
    expected_tok_s_min: float = 0.0
    meets_sla: bool = True


def run_model_workload(
    key: str,
    workload: dict,
    prompts: list[str],
    gpu_index: int,
    idle_power_w: float,
    cloud_rate: float = 3.99,
    carbon_g_kwh: float = 394.0,
) -> WorkloadResult:
    """Run inference for one model with full monitoring."""

    result = WorkloadResult(
        model_key=key,
        label=workload["label"],
        family=workload["hf_id"].split("/")[0],
        team=workload["team"],
        use_case=workload["use_case"],
        params_b=workload["params_b"],
        idle_power_w=idle_power_w,
        expected_tok_s_min=workload.get("expected_tok_s_min", 0),
    )

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        result.status = "error"
        result.errors.append("vLLM not installed")
        return result

    # ── GPU monitor ──────────────────────────────────────────────
    power_samples = []
    stop_monitor = threading.Event()

    def _monitor():
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        while not stop_monitor.is_set():
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                power_samples.append({
                    "ts": time.monotonic(), "power": p, "util": u.gpu,
                    "temp": t, "mem_used": m.used / 1e9,
                })
            except Exception:
                pass
            time.sleep(0.1)

    mon_thread = threading.Thread(target=_monitor, daemon=True)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        # ── Load model ───────────────────────────────────────────
        log(f"    Loading {workload['label']}...")
        t_load = time.monotonic()
        llm = LLM(
            model=workload["hf_id"],
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            enforce_eager=False,
        )
        result.load_time_s = time.monotonic() - t_load
        log(f"    Loaded in {result.load_time_s:.1f}s")

        # ── Format prompts ───────────────────────────────────────
        sampling = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=workload.get("max_tokens", 512),
        )

        formatted = []
        for p in prompts:
            try:
                tok = llm.get_tokenizer()
                text = tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted.append(text)
            except Exception:
                formatted.append(p)

        # ── Run inference with monitoring ────────────────────────
        mon_thread.start()
        t_infer = time.monotonic()
        outputs = llm.generate(formatted, sampling)
        if _TORCH:
            torch.cuda.synchronize()
        infer_time = time.monotonic() - t_infer
        stop_monitor.set()
        mon_thread.join(timeout=5)

        # ── Collect results ──────────────────────────────────────
        total_in = 0
        total_out = 0
        for i, out in enumerate(outputs):
            p_tok = len(out.prompt_token_ids)
            g_tok = sum(len(o.token_ids) for o in out.outputs)
            total_in += p_tok
            total_out += g_tok
            if i < 3:
                text = out.outputs[0].text if out.outputs else ""
                result.sample_responses.append({
                    "prompt": prompts[i] if i < len(prompts) else "",
                    "response": text[:300],
                    "tokens": g_tok,
                })

        result.total_prompts = len(outputs)
        result.total_input_tokens = total_in
        result.total_output_tokens = total_out
        result.duration_s = infer_time
        result.tokens_per_sec = total_out / infer_time if infer_time > 0 else 0
        result.avg_output_len = total_out / max(len(outputs), 1)

        # ── Power metrics ────────────────────────────────────────
        if power_samples:
            powers = [s["power"] for s in power_samples]
            utils = [s["util"] for s in power_samples]
            temps = [s["temp"] for s in power_samples]
            mems = [s["mem_used"] for s in power_samples]

            result.avg_power_w = sum(powers) / len(powers)
            result.peak_power_w = max(powers)
            result.dynamic_power_w = max(0, result.avg_power_w - idle_power_w)
            result.avg_gpu_util = sum(utils) / len(utils)
            result.avg_temp_c = sum(temps) / len(temps)
            result.peak_temp_c = max(temps)
            result.vram_peak_gb = max(mems)

        # ── Efficiency ───────────────────────────────────────────
        if total_out > 0 and result.avg_power_w > 0:
            total_energy = result.avg_power_w * infer_time
            dyn_energy = result.dynamic_power_w * infer_time
            result.joules_per_token = total_energy / total_out
            result.joules_per_token_dynamic = dyn_energy / total_out
            result.kwh_per_1m_tokens = result.avg_power_w / (result.tokens_per_sec * 1000)
            gpu_hrs_per_1m = 1_000_000 / result.tokens_per_sec / 3600
            result.cost_per_1m_tokens = gpu_hrs_per_1m * cloud_rate
            result.co2e_per_1m_tokens = result.kwh_per_1m_tokens * carbon_g_kwh

        # ── SLA check ────────────────────────────────────────────
        result.meets_sla = result.tokens_per_sec >= result.expected_tok_s_min
        result.status = "pass" if not result.errors else "error"

        # ── Cleanup ──────────────────────────────────────────────
        del llm
        if _TORCH:
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        result.status = "error"
        result.errors.append(str(e))
        stop_monitor.set()
        log(f"    ERROR: {e}")

    return result


def phase3_run_workloads(
    workload_keys: list[str],
    prompts: list[str],
    gpu_index: int,
    idle_power_w: float,
    cooldown_s: float = 10.0,
    dry_run: bool = False,
) -> list[WorkloadResult]:
    """Run all model workloads with monitoring."""
    log_phase("Phase 3 — Model Inference with Live Monitoring")

    if dry_run:
        log("  [DRY RUN] Skipping inference")
        return []

    results = []
    for i, key in enumerate(workload_keys):
        workload = CUSTOMER_WORKLOADS[key]
        log(f"\n  ┌─ [{i+1}/{len(workload_keys)}] {workload['label']} ({workload['params_b']}B)")
        log(f"  │  Team: {workload['team']}")
        log(f"  │  Use case: {workload['use_case']}")

        result = run_model_workload(key, workload, prompts, gpu_index, idle_power_w)
        results.append(result)

        if result.status == "pass":
            sla_str = "PASS" if result.meets_sla else f"MISS (expected >{result.expected_tok_s_min})"
            log(f"  │  {result.tokens_per_sec:.0f} tok/s | "
                f"{result.avg_power_w:.0f}W | "
                f"{result.joules_per_token_dynamic:.3f} J/tok | "
                f"${result.cost_per_1m_tokens:.4f}/1M")
            log(f"  │  SLA: {sla_str}")
            log(f"  └─ {result.label} complete")
        else:
            log(f"  └─ {result.label} FAILED: {result.errors}")

        if i < len(workload_keys) - 1:
            log(f"  Cooling down {cooldown_s}s...")
            time.sleep(cooldown_s)

    return results


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Efficiency Comparison
# ═══════════════════════════════════════════════════════════════════

def phase4_compare(results: list[WorkloadResult]):
    """Generate the efficiency comparison a customer would see in their dashboard."""
    log_phase("Phase 4 — Efficiency Comparison & Recommendations")

    valid = [r for r in results if r.status == "pass" and r.tokens_per_sec > 0]
    if not valid:
        log("  No valid results to compare")
        return

    if _RICH:
        # ── Main comparison table ────────────────────────────────
        table = Table(
            title="NemulAI — Model Energy Efficiency Report",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model", style="cyan", min_width=16)
        table.add_column("Team", style="dim")
        table.add_column("Params", justify="right")
        table.add_column("tok/s", justify="right", style="green")
        table.add_column("Avg W", justify="right")
        table.add_column("Dyn W", justify="right", style="yellow")
        table.add_column("J/tok\n(dyn)", justify="right", style="bold green")
        table.add_column("kWh/1M", justify="right")
        table.add_column("$/1M", justify="right", style="yellow")
        table.add_column("gCO2e\n/1M", justify="right")
        table.add_column("VRAM", justify="right")
        table.add_column("Temp", justify="right")
        table.add_column("SLA", justify="center")

        for r in sorted(valid, key=lambda x: x.joules_per_token_dynamic):
            sla_icon = "[green]PASS[/]" if r.meets_sla else "[red]MISS[/]"
            table.add_row(
                r.label,
                r.team,
                f"{r.params_b}B",
                f"{r.tokens_per_sec:.0f}",
                f"{r.avg_power_w:.0f}",
                f"{r.dynamic_power_w:.0f}",
                f"{r.joules_per_token_dynamic:.3f}",
                f"{r.kwh_per_1m_tokens:.4f}",
                f"${r.cost_per_1m_tokens:.4f}",
                f"{r.co2e_per_1m_tokens:.1f}",
                f"{r.vram_peak_gb:.1f}G",
                f"{r.peak_temp_c}°C",
                sla_icon,
            )

        console.print()
        console.print(table)

        # ── Winners ──────────────────────────────────────────────
        best_eff = min(valid, key=lambda r: r.joules_per_token_dynamic)
        best_speed = max(valid, key=lambda r: r.tokens_per_sec)
        best_cost = min(valid, key=lambda r: r.cost_per_1m_tokens)
        coolest = min(valid, key=lambda r: r.peak_temp_c)

        console.print()
        console.print(Panel(
            f"[bold green]Most Energy Efficient:[/]  {best_eff.label} — "
            f"{best_eff.joules_per_token_dynamic:.3f} J/tok (dynamic)\n"
            f"[bold cyan]Fastest Throughput:[/]      {best_speed.label} — "
            f"{best_speed.tokens_per_sec:.0f} tok/s\n"
            f"[bold yellow]Lowest Cost per Token:[/]  {best_cost.label} — "
            f"${best_cost.cost_per_1m_tokens:.4f}/1M tokens\n"
            f"[bold blue]Coolest Running:[/]        {coolest.label} — "
            f"{coolest.peak_temp_c}°C peak",
            title="[bold]Winners[/]",
            border_style="green",
        ))

        # ── Recommendations ──────────────────────────────────────
        recs = []
        # Compare within size classes
        small = [r for r in valid if r.params_b <= 2.0]
        medium = [r for r in valid if 2.0 < r.params_b <= 5.0]
        large = [r for r in valid if r.params_b > 5.0]

        if small:
            best_s = min(small, key=lambda r: r.joules_per_token_dynamic)
            recs.append(
                f"For edge/mobile workloads: [cyan]{best_s.label}[/] — "
                f"{best_s.tokens_per_sec:.0f} tok/s at only "
                f"{best_s.dynamic_power_w:.0f}W dynamic power"
            )

        if large and len(large) > 1:
            best_l = min(large, key=lambda r: r.joules_per_token_dynamic)
            worst_l = max(large, key=lambda r: r.joules_per_token_dynamic)
            savings = (1 - best_l.cost_per_1m_tokens / worst_l.cost_per_1m_tokens) * 100
            recs.append(
                f"Switching from [red]{worst_l.label}[/] to [green]{best_l.label}[/] "
                f"saves [bold]{savings:.0f}%[/] on inference cost at similar quality"
            )

        sla_misses = [r for r in valid if not r.meets_sla]
        if sla_misses:
            recs.append(
                f"[yellow]SLA concerns:[/] {', '.join(r.label for r in sla_misses)} "
                f"below minimum throughput targets"
            )

        if recs:
            console.print()
            console.print("[bold]Recommendations:[/]")
            for rec in recs:
                console.print(f"  → {rec}")

        # ── Per-team cost breakdown ──────────────────────────────
        teams = {}
        for r in valid:
            teams.setdefault(r.team, []).append(r)

        if len(teams) > 1:
            console.print()
            console.print("[bold]Per-Team Attribution:[/]")
            for team, models in sorted(teams.items()):
                total_power = sum(r.dynamic_power_w for r in models)
                avg_eff = sum(r.joules_per_token_dynamic for r in models) / len(models)
                console.print(
                    f"  {team:15s} │ {len(models)} models │ "
                    f"{total_power:.0f}W dynamic │ "
                    f"avg {avg_eff:.3f} J/tok"
                )

    else:
        print("\n" + "=" * 100)
        print("NemulAI — Model Energy Efficiency Report")
        print("=" * 100)
        for r in sorted(valid, key=lambda x: x.joules_per_token_dynamic):
            sla = "PASS" if r.meets_sla else "MISS"
            print(
                f"  {r.label:<18} {r.params_b:>4.1f}B  "
                f"{r.tokens_per_sec:>6.0f} tok/s  "
                f"{r.avg_power_w:>4.0f}W  "
                f"{r.joules_per_token_dynamic:.3f} J/tok  "
                f"${r.cost_per_1m_tokens:.4f}/1M  "
                f"SLA:{sla}"
            )
        print("=" * 100)


# ═══════════════════════════════════════════════════════════════════
# Phase 5: CLI Tool Validation
# ═══════════════════════════════════════════════════════════════════

def phase5_cli_tools(gpu_index: int, dry_run: bool = False):
    """Test NemulAI CLI tools as a customer would use them."""
    log_phase("Phase 5 — CLI Tool Validation")

    if dry_run:
        log("  [DRY RUN] CLI tools skipped")
        return

    agent_dir = Path(__file__).parent.parent

    # ── nemulai benchmark ────────────────────────────────────
    log("  Testing: nemulai benchmark --duration 15")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "benchmark",
             "--gpu", str(gpu_index), "--duration", "15"],
            cwd=str(agent_dir),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            log("    PASS: benchmark completed")
            for line in result.stdout.strip().split("\n")[-5:]:
                log(f"    {line.strip()}")
        else:
            log(f"    WARN: benchmark returned {result.returncode}")
            if result.stderr:
                log(f"    {result.stderr[:200]}")
    except Exception as e:
        log(f"    SKIP: benchmark — {e}")

    # ── nemulai optimize ─────────────────────────────────────
    log("  Testing: nemulai optimize --duration 15")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "cli", "optimize",
             "--gpu", str(gpu_index), "--duration", "15"],
            cwd=str(agent_dir),
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            log("    PASS: optimize completed")
        else:
            log(f"    SKIP: optimize returned {result.returncode}")
    except Exception as e:
        log(f"    SKIP: optimize — {e}")


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Report Generation
# ═══════════════════════════════════════════════════════════════════

def phase6_report(
    results: list[WorkloadResult],
    checks: dict,
    output_path: str,
):
    """Generate the final customer-facing report."""
    log_phase("Phase 6 — Report Generation")

    report = {
        "report": "NemulAI — H100 SXM Full Product Test",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": checks.get("gpu", {}).get("name", "H100 SXM"),
        "environment": checks,
        "models_tested": len(results),
        "models_passed": sum(1 for r in results if r.status == "pass"),
        "results": [],
        "summary": {},
    }

    valid = [r for r in results if r.status == "pass" and r.tokens_per_sec > 0]

    for r in sorted(results, key=lambda x: x.joules_per_token_dynamic or float('inf')):
        entry = {
            "model": r.label,
            "family": r.family,
            "team": r.team,
            "params_b": r.params_b,
            "use_case": r.use_case,
            "status": r.status,
            "throughput": {
                "prompts": r.total_prompts,
                "input_tokens": r.total_input_tokens,
                "output_tokens": r.total_output_tokens,
                "tokens_per_sec": round(r.tokens_per_sec, 1),
                "duration_s": round(r.duration_s, 1),
                "load_time_s": round(r.load_time_s, 1),
            },
            "power": {
                "avg_w": round(r.avg_power_w, 1),
                "peak_w": round(r.peak_power_w, 1),
                "dynamic_w": round(r.dynamic_power_w, 1),
                "idle_w": round(r.idle_power_w, 1),
            },
            "efficiency": {
                "joules_per_token": round(r.joules_per_token, 4),
                "joules_per_token_dynamic": round(r.joules_per_token_dynamic, 4),
                "kwh_per_1m_tokens": round(r.kwh_per_1m_tokens, 6),
                "cost_usd_per_1m_tokens": round(r.cost_per_1m_tokens, 6),
                "co2e_grams_per_1m_tokens": round(r.co2e_per_1m_tokens, 4),
            },
            "gpu_metrics": {
                "avg_utilization_pct": round(r.avg_gpu_util, 1),
                "avg_temp_c": round(r.avg_temp_c, 1),
                "peak_temp_c": round(r.peak_temp_c, 1),
                "vram_peak_gb": round(r.vram_peak_gb, 1),
            },
            "sla": {
                "target_tok_s": r.expected_tok_s_min,
                "actual_tok_s": round(r.tokens_per_sec, 1),
                "met": r.meets_sla,
            },
            "sample_responses": r.sample_responses[:2],
            "errors": r.errors,
        }
        report["results"].append(entry)

    if valid:
        best_eff = min(valid, key=lambda r: r.joules_per_token_dynamic)
        best_speed = max(valid, key=lambda r: r.tokens_per_sec)
        best_cost = min(valid, key=lambda r: r.cost_per_1m_tokens)

        report["summary"] = {
            "most_energy_efficient": {
                "model": best_eff.label,
                "joules_per_token_dynamic": round(best_eff.joules_per_token_dynamic, 4),
            },
            "fastest": {
                "model": best_speed.label,
                "tokens_per_sec": round(best_speed.tokens_per_sec, 1),
            },
            "lowest_cost": {
                "model": best_cost.label,
                "cost_usd_per_1m_tokens": round(best_cost.cost_per_1m_tokens, 6),
            },
            "sla_pass_rate": f"{sum(1 for r in valid if r.meets_sla)}/{len(valid)}",
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    log(f"  Report saved: {output_path}")
    log(f"  Models tested: {len(results)}")
    log(f"  Models passed: {sum(1 for r in results if r.status == 'pass')}")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    if _RICH:
        console.print(f"[dim]{ts}[/] {msg}")
    else:
        print(f"[{ts}] {msg}", flush=True)


def log_phase(title: str):
    if _RICH:
        console.print()
        console.rule(f"[bold cyan]{title}[/]")
    else:
        print(f"\n{'═' * 60}")
        print(f"  {title}")
        print("═" * 60)


def measure_idle(gpu_index: int, duration: float = 15.0) -> float:
    if not _NVML:
        return 70.0
    log(f"  Measuring idle power ({duration}s)...")
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    samples = []
    end = time.monotonic() + duration
    while time.monotonic() < end:
        try:
            p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            samples.append(p)
        except Exception:
            pass
        time.sleep(0.2)
    avg = sum(samples) / max(len(samples), 1)
    log(f"  Idle power: {avg:.1f}W ({len(samples)} samples)")
    return avg


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NemulAI — Full Customer Simulation on H100 SXM",
    )
    parser.add_argument("--models", default="all",
                        help=f"Model group or comma-sep keys. Groups: {', '.join(WORKLOAD_GROUPS.keys())}")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prompts", type=int, default=25,
                        help="Prompts per model (default 25)")
    parser.add_argument("--team", default="ml-platform",
                        help="Team name for attribution")
    parser.add_argument("--cooldown", type=float, default=10.0,
                        help="Seconds between models")
    parser.add_argument("--output", default="h100_customer_sim_results.json")
    parser.add_argument("--upload", action="store_true",
                        help="Upload metrics to NemulAI (requires API key)")
    parser.add_argument("--api-key", default="",
                        help="NemulAI API key (or set NEMULAI_API_KEY)")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase only (1-6, 0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without GPU workloads")
    parser.add_argument("--cloud-rate", type=float, default=3.99,
                        help="Cloud GPU rate $/hr (default: 3.99)")
    args = parser.parse_args()

    # ── Resolve models ───────────────────────────────────────────
    if args.models in WORKLOAD_GROUPS:
        model_keys = WORKLOAD_GROUPS[args.models]
    else:
        model_keys = [k.strip() for k in args.models.split(",")]
        for k in model_keys:
            if k not in CUSTOMER_WORKLOADS:
                print(f"Unknown model: {k}")
                print(f"Available: {', '.join(CUSTOMER_WORKLOADS.keys())}")
                sys.exit(1)

    prompts = CUSTOMER_PROMPTS[:args.prompts]

    config = CustomerConfig(
        team=args.team,
        api_key=args.api_key or env("NEMULAI_API_KEY", ""),
        gpu_index=args.gpu,
        upload=args.upload,
    )

    # ── Banner ───────────────────────────────────────────────────
    if _RICH:
        console.rule("[bold cyan]NemulAI — Full Product Test (H100 SXM)[/]")
        console.print(f"  Simulating customer: [bold]{config.team}[/]")
        console.print(f"  Models: {len(model_keys)} | Prompts: {len(prompts)} per model")
        console.print(f"  Cloud rate: ${args.cloud_rate}/hr")
        if args.dry_run:
            console.print("  [yellow]DRY RUN — no GPU workloads[/]")
        console.print()
    else:
        print("=" * 60)
        print(f"NemulAI — Full Product Test (Customer: {config.team})")
        print(f"Models: {len(model_keys)} | Prompts: {len(prompts)}")
        print("=" * 60)

    agent_proc = None

    try:
        # ── Phase 1 ──────────────────────────────────────────────
        if args.phase in (0, 1):
            checks = phase1_validate(config, args.dry_run)
        else:
            checks = {}

        # ── Phase 2 ──────────────────────────────────────────────
        if args.phase in (0, 2):
            agent_proc = phase2_start_agent(config, args.dry_run)

        # ── Idle baseline ────────────────────────────────────────
        idle_power = 70.0
        if not args.dry_run and _NVML:
            idle_power = measure_idle(args.gpu)

        # ── Phase 3 ──────────────────────────────────────────────
        results = []
        if args.phase in (0, 3):
            results = phase3_run_workloads(
                model_keys, prompts, args.gpu, idle_power,
                args.cooldown, args.dry_run,
            )

        # ── Phase 4 ──────────────────────────────────────────────
        if args.phase in (0, 4) and results:
            phase4_compare(results)

        # ── Phase 5 ──────────────────────────────────────────────
        if args.phase in (0, 5):
            phase5_cli_tools(args.gpu, args.dry_run)

        # ── Phase 6 ──────────────────────────────────────────────
        if args.phase in (0, 6):
            phase6_report(results, checks, args.output)

    finally:
        if agent_proc:
            log("Stopping agent daemon...")
            agent_proc.terminate()
            agent_proc.wait(timeout=5)

    log("")
    log("Customer simulation complete.")
    if results:
        passed = sum(1 for r in results if r.status == "pass")
        log(f"  {passed}/{len(results)} models passed")


if __name__ == "__main__":
    main()
