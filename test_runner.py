"""
nemulai test — one-command model energy testing.

Three modes:
  nemulai test                     Monitor whatever's running on the GPU now
  nemulai test --model X           Load a model, benchmark it, get recommendations
  nemulai test --endpoint URL      Load test an existing inference server
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

try:
    import pynvml
    _NVML = True
except ImportError:
    _NVML = False

try:
    from rich.console import Console
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None


# ═══════════════════════════════════════════════════════════════════
# Built-in Prompts
# ═══════════════════════════════════════════════════════════════════

BUILTIN_PROMPTS = [
    "What is the capital of France?",
    "Explain the difference between TCP and UDP in networking.",
    "Write a Python function that finds the longest palindromic substring.",
    "Compare merge sort and quicksort: time complexity, space, and when each is preferred.",
    "What are five ways to reduce the carbon footprint of training large language models?",
    "Write a SQL query to find the top 5 customers by total order value.",
    "Explain quantum computing to a 10-year-old using analogies.",
    "Design a database schema for a GPU monitoring system.",
    "What is the roofline model in high-performance computing?",
    "Write a bash one-liner that finds Python files modified in the last 24 hours.",
    "A farmer has chickens and rabbits. 20 heads and 56 legs. How many of each?",
    "Write a REST API endpoint in FastAPI that accepts GPU telemetry data.",
    "Explain the difference between FP16, BF16, and FP8 precision in GPU computing.",
    "How does vLLM achieve high throughput compared to naive implementations?",
    "Write a Dockerfile for a Python ML inference service with CUDA 12.4.",
    "If a GPU uses 700W and processes 1000 tok/s, calculate energy cost per million tokens at $0.12/kWh.",
    "What are the SOLID principles? Give a one-line example for each.",
    "Explain recursion using recursion.",
    "List exactly 7 programming languages sorted alphabetically with one sentence about each.",
    "Write a product description for a smart water bottle that tracks hydration.",
    "Compare GPT-4, Claude, Gemini, and Llama across reasoning, coding, and cost.",
    "What is Big O notation for binary search and why?",
    "Explain horizontal vs vertical scaling with examples.",
    "How does GPU power consumption relate to compute utilization in data centers?",
    "Summarize in 3 bullet points: ML models need data, can be supervised or unsupervised, transfer learning reduces data needs.",
]


# ═══════════════════════════════════════════════════════════════════
# GPU Monitor Thread
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


class GPUMonitor:
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
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[PowerSample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return list(self._samples)

    def _loop(self):
        while not self._stop.is_set():
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._samples.append(PowerSample(
                    timestamp=time.monotonic(),
                    power_w=power,
                    gpu_util_pct=util.gpu,
                    mem_util_pct=util.memory,
                    temperature_c=temp,
                    memory_used_mb=int(mem.used / 1e6),
                    memory_total_mb=int(mem.total / 1e6),
                ))
            except Exception:
                pass
            time.sleep(self._interval)


# ═══════════════════════════════════════════════════════════════════
# Test Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    mode: str
    gpu_name: str
    gpu_index: int
    model: Optional[str] = None

    # Throughput (model/endpoint modes only)
    total_tokens: int = 0
    duration_s: float = 0.0
    tokens_per_sec: float = 0.0

    # Power
    idle_power_w: float = 0.0
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    dynamic_power_w: float = 0.0

    # Efficiency
    joules_per_token: float = 0.0
    joules_per_token_dynamic: float = 0.0
    kwh_per_1m_tokens: float = 0.0
    cost_per_1m_tokens_usd: float = 0.0
    co2e_per_1m_tokens_g: float = 0.0

    # GPU
    avg_util_pct: float = 0.0
    avg_mem_util_pct: float = 0.0
    avg_temp_c: float = 0.0
    peak_temp_c: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0

    # Recommendations
    recommendations: list = field(default_factory=list)
    workload_regime: str = ""

    # Baseline (--compare mode)
    baseline_tokens_per_sec: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# GPU Info
# ═══════════════════════════════════════════════════════════════════

def _get_gpu_info(gpu_index: int) -> dict:
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
    try:
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
    except pynvml.NVMLError:
        power_limit = 0
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
    except Exception:
        driver = "unknown"
    return {
        "name": name, "vram_total_gb": round(mem.total / 1e9, 1),
        "vram_free_gb": round(mem.free / 1e9, 1), "idle_power_w": round(power, 1),
        "power_limit_w": round(power_limit), "driver": driver,
    }


def _measure_idle(gpu_index: int, duration_s: float = 10.0) -> float:
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    samples = []
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        try:
            samples.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        except Exception:
            pass
        time.sleep(0.2)
    return sum(samples) / max(len(samples), 1)


def _get_cloud_rate(gpu_name: str) -> float:
    try:
        from efficiency.cloud_detect import GPU_HOURLY_RATES
        for key, rate in GPU_HOURLY_RATES.items():
            if key.lower() in gpu_name.lower() or gpu_name.lower() in key.lower():
                return rate
    except ImportError:
        pass
    return 3.99


# ═══════════════════════════════════════════════════════════════════
# Recommendations (reuses optimize.py WorkloadAnalyzer)
# ═══════════════════════════════════════════════════════════════════

def _get_recommendations(samples: list[PowerSample], gpu_name: str, gpu_index: int, duration_s: float) -> tuple[list, str]:
    try:
        from efficiency.gpu_specs import resolve_arch
        from optimize import WorkloadAnalyzer
        arch = resolve_arch(gpu_name)
        analyzer = WorkloadAnalyzer(arch_spec=arch)
        sample_dicts = [{
            "power_draw_w": s.power_w,
            "utilization_gpu_pct": s.gpu_util_pct,
            "utilization_memory_pct": s.mem_util_pct,
            "temperature_c": s.temperature_c,
            "power_limit_w": 0,
        } for s in samples]
        result = analyzer.analyze(sample_dicts, gpu_name, gpu_index, duration_s)
        recs = [{"priority": r.priority, "description": r.description, "action": r.action,
                 "savings_pct": r.estimated_savings_pct} for r in result.recommendations]
        regime = "memory-bound" if result.is_memory_bound else "compute-bound" if result.is_memory_bound is False else "unknown"
        return recs, regime
    except Exception:
        return [], "unknown"


# ═══════════════════════════════════════════════════════════════════
# Power Cap A/B (graceful)
# ═══════════════════════════════════════════════════════════════════

def _try_power_cap_ab(gpu_index: int, baseline_watts: int, cap_watts: int, duration_s: float = 20.0) -> Optional[dict]:
    try:
        from efficiency.power_control import set_power_limit, get_power_limit
        original = get_power_limit(gpu_index)
        if original <= 0:
            return None
        ok = set_power_limit(gpu_index, cap_watts, quiet=True)
        if not ok:
            return None
        time.sleep(duration_s)
        monitor = GPUMonitor(gpu_index, 0.2)
        monitor.start()
        time.sleep(duration_s)
        capped_samples = monitor.stop()
        set_power_limit(gpu_index, original, quiet=True)
        if not capped_samples:
            return None
        capped_avg = sum(s.power_w for s in capped_samples) / len(capped_samples)
        savings = (1.0 - capped_avg / baseline_watts) * 100 if baseline_watts > 0 else 0
        return {"cap_watts": cap_watts, "capped_avg_w": round(capped_avg, 1), "savings_pct": round(savings, 1)}
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Mode 1: Monitor (no args)
# ═══════════════════════════════════════════════════════════════════

def _run_monitor(gpu_index: int, duration_s: float, idle_power: float, gpu_info: dict) -> Optional[TestResult]:
    _log(f"Monitoring GPU {gpu_index}...")

    # Early idle detection: sample for 5s, bail if nothing's running
    monitor = GPUMonitor(gpu_index, 0.2)
    monitor.start()
    time.sleep(5.0)
    early = monitor.stop()
    if early:
        avg_util = sum(s.gpu_util_pct for s in early) / len(early)
        if avg_util < 5:
            _log("")
            _log("GPU is idle — nothing to measure.")
            _log("Start your workload, then run again. Or use:")
            _log("  nemulai test --model Qwen/Qwen2.5-7B-Instruct")
            _log("  nemulai test --endpoint http://localhost:8000")
            return None

    # GPU is active — monitor for the full duration
    _log(f"Workload detected ({avg_util:.0f}% util). Monitoring for {duration_s:.0f}s...")
    monitor = GPUMonitor(gpu_index, 0.1)
    monitor.start()
    time.sleep(duration_s)
    samples = monitor.stop()
    return _build_result("monitor", samples, gpu_info, gpu_index, idle_power, duration_s)


# ═══════════════════════════════════════════════════════════════════
# Mode 2: Model (--model X)
# ═══════════════════════════════════════════════════════════════════

def _load_prompts(args) -> list[str]:
    if args.prompts_file:
        prompts = []
        with open(args.prompts_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        prompts.append(json.loads(line))
                    except json.JSONDecodeError:
                        prompts.append(line)
        return prompts
    return BUILTIN_PROMPTS[:args.prompts]


def _run_inference_transformers(model_id: str, prompts: list[str], max_tokens: int, gpu_index: int) -> tuple[int, float]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _log(f"Loading {model_id} via transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=f"cuda:{gpu_index}",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log(f"Running {len(prompts)} prompts (max_tokens={max_tokens})...")
    total_tokens = 0
    t0 = time.time()
    for prompt in prompts:
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.9)
        gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += gen_tokens
    elapsed = time.time() - t0

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return total_tokens, elapsed


def _run_inference_vllm(model_id: str, prompts: list[str], max_tokens: int, gpu_index: int) -> tuple[int, float]:
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    _log(f"Loading {model_id} via vLLM...")
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=4096, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)

    formatted = []
    for p in prompts:
        try:
            tok = llm.get_tokenizer()
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            formatted.append(text)
        except Exception:
            formatted.append(p)

    _log(f"Running {len(formatted)} prompts (max_tokens={max_tokens})...")
    t0 = time.time()
    outputs = llm.generate(formatted, sampling)
    elapsed = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return total_tokens, elapsed


def _run_model(args, gpu_index: int, idle_power: float, gpu_info: dict) -> TestResult:
    prompts = _load_prompts(args)
    use_vllm = args.fast
    if use_vllm:
        try:
            import vllm  # noqa: F401
        except ImportError:
            _log("vLLM not installed, falling back to transformers")
            use_vllm = False

    if not use_vllm:
        try:
            import transformers  # noqa: F401
        except ImportError:
            _log("ERROR: Neither vLLM nor transformers installed. Install one:")
            _log("  pip install transformers accelerate")
            _log("  pip install vllm  (for --fast mode)")
            return TestResult(mode="model", gpu_name=gpu_info["name"], gpu_index=gpu_index, model=args.model)

    # Baseline (--compare)
    baseline_tps = 0.0
    if args.compare:
        _phase("Baseline (no monitoring)")
        if use_vllm:
            tokens, elapsed = _run_inference_vllm(args.model, prompts, args.max_tokens, gpu_index)
        else:
            tokens, elapsed = _run_inference_transformers(args.model, prompts, args.max_tokens, gpu_index)
        baseline_tps = tokens / elapsed if elapsed > 0 else 0
        _log(f"  Throughput:  {baseline_tps:.1f} tok/s")
        _log(f"  Time:        {elapsed:.1f}s ({tokens} tokens)")
        _log(f"  Power:       ???")
        _log(f"  Cost:        ???")
        _log("  This is what running blind looks like.\n")
        time.sleep(3)

    # Monitored run
    _phase("Monitored Run")
    monitor = GPUMonitor(gpu_index, 0.1)
    monitor.start()
    if use_vllm:
        tokens, elapsed = _run_inference_vllm(args.model, prompts, args.max_tokens, gpu_index)
    else:
        tokens, elapsed = _run_inference_transformers(args.model, prompts, args.max_tokens, gpu_index)
    samples = monitor.stop()

    result = _build_result("model", samples, gpu_info, gpu_index, idle_power, elapsed, tokens)
    result.model = args.model
    result.baseline_tokens_per_sec = baseline_tps
    return result


# ═══════════════════════════════════════════════════════════════════
# Mode 3: Endpoint (--endpoint URL)
# ═══════════════════════════════════════════════════════════════════

def _send_prompt(url: str, prompt: str, max_tokens: int, model: str = "default") -> int:
    """Send one prompt to an OpenAI-compatible endpoint. Returns token count."""
    # Try chat completions first, fall back to completions
    for path_suffix, make_body in [
        ("/v1/chat/completions", lambda: {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.7}),
        ("/v1/completions", lambda: {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7}),
    ]:
        try:
            target = url.rstrip("/") + path_suffix if not url.rstrip("/").endswith(path_suffix.lstrip("/")) else url
            payload = json.dumps(make_body()).encode()
            req = urllib.request.Request(target, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
                usage = body.get("usage", {})
                tokens = usage.get("completion_tokens", 0)
                if tokens == 0 and "choices" in body:
                    choice = body["choices"][0]
                    text = choice.get("text", "") or choice.get("message", {}).get("content", "")
                    tokens = len(text.split()) * 1.3
                return int(tokens)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
    return 0


def _run_endpoint(args, gpu_index: int, idle_power: float, gpu_info: dict) -> TestResult:
    prompts = _load_prompts(args)
    url = args.endpoint.rstrip("/")

    _phase("Load Testing Endpoint")
    _log(f"Target: {url}")
    _log(f"Sending {len(prompts)} prompts (max_tokens={args.max_tokens})...")

    monitor = GPUMonitor(gpu_index, 0.1)
    monitor.start()

    total_tokens = 0
    errors = 0
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        try:
            tokens = _send_prompt(url, prompt, args.max_tokens)
            total_tokens += tokens
            if (i + 1) % 5 == 0:
                elapsed_so_far = time.time() - t0
                rate = total_tokens / elapsed_so_far if elapsed_so_far > 0 else 0
                _log(f"  [{i+1}/{len(prompts)}] {total_tokens} tokens, {rate:.0f} tok/s")
        except Exception as e:
            errors += 1
            if errors <= 3:
                _log(f"  Request {i+1} failed: {e}")
            if errors == 3:
                _log(f"  (suppressing further errors)")

    elapsed = time.time() - t0
    samples = monitor.stop()

    if errors == len(prompts):
        _log(f"All {errors} requests failed. Check the endpoint URL and that the server is running.")

    _log(f"  Done: {total_tokens} tokens in {elapsed:.1f}s ({total_tokens / elapsed:.0f} tok/s)" if elapsed > 0 else "")

    result = _build_result("endpoint", samples, gpu_info, gpu_index, idle_power, elapsed, total_tokens)
    result.model = url
    return result


# ═══════════════════════════════════════════════════════════════════
# Result Builder
# ═══════════════════════════════════════════════════════════════════

def _build_result(
    mode: str, samples: list[PowerSample], gpu_info: dict,
    gpu_index: int, idle_power: float, duration_s: float,
    total_tokens: int = 0,
) -> TestResult:
    result = TestResult(mode=mode, gpu_name=gpu_info["name"], gpu_index=gpu_index)
    result.idle_power_w = idle_power
    result.duration_s = duration_s
    result.total_tokens = total_tokens
    result.vram_total_gb = gpu_info["vram_total_gb"]

    if total_tokens > 0 and duration_s > 0:
        result.tokens_per_sec = total_tokens / duration_s

    if samples:
        powers = [s.power_w for s in samples]
        result.avg_power_w = sum(powers) / len(powers)
        result.peak_power_w = max(powers)
        result.dynamic_power_w = max(0, result.avg_power_w - idle_power)
        result.avg_util_pct = sum(s.gpu_util_pct for s in samples) / len(samples)
        result.avg_mem_util_pct = sum(s.mem_util_pct for s in samples) / len(samples)
        temps = [s.temperature_c for s in samples]
        result.avg_temp_c = sum(temps) / len(temps)
        result.peak_temp_c = max(temps)
        result.vram_used_gb = max(s.memory_used_mb for s in samples) / 1000.0

    if total_tokens > 0 and result.avg_power_w > 0 and duration_s > 0:
        total_energy = result.avg_power_w * duration_s
        dynamic_energy = result.dynamic_power_w * duration_s
        result.joules_per_token = total_energy / total_tokens
        result.joules_per_token_dynamic = dynamic_energy / total_tokens
        result.kwh_per_1m_tokens = result.avg_power_w / (result.tokens_per_sec * 1000)
        gpu_hrs_per_1m = 1_000_000 / result.tokens_per_sec / 3600
        result.cost_per_1m_tokens_usd = gpu_hrs_per_1m * _get_cloud_rate(gpu_info["name"])
        result.co2e_per_1m_tokens_g = result.kwh_per_1m_tokens * 394.0

    recs, regime = _get_recommendations(samples, gpu_info["name"], gpu_index, duration_s)
    result.recommendations = recs
    result.workload_regime = regime
    return result


# ═══════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════

def _log(msg: str):
    if _RICH:
        console.print(f"  {msg}")
    else:
        print(f"  {msg}", flush=True)


def _phase(title: str):
    if _RICH:
        console.print(f"\n[bold cyan]{title}[/]")
    else:
        print(f"\n{title}")


def _print_report(result: TestResult, power_cap_result: Optional[dict] = None):
    model_str = result.model or "live workload"
    if _RICH:
        console.print()
        console.print(f"[bold]═══════════════════════════════════════════════════════════════[/]")
        console.print(f"[bold]  NemulAI — GPU Energy Report[/]")
        console.print(f"  GPU: [cyan]{result.gpu_name}[/] | {model_str}")
        console.print(f"[bold]═══════════════════════════════════════════════════════════════[/]")
    else:
        print()
        print("═" * 63)
        print("  NemulAI — GPU Energy Report")
        print(f"  GPU: {result.gpu_name} | {model_str}")
        print("═" * 63)

    if result.baseline_tokens_per_sec > 0:
        _log("")
        _log(f"[dim]Baseline (blind):[/]  {result.baseline_tokens_per_sec:.1f} tok/s | Power: ??? | Cost: ???" if _RICH else
             f"Baseline (blind):  {result.baseline_tokens_per_sec:.1f} tok/s | Power: ??? | Cost: ???")

    _log("")
    if result.tokens_per_sec > 0:
        delta = ""
        if result.baseline_tokens_per_sec > 0:
            pct = ((result.tokens_per_sec / result.baseline_tokens_per_sec) - 1) * 100
            delta = f" ({pct:+.1f}%)"
        _log(f"Throughput:   {result.tokens_per_sec:.1f} tok/s{delta}")

    _log(f"Avg Power:    {result.avg_power_w:.0f}W ({result.dynamic_power_w:.0f}W dynamic, {result.idle_power_w:.0f}W idle)")
    _log(f"Peak Power:   {result.peak_power_w:.0f}W")
    _log(f"GPU Util:     {result.avg_util_pct:.0f}% avg")
    _log(f"Temperature:  {result.avg_temp_c:.0f}°C avg, {result.peak_temp_c:.0f}°C peak")
    _log(f"VRAM:         {result.vram_used_gb:.1f} / {result.vram_total_gb:.1f} GB")

    if result.tokens_per_sec > 0:
        _log("")
        _log(f"J/token:      {result.joules_per_token:.3f} ({result.joules_per_token_dynamic:.3f} dynamic)")
        _log(f"kWh/1M tok:   {result.kwh_per_1m_tokens:.4f}")
        _log(f"$/1M tokens:  ${result.cost_per_1m_tokens_usd:.4f}")
        _log(f"CO2/1M tok:   {result.co2e_per_1m_tokens_g:.1f} gCO2e")
    else:
        _log("")
        energy_j = result.avg_power_w * result.duration_s
        _log(f"Energy:       {energy_j:,.0f} J total ({result.dynamic_power_w * result.duration_s:,.0f} J dynamic)")
        cloud_rate = _get_cloud_rate(result.gpu_name)
        _log(f"Cost Rate:    ${result.avg_power_w / 1000 * 0.12:.3f}/hr energy | ${cloud_rate:.2f}/hr cloud")

    if result.recommendations:
        _log("")
        regime_str = f" ({result.workload_regime})" if result.workload_regime != "unknown" else ""
        if _RICH:
            console.print(f"  [bold]Optimization{regime_str}[/]")
        else:
            _log(f"Optimization{regime_str}")
        for rec in result.recommendations[:5]:
            _log(f"  [{rec['priority']}] {rec['description']}")
            if rec.get("action"):
                _log(f"       {rec['action']}")

    if power_cap_result:
        _log("")
        _log(f"Power Cap Test: {result.avg_power_w:.0f}W → {power_cap_result['cap_watts']}W")
        _log(f"  Capped avg: {power_cap_result['capped_avg_w']}W | Savings: {power_cap_result['savings_pct']}%")

    if _RICH:
        console.print(f"\n[bold]═══════════════════════════════════════════════════════════════[/]")
    else:
        print("\n" + "═" * 63)


def _save_json(result: TestResult, path: str, power_cap: Optional[dict] = None):
    data = {
        "nemulai_test": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": result.mode,
        "gpu": result.gpu_name,
        "model": result.model,
        "throughput": {"tokens": result.total_tokens, "tok_per_sec": round(result.tokens_per_sec, 1), "duration_s": round(result.duration_s, 1)},
        "power": {"avg_w": round(result.avg_power_w, 1), "peak_w": round(result.peak_power_w, 1), "dynamic_w": round(result.dynamic_power_w, 1), "idle_w": round(result.idle_power_w, 1)},
        "efficiency": {"j_per_token": round(result.joules_per_token, 4), "j_per_token_dynamic": round(result.joules_per_token_dynamic, 4), "kwh_per_1m": round(result.kwh_per_1m_tokens, 6), "cost_usd_per_1m": round(result.cost_per_1m_tokens_usd, 6), "co2e_g_per_1m": round(result.co2e_per_1m_tokens_g, 4)},
        "gpu_metrics": {"util_pct": round(result.avg_util_pct, 1), "mem_util_pct": round(result.avg_mem_util_pct, 1), "temp_avg_c": round(result.avg_temp_c, 1), "temp_peak_c": round(result.peak_temp_c, 1), "vram_used_gb": round(result.vram_used_gb, 1), "vram_total_gb": round(result.vram_total_gb, 1)},
        "recommendations": result.recommendations,
        "workload_regime": result.workload_regime,
        "power_cap_test": power_cap,
    }
    if result.baseline_tokens_per_sec > 0:
        data["baseline_tok_per_sec"] = round(result.baseline_tokens_per_sec, 1)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    _log(f"Results saved to {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nemulai test",
        description="One-command GPU energy testing. Monitor, benchmark, optimize.",
    )
    p.add_argument("--model", default=None, help="HuggingFace model ID to load and benchmark")
    p.add_argument("--endpoint", default=None, help="URL of a running OpenAI-compatible inference server")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    p.add_argument("--duration", type=int, default=60, help="Monitor duration in seconds (no-args mode, default: 60)")
    p.add_argument("--prompts", type=int, default=25, help="Number of built-in prompts (default: 25)")
    p.add_argument("--prompts-file", default=None, help="Custom prompts file (one per line, or JSONL)")
    p.add_argument("--max-tokens", type=int, default=512, help="Max output tokens per prompt (default: 512)")
    p.add_argument("--compare", action="store_true", help="Show blind baseline before monitored run (demo mode)")
    p.add_argument("--fast", action="store_true", help="Use vLLM instead of transformers (if installed)")
    p.add_argument("--power-cap", type=int, default=None, help="Test a specific power cap (watts)")
    p.add_argument("--team", default=None, help="Team tag for attribution")
    p.add_argument("--output", default=None, help="Save results to JSON file")
    p.add_argument("--quiet", action="store_true", help="Minimal output")
    return p


def run_test(args: argparse.Namespace) -> int:
    if not _NVML:
        print("ERROR: pynvml not installed. Run: pip install nvidia-ml-py")
        return 1

    try:
        pynvml.nvmlInit()
    except Exception as e:
        print(f"ERROR: NVML init failed — {e}")
        print("Run 'nemulai doctor' to diagnose.")
        return 1

    # Environment
    _phase("Environment")
    gpu_info = _get_gpu_info(args.gpu)
    _log(f"GPU: {gpu_info['name']} | {gpu_info['vram_total_gb']} GB VRAM | Idle: {gpu_info['idle_power_w']}W | Driver: {gpu_info['driver']}")

    if args.team:
        os.environ["ALUMINATAI_TEAM"] = args.team

    # Idle baseline
    idle_power = _measure_idle(args.gpu, 10.0)
    _log(f"Idle baseline: {idle_power:.1f}W (10s)")

    # Dispatch
    if args.model:
        _phase("Model Test")
        result = _run_model(args, args.gpu, idle_power, gpu_info)
    elif args.endpoint:
        result = _run_endpoint(args, args.gpu, idle_power, gpu_info)
    else:
        result = _run_monitor(args.gpu, args.duration, idle_power, gpu_info)

    if result is None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return 0

    # Power cap test
    cap_result = None
    if args.power_cap and result.avg_power_w > 0:
        _phase("Power Cap Test")
        _log(f"Testing {result.avg_power_w:.0f}W → {args.power_cap}W...")
        cap_result = _try_power_cap_ab(args.gpu, int(result.avg_power_w), args.power_cap)
        if cap_result is None:
            _log("Power cap test requires elevated privileges — skipped.")
            _log("Run with sudo for A/B power cap testing.")
    elif not args.power_cap and result.recommendations:
        cap_recs = [r for r in result.recommendations if "power cap" in r.get("description", "").lower() or "power cap" in r.get("action", "").lower()]
        if cap_recs:
            _log("")
            _log("Tip: run with --power-cap WATTS to test the recommended power cap.")

    # Report
    _print_report(result, cap_result)

    # Save
    if args.output:
        _save_json(result, args.output, cap_result)

    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return 0
