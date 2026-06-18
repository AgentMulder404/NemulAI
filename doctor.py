"""
nemulai doctor — pre-flight environment check.

Validates GPU access, NVML, disk space, VRAM, Python deps, and CUDA
version before wasting time on a broken setup.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

try:
    from rich.console import Console
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None


def _check(label: str, passed: bool, detail: str, warn: bool = False):
    if _RICH:
        icon = "[green]✓[/]" if passed else ("[yellow]─[/]" if warn else "[red]✗[/]")
        console.print(f"  {label + ':':<20s} {detail:<36s} {icon}")
    else:
        icon = "OK" if passed else ("--" if warn else "FAIL")
        print(f"  {label + ':':<20s} {detail:<36s} {icon}")


def run_doctor(args: argparse.Namespace) -> int:
    if _RICH:
        console.print("\n[bold]NemulAI Pre-Flight Check[/]")
        console.print("─" * 60)
    else:
        print("\nNemulAI Pre-Flight Check")
        print("─" * 60)

    all_ok = True

    # Python
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    _check("Python", True, py_ver)

    # pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        vram_gb = mem.total / 1e9
        free_gb = mem.free / 1e9
        _check("GPU", True, f"{name} ({count} detected)")
        _check("VRAM", True, f"{vram_gb:.0f} GB total, {free_gb:.0f} GB free")
        _check("Idle power", True, f"{power:.0f}W")
        _check("Driver", True, driver)
        _check("NVML", True, f"pynvml {pynvml.__version__}" if hasattr(pynvml, '__version__') else "pynvml installed")
        pynvml.nvmlShutdown()
    except ImportError:
        _check("NVML", False, "pip install nvidia-ml-py")
        all_ok = False
    except Exception as e:
        _check("GPU", False, str(e)[:40])
        all_ok = False

    # PyTorch + CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        cuda_ver = torch.version.cuda or "n/a"
        _check("PyTorch", True, f"{torch.__version__} (CUDA {cuda_ver})")
        if not cuda_ok:
            _check("CUDA", False, "torch.cuda not available")
            all_ok = False
    except ImportError:
        _check("PyTorch", False, "not installed", warn=True)

    # vLLM (optional)
    try:
        import vllm
        _check("vLLM", True, f"{vllm.__version__} (--fast mode)")
    except ImportError:
        _check("vLLM", True, "not installed (optional)", warn=True)

    # transformers (optional)
    try:
        import transformers
        _check("transformers", True, transformers.__version__)
    except ImportError:
        _check("transformers", True, "not installed (optional)", warn=True)

    # Disk space
    disk = shutil.disk_usage(os.getcwd())
    free_disk_gb = disk.free / 1e9
    disk_ok = free_disk_gb > 5.0
    _check("Disk", disk_ok, f"{free_disk_gb:.1f} GB free")
    if not disk_ok:
        all_ok = False

    # Summary
    if _RICH:
        console.print("─" * 60)
        if all_ok:
            console.print("  [bold green]Ready to go.[/]\n")
        else:
            console.print("  [bold red]Issues found — fix the items above before running nemulai test.[/]\n")
    else:
        print("─" * 60)
        print("  Ready to go." if all_ok else "  Issues found — fix the items above.")
        print()

    return 0 if all_ok else 1


def make_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="nemulai doctor",
        description="Pre-flight environment check for NemulAI.",
    )
