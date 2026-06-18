#!/usr/bin/env bash
# NemulAI — H100 SXM Model Benchmark Environment Setup
# Run this on a fresh RunPod H100 SXM pod (or any H100 SXM instance).
#
# Usage:
#   chmod +x h100_setup.sh && ./h100_setup.sh

set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "  NemulAI — H100 SXM Benchmark Environment Setup"
echo "═══════════════════════════════════════════════════════════════"

# ── System info ──────────────────────────────────────────────────
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo "CUDA version:"
nvcc --version 2>/dev/null | tail -1 || echo "nvcc not found (will use PyTorch bundled CUDA)"
echo ""

# ── Python environment ───────────────────────────────────────────
echo "Setting up Python environment..."

python3 -m pip install --upgrade pip

# Core ML stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# vLLM for high-performance inference on H100
pip install vllm

# NemulAI agent + dependencies
pip install nvidia-ml-py rich requests python-dotenv numpy

# HuggingFace for model downloads
pip install huggingface_hub transformers accelerate

# ── Clone NemulAI agent ──────────────────────────────────────
AGENT_DIR="/workspace/nemulai-agent"
if [ ! -d "$AGENT_DIR" ]; then
    echo "Cloning NemulAI agent..."
    git clone https://github.com/AgentMulder404/NemulAI.git "$AGENT_DIR"
fi

cd "$AGENT_DIR/agent"
pip install -e ".[all]"

# ── Pre-download models (parallel) ──────────────────────────────
echo ""
echo "Pre-downloading models (this takes a while on first run)..."

MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "teknium/OpenHermes-2.5-Mistral-7B"
    "NousResearch/Hermes-3-Llama-3.1-8B"
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
)

for model in "${MODELS[@]}"; do
    echo "  Downloading: $model"
    huggingface-cli download "$model" --quiet &
done

echo "  Waiting for all downloads to complete..."
wait
echo "  All models downloaded."

# ── Verify GPU is healthy ────────────────────────────────────────
echo ""
echo "Running quick GPU health check..."
python3 -c "
import torch
import pynvml

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
name = pynvml.nvmlDeviceGetName(h)
if isinstance(name, bytes): name = name.decode()
mem = pynvml.nvmlDeviceGetMemoryInfo(h)
power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
print(f'  GPU: {name}')
print(f'  Memory: {mem.total / 1e9:.0f} GB total, {mem.free / 1e9:.0f} GB free')
print(f'  Power: {power:.1f} W (idle)')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  PyTorch CUDA: {torch.version.cuda}')

# Quick matmul test
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
torch.cuda.synchronize()
import time
t0 = time.monotonic()
for _ in range(100):
    _ = a @ b
torch.cuda.synchronize()
elapsed = time.monotonic() - t0
tflops = (100 * 2 * 4096**3) / elapsed / 1e12
print(f'  BF16 matmul: {tflops:.1f} TFLOPS')
pynvml.nvmlShutdown()
print()
print('GPU health check passed.')
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete! Run the benchmark with:"
echo ""
echo "    cd $AGENT_DIR/agent/tests"
echo "    python3 h100_model_benchmark.py"
echo ""
echo "  Options:"
echo "    --models all              # run all models (default)"
echo "    --models qwen             # only Qwen family"
echo "    --duration 60             # seconds per model (default: 60)"
echo "    --prompts 50              # prompts per model (default: 50)"
echo "    --upload                  # submit to NemulAI leaderboard"
echo "    --output results.json     # custom output path"
echo "═══════════════════════════════════════════════════════════════"
