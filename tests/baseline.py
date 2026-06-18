import os, time, gc, torch
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
os.environ.setdefault("XDG_CONFIG_HOME", "/workspace/.config")
from vllm import LLM, SamplingParams
print("Loading Qwen2.5-0.5B BASELINE...")
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16", max_model_len=4096)
prompts = ["Explain how GPU power relates to compute utilization.", "Write a Python function that monitors GPU temp using pynvml.", "Compare energy efficiency of H100 vs A100 for LLM inference.", "What are five ways to reduce carbon footprint of training LLMs?", "Design a monitoring alert system for GPU temperature and power.", "Write a REST API endpoint that accepts GPU telemetry data.", "Explain the roofline model in high-performance computing.", "A company runs 100 req/s on 4 H100s. Calculate monthly energy cost.", "What is the difference between FP16 BF16 and FP8 precision?", "Write a Dockerfile for a vLLM inference service with CUDA 12.4."]
tok = llm.get_tokenizer()
formatted = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
t0 = time.time()
outputs = llm.generate(formatted, params)
elapsed = time.time() - t0
total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
print()
print("=" * 60)
print("  BASELINE (no NemulAI)")
print("=" * 60)
print(f"  Model:       Qwen2.5-0.5B-Instruct")
print(f"  Throughput:  {total_out / elapsed:.1f} tok/s")
print(f"  Time:        {elapsed:.1f}s")
print(f"  Tokens out:  {total_out}")
print(f"  Power:       ??? (no monitoring)")
print(f"  J/token:     ??? (no monitoring)")
print(f"  Cost/1M tok: ??? (no monitoring)")
print("=" * 60)
del llm
torch.cuda.empty_cache()
gc.collect()
