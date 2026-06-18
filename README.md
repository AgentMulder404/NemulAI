<div align="center">

# ⚡ NemulAI Agent

**Know exactly what every GPU job costs — and which ones are wasting money.**

The open-source GPU cost-attribution agent for [NemulAI](https://nemulai.com).
Read-only by default. One `pip install`. Works on NVIDIA, AMD, Intel, and Apple Silicon.

[![PyPI version](https://img.shields.io/pypi/v/nemulai.svg)](https://pypi.org/project/nemulai/)
[![Python](https://img.shields.io/pypi/pyversions/nemulai.svg)](https://pypi.org/project/nemulai/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/AgentMulder404/NemulAI?style=social)](https://github.com/AgentMulder404/NemulAI/stargazers)

[Website](https://nemulai.com) · [Docs](https://nemulai.com/docs/agent) · [Dashboard](https://nemulai.com/dashboard) · [Report a bug](https://github.com/AgentMulder404/NemulAI/issues)

</div>

---

## The problem

Your A100 burns ~$40/hour. **Do you know which training run was worth it?**

Most AI teams can't answer that. `nvidia-smi` shows real-time watts. Your cloud provider shows a monthly bill. Neither tells you which *specific job* — or team, or model — your money went to, or which GPUs sat idle while the meter ran.

The waste compounds quietly:

- Training jobs left running overnight, long after convergence stalled
- GPUs sitting at 3% utilization, drawing near-full power
- No per-team attribution → no accountability → no fix
- Finance asks "can we cut GPU spend?" and nobody has the data to answer

**NemulAI closes that gap.** A lightweight agent runs on your GPU machines, attributes energy to individual jobs in real time, prices it at *your* GPU rate, and flags idle waste — before the invoice surprises you.

## What it does

- **Per-job cost attribution** — energy and dollars per training/inference run, not just per machine
- **Idle & underutilization detection** — flags GPUs running idle (<1%) or underused (<10%), with the dollars wasted
- **Team / model / customer chargeback** — split spend with one environment variable (`NEMULAI_TEAM`, `NEMULAI_MODEL`)
- **Real-time power monitoring** — samples NVML every 5 seconds via `nvidia-ml-py`
- **Multi-vendor** — NVIDIA, AMD (ROCm), Intel Gaudi, Intel Arc, Apple Silicon, and CPU-only (RAPL)
- **WAL-backed reliability** — metrics buffer locally during API outages and replay on reconnect
- **Scheduler-aware** — Kubernetes, Slurm, Run:ai, or manual tagging
- **MLflow & W&B callbacks** — tag experiment runs with their energy cost automatically
- **Prometheus endpoint** — expose metrics to your existing Grafana stack
- **Read-only by default** — collects telemetry only; never touches your workloads unless you opt in (see [Operating modes](#operating-modes))
- **Near-zero overhead** — ~0% CPU, ~50 MB RAM, single `pip install`

## Quick start

```bash
pip install nemulai
```

```bash
export NEMULAI_API_KEY=alum_your_key_here
nemulai
```

That's it — the agent auto-detects your hardware and starts streaming cost data to your dashboard. Get a key at [nemulai.com/dashboard](https://nemulai.com/dashboard) (free tier, no credit card).

### Docker

```bash
docker run --rm --runtime=nvidia --pid=host \
  -e NEMULAI_API_KEY=alum_your_key_here \
  ghcr.io/agentmulder404/nemulai-agent:latest
```

## Supported hardware

| Backend | GPUs | Primary SDK | Fallback |
|---------|------|-------------|----------|
| **NVIDIA** | A100, H100, H200, L40S, RTX 4090, T4, V100, … | `nvidia-ml-py` (NVML) | — |
| **AMD** | MI300X/A, MI325X, MI250X, MI210, MI100, … | `amdsmi` | `rocm-smi` |
| **Intel Gaudi** | Gaudi, Gaudi2, Gaudi3 | `pyhlml` | `hl-smi` |
| **Intel Arc** | A770/750/580, B580, Flex, Max | `xpu-smi` | hwmon + `intel_gpu_top` |
| **Apple Silicon** | M1–M5, Pro/Max/Ultra | `powermetrics` | `ioreg` |
| **CPU-only** | Any x86 (Intel/AMD) | RAPL sysfs | — |

Detection cascade runs automatically at startup: NVIDIA → AMD → Gaudi → Intel Arc → Apple Silicon → RAPL. No configuration required.

## Configuration

All settings are environment variables — no config files required.

| Variable | Default | Description |
|----------|---------|-------------|
| `NEMULAI_API_KEY` | *(required)* | Your API key from the dashboard |
| `NEMULAI_API_ENDPOINT` | `https://nemulai.com/api/metrics/ingest` | Ingest endpoint (change to self-host) |
| `NEMULAI_TEAM` | *(none)* | Team tag for chargeback attribution |
| `NEMULAI_MODEL` | *(none)* | Model tag for per-experiment tracking |
| `SAMPLE_INTERVAL` | `5.0` | Seconds between NVML samples |
| `UPLOAD_INTERVAL` | `60` | Seconds between metric flushes |
| `METRICS_PORT` | `9100` | Prometheus scrape port (`0` = disabled) |
| `OFFLINE_MODE` | `0` | `1` = WAL only, no HTTP (air-gapped clusters) |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

See [`deploy/nemulai-agent.env.example`](deploy/nemulai-agent.env.example) for the full reference.

## Job attribution

Tag workloads at launch for per-job cost breakdown:

```bash
NEMULAI_TEAM=nlp-team \
NEMULAI_MODEL=llama3-finetune \
NEMULAI_API_KEY=alum_... \
python train.py
```

Or wire it into your experiment tracker:

```python
# MLflow
from nemulai.integrations.mlflow_callback import NemulMLflowCallback
with mlflow.start_run():
    trainer.add_callback(NemulMLflowCallback())

# Weights & Biases
from nemulai.integrations.wandb_callback import NemulWandbCallback
wandb.init(project="my-project")
trainer.add_callback(NemulWandbCallback())
```

On Kubernetes and Slurm, job and team metadata is detected automatically from scheduler labels and environment.

## Operating modes

The agent ships one binary with three modes. Higher tiers are **strictly opt-in** — the default is read-only.

| Mode | Default? | What it does |
|------|----------|--------------|
| **Monitor** | ✅ Yes | Read-only metrics, cost attribution, waste detection, Prometheus |
| **Advisor** | Opt-in | Surfaces recommendations ("GPU 3 is 40% idle — cap to 200 W?") with one-click apply **and automatic rollback** |
| **Swarm** | Opt-in | Fleet-wide power capping, thermal balancing, carbon-aware scheduling |

```bash
# Monitor (default — no extra config)
nemulai

# Advisor — uploads recommendations, polls for approved commands only
AUTO_TUNE_ENABLED=1 COMMAND_POLL_ENABLED=1 nemulai
```

Any optimization action opens an observation window and **rolls back automatically** if throughput drops. You stay in control.

## Architecture

```
┌──────────────────────── GPU machine ────────────────────────┐
│  NVML / vendor SDK ──▶ Sampler (5s) ──▶ Attributor ──▶ WAL   │
│                                         (job/team)   buffer  │
└───────────────────────────────────────────────────────┬─────┘
                                                          │ HTTPS
                                                          ▼
                                          nemulai.com /api/metrics/ingest
                                                          │
                                                          ▼
                                   Dashboard: watts → $ per job,
                                   team chargeback, waste alerts
```

## Deployment

### systemd (recommended for production)

```bash
sudo cp deploy/nemulai-agent.service /etc/systemd/system/
echo "NEMULAI_API_KEY=alum_your_key_here" | sudo tee /etc/nemulai/agent.env
sudo chmod 600 /etc/nemulai/agent.env
sudo systemctl enable --now nemulai-agent
```

### Kubernetes DaemonSet

```bash
kubectl apply -f https://raw.githubusercontent.com/AgentMulder404/NemulAI/main/deploy/k8s/daemonset.yaml
```

### Slurm (prolog)

```bash
# /etc/slurm/prolog.d/nemulai.sh
source /etc/nemulai/agent.env
nemulai &
```

## Self-hosting

The agent is fully functional without the hosted dashboard. Point it at your own ingest endpoint:

```bash
NEMULAI_API_ENDPOINT=https://your-internal-api.com/api/metrics/ingest \
NEMULAI_API_KEY=your_key \
nemulai
```

Or run fully offline (air-gapped) with `OFFLINE_MODE=1` and scrape the local Prometheus endpoint.

## Why open source?

GPU cost visibility should be a solved problem, not a proprietary feature gate. The monitoring space is full of tools that show you what's *happening* (`nvidia-smi`, Grafana) or what *happened* (cloud billing). NemulAI is the missing link: **what each specific job cost, in real time, in dollars.**

Open-sourcing the agent means anyone can:

- Audit exactly what's collected — it's power draw and metadata you tag, nothing else
- Run a fully self-hosted stack against their own endpoint
- Contribute integrations for their scheduler, tracker, or cloud
- Build on the primitives for their own cost tooling

The hosted dashboard at [nemulai.com](https://nemulai.com) sustains the project. **The agent that collects your data will always be free and open.**

## Contributing

Contributions are welcome — fork → branch → PR against `main`. Good first issues: scheduler integrations, MLflow/W&B/OTEL hooks, packaging, docs.

By contributing, you agree your code is licensed under Apache-2.0 and credited in [`NOTICE`](NOTICE).

## Security

Found a vulnerability? Please **don't** open a public issue — see [`SECURITY.md`](SECURITY.md) for responsible disclosure.

## Citation

```bibtex
@software{nemulai2026,
  title   = {NemulAI: Per-Job GPU Cost Attribution and Waste Detection},
  author  = {NemulAI},
  year    = {2026},
  url     = {https://github.com/AgentMulder404/NemulAI},
  version = {0.4.0}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable format.

## License

[Apache-2.0](LICENSE) — use it, fork it, build on it, sell products with it. Keep the copyright notice, don't call your fork "NemulAI", and don't claim you wrote it. See [`NOTICE`](NOTICE) for trademark and attribution terms.

<div align="center">

Built for AI teams who want to know where their GPU money goes.
**Star ⭐ if this saves you money.**

</div>
