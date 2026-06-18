# Self-Learning Optimization Agent — Phase 1: Experience Logger

Right now, NemulAI's agent makes optimization recommendations using **hardcoded rules** — "if utilization < 40%, flag it" or "if power exceeds 80% TDP, suggest a cap." Those rules work, but they never get smarter. The recommendation table already tracks whether users accept or reject suggestions, and what the actual savings were — but that data goes nowhere. It's a dead end.

Phase 1 closes that loop by logging every **(context, action, outcome)** tuple into a training corpus.

---

## The Three Data Structures

**WorkloadContext** — a snapshot of what's happening on the GPU right when a recommendation fires:
- GPU type, workload class (e.g. `llm-inference-bf16`), utilization, memory pressure, power draw, temperature, power ratio

**ActionTaken** — what the heuristic engine recommended:
- Action type (power cap, precision change, etc.), the recommended value vs current value, estimated savings

**ActionOutcome** — what actually happened 5 minutes later:
- Energy before vs after, throughput before vs after, whether the user accepted it, and the actual savings percentage

These three get bundled into an **ExperienceTuple** with a computed **reward** (0-1 scale: energy saved minus throughput regression penalty).

---

## How It Works in the Agent Loop

The agent already runs a 20-stage sampling pipeline every 5 seconds. We hooked into three points:

1. **After the auto-tuner fires** (~every 5 min) — when it recommends a power cap, the logger captures the current GPU context and the action taken
2. **Every outcome window** (default 300s) — resolves pending tuples by comparing current energy/throughput to the pre-action snapshot, computes the reward
3. **During upload flush** — ships completed tuples to the fleet aggregation API

Everything is **off by default**. Set `ALUMINATAI_LEARNER_ENABLED=1` to start collecting.

---

## Storage & Durability

Tuples are stored as **flock-protected JSONL** in `DATA_DIR/experience/experience.wal` — the exact same WAL pattern the metrics uploader uses. Survives crashes, handles concurrent writers (K8s rolling updates), and replays on restart.

For fleet-wide aggregation, completed tuples upload to a new `/api/agent/experience` endpoint backed by a new `experience_log` Supabase table with RLS.

---

## The Reward Function

Isolated in `reward.py` for easy tuning:

```
reward = energy_improvement x throughput_factor
```

- **Energy improvement**: `1 - (energy_after / energy_before)`, clamped to [0, 1]
- **Throughput factor**: multiplicative penalty if utilization dropped (weight = 0.3)

So a 50% energy reduction with no throughput loss = 0.5 reward. Same reduction with 50% throughput loss = 0.5 x 0.85 = 0.425.

---

## The Feature Encoder

`classify_workload()` turns raw model tags + precision + utilization patterns into human-readable classes like `llm-training-fp16` or `vision-inference-int8`. It recognizes ~20 model families (llama, mistral, qwen, stable-diffusion, whisper, resnet, etc.) and all common precision formats.

`encode_context()` produces flat numerical features bucketized to 5% increments — matching the efficiency curve builder's existing bucketing — ready for the bandit in Phase 2.

---

## The CLI

```
nemulai learn status        # corpus stats per GPU class, workload, action type
nemulai learn corpus-size   # progress bars showing tuples vs 10K target
nemulai learn export        # dump corpus as JSONL or CSV for offline analysis
```

---

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `ALUMINATAI_LEARNER_ENABLED` | `false` | Enable experience logging |
| `ALUMINATAI_LEARNER_OUTCOME_WINDOW` | `300` | Seconds to wait before measuring outcome (60-1800) |
| `ALUMINATAI_LEARNER_UPLOAD` | `false` | Upload completed tuples to fleet API |

---

## Files

| File | Purpose |
|------|---------|
| `agent/learner/__init__.py` | Package exports |
| `agent/learner/experience_logger.py` | Core (context, action, outcome) WAL logger |
| `agent/learner/feature_encoder.py` | Workload classification + context encoding |
| `agent/learner/reward.py` | Isolated [0,1] reward computation |
| `agent/learner/cli.py` | `nemulai learn` subcommand handler |
| `agent/tests/test_experience_logger.py` | 25 unit tests |
| `database/migrations/051_experience_log.sql` | Supabase table + RLS |
| `app/api/agent/experience/route.ts` | Fleet aggregation API endpoint |

---

## Phase 2: Contextual Bandit

The contextual bandit learns which power cap level works best for each workload context. It runs alongside the heuristic engine — never replacing it, only supplementing.

### How It Works

The bandit observes (context, action, reward) triples from Phase 1's experience corpus and learns a policy mapping workload contexts to optimal power cap levels.

**Action space** — 7 discrete power cap levels:
- `cap_40pct` through `cap_100pct` (fractions of TDP)
- Safety invariant: never below 40% TDP, never above 100% TDP

**Exploration** — epsilon-greedy (default e=0.1):
- 90% of the time: exploit the best-known action for this context
- 10% of the time: explore a random action to discover better policies

**Backends**:
- **Vowpal Wabbit** (`--cb_explore_adf`) — full contextual bandit with action-dependent features, online learning, and model persistence. Used when `pip install nemulai[learner]` installs VW.
- **Simple fallback** — pure-Python epsilon-greedy with incremental mean updates. Works without any ML dependencies. Less accurate but functional.

### Lifecycle

1. **Warm start**: `nemulai learn bandit-train` consumes the Phase 1 corpus
2. **Online learning**: Each time the agent applies a power cap and observes the outcome, the bandit updates its policy
3. **Checkpointing**: Model saved every 500 interactions (configurable)
4. **Offline evaluation**: `nemulai learn bandit-eval` uses doubly-robust estimation to evaluate the current policy against logged data before promoting

### Agent Integration

After the auto-tuner fires in the main loop, the bandit runs for each GPU:
1. Encodes the current workload context (utilization, memory, power, temperature)
2. Calls `bandit.suggest()` to get a power cap recommendation
3. If the suggestion differs meaningfully from the current cap, submits it as a `source="bandit"` recommendation
4. The recommendation goes through the normal approval workflow (unless `ALUMINATAI_BANDIT_AUTO_APPLY=1`)

### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `ALUMINATAI_BANDIT_ENABLED` | `false` | Enable the contextual bandit |
| `ALUMINATAI_BANDIT_EPSILON` | `0.1` | Exploration rate (0.01-0.5) |
| `ALUMINATAI_BANDIT_RETRAIN_EVERY` | `500` | Checkpoint interval (interactions) |
| `ALUMINATAI_BANDIT_AUTO_APPLY` | `false` | Auto-apply suggestions (skip approval) |
| `ALUMINATAI_BANDIT_MIN_CORPUS` | `1000` | Minimum tuples before activation |

### CLI

```
nemulai learn bandit-status   # backend, corpus size, model version, readiness
nemulai learn bandit-train    # warm-start on experience corpus
nemulai learn bandit-eval     # offline doubly-robust policy evaluation
```

### Files

| File | Purpose |
|------|---------|
| `agent/learner/bandit.py` | EnergyBandit + VW/simple backends |
| `agent/tests/test_bandit.py` | 15 unit tests |
| `database/migrations/052_bandit_models.sql` | Fleet model sharing table |
