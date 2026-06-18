#!/usr/bin/env bash
# NemulAI — H100 SXM Quick-Start Runner
#
# Copy this entire agent/tests/ directory to your H100 pod, then run:
#   chmod +x h100_run.sh && ./h100_run.sh
#
# Or run specific stages:
#   ./h100_run.sh setup          # install dependencies + download models
#   ./h100_run.sh benchmark      # raw model benchmark only
#   ./h100_run.sh customer       # full customer simulation
#   ./h100_run.sh quick          # fast test (3 models only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S)"

usage() {
    echo "Usage: $0 [setup|benchmark|customer|quick|all]"
    echo ""
    echo "  setup      — Install deps, download models"
    echo "  benchmark  — Raw model benchmark (all 12 models)"
    echo "  customer   — Full customer simulation (6 phases)"
    echo "  quick      — Quick test (3 models: Qwen2.5-0.5B, 7B, Mistral-7B)"
    echo "  all        — Setup + customer simulation (default)"
    echo ""
    echo "Results saved to: $RESULTS_DIR/"
}

run_setup() {
    echo "═══ Setting up environment... ═══"
    bash "$SCRIPT_DIR/h100_setup.sh"
}

run_benchmark() {
    mkdir -p "$RESULTS_DIR"
    echo ""
    echo "═══ Running model benchmark... ═══"
    cd "$AGENT_DIR"
    python3 "$SCRIPT_DIR/h100_model_benchmark.py" \
        --models "${1:-all}" \
        --prompts 50 \
        --max-tokens 512 \
        --cooldown 10 \
        --output "$RESULTS_DIR/benchmark_results.json" \
        --save-traces \
        2>&1 | tee "$RESULTS_DIR/benchmark.log"
    echo ""
    echo "Benchmark results: $RESULTS_DIR/benchmark_results.json"
}

run_customer() {
    mkdir -p "$RESULTS_DIR"
    echo ""
    echo "═══ Running customer simulation... ═══"
    cd "$AGENT_DIR"
    python3 "$SCRIPT_DIR/h100_customer_sim.py" \
        --models "${1:-all}" \
        --prompts 25 \
        --team "nemulai-test" \
        --cooldown 10 \
        --output "$RESULTS_DIR/customer_sim_results.json" \
        2>&1 | tee "$RESULTS_DIR/customer_sim.log"
    echo ""
    echo "Customer sim results: $RESULTS_DIR/customer_sim_results.json"
}

run_quick() {
    mkdir -p "$RESULTS_DIR"
    echo ""
    echo "═══ Quick test (3 models)... ═══"
    cd "$AGENT_DIR"
    python3 "$SCRIPT_DIR/h100_customer_sim.py" \
        --models quick \
        --prompts 10 \
        --team "nemulai-quick-test" \
        --cooldown 5 \
        --output "$RESULTS_DIR/quick_test_results.json" \
        2>&1 | tee "$RESULTS_DIR/quick_test.log"
    echo ""
    echo "Quick test results: $RESULTS_DIR/quick_test_results.json"
}

# ── Main ─────────────────────────────────────────────────────────

MODE="${1:-all}"

case "$MODE" in
    setup)
        run_setup
        ;;
    benchmark)
        run_benchmark "${2:-all}"
        ;;
    customer)
        run_customer "${2:-all}"
        ;;
    quick)
        run_quick
        ;;
    all)
        run_setup
        run_customer "all"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown mode: $MODE"
        usage
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done! Results in: $RESULTS_DIR/"
echo "═══════════════════════════════════════════════════════════════"
