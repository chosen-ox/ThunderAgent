#!/usr/bin/env bash
#
# Sequential SWE-bench Lite baseline experiments (no ThunderAgent): 32 tasks then 100 tasks.
# vLLM listens on port 9000 so existing config works unchanged.
# Each experiment restarts vLLM for clean metrics.
#
# Usage:
#   bash scripts/run_sequential_experiments_baseline.sh
#
# Prerequisites:
#   conda activate thunder
#   module load CUDA/12.6.0
#   module load apptainer
#   export APPTAINER_SANDBOX_DIR="/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox"
#   export APPTAINER_CACHEDIR="/projects/pearl/Storage_Jiakun/apptainer_cache"

set -euo pipefail

cd /home/jiakunfan/ThunderAgent

# ── Config ──────────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3-8B"
CONFIG="examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen3_8b.yaml"
WORKERS=32
# vLLM listens on 9000 (same port ThunderAgent would use) so config works unchanged
VLLM_PORT=9000

DIR_32="results/qwen3_8b_lite_32_0415_vllm"
DIR_100="results/qwen3_8b_lite_100_0415_vllm"

EXPERIMENTS=(
  "32:${DIR_32}:0:32"
  "100:${DIR_100}:0:100"
)

# ── Helper functions ────────────────────────────────────────────────────────

wait_for() {
  local url=$1 name=$2 max_wait=${3:-300}
  echo "[WAIT] Waiting for ${name} at ${url} ..."
  local elapsed=0
  while ! curl -sf "${url}" > /dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed >= max_wait )); then
      echo "[ERROR] ${name} did not become ready within ${max_wait}s"
      return 1
    fi
  done
  echo "[OK] ${name} is ready (${elapsed}s)"
}

kill_services() {
  echo "[KILL] Stopping ThunderAgent (in case) ..."
  pkill -f "ThunderAgent" 2>/dev/null || true
  sleep 2

  echo "[KILL] Stopping vLLM ..."
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3

  # Make sure they're really gone
  if pgrep -f "vllm serve" > /dev/null 2>&1; then
    echo "[WARN] vLLM still running, force killing ..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 2
  fi
  echo "[OK] Services stopped."
}

start_vllm() {
  local log_file=$1
  echo "[START] Starting vLLM (${MODEL}) on port ${VLLM_PORT} ..."
  vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --enable-prompt-tokens-details \
    2>&1 | tee "${log_file}" &
  VLLM_PID=$!
  wait_for "http://localhost:${VLLM_PORT}/health" "vLLM" 600
}

dump_metrics() {
  local output_dir=$1
  echo "[DUMP] Saving metrics to ${output_dir} ..."
  curl -sf "http://localhost:${VLLM_PORT}/metrics" 2>/dev/null \
    > "${output_dir}/vllm_metrics.txt" 2>/dev/null || echo "[WARN] Failed to dump vLLM metrics"
  echo "[OK] Metrics dumped."
}

run_experiment() {
  local n_tasks=$1 output_dir=$2 slice=$3
  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo "  BASELINE EXPERIMENT: ${n_tasks} tasks → ${output_dir} (slice ${slice})"
  echo "════════════════════════════════════════════════════════════════════"
  echo ""

  mkdir -p "${output_dir}"

  # Clean start
  kill_services

  # Start vLLM only (no ThunderAgent)
  start_vllm "${output_dir}/vllm.log"

  echo ""
  echo "[RUN] Starting mini-extra swebench (${n_tasks} tasks, ${WORKERS} workers, slice ${slice}) ..."
  echo ""

  mini-extra swebench \
    --subset lite \
    --split test \
    --slice "${slice}" \
    --workers "${WORKERS}" \
    -o "${output_dir}" \
    --config "${CONFIG}"

  echo ""
  echo "[DONE] mini-extra finished for ${n_tasks} tasks."

  # Dump metrics
  dump_metrics "${output_dir}"

  echo ""
  echo "[COMPLETE] Baseline experiment ${n_tasks} tasks done. Results in ${output_dir}"
  echo ""
}

# ── Main ────────────────────────────────────────────────────────────────────

echo "=============================================="
echo " Sequential SWE-bench Lite BASELINE Experiments (no TA)"
echo " $(date)"
echo "=============================================="

# Run experiments sequentially
for exp in "${EXPERIMENTS[@]}"; do
  IFS=':' read -r n_tasks output_dir slice <<< "${exp}"
  run_experiment "${n_tasks}" "${output_dir}" "${slice}"
done

# Final cleanup
kill_services

echo ""
echo "=============================================="
echo " ALL BASELINE EXPERIMENTS COMPLETE"
echo " $(date)"
echo "=============================================="
echo ""
echo "Results:"
echo "  32 tasks:  ${DIR_32}/"
echo "  100 tasks: ${DIR_100}/"
