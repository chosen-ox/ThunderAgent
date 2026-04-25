#!/usr/bin/env bash
#
# Sequential SWE-bench Lite experiments: 32 tasks then 100 tasks.
# Each experiment restarts vLLM and ThunderAgent for clean metrics.
#
# Usage:
#   bash scripts/run_sequential_experiments.sh
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
VLLM_PORT=8000
TA_PORT=9000

DATE_TAG="0416"
DIR_32="results/qwen3_8b_lite_32_${DATE_TAG}"
DIR_100="results/qwen3_8b_lite_100_${DATE_TAG}"

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
  echo "[KILL] Stopping ThunderAgent ..."
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
  if pgrep -f "ThunderAgent" > /dev/null 2>&1; then
    echo "[WARN] ThunderAgent still running, force killing ..."
    pkill -9 -f "ThunderAgent" 2>/dev/null || true
    sleep 2
  fi

  # Wait for ports to be fully released
  echo "[WAIT] Waiting for ports ${VLLM_PORT} and ${TA_PORT} to be released ..."
  for i in $(seq 1 30); do
    vllm_gone=true; ta_gone=true
    ss -tlnpH "sport = :${VLLM_PORT}" 2>/dev/null | grep -q LISTEN && vllm_gone=false
    ss -tlnpH "sport = :${TA_PORT}"   2>/dev/null | grep -q LISTEN && ta_gone=false
    ${vllm_gone} && ${ta_gone} && break
    sleep 2
  done

  echo "[OK] Services stopped."
}

start_vllm() {
  local log_file=$1
  echo "[START] Starting vLLM (${MODEL}) ..."
  nohup vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --enable-prompt-tokens-details \
    >"${log_file}" 2>&1 &
  VLLM_PID=$!
  wait_for "http://localhost:${VLLM_PORT}/health" "vLLM" 600
}

start_thunderagent() {
  local output_dir=$1 log_file=$2
  echo "[START] Starting ThunderAgent (port ${TA_PORT}) ..."
  nohup python -m ThunderAgent \
    --backend-type vllm \
    --backends "http://localhost:${VLLM_PORT}" \
    --port "${TA_PORT}" \
    --metrics \
    --profile \
    --profile-dir "${output_dir}" \
    >"${log_file}" 2>&1 &
  TA_PID=$!
  wait_for "http://localhost:${TA_PORT}/health" "ThunderAgent" 120
}

dump_metrics() {
  local output_dir=$1
  echo "[DUMP] Saving metrics to ${output_dir} ..."

  if ! curl -sf "http://localhost:${TA_PORT}/health" > /dev/null 2>&1; then
    echo "[ERROR] ThunderAgent is not running on port ${TA_PORT}! Cannot dump metrics."
    return 1
  fi

  curl -sf "http://localhost:${TA_PORT}/metrics" 2>/dev/null | python3 -m json.tool \
    > "${output_dir}/thunderagent_metrics.json" 2>/dev/null || echo "[WARN] Failed to dump TA metrics"
  curl -sf "http://localhost:${TA_PORT}/health" 2>/dev/null | python3 -m json.tool \
    > "${output_dir}/thunderagent_health.json" 2>/dev/null || echo "[WARN] Failed to dump TA health"
  curl -sf "http://localhost:${VLLM_PORT}/metrics" 2>/dev/null \
    > "${output_dir}/vllm_metrics.txt" 2>/dev/null || echo "[WARN] Failed to dump vLLM metrics"
  echo "[OK] Metrics dumped."
}

run_experiment() {
  local n_tasks=$1 output_dir=$2 slice=$3
  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo "  EXPERIMENT: ${n_tasks} tasks → ${output_dir} (slice ${slice})"
  echo "════════════════════════════════════════════════════════════════════"
  echo ""

  mkdir -p "${output_dir}"

  # Clean start
  kill_services

  # Start services
  start_vllm "${output_dir}/vllm.log"
  start_thunderagent "${output_dir}" "${output_dir}/thunderagent.log"

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
  echo "[COMPLETE] Experiment ${n_tasks} tasks done. Results in ${output_dir}"
  echo ""
}

# ── Main ────────────────────────────────────────────────────────────────────

echo "=============================================="
echo " Sequential SWE-bench Lite Experiments"
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
echo " ALL EXPERIMENTS COMPLETE"
echo " $(date)"
echo "=============================================="
echo ""
echo "Results:"
echo "  32 tasks:  ${DIR_32}/"
echo "  100 tasks: ${DIR_100}/"
