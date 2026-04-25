#!/usr/bin/env bash
#
# Sequential: Baseline (vLLM only) then ThunderAgent, 48 workers, 200 tasks.
#
# Usage:
#   bash scripts/run_0427_48_200_sequential.sh

set -euo pipefail

cd /home/jiakunfan/ThunderAgent

# ── Config ──────────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3-8B"
CONFIG="examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen3_8b.yaml"
WORKERS=48
VLLM_PORT=8001
TA_PORT=9000
BASELINE_PORT=9000

DIR_BASELINE="results/qwen3_8b_lite_48_0427_200_vllm"
DIR_THUNDER="results/qwen3_8b_lite_48_0427_200_thunder"

export APPTAINER_SANDBOX_DIR="${APPTAINER_SANDBOX_DIR:-/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/projects/pearl/Storage_Jiakun/apptainer_cache}"

# ── Helper functions ────────────────────────────────────────────────────────

wait_for() {
  local url=$1 name=$2 max_wait=${3:-600}
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

  for i in $(seq 1 30); do
    vllm_gone=true; ta_gone=true; base_gone=true
    ss -tlnpH "sport = :${VLLM_PORT}" 2>/dev/null | grep -q LISTEN && vllm_gone=false
    ss -tlnpH "sport = :${TA_PORT}"   2>/dev/null | grep -q LISTEN && ta_gone=false
    ss -tlnpH "sport = :${BASELINE_PORT}" 2>/dev/null | grep -q LISTEN && base_gone=false
    ${vllm_gone} && ${ta_gone} && ${base_gone} && break
    sleep 2
  done
  echo "[OK] Services stopped."
}

# ── Experiment 1: Baseline (vLLM only on port 9000) ─────────────────────────

run_baseline() {
  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo "  BASELINE: 200 tasks, ${WORKERS} workers → ${DIR_BASELINE}"
  echo "════════════════════════════════════════════════════════════════════"
  echo ""

  mkdir -p "${DIR_BASELINE}"
  kill_services

  echo "[START] Starting vLLM on port ${BASELINE_PORT} ..."
  vllm serve "${MODEL}" \
    --port "${BASELINE_PORT}" \
    --enable-prompt-tokens-details \
    2>&1 | tee "${DIR_BASELINE}/vllm.log" &
  wait_for "http://localhost:${BASELINE_PORT}/health" "vLLM (baseline)" 600

  echo ""
  echo "[RUN] Starting mini-extra swebench (200 tasks, ${WORKERS} workers) ..."
  echo ""

  mini-extra swebench \
    --subset lite \
    --split test \
    --slice "0:200" \
    --workers "${WORKERS}" \
    -o "${DIR_BASELINE}" \
    --config "${CONFIG}" \
    2>&1 | tee "${DIR_BASELINE}/eval.log"

  echo ""
  echo "[DONE] Baseline finished."

  curl -sf "http://localhost:${BASELINE_PORT}/metrics" 2>/dev/null \
    > "${DIR_BASELINE}/vllm_metrics.txt" || echo "[WARN] Failed to dump vLLM metrics"

  kill_services
  echo "[COMPLETE] Baseline done. Results in ${DIR_BASELINE}"
}

# ── Experiment 2: ThunderAgent (vLLM:8000 → TA:9000) ───────────────────────

run_thunderagent() {
  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo "  THUNDERAGENT: 200 tasks, ${WORKERS} workers → ${DIR_THUNDER}"
  echo "════════════════════════════════════════════════════════════════════"
  echo ""

  mkdir -p "${DIR_THUNDER}"
  kill_services

  echo "[START] Starting vLLM on port ${VLLM_PORT} ..."
  vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --enable-prompt-tokens-details \
    2>&1 | tee "${DIR_THUNDER}/vllm.log" &
  wait_for "http://localhost:${VLLM_PORT}/health" "vLLM" 600

  echo "[START] Starting ThunderAgent on port ${TA_PORT} ..."
  python -m ThunderAgent \
    --backend-type vllm \
    --backends "http://localhost:${VLLM_PORT}" \
    --port "${TA_PORT}" \
    --metrics \
    --profile \
    --profile-dir "${DIR_THUNDER}" \
    > "${DIR_THUNDER}/thunderagent.log" 2>&1 &
  wait_for "http://localhost:${TA_PORT}/health" "ThunderAgent" 120

  echo ""
  echo "[RUN] Starting mini-extra swebench (200 tasks, ${WORKERS} workers) ..."
  echo ""

  mini-extra swebench \
    --subset lite \
    --split test \
    --slice "0:200" \
    --workers "${WORKERS}" \
    -o "${DIR_THUNDER}" \
    --config "${CONFIG}"

  echo ""
  echo "[DONE] ThunderAgent experiment finished."

  curl -sf "http://localhost:${TA_PORT}/metrics" 2>/dev/null | python3 -m json.tool \
    > "${DIR_THUNDER}/thunderagent_metrics.json" || echo "[WARN] Failed to dump TA metrics"
  curl -sf "http://localhost:${TA_PORT}/health" 2>/dev/null | python3 -m json.tool \
    > "${DIR_THUNDER}/thunderagent_health.json" || echo "[WARN] Failed to dump TA health"
  curl -sf "http://localhost:${VLLM_PORT}/metrics" 2>/dev/null \
    > "${DIR_THUNDER}/vllm_metrics.txt" || echo "[WARN] Failed to dump vLLM metrics"

  kill_services
  echo "[COMPLETE] ThunderAgent done. Results in ${DIR_THUNDER}"
}

# ── Main ────────────────────────────────────────────────────────────────────

module load CUDA/12.6.0 2>/dev/null || true
module load apptainer 2>/dev/null || true

echo "=============================================="
echo " Sequential: Baseline → ThunderAgent"
echo " 48 workers, 200 tasks each"
echo " $(date)"
echo "=============================================="

run_baseline
run_thunderagent

echo ""
echo "=============================================="
echo " ALL EXPERIMENTS COMPLETE"
echo " $(date)"
echo "=============================================="
echo ""
echo "Results:"
echo "  Baseline:     ${DIR_BASELINE}/"
echo "  ThunderAgent: ${DIR_THUNDER}/"
