#!/usr/bin/env bash
set -euo pipefail

# Sequential pipeline: baseline (vLLM only) then ThunderAgent experiment.
# Fixed for 2026-04-23 run: 100 tasks, 48 workers, gpu_memory_utilization=0.55
#
# Usage:
#   bash scripts/pipeline_0423.sh [PULL_PID]
#
# Optional argument:
#   PULL_PID  If given, wait for this PID to finish before starting experiments.

PULL_PID="${1:-}"

TASK_COUNT=100
WORKERS=48
GPU_MEM_UTIL=0.55
MODEL_REPO="Qwen/Qwen3-8B"
HEALTH_TIMEOUT_S=600
SWEBENCH_SUBSET="lite"
SWEBENCH_SPLIT="test"
SWEBENCH_SLICE="0:${TASK_COUNT}"
CONFIG_FILE="examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen3_8b.yaml"

OUTPUT_VLLM="results/qwen3_8b_lite_48_0423_vllm"
OUTPUT_THUNDER="results/qwen3_8b_lite_48_0423_thunder"

APPTAINER_SANDBOX_DIR="${APPTAINER_SANDBOX_DIR:-/localscratch-nvme/${SLURM_JOB_ID:-0}/apptainer_sandbox}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/projects/pearl/Storage_Jiakun/apptainer_cache}"

PIPELINE_LOG="results/pipeline_0423.log"

log_info()  { echo "[INFO]  $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "${PIPELINE_LOG}"; }
log_error() { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "${PIPELINE_LOG}" >&2; }
die()       { log_error "$*"; exit 1; }

wait_for_health() {
  local url="$1" name="$2" timeout_s="${3:-600}"
  local start now
  start="$(date +%s)"
  while true; do
    if curl -sS -m 2 "${url}" >/dev/null 2>&1; then
      log_info "${name} is healthy."
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      die "${name} health check timed out after ${timeout_s}s"
    fi
    sleep 5
  done
}

kill_all() {
  log_info "Stopping all vLLM and ThunderAgent processes..."
  pkill -f "python -m ThunderAgent" 2>/dev/null || true
  sleep 2
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  for pid in $pids; do
    local owner
    owner=$(ps -o user= -p "${pid}" 2>/dev/null || true)
    if [[ "${owner}" == "$(whoami)" ]]; then
      log_info "  Force killing stale GPU process PID=${pid}"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
}

# ──────────────────────────────────────────────
# PHASE 0: Wait for sandbox pull
# ──────────────────────────────────────────────
mkdir -p results
if [[ -n "${PULL_PID}" ]]; then
  log_info "Waiting for sandbox pull (PID ${PULL_PID}) to complete..."
  if kill -0 "${PULL_PID}" 2>/dev/null; then
    wait "${PULL_PID}" || log_error "Pull sandbox exited with non-zero (continuing anyway)"
  else
    log_info "  PID ${PULL_PID} already finished."
  fi
  log_info "Sandbox pull done."
fi

module load CUDA/12.6.0 2>/dev/null || true
module load apptainer   2>/dev/null || true
export APPTAINER_SANDBOX_DIR
export APPTAINER_CACHEDIR

# ──────────────────────────────────────────────
# PHASE 1: Baseline — vLLM on port 9000
# ──────────────────────────────────────────────
log_info "============================================================"
log_info "PHASE 1: Baseline (vLLM only) -> ${OUTPUT_VLLM}"
log_info "  tasks=${TASK_COUNT}  workers=${WORKERS}  gpu_mem=${GPU_MEM_UTIL}"
log_info "============================================================"

mkdir -p "${OUTPUT_VLLM}"
kill_all

log_info "Starting vLLM on port 9000..."
VLLM_PID_VLLM=""
nohup vllm serve "${MODEL_REPO}" \
  --port 9000 \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  >"${OUTPUT_VLLM}/vllm.log" 2>&1 &
VLLM_PID_VLLM="$!"
log_info "  vLLM PID: ${VLLM_PID_VLLM}"

wait_for_health "http://localhost:9000/health" "vLLM(9000)" "${HEALTH_TIMEOUT_S}"

log_info "Running SWE-bench (baseline)..."
mini-extra swebench \
  --subset "${SWEBENCH_SUBSET}" \
  --split  "${SWEBENCH_SPLIT}" \
  --slice  "${SWEBENCH_SLICE}" \
  --workers "${WORKERS}" \
  --output "${OUTPUT_VLLM}" \
  --config "${CONFIG_FILE}" \
  2>&1 | tee "${OUTPUT_VLLM}/eval.log"

log_info "Collecting baseline metrics..."
curl -s http://localhost:9000/metrics > "${OUTPUT_VLLM}/vllm_metrics.txt" || true

log_info "Stopping vLLM after baseline..."
kill "${VLLM_PID_VLLM}" 2>/dev/null || true
sleep 5

log_info "PHASE 1 DONE. Results in ${OUTPUT_VLLM}"

# ──────────────────────────────────────────────
# PHASE 2: ThunderAgent — vLLM on 8000, TA on 9000
# ──────────────────────────────────────────────
log_info "============================================================"
log_info "PHASE 2: ThunderAgent -> ${OUTPUT_THUNDER}"
log_info "  tasks=${TASK_COUNT}  workers=${WORKERS}  gpu_mem=${GPU_MEM_UTIL}"
log_info "============================================================"

mkdir -p "${OUTPUT_THUNDER}"
kill_all

log_info "Starting vLLM on port 8000..."
VLLM_PID_TA=""
nohup vllm serve "${MODEL_REPO}" \
  --port 8000 \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  >"${OUTPUT_THUNDER}/vllm.log" 2>&1 &
VLLM_PID_TA="$!"
log_info "  vLLM PID: ${VLLM_PID_TA}"

wait_for_health "http://localhost:8000/health" "vLLM(8000)" "${HEALTH_TIMEOUT_S}"

log_info "Starting ThunderAgent on port 9000..."
TA_PID=""
nohup python -m ThunderAgent \
  --backend-type vllm \
  --backends http://localhost:8000 \
  --port 9000 \
  --metrics \
  --profile \
  --profile-dir "${OUTPUT_THUNDER}" \
  >"${OUTPUT_THUNDER}/thunderagent.log" 2>&1 &
TA_PID="$!"
log_info "  ThunderAgent PID: ${TA_PID}"

wait_for_health "http://localhost:9000/health" "ThunderAgent(9000)" "${HEALTH_TIMEOUT_S}"

log_info "Running SWE-bench (ThunderAgent)..."
mini-extra swebench \
  --subset "${SWEBENCH_SUBSET}" \
  --split  "${SWEBENCH_SPLIT}" \
  --slice  "${SWEBENCH_SLICE}" \
  --workers "${WORKERS}" \
  --output "${OUTPUT_THUNDER}" \
  --config "${CONFIG_FILE}" \
  2>&1 | tee "${OUTPUT_THUNDER}/eval.log"

log_info "Collecting ThunderAgent metrics..."
curl -s http://localhost:9000/metrics | python3 -m json.tool > "${OUTPUT_THUNDER}/thunderagent_metrics.json" || true
curl -s http://localhost:9000/health  | python3 -m json.tool > "${OUTPUT_THUNDER}/thunderagent_health.json"  || true
curl -s http://localhost:8000/metrics > "${OUTPUT_THUNDER}/vllm_metrics.txt" || true

log_info "Stopping ThunderAgent and vLLM..."
kill "${TA_PID}"      2>/dev/null || true
kill "${VLLM_PID_TA}" 2>/dev/null || true

log_info "PHASE 2 DONE. Results in ${OUTPUT_THUNDER}"

log_info "============================================================"
log_info "ALL DONE. Both experiments complete."
log_info "  Baseline:     ${OUTPUT_VLLM}"
log_info "  ThunderAgent: ${OUTPUT_THUNDER}"
log_info "============================================================"
