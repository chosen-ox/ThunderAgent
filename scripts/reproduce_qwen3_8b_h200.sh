#!/usr/bin/env bash
set -euo pipefail

# Reproduce Qwen3-8B + vLLM + ThunderAgent + mini-SWE-agent SWE-bench Lite on single H200.
# Run this script from repository root.
#
# Usage:
#   bash scripts/reproduce_qwen3_8b_h200.sh [TASK_COUNT] [WORKERS]
#
# Arguments:
#   TASK_COUNT  Number of SWE-bench instances to run (default: 300, i.e. full Lite)
#   WORKERS     Number of parallel workers (default: 64)
#
# H200 vs A100 differences:
#   - H200 SXM has 141GB HBM3e vs A100's 80GB
#   - Qwen3-8B BF16 (~16GB) leaves ~125GB for KV cache (vs ~64GB on A100)
#   - Default workers raised to 64 (vs 32 on A100)

# =========================
# User-facing configuration
# =========================
TASK_COUNT="${1:-300}"
WORKERS="${2:-64}"

# Model
MODEL_REPO="Qwen/Qwen3-8B"

# Runtime services
VLLM_PORT="8000"
TA_PORT="9000"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
HEALTH_TIMEOUT_S="600"

# SWE-bench run
SWEBENCH_SUBSET="lite"
SWEBENCH_SPLIT="test"
SWEBENCH_SLICE="0:${TASK_COUNT}"
CONFIG_FILE="examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen3_8b.yaml"

# Output
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/qwen3_8b_h200_lite_${TASK_COUNT}_${TIMESTAMP}"

# Apptainer sandbox cache (pre-built sandboxes skip apptainer build)
APPTAINER_SANDBOX_DIR="${APPTAINER_SANDBOX_DIR:-/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/projects/pearl/Storage_Jiakun/apptainer_cache}"

# =========================
# Internal wiring
# =========================
VLLM_LOG="${OUTPUT_DIR}/vllm.log"
TA_LOG="${OUTPUT_DIR}/thunderagent.log"
EVAL_LOG="${OUTPUT_DIR}/eval.log"

VLLM_HEALTH_URL="http://localhost:${VLLM_PORT}/health"
TA_HEALTH_URL="http://localhost:${TA_PORT}/health"

VLLM_PID=""
TA_PID=""

# =========================
# Utility functions
# =========================
log_info()  { echo "[INFO]  $(date +%H:%M:%S) $*"; }
log_error() { echo "[ERROR] $(date +%H:%M:%S) $*" >&2; }
die()       { log_error "$*"; exit 1; }

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    die "Missing required command: ${cmd}"
  fi
}

wait_for_health() {
  local url="$1" name="$2" timeout_s="${3:-600}"
  local start now
  start="$(date +%s)"
  while true; do
    if curl -sS -m 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      die "${name} health check failed: timeout after ${timeout_s}s"
    fi
    sleep 5
  done
}

kill_old_processes() {
  log_info "Killing old ThunderAgent processes..."
  pkill -f "python -m ThunderAgent" 2>/dev/null || true
  sleep 2

  log_info "Killing old vLLM processes..."
  pkill -f "vllm serve.*${MODEL_REPO}" 2>/dev/null || true
  sleep 3

  # Force kill any remaining engine core processes on our GPU
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  for pid in $pids; do
    local owner
    owner=$(ps -o user= -p "${pid}" 2>/dev/null || true)
    if [[ "${owner}" == "$(whoami)" ]]; then
      log_info "Force killing stale GPU process: PID=${pid}"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
}

dump_metrics() {
  log_info "Dumping final metrics..."
  local out="$1"

  curl -s "http://localhost:${TA_PORT}/metrics" \
    | python3 -m json.tool > "${out}/thunderagent_metrics.json" 2>/dev/null \
    && log_info "  thunderagent_metrics.json saved" \
    || log_error "  Failed to dump ThunderAgent metrics"

  curl -s "http://localhost:${TA_PORT}/health" \
    | python3 -m json.tool > "${out}/thunderagent_health.json" 2>/dev/null \
    && log_info "  thunderagent_health.json saved" \
    || log_error "  Failed to dump ThunderAgent health"

  curl -s "http://localhost:${VLLM_PORT}/metrics" \
    > "${out}/vllm_metrics.txt" 2>/dev/null \
    && log_info "  vllm_metrics.txt saved" \
    || log_error "  Failed to dump vLLM metrics"
}

cleanup() {
  set +e
  if [[ -n "${TA_PID}" ]] && kill -0 "${TA_PID}" 2>/dev/null; then
    log_info "Stopping ThunderAgent (PID ${TA_PID})..."
    kill "${TA_PID}" 2>/dev/null
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    log_info "Stopping vLLM (PID ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

# =========================
# Main steps
# =========================
start_vllm() {
  log_info "Starting vLLM on GPU ${GPU_ID}, port ${VLLM_PORT} (log: ${VLLM_LOG})"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup vllm serve "${MODEL_REPO}" \
    --port "${VLLM_PORT}" \
    --enable-prompt-tokens-details \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"

  log_info "Waiting for vLLM health: ${VLLM_HEALTH_URL}"
  wait_for_health "${VLLM_HEALTH_URL}" "vLLM" "${HEALTH_TIMEOUT_S}"
}

start_thunderagent() {
  log_info "Starting ThunderAgent on port ${TA_PORT} (log: ${TA_LOG})"
  nohup python -m ThunderAgent \
    --backend-type vllm \
    --backends "http://localhost:${VLLM_PORT}" \
    --port "${TA_PORT}" \
    --metrics \
    --profile \
    --profile-dir "${OUTPUT_DIR}" \
    >"${TA_LOG}" 2>&1 &
  TA_PID="$!"

  log_info "Waiting for ThunderAgent health: ${TA_HEALTH_URL}"
  wait_for_health "${TA_HEALTH_URL}" "ThunderAgent" "${HEALTH_TIMEOUT_S}"
}

run_swebench() {
  log_info "Running SWE-bench Lite (slice ${SWEBENCH_SLICE}, workers ${WORKERS})"
  mini-extra swebench \
    --subset "${SWEBENCH_SUBSET}" \
    --split "${SWEBENCH_SPLIT}" \
    --slice "${SWEBENCH_SLICE}" \
    --workers "${WORKERS}" \
    --output "${OUTPUT_DIR}" \
    --config "${CONFIG_FILE}" \
    2>&1 | tee "${EVAL_LOG}"
}

main() {
  log_info "============================================"
  log_info "Qwen3-8B SWE-bench Lite reproduce (H200)"
  log_info "  Tasks: ${TASK_COUNT}  Workers: ${WORKERS}"
  log_info "  Output: ${OUTPUT_DIR}"
  log_info "  GPU: ${GPU_ID}"
  log_info "============================================"

  [[ -d "./examples/inference/mini-swe-agent" ]] || die "Please run from repository root."
  [[ -f "${CONFIG_FILE}" ]] || die "Config not found: ${CONFIG_FILE}"
  require_cmd vllm
  require_cmd curl
  require_cmd mini-extra
  require_cmd python3

  # Setup
  module load CUDA/12.6.0 2>/dev/null || true
  module load apptainer 2>/dev/null || true
  export APPTAINER_SANDBOX_DIR
  export APPTAINER_CACHEDIR
  mkdir -p "${OUTPUT_DIR}"

  # Run
  kill_old_processes
  start_vllm
  start_thunderagent
  run_swebench

  # Collect
  dump_metrics "${OUTPUT_DIR}"

  log_info "============================================"
  log_info "Done! Results in: ${OUTPUT_DIR}"
  log_info "  - Instance trajectories: ${OUTPUT_DIR}/*/"
  log_info "  - Step profiles: ${OUTPUT_DIR}/step_profiles.csv"
  log_info "  - ThunderAgent metrics: ${OUTPUT_DIR}/thunderagent_metrics.json"
  log_info "  - vLLM metrics: ${OUTPUT_DIR}/vllm_metrics.txt"
  log_info "  - Eval log: ${EVAL_LOG}"
  log_info "============================================"
}

main "$@"
