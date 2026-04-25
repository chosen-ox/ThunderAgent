#!/usr/bin/env bash
# Sequential run: vLLM baseline -> ThunderAgent
# 100 tasks, 32 workers, gpu_memory_utilization=0.55
# Output: results/qwen3_8b_lite_32_0424_0.55_vllm  /  results/qwen3_8b_lite_32_0424_0.55_thunder
set -euo pipefail

TASK_COUNT=100
WORKERS=32
GPU_MEMORY_UTIL=0.55
MODEL_REPO="Qwen/Qwen3-8B"
CONFIG_FILE="examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen3_8b.yaml"
SWEBENCH_SUBSET="lite"
SWEBENCH_SPLIT="test"
SWEBENCH_SLICE="0:${TASK_COUNT}"
HEALTH_TIMEOUT_S=600
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

VLLM_OUTPUT_DIR="results/qwen3_8b_lite_32_0424_0.55_vllm"
THUNDER_OUTPUT_DIR="results/qwen3_8b_lite_32_0424_0.55_thunder"

APPTAINER_SANDBOX_DIR="${APPTAINER_SANDBOX_DIR:-/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/projects/pearl/Storage_Jiakun/apptainer_cache}"

log_info()  { echo "[INFO]  $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a /home/jiakunfan/ThunderAgent/run_0424_sequential.log; }
log_error() { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a /home/jiakunfan/ThunderAgent/run_0424_sequential.log >&2; }
die()       { log_error "$*"; exit 1; }

wait_for_health() {
  local url="$1" name="$2"
  local start; start="$(date +%s)"
  while true; do
    if curl -sS -m 2 "${url}" >/dev/null 2>&1; then return 0; fi
    (( $(date +%s) - start >= HEALTH_TIMEOUT_S )) && die "${name} health timeout"
    sleep 5
  done
}

kill_all() {
  log_info "Killing ThunderAgent and vLLM..."
  pkill -f "python -m ThunderAgent" 2>/dev/null || true
  sleep 2
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  for pid in $pids; do
    local owner; owner=$(ps -o user= -p "${pid}" 2>/dev/null || true)
    if [[ "${owner}" == "$(whoami)" ]]; then kill -9 "${pid}" 2>/dev/null || true; fi
  done
  sleep 2
}

module load CUDA/12.6.0 2>/dev/null || true
module load apptainer 2>/dev/null || true
export APPTAINER_SANDBOX_DIR APPTAINER_CACHEDIR

log_info "====== Starting sequential experiment run 0424 ======"
log_info "Tasks=${TASK_COUNT}  Workers=${WORKERS}  GPU_MEM_UTIL=${GPU_MEMORY_UTIL}"

# =========================================================
# Phase 1: vLLM baseline (port 9000, no ThunderAgent)
# =========================================================
log_info "--- Phase 1: vLLM baseline ---"
mkdir -p "${VLLM_OUTPUT_DIR}"
touch "${VLLM_OUTPUT_DIR}/.phase1_started"

kill_all

log_info "Starting vLLM on port 9000 (baseline)"
CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup vllm serve "${MODEL_REPO}" \
  --port 9000 \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
  >"${VLLM_OUTPUT_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

wait_for_health "http://localhost:9000/health" "vLLM-baseline"
log_info "vLLM baseline ready. Running SWE-bench..."

mini-extra swebench \
  --subset "${SWEBENCH_SUBSET}" \
  --split "${SWEBENCH_SPLIT}" \
  --slice "${SWEBENCH_SLICE}" \
  --workers "${WORKERS}" \
  --output "${VLLM_OUTPUT_DIR}" \
  --config "${CONFIG_FILE}" \
  2>&1 | tee "${VLLM_OUTPUT_DIR}/eval.log"

log_info "Dumping vLLM baseline metrics..."
curl -s http://localhost:9000/metrics > "${VLLM_OUTPUT_DIR}/vllm_metrics.txt" || true
touch "${VLLM_OUTPUT_DIR}/.phase1_done"

kill_all
log_info "--- Phase 1 complete ---"

# =========================================================
# Phase 2: ThunderAgent (vLLM port 8000, TA port 9000)
# =========================================================
log_info "--- Phase 2: ThunderAgent ---"
mkdir -p "${THUNDER_OUTPUT_DIR}"
touch "${THUNDER_OUTPUT_DIR}/.phase2_started"

kill_all

log_info "Starting vLLM on port 8000 (ThunderAgent backend)"
CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup vllm serve "${MODEL_REPO}" \
  --port 8000 \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
  >"${THUNDER_OUTPUT_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

wait_for_health "http://localhost:8000/health" "vLLM-thunder"
log_info "vLLM ready. Starting ThunderAgent on port 9000..."

nohup python -m ThunderAgent \
  --backend-type vllm \
  --backends http://localhost:8000 \
  --port 9000 \
  --metrics \
  --profile \
  --profile-dir "${THUNDER_OUTPUT_DIR}" \
  >"${THUNDER_OUTPUT_DIR}/thunderagent.log" 2>&1 &
TA_PID=$!

wait_for_health "http://localhost:9000/health" "ThunderAgent"
log_info "ThunderAgent ready. Running SWE-bench..."

mini-extra swebench \
  --subset "${SWEBENCH_SUBSET}" \
  --split "${SWEBENCH_SPLIT}" \
  --slice "${SWEBENCH_SLICE}" \
  --workers "${WORKERS}" \
  --output "${THUNDER_OUTPUT_DIR}" \
  --config "${CONFIG_FILE}" \
  2>&1 | tee "${THUNDER_OUTPUT_DIR}/eval.log"

log_info "Dumping ThunderAgent metrics..."
curl -s http://localhost:9000/metrics | python3 -m json.tool > "${THUNDER_OUTPUT_DIR}/thunderagent_metrics.json" || true
curl -s http://localhost:9000/health  | python3 -m json.tool > "${THUNDER_OUTPUT_DIR}/thunderagent_health.json"  || true
curl -s http://localhost:8000/metrics > "${THUNDER_OUTPUT_DIR}/vllm_metrics.txt" || true
touch "${THUNDER_OUTPUT_DIR}/.phase2_done"

kill_all
log_info "--- Phase 2 complete ---"
log_info "====== All experiments done ======"
log_info "  Baseline:     ${VLLM_OUTPUT_DIR}/"
log_info "  ThunderAgent: ${THUNDER_OUTPUT_DIR}/"
touch /home/jiakunfan/ThunderAgent/.run_0424_all_done
