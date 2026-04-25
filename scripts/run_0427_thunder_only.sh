#!/usr/bin/env bash
#
# ThunderAgent only: 48 workers, 200 tasks, vLLM on port 8001.
# Run AFTER baseline completes.

set -euo pipefail

cd /home/jiakunfan/ThunderAgent

MODEL="Qwen/Qwen3-8B"
CONFIG="examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen3_8b.yaml"
WORKERS=48
VLLM_PORT=8001
TA_PORT=9000

DIR_THUNDER="results/qwen3_8b_lite_48_0427_200_thunder"

export APPTAINER_SANDBOX_DIR="${APPTAINER_SANDBOX_DIR:-/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/projects/pearl/Storage_Jiakun/apptainer_cache}"

wait_for() {
  local url=$1 name=$2 max_wait=${3:-600}
  echo "[WAIT] Waiting for ${name} at ${url} ..."
  local elapsed=0
  while ! curl -sf "${url}" > /dev/null 2>&1; do
    sleep 5; elapsed=$((elapsed + 5))
    if (( elapsed >= max_wait )); then echo "[ERROR] timeout"; return 1; fi
  done
  echo "[OK] ${name} ready (${elapsed}s)"
}

kill_services() {
  pkill -f "ThunderAgent" 2>/dev/null || true; sleep 2
  pkill -f "vllm serve.*${MODEL}" 2>/dev/null || true; sleep 3
  pkill -9 -f "vllm serve.*${MODEL}" 2>/dev/null || true; sleep 2
  for i in $(seq 1 30); do
    vllm_gone=true; ta_gone=true
    ss -tlnpH "sport = :${VLLM_PORT}" 2>/dev/null | grep -q LISTEN && vllm_gone=false
    ss -tlnpH "sport = :${TA_PORT}"   2>/dev/null | grep -q LISTEN && ta_gone=false
    ${vllm_gone} && ${ta_gone} && break
    sleep 2
  done
  echo "[OK] Services stopped."
}

module load CUDA/12.6.0 2>/dev/null || true
module load apptainer 2>/dev/null || true

mkdir -p "${DIR_THUNDER}"
kill_services

echo "[START] vLLM on port ${VLLM_PORT} ..."
vllm serve "${MODEL}" \
  --port "${VLLM_PORT}" \
  --enable-prompt-tokens-details \
  2>&1 | tee "${DIR_THUNDER}/vllm.log" &
wait_for "http://localhost:${VLLM_PORT}/health" "vLLM" 600

echo "[START] ThunderAgent on port ${TA_PORT} ..."
python -m ThunderAgent \
  --backend-type vllm \
  --backends "http://localhost:${VLLM_PORT}" \
  --port "${TA_PORT}" \
  --metrics \
  --profile \
  --profile-dir "${DIR_THUNDER}" \
  > "${DIR_THUNDER}/thunderagent.log" 2>&1 &
wait_for "http://localhost:${TA_PORT}/health" "ThunderAgent" 120

echo "[RUN] mini-extra swebench 200 tasks, 48 workers ..."
mini-extra swebench \
  --subset lite \
  --split test \
  --slice "0:200" \
  --workers "${WORKERS}" \
  -o "${DIR_THUNDER}" \
  --config "${CONFIG}" \
  2>&1 | tee "${DIR_THUNDER}/eval.log"

echo "[METRICS] Dumping ..."
curl -sf "http://localhost:${TA_PORT}/metrics" | python3 -m json.tool \
  > "${DIR_THUNDER}/thunderagent_metrics.json" || true
curl -sf "http://localhost:${TA_PORT}/health" | python3 -m json.tool \
  > "${DIR_THUNDER}/thunderagent_health.json" || true
curl -sf "http://localhost:${VLLM_PORT}/metrics" \
  > "${DIR_THUNDER}/vllm_metrics.txt" || true

pkill -f "ThunderAgent" 2>/dev/null || true
pkill -f "vllm serve.*${MODEL}" 2>/dev/null || true

echo "[COMPLETE] ThunderAgent done → ${DIR_THUNDER}"
