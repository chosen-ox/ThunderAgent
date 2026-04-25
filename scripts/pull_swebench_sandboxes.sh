#!/usr/bin/env bash
set -euo pipefail

# Pre-build Apptainer sandboxes for first N SWE-bench Lite instances.
# Builds on local NVMe (fast), then copies to Pearl (persistent).
# Usage: bash scripts/pull_swebench_sandboxes.sh [COUNT] [PARALLEL]
#   COUNT defaults to 100, PARALLEL defaults to 4

COUNT="${1:-100}"
PARALLEL="${2:-4}"
SANDBOX_DIR="/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox"
export APPTAINER_CACHEDIR="/projects/pearl/Storage_Jiakun/apptainer_cache"

module load apptainer 2>/dev/null || true
mkdir -p "${SANDBOX_DIR}" "${APPTAINER_CACHEDIR}"

# Write image list to a temp file
LIST_FILE=$(mktemp)
/home/jiakunfan/miniconda3/envs/thunder/bin/python3 -c "
from datasets import load_dataset
instances = list(load_dataset('princeton-nlp/SWE-Bench_Lite', split='test'))[:${COUNT}]
for inst in instances:
    iid = inst['instance_id']
    id_docker = iid.replace('__', '_1776_')
    image = f'docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest'.lower()
    print(f'{iid} docker://{image}')
" > "$LIST_FILE"

total=$(wc -l < "$LIST_FILE")
echo "Pre-building ${total} sandboxes (parallel=${PARALLEL})"
echo "  Sandbox dir (NVMe): ${SANDBOX_DIR}"
echo "  Cache dir (Pearl): ${SANDBOX_DIR}"

build_one() {
    local line="$1"
    local iid="${line%% *}"
    local image="${line#* }"
    local target_dir="${SANDBOX_DIR}/${iid}"

    # Skip if already cached (has /testbed)
    if [[ -d "${target_dir}/testbed" ]]; then
        echo "[skip] ${iid}"
        return 0
    fi
    # Remove incomplete directory (missing /testbed)
    if [[ -d "${target_dir}" ]]; then
        rm -rf "${target_dir}"
    fi

    # Build directly to target on NVMe
    echo "[build] ${iid}"
    local log_file
    log_file=$(mktemp)
    if apptainer build --sandbox "${target_dir}" "${image}" >"${log_file}" 2>&1; then
        echo "[done] ${iid}"
    else
        echo "[fail] ${iid}"
        tail -1 "${log_file}"
        rm -rf "${target_dir}"
    fi
    rm -f "${log_file}"
}

export SANDBOX_DIR APPTAINER_CACHEDIR
export -f build_one

cat "$LIST_FILE" | xargs -P "${PARALLEL}" -I {} bash -c 'build_one "$@"' _ {}

rm -f "$LIST_FILE"

echo "Finished."
ls "${SANDBOX_DIR}" | wc -l | xargs echo "Total sandboxes built:"
