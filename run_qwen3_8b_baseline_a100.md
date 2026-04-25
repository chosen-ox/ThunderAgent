# Qwen3-8B + vLLM + mini-SWE-agent (Baseline, no ThunderAgent) on A100

## Architecture

```
mini-SWE-agent --> vLLM (port 9000, A100 GPU)
```

- **vLLM**: Model inference backend, loads Qwen3-8B on A100 GPU, listens on port 9000
- **mini-SWE-agent**: SWE-bench evaluation client, connects directly to vLLM

This is the **baseline** (no ThunderAgent). Compare against `run_qwen3_8b_a100.md` (with ThunderAgent).

**关键设计**：vLLM 监听 9000 端口（与 ThunderAgent 相同），这样现有 config 的 `base_url: http://localhost:9000/v1` 完全不用改。

## Prerequisites

### 1. Environment

```bash
# Use the thunder conda environment
conda activate thunder

# Load required modules
module load CUDA/12.6.0
module load apptainer
```

### 2. Environment Variables (每次新 terminal 必须执行)

```bash
# Apptainer sandbox 缓存 — 预构建的 sandbox 存在 Pearl 存储，避免重复拉取/构建
export APPTAINER_SANDBOX_DIR="/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox"
export APPTAINER_CACHEDIR="/projects/pearl/Storage_Jiakun/apptainer_cache"
```

### 3. Model Weights (already downloaded)

Qwen3-8B weights are cached at:

```
/projects/pearl/Storage_Jiakun/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/
```

No additional download needed. `HF_HOME` is set to `/projects/pearl/Storage_Jiakun/huggingface`.

### 4. Installed Packages (already installed)

| Package | Version |
|---------|---------|
| vLLM    | 0.19.0  |
| mini-SWE-agent | 1.14.4 (`pip install -e examples/inference/mini-swe-agent[full]`) |

If reinstalling is needed:

```bash
cd /home/jiakunfan/ThunderAgent
pip install -e "examples/inference/mini-swe-agent[full]"
```

### 5. Pre-build Apptainer Sandboxes (推荐，首次运行前执行)

SWE-bench 的每个实例都需要一个 Apptainer sandbox（~2.7GB/个）。预构建可避免评测时重复拉取和构建。

```bash
# 预构建前 N 个实例的 sandbox，4 并行拉取
bash scripts/pull_swebench_sandboxes.sh 100 4
```

Sandboxes 存储在 `/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox/{instance_id}/`。

如需构建全部 300 个（SWE-bench Lite）：

```bash
bash scripts/pull_swebench_sandboxes.sh 300 4
```

## Running

> **!!! 每次实验前必须重启 vLLM !!!**
>
> vLLM 的 metrics 仅存于内存，跨 run 会累积。如果不重启，cache hit rate 等累积指标会被前一次 run 污染，导致结果不准确。
>
> **做法：先 `pkill -f "vllm serve"` 杀掉旧进程，再按下方 Step 1 重新启动。**

### One-Click Script

最简单的方式 — 一键运行全部流程：

```bash
cd /home/jiakunfan/ThunderAgent
module load CUDA/12.6.0
module load apptainer

export APPTAINER_SANDBOX_DIR="/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox"
export APPTAINER_CACHEDIR="/projects/pearl/Storage_Jiakun/apptainer_cache"

# 默认: 300 tasks, 32 workers
bash scripts/reproduce_qwen3_8b_baseline_a100.sh

# 自定义: 前 100 个 tasks, 32 workers
bash scripts/reproduce_qwen3_8b_baseline_a100.sh 100 32
```

脚本自动完成：kill 旧进程 -> 启动 vLLM (port 9000) -> 运行 SWE-bench -> dump metrics -> 输出目录。

### Manual Step-by-Step

如需分步手动操作（调试/监控用）：

### Output Directory

All outputs are consolidated into a single directory:

```bash
OUTPUT_DIR="results/qwen3_8b_baseline_lite"
mkdir -p $OUTPUT_DIR
```

This directory will contain:
- `{instance_id}/` — per-instance trajectory JSONs (from mini-SWE-agent)
- `vllm_metrics.txt` — vLLM Prometheus metrics snapshot

### Step 1: Start vLLM (port 9000)

**先杀掉旧的 vLLM 进程（必须）：**

```bash
pkill -f "vllm serve"
sleep 3
```

vLLM 监听 9000 端口，复用现有 config 的 `base_url: http://localhost:9000/v1`：

```bash
module load CUDA/12.6.0
vllm serve Qwen/Qwen3-8B \
  --port 9000 \
  --enable-prompt-tokens-details \
  2>&1 | tee $OUTPUT_DIR/vllm.log
```

参数说明：

- `--port 9000`: **关键**。与 ThunderAgent 实验使用相同端口，config 无需修改。
- `--enable-prompt-tokens-details`: 用于返回 `usage.prompt_tokens_details.cached_tokens`，便于计算 token-level KV cache hit rate。
- `--enable-prefix-caching`: vLLM 0.19.0 默认已开启，无需手动添加。

Wait until vLLM reports ready:

```bash
curl http://localhost:9000/health
```

### Step 2: Run SWE-bench Evaluation

直接复用现有 config，无需修改：

```bash
cd /home/jiakunfan/ThunderAgent
module load apptainer

export APPTAINER_SANDBOX_DIR="/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox"
export APPTAINER_CACHEDIR="/projects/pearl/Storage_Jiakun/apptainer_cache"

OUTPUT_DIR="results/qwen3_8b_baseline_lite"

mini-extra swebench \
  --subset lite \
  --split test \
  --slice "0:100" \
  --workers 32 \
  -o $OUTPUT_DIR \
  --config examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen3_8b.yaml
```

参数说明：

- `--slice "0:100"`: **重要**。限制只运行前 100 个实例。不加此参数会运行全部 300 个（SWE-bench Lite 完整数据集）。格式为 `"start:end"`（Python slice 语义，不含 end）。

Notes on `--workers`:
- A100 80GB with Qwen3-8B BF16 leaves ~64GB for KV cache
- 32 concurrent workers is a reasonable starting point
- **与 ThunderAgent 实验不同**：没有 ThunderAgent 的 KV cache 调度，高并发时可能出现 OOM，需根据实际情况调整

### Step 3: Dump Final Metrics

After evaluation completes, save the final metrics snapshot:

```bash
OUTPUT_DIR="results/qwen3_8b_baseline_lite"

curl -s http://localhost:9000/metrics \
  > $OUTPUT_DIR/vllm_metrics.txt
```

### Output Directory Structure

After a complete run, `$OUTPUT_DIR` contains:

```
results/qwen3_8b_baseline_lite/
├── astropy__astropy-14182/
│   └── astropy__astropy-14182.traj.json    # instance trajectory
├── django__django-10914/
│   └── django__django-10914.traj.json
├── ...                                     # instance directories
├── vllm.log                                # vLLM stdout/stderr log
└── vllm_metrics.txt                        # vLLM Prometheus metrics snapshot
```

**与 ThunderAgent 实验的差异**：无 `step_profiles.csv`、`thunderagent_metrics.json`、`thunderagent_health.json`。

### Config File

**完全复用** `swebench_qwen3_8b.yaml`，无需任何修改。vLLM 监听 9000 端口，config 中 `base_url: http://localhost:9000/v1` 自然匹配。

## Monitoring & Metrics

### Runtime Monitoring

During evaluation, check progress in real-time:

```bash
# vLLM Prometheus metrics (prefill/decode tokens, cache hit rate, etc.)
curl -s http://localhost:9000/metrics | grep -E "vllm:(prompt_tokens|generation_tokens|prefix_cache|num_requests|kv_cache)"
```

**与 ThunderAgent 实验的差异**：无 `/health`、`/metrics`、`/programs`、`/profiles` 端点（这些是 ThunderAgent 提供的）。

### Key Metrics Explained

| Metric | Source | Meaning |
|--------|--------|---------|
| `prompt_tokens_by_source{local_cache_hit}` | `:9000/metrics` | Token-level KV cache reuse count |
| `prompt_tokens_total` | `:9000/metrics` | Total prompt tokens processed |
| `generation_tokens_total` | `:9000/metrics` | Total tokens generated |
| `vllm:num_requests_running` | `:9000/metrics` | Currently running requests |
| Token-level reuse rate | computed | `local_cache_hit / prompt_tokens_total` |

### Prefix Cache Hit Rate

Baseline 只有 vLLM 原生的 cache reuse，计算方式：

```
Token-level reuse = prompt_tokens_by_source{source="local_cache_hit"} / prompt_tokens_total
```

与 ThunderAgent 实验对比时，关注：
- **Token-level reuse rate**：Baseline vs ThunderAgent 的差异体现 ThunderAgent 的 KV cache 调度效果
- **总推理时间**：Baseline vs ThunderAgent 的 wall-clock time 差异
- **OOM 频率**：Baseline 在高并发下是否出现 OOM（ThunderAgent 有 preemption 机制避免 OOM）

## Known Issues

### Qwen3-8B context window limitation

Qwen3-8B maximum context length is 40960 tokens. Some SWE-bench instances have very long problem statements that quickly exceed this limit, causing `ContextLengthExceeded` errors (~37% of Lite instances in testing).

### OOM under high concurrency

**Baseline 特有问题**：没有 ThunderAgent 的 KV cache 调度，32 workers 同时推理时可能触发 OOM。如果遇到 OOM，降低 `--workers`（如 16 或 8）。

### Apptainer image pull failures on concurrent builds

When many workers build sandboxes simultaneously, network bandwidth can be saturated causing `conveyor failed to get: unexpected end of JSON input` errors. The code retries 3 times. Pre-building sandboxes with `scripts/pull_swebench_sandboxes.sh` avoids this entirely.

### Model behavior

Qwen3-8B tends to submit fixes after very few steps (avg ~6 steps) without adequate testing or verification, compared to larger models like GLM-4.7 (avg ~33 steps) which follow the recommended workflow more closely.
