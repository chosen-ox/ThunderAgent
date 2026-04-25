# ThunderAgent + Qwen3-8B + mini-SWE-agent on H200

## Architecture

```
mini-SWE-agent --> ThunderAgent (port 9000) --> vLLM (port 8000, H200 GPU)
```

- **vLLM**: Model inference backend, loads Qwen3-8B on H200 GPU
- **ThunderAgent**: Program-aware proxy/scheduler, improves agentic inference throughput
- **mini-SWE-agent**: SWE-bench evaluation client, sends `program_id` via `extra_body`

## H200 vs A100

| | A100 80GB | H200 141GB |
|---|---|---|
| HBM | 80GB HBM2e | 141GB HBM3e |
| Model (BF16) | ~16GB | ~16GB |
| KV cache headroom | ~64GB | ~125GB |
| Default workers | 32 | 64 |
| Memory bandwidth | 2.0 TB/s | 4.8 TB/s |

H200 的 141GB HBM3e 留给 KV cache 的空间是 A100 的约 2 倍，可以支撑 64 个并发 worker，显著提升 agentic inference 吞吐。

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

- `APPTAINER_SANDBOX_DIR`: 预构建 sandbox 的目录。mini-SWE-agent 会优先查找此目录下已有的 sandbox（按 instance_id 命名），找到则直接复用，跳过 `apptainer build --sandbox`。
- `APPTAINER_CACHEDIR`: Apptainer 的 blob 层缓存目录，拉取过的镜像层会缓存，加速后续 build。

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
| ThunderAgent | local (repo root) |
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

> **!!! 每次实验前必须重启 vLLM 和 ThunderAgent !!!**
>
> vLLM 和 ThunderAgent 的 metrics 仅存于内存，跨 run 会累积。如果不重启，prefix cache hit rate 等累积指标会被前一次 run 污染，导致结果不准确。
>
> **做法：先 `pkill -f "vllm serve"` 和 `pkill -f "ThunderAgent"` 杀掉旧进程，再按下方 Step 1/2 重新启动。**
>
> 每一组实验（不同 task 数量 / 不同配置）都需要独立重启。

### One-Click Script

最简单的方式 — 一键运行全部流程：

```bash
cd /home/jiakunfan/ThunderAgent
module load CUDA/12.6.0
module load apptainer

export APPTAINER_SANDBOX_DIR="/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox"
export APPTAINER_CACHEDIR="/projects/pearl/Storage_Jiakun/apptainer_cache"

# 默认: 300 tasks, 64 workers
bash scripts/reproduce_qwen3_8b_h200.sh

# 自定义: 前 100 个 tasks, 64 workers
bash scripts/reproduce_qwen3_8b_h200.sh 100 64
```

脚本自动完成：kill 旧进程 -> 启动 vLLM -> 启动 ThunderAgent -> 运行 SWE-bench -> dump metrics -> 输出目录。

### Manual Step-by-Step

如需分步手动操作（调试/监控用）：

### Output Directory

All outputs are consolidated into a single directory:

```bash
OUTPUT_DIR="results/qwen3_8b_h200_lite"
mkdir -p $OUTPUT_DIR
```

This directory will contain:
- `{instance_id}/` — per-instance trajectory JSONs (from mini-SWE-agent)
- `step_profiles.csv` — per-step timing/token data (from ThunderAgent `--profile-dir`)
- `thunderagent_metrics.json` — ThunderAgent aggregate metrics (dumped post-run)
- `thunderagent_health.json` — ThunderAgent final health state
- `vllm_metrics.txt` — vLLM Prometheus metrics snapshot

### Step 1: Start vLLM

**先杀掉旧的 vLLM 进程（必须）：**

```bash
pkill -f "vllm serve"
# 等待进程完全退出
sleep 3
```

Single H200 141GB runs Qwen3-8B (~16GB BF16) with ~125GB for KV cache:

```bash
module load CUDA/12.6.0
vllm serve Qwen/Qwen3-8B \
  --port 8000 \
  --enable-prompt-tokens-details \
  2>&1 | tee $OUTPUT_DIR/vllm.log
```

参数说明：

- `--enable-prompt-tokens-details`: **必须**。启用后 vLLM 会在每个 API response 的 `usage.prompt_tokens_details.cached_tokens` 中返回 KV cache 命中的 token 数，ThunderAgent 据此计算 per-step 和 per-program 的 token-level KV cache hit rate，写入 `step_profiles.csv`。不加此参数则 `cached_tokens` 和 `kv_hit_rate` 列始终为空。
- `--enable-prefix-caching`: vLLM 0.19.0 默认已开启（`CacheConfig.enable_prefix_caching = True`），无需手动添加。
- `--enable-force-include-usage`: 实测不加也能正常返回 usage 字段，无需手动添加。

Wait until vLLM reports ready:

```bash
curl http://localhost:8000/health
```

### Step 2: Start ThunderAgent

**先杀掉旧的 ThunderAgent 进程（必须）：**

```bash
pkill -f "ThunderAgent"
# 等待进程完全退出
sleep 2
```

In another terminal:

```bash
cd /home/jiakunfan/ThunderAgent
module load CUDA/12.6.0

OUTPUT_DIR="results/qwen3_8b_h200_lite"

python -m ThunderAgent \
  --backend-type vllm \
  --backends http://localhost:8000 \
  --port 9000 \
  --metrics \
  --profile \
  --profile-dir $OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/thunderagent.log
```

Wait until ThunderAgent reports ready:

```bash
curl http://localhost:9000/health
```

### Step 3: Run SWE-bench Evaluation

In another terminal:

```bash
cd /home/jiakunfan/ThunderAgent

module load CUDA/12.6.0
module load apptainer

# 确保环境变量已设置
export APPTAINER_SANDBOX_DIR="/localscratch-nvme/${SLURM_JOB_ID}/apptainer_sandbox"
export APPTAINER_CACHEDIR="/projects/pearl/Storage_Jiakun/apptainer_cache"

OUTPUT_DIR="results/qwen3_8b_h200_lite"

mini-extra swebench \
  --subset lite \
  --split test \
  --workers 64 \
  -o $OUTPUT_DIR \
  --config examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen3_8b.yaml
```

Notes on `--workers`:
- H200 141GB with Qwen3-8B BF16 leaves ~125GB for KV cache
- 64 concurrent workers 充分利用 H200 的大显存和高带宽
- ThunderAgent 会自动管理 KV cache 容量，超出时自动 pause/resume 程序
- 可根据实际 GPU 利用率适当增减

### Step 4: Dump Final Metrics

After evaluation completes, save the final metrics snapshot:

```bash
OUTPUT_DIR="results/qwen3_8b_h200_lite"

curl -s http://localhost:9000/metrics | python3 -m json.tool \
  > $OUTPUT_DIR/thunderagent_metrics.json

curl -s http://localhost:9000/health | python3 -m json.tool \
  > $OUTPUT_DIR/thunderagent_health.json

curl -s http://localhost:8000/metrics \
  > $OUTPUT_DIR/vllm_metrics.txt
```

### Output Directory Structure

After a complete run, `$OUTPUT_DIR` contains:

```
results/qwen3_8b_h200_lite/
├── astropy__astropy-14182/
│   └── astropy__astropy-14182.traj.json    # instance trajectory
├── django__django-10914/
│   └── django__django-10914.traj.json
├── ...                                     # 300 instance directories
├── step_profiles.csv                       # per-step timing & token data (ThunderAgent)
├── vllm.log                                # vLLM stdout/stderr log
├── thunderagent.log                        # ThunderAgent stdout/stderr log
├── thunderagent_metrics.json               # ThunderAgent aggregate metrics
├── thunderagent_health.json                # ThunderAgent final health state
└── vllm_metrics.txt                        # vLLM Prometheus metrics snapshot
```

### Config File

`examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen3_8b.yaml`

Key differences from default `swebench.yaml`:

| Parameter | swebench.yaml (GLM-4.6) | swebench_qwen3_8b.yaml |
|-----------|------------------------|----------------------|
| `model_name` | `zai-org/GLM-4.6-FP8` | `Qwen/Qwen3-8B` |
| `base_url` | `http://localhost:8000/v1` | `http://localhost:9000/v1` |
| `environment_class` | `docker` | `singularity` |
| `executable` | (default: docker) | `apptainer` |

## Monitoring & Metrics

### Runtime Monitoring

During evaluation, check progress in real-time:

```bash
# ThunderAgent status (active programs, reasoning/acting/paused counts)
curl -s http://localhost:9000/health | python3 -m json.tool

# ThunderAgent metrics (KV cache, prefix cache, request stats)
curl -s http://localhost:9000/metrics | python3 -m json.tool

# Per-program state (tokens, steps, status)
curl -s http://localhost:9000/programs | python3 -m json.tool

# Per-program timing profiles (averages)
curl -s http://localhost:9000/profiles | python3 -m json.tool

# vLLM Prometheus metrics (prefill/decode tokens, cache hit rate, etc.)
curl -s http://localhost:8000/metrics | grep -E "vllm:(prompt_tokens|generation_tokens|prefix_cache|num_requests|kv_cache)"
```

### step_profiles.csv

Written in real-time during the run. Each row is one completed inference step:

| Column | Description |
|--------|-------------|
| `program_id` | SWE-bench instance identifier |
| `step_id` | Step number within the program |
| `prefill_s` | Prefill latency (seconds) |
| `decode_s` | Decode latency (seconds) |
| `pause_s` | Time paused waiting for KV cache capacity (seconds) |
| `tool_call_s` | Time between response end and next request arrival (seconds) |
| `prompt_tokens` | Prompt token count |
| `completion_tokens` | Generated token count |
| `cached_tokens` | KV cache hit token count |
| `kv_hit_rate` | `cached_tokens / prompt_tokens` |
| `completed_at` | Unix timestamp of step completion |

### Key Metrics Explained

| Metric | Endpoint | Meaning |
|--------|----------|---------|
| `active_program_tokens` | `/metrics` | Total KV cache tokens used by all active programs |
| `active_program_tokens_ratio` | `/metrics` | KV cache utilization (0-1) |
| `capacity_overflow` | `/metrics` | Times KV cache overflowed (should be 0) |
| `num_preemptions` | `/metrics` | Times programs were paused/resumed due to cache pressure |
| `prefix_cache_hit_rate` | `/metrics` | Block-level prefix cache hit rate (see below) |
| `prompt_tokens_by_source{local_cache_hit}` | `:8000/metrics` | **Token-level** KV cache reuse count (more accurate than block-level) |
| `prompt_tokens_total` | `/metrics` | Total prompt tokens processed (including cache hits) |
| `generation_tokens_total` | `/metrics` | Total tokens generated |
| `requests_completed` | `/metrics` | Total vLLM inference requests completed |
| `programs_count` | `/health` | Number of active programs |
| `reasoning_count` / `acting_count` | `/health` | Programs in reasoning vs acting phase |

### Prefix Cache Hit Rate

Two metrics measure cache reuse:

- **Block-level hit rate** (`prefix_cache_hit_rate` in `/metrics`): measures how often vLLM's radix tree block lookups hit cached KV blocks. This counts every internal engine iteration query, so it's diluted.
- **Token-level reuse rate** (from `:8000/metrics`): `prompt_tokens_by_source{source="local_cache_hit"}` / `prompt_tokens_total`. This measures the actual fraction of prompt tokens served from cache without recomputation. This is the more meaningful metric.

H200 的更大 KV cache 容量预期会带来更高的 cache hit rate 和更少的 preemption。

## Known Issues

### Qwen3-8B context window limitation

Qwen3-8B maximum context length is 40960 tokens. Some SWE-bench instances have very long problem statements that quickly exceed this limit, causing `ContextLengthExceeded` errors (~37% of Lite instances in testing). GLM-4.6 has 128k context and does not have this issue.

### Apptainer image pull failures on concurrent builds

When many workers build sandboxes simultaneously, network bandwidth can be saturated causing `conveyor failed to get: unexpected end of JSON input` errors. The code retries 3 times. Pre-building sandboxes with `scripts/pull_swebench_sandboxes.sh` avoids this entirely. 64 workers 并发时这个问题更容易触发，强烈建议提前构建 sandbox。

### Model behavior

Qwen3-8B tends to submit fixes after very few steps (avg ~6 steps) without adequate testing or verification, compared to larger models like GLM-4.7 (avg ~33 steps) which follow the recommended workflow more closely.

## Sandbox Caching Mechanism

The modified `singularity.py` supports sandbox reuse:

1. **Cache hit**: If `APPTAINER_SANDBOX_DIR/{instance_id}/` exists, use it directly (skip build)
2. **Cache miss**: Build sandbox as usual, then copy to cache directory for future runs
3. **Cleanup**: Cached sandboxes are NOT deleted after instance completion; only newly built (uncached) sandboxes are cleaned up

This means the first run builds and caches, subsequent runs reuse cached sandboxes instantly.
