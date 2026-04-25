#!/usr/bin/env python3
"""
Comprehensive statistical analysis of SWE-bench experiments:
  ThunderAgent vs Baseline vLLM (both gpu_memory_utilization=0.55, KV=200,880 tokens)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
THUNDER_DIR = Path("/home/jiakunfan/ThunderAgent/results/qwen3_8b_lite_32_0423_thunder")
BASELINE_DIR = Path("/home/jiakunfan/ThunderAgent/results/qwen3_8b_lite_32_0423_vllm")
STEP_PROFILES_CSV = THUNDER_DIR / "step_profiles.csv"

CAT_THRESHOLD = 0.05  # "catastrophic" kv_hit_rate threshold


def fmt(series_or_val, decimals=2):
    """Format a number or describe a series."""
    if isinstance(series_or_val, (pd.Series, np.ndarray)):
        s = pd.Series(series_or_val).dropna()
        return {
            "count": int(len(s)),
            "mean": round(float(s.mean()), decimals),
            "std": round(float(s.std()), decimals),
            "median": round(float(s.median()), decimals),
            "p10": round(float(s.quantile(0.10)), decimals),
            "p25": round(float(s.quantile(0.25)), decimals),
            "p75": round(float(s.quantile(0.75)), decimals),
            "p90": round(float(s.quantile(0.90)), decimals),
            "p99": round(float(s.quantile(0.99)), decimals),
            "min": round(float(s.min()), decimals),
            "max": round(float(s.max()), decimals),
        }
    return round(float(series_or_val), decimals)


def print_stats(label, stats_dict, indent=2):
    """Pretty-print a stats dictionary."""
    pad = " " * indent
    if isinstance(stats_dict, dict) and "mean" in stats_dict:
        print(f"{pad}{label}:")
        print(f"{pad}  n={stats_dict['count']}  mean={stats_dict['mean']}  std={stats_dict['std']}")
        print(f"{pad}  median={stats_dict['median']}  p10={stats_dict['p10']}  p25={stats_dict['p25']}  p75={stats_dict['p75']}")
        print(f"{pad}  p90={stats_dict['p90']}  p99={stats_dict['p99']}")
        print(f"{pad}  min={stats_dict['min']}  max={stats_dict['max']}")
    else:
        print(f"{pad}{label}: {stats_dict}")


# ══════════════════════════════════════════════════════════════════════
# PART 1: ThunderAgent Analysis (from step_profiles.csv)
# ══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("PART 1: THUNDERAGENT ANALYSIS (step_profiles.csv)")
print("=" * 80)

df_thunder = pd.read_csv(STEP_PROFILES_CSV)
print(f"\nTotal rows in step_profiles.csv: {len(df_thunder)}")
print(f"Columns: {df_thunder.columns.tolist()}")
print(f"Unique programs (tasks): {df_thunder['program_id'].nunique()}")

# ── 1a. Per-step latency stats ────────────────────────────────────────
print("\n── 1a. Per-step Latency Statistics (Thunder) ──")
for col in ["prefill_s", "decode_s", "pause_s", "tool_call_s"]:
    s = fmt(df_thunder[col])
    print_stats(col, s)

# ── 1b. Per-step token stats ──────────────────────────────────────────
print("\n── 1b. Per-step Token Statistics (Thunder) ──")
for col in ["prompt_tokens", "completion_tokens"]:
    s = fmt(df_thunder[col])
    print_stats(col, s)

# ── 1c. KV hit rate ───────────────────────────────────────────────────
print("\n── 1c. KV Hit Rate Statistics (Thunder) ──")
kvhr = df_thunder["kv_hit_rate"].dropna()
print_stats("kv_hit_rate", fmt(kvhr))

cat_steps_t = (kvhr < CAT_THRESHOLD).sum()
print(f"\n  Catastrophic steps (kv_hit_rate < {CAT_THRESHOLD}): {cat_steps_t} / {len(kvhr)} "
      f"({100*cat_steps_t/len(kvhr):.1f}%)")

# Breakdown by range
print("\n  KV hit rate distribution:")
bins = [0, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 1.01]
labels = ["<5%", "5-25%", "25-50%", "50-75%", "75-90%", "90-95%", "95-100%"]
kv_binned = pd.cut(kvhr, bins=bins, labels=labels, right=False)
for label in labels:
    cnt = (kv_binned == label).sum()
    pct = 100 * cnt / len(kvhr)
    print(f"    {label:>8s}: {cnt:5d} steps ({pct:5.1f}%)")

# ── 1d. Pause analysis ────────────────────────────────────────────────
print("\n── 1d. Pause Duration Analysis (Thunder) ──")
pause = df_thunder["pause_s"]
print(f"  Steps with pause_s > 0.001s:   {(pause > 0.001).sum():5d} ({100*(pause > 0.001).mean():.1f}%)")
print(f"  Steps with pause_s > 0.1s:     {(pause > 0.1).sum():5d} ({100*(pause > 0.1).mean():.1f}%)")
print(f"  Steps with pause_s > 1s:       {(pause > 1).sum():5d} ({100*(pause > 1).mean():.1f}%)")
print(f"  Steps with pause_s > 10s:      {(pause > 10).sum():5d} ({100*(pause > 10).mean():.1f}%)")
print(f"  Steps with pause_s > 60s:      {(pause > 60).sum():5d} ({100*(pause > 60).mean():.1f}%)")
print(f"  Steps with pause_s > 300s:     {(pause > 300).sum():5d} ({100*(pause > 300).mean():.1f}%)")
print(f"  Steps with pause_s > 600s:     {(pause > 600).sum():5d} ({100*(pause > 600).mean():.1f}%)")
print_stats("pause_s (all)", fmt(pause))
print_stats("pause_s (>0.001 only)", fmt(pause[pause > 0.001]))

# ── 1e. Total wall-clock time ─────────────────────────────────────────
print("\n── 1e. Wall-Clock Time (Thunder) ──")
t_min = df_thunder["completed_at"].min()
t_max = df_thunder["completed_at"].max()
wall_s = t_max - t_min
wall_m = wall_s / 60
wall_h = wall_s / 3600
print(f"  First completed_at: {t_min:.2f}")
print(f"  Last  completed_at: {t_max:.2f}")
print(f"  Wall-clock: {wall_s:.1f}s = {wall_m:.1f} min = {wall_h:.2f} hours")

# ── 1f. Per-task stats ────────────────────────────────────────────────
print("\n── 1f. Steps per Task (Thunder) ──")
steps_per_task = df_thunder.groupby("program_id").size()
print_stats("steps_per_task", fmt(steps_per_task))

# ── 1g. Aggregate token totals ────────────────────────────────────────
print("\n── 1g. Aggregate Token Totals (Thunder, from CSV) ──")
total_prompt_t = df_thunder["prompt_tokens"].sum()
total_comp_t = df_thunder["completion_tokens"].sum()
total_cached_t = df_thunder["cached_tokens"].sum()
print(f"  Total prompt_tokens:       {total_prompt_t:>12,}")
print(f"  Total completion_tokens:   {total_comp_t:>12,}")
print(f"  Total cached_tokens:       {total_cached_t:>12,}")
print(f"  Overall cache hit rate:    {total_cached_t/total_prompt_t*100:.2f}%")
compute_t = total_prompt_t - total_cached_t
print(f"  Cache-compute tokens:      {compute_t:>12,}")


# ══════════════════════════════════════════════════════════════════════
# PART 2: Baseline vLLM Analysis (from traj.json files)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 2: BASELINE vLLM ANALYSIS (traj.json files)")
print("=" * 80)

baseline_rows = []

for instance_dir in sorted(BASELINE_DIR.iterdir()):
    if not instance_dir.is_dir():
        continue
    instance_id = instance_dir.name
    traj_path = instance_dir / f"{instance_id}.traj.json"
    if not traj_path.exists():
        continue
    with open(traj_path) as f:
        data = json.load(f)
    msgs = data.get("messages", [])
    step_idx = 0
    for msg in msgs:
        if msg.get("role") != "assistant" or not msg.get("extra"):
            continue
        step_idx += 1
        resp = msg["extra"]["response"]
        usage = resp.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        comp_tok = usage.get("completion_tokens", 0)
        cached_tok = 0
        ptd = usage.get("prompt_tokens_details")
        if ptd and ptd.get("cached_tokens") is not None:
            cached_tok = ptd["cached_tokens"]
        created = resp.get("created", 0)

        baseline_rows.append({
            "instance_id": instance_id,
            "step_id": step_idx,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "cached_tokens": cached_tok,
            "created": created,
        })

df_baseline = pd.DataFrame(baseline_rows)
print(f"\nTotal assistant steps extracted: {len(df_baseline)}")
print(f"Unique instances: {df_baseline['instance_id'].nunique()}")

# ── 2a. KV hit rate per step ──────────────────────────────────────────
print("\n── 2a. KV Hit Rate Statistics (Baseline) ──")
df_baseline["kv_hit_rate"] = df_baseline["cached_tokens"] / df_baseline["prompt_tokens"].replace(0, np.nan)
kvhr_b = df_baseline["kv_hit_rate"].dropna()
print_stats("kv_hit_rate", fmt(kvhr_b))

cat_steps_b = (kvhr_b < CAT_THRESHOLD).sum()
print(f"\n  Catastrophic steps (kv_hit_rate < {CAT_THRESHOLD}): {cat_steps_b} / {len(kvhr_b)} "
      f"({100*cat_steps_b/len(kvhr_b):.1f}%)")

# KV hit rate distribution
print("\n  KV hit rate distribution:")
kv_binned_b = pd.cut(kvhr_b, bins=bins, labels=labels, right=False)
for label in labels:
    cnt = (kv_binned_b == label).sum()
    pct = 100 * cnt / len(kvhr_b)
    print(f"    {label:>8s}: {cnt:5d} steps ({pct:5.1f}%)")

# ── 2b. Per-step token stats ──────────────────────────────────────────
print("\n── 2b. Per-step Token Statistics (Baseline) ──")
for col in ["prompt_tokens", "completion_tokens"]:
    s = fmt(df_baseline[col])
    print_stats(col, s)

# ── 2c. Timing estimation from created timestamps ─────────────────────
print("\n── 2c. Timing Estimation from `created` timestamps (Baseline) ──")
# For each step, compute inter-step time (created[i] - created[i-1]) per instance
df_baseline = df_baseline.sort_values(["instance_id", "step_id"]).reset_index(drop=True)
df_baseline["prev_created"] = df_baseline.groupby("instance_id")["created"].shift(1)
df_baseline["inter_step_s"] = df_baseline["created"] - df_baseline["prev_created"]
inter_step_b = df_baseline["inter_step_s"].dropna()
# Remove first step of each task (no prev)
inter_step_b = inter_step_b[inter_step_b > 0]  # sanity check
print_stats("inter_step_s (time between consecutive steps)", fmt(inter_step_b))

# Wall-clock
b_min = df_baseline["created"].min()
b_max = df_baseline["created"].max()
wall_s_b = b_max - b_min
print(f"\n  First created: {b_min:.2f}")
print(f"  Last  created: {b_max:.2f}")
print(f"  Wall-clock: {wall_s_b:.1f}s = {wall_s_b/60:.1f} min = {wall_s_b/3600:.2f} hours")

# ── 2d. Steps per task ────────────────────────────────────────────────
print("\n── 2d. Steps per Task (Baseline) ──")
steps_per_task_b = df_baseline.groupby("instance_id").size()
print_stats("steps_per_task", fmt(steps_per_task_b))

# ── 2e. Aggregate token totals ────────────────────────────────────────
print("\n── 2e. Aggregate Token Totals (Baseline) ──")
total_prompt_b = df_baseline["prompt_tokens"].sum()
total_comp_b = df_baseline["completion_tokens"].sum()
total_cached_b = df_baseline["cached_tokens"].sum()
print(f"  Total prompt_tokens:       {total_prompt_b:>12,}")
print(f"  Total completion_tokens:   {total_comp_b:>12,}")
print(f"  Total cached_tokens:       {total_cached_b:>12,}")
print(f"  Overall cache hit rate:    {total_cached_b/total_prompt_b*100:.2f}%")
compute_b = total_prompt_b - total_cached_b
print(f"  Cache-compute tokens:      {compute_b:>12,}")

# Steps with cached_tokens == 0 vs >0
print(f"\n  Steps with cached_tokens == 0:   {(df_baseline['cached_tokens'] == 0).sum()}")
print(f"  Steps with cached_tokens == 112:  {(df_baseline['cached_tokens'] == 112).sum()}")
print(f"  Steps with cached_tokens > 112:   {(df_baseline['cached_tokens'] > 112).sum()}")


# ══════════════════════════════════════════════════════════════════════
# PART 3: Cross-Experiment Comparison
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 3: CROSS-EXPERIMENT COMPARISON")
print("=" * 80)

# Also load Thunder traj.json for a fair comparison of kv_hit_rate from traj
print("\n── 3a. Token-Level Cache Hit Rate ──")
# Thunder from CSV
overall_t = total_cached_t / total_prompt_t * 100
overall_b = total_cached_b / total_prompt_b * 100
print(f"  Thunder:  {total_cached_t:>12,} / {total_prompt_t:>12,} = {overall_t:.2f}%")
print(f"  Baseline: {total_cached_b:>12,} / {total_prompt_b:>12,} = {overall_b:.2f}%")
print(f"  Delta:    {overall_t - overall_b:+.2f} pp")

# Cache-compute comparison
print(f"\n  Cache-compute tokens (prompt - cached):")
print(f"    Thunder:  {compute_t:>12,}")
print(f"    Baseline: {compute_b:>12,}")
print(f"    Delta:    {compute_t - compute_b:+12,} ({(compute_t/compute_b - 1)*100:+.1f}%)")

# Prompt tokens comparison
print(f"\n  Total prompt tokens:")
print(f"    Thunder:  {total_prompt_t:>12,}")
print(f"    Baseline: {total_prompt_b:>12,}")
print(f"    Delta:    {total_prompt_t - total_prompt_b:+12,} ({(total_prompt_t/total_prompt_b - 1)*100:+.1f}%)")

# Gen tokens comparison
print(f"\n  Total generation tokens:")
print(f"    Thunder:  {total_comp_t:>12,}")
print(f"    Baseline: {total_comp_b:>12,}")
print(f"    Delta:    {total_comp_t - total_comp_b:+12,} ({(total_comp_t/total_comp_b - 1)*100:+.1f}%)")

print("\n── 3b. Catastrophic Steps ──")
print(f"  Thunder:  {cat_steps_t:5d} / {len(kvhr):5d} steps ({100*cat_steps_t/len(kvhr):.1f}%)")
print(f"  Baseline: {cat_steps_b:5d} / {len(kvhr_b):5d} steps ({100*cat_steps_b/len(kvhr_b):.1f}%)")

# For baseline, also count steps with cached_tokens == 0 separately
zero_cache_b = (df_baseline["cached_tokens"] == 0).sum()
print(f"  Baseline steps with cached_tokens=0: {zero_cache_b}")

print("\n── 3c. Steps per Task ──")
print_stats("Thunder steps/task", fmt(steps_per_task))
print()
print_stats("Baseline steps/task", fmt(steps_per_task_b))

print("\n── 3d. Per-Step KV Hit Rate Side-by-Side ──")
# Show percentiles side by side
percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print(f"  {'Percentile':>12s}  {'Thunder':>10s}  {'Baseline':>10s}")
for p in percentiles:
    tv = kvhr.quantile(p)
    bv = kvhr_b.quantile(p)
    print(f"  {p*100:>11.1f}%  {tv:>10.4f}  {bv:>10.4f}")

print(f"\n  {'Mean':>12s}  {kvhr.mean():>10.4f}  {kvhr_b.mean():>10.4f}")
print(f"  {'Median':>12s}  {kvhr.median():>10.4f}  {kvhr_b.median():>10.4f}")
print(f"  {'Std':>12s}  {kvhr.std():>10.4f}  {kvhr_b.std():>10.4f}")


# ══════════════════════════════════════════════════════════════════════
# PART 4: Per-Instance Level Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 4: PER-INSTANCE AGGREGATE COMPARISON")
print("=" * 80)

# Thunder per-instance from CSV
t_inst = df_thunder.groupby("program_id").agg(
    steps=("step_id", "count"),
    total_prompt=("prompt_tokens", "sum"),
    total_comp=("completion_tokens", "sum"),
    total_cached=("cached_tokens", "sum"),
    total_prefill=("prefill_s", "sum"),
    total_decode=("decode_s", "sum"),
    total_pause=("pause_s", "sum"),
).reset_index()
t_inst["kv_hit_rate"] = t_inst["total_cached"] / t_inst["total_prompt"]

# Baseline per-instance
b_inst = df_baseline.groupby("instance_id").agg(
    steps=("step_id", "count"),
    total_prompt=("prompt_tokens", "sum"),
    total_comp=("completion_tokens", "sum"),
    total_cached=("cached_tokens", "sum"),
).reset_index()
b_inst["kv_hit_rate"] = b_inst["total_cached"] / b_inst["total_prompt"]

print("\n── Per-Instance Aggregate KV Hit Rate ──")
print_stats("Thunder per-instance KV hit rate", fmt(t_inst["kv_hit_rate"]))
print()
print_stats("Baseline per-instance KV hit rate", fmt(b_inst["kv_hit_rate"]))

print("\n── Per-Instance Total Prompt Tokens ──")
print_stats("Thunder", fmt(t_inst["total_prompt"]))
print()
print_stats("Baseline", fmt(b_inst["total_prompt"]))

print("\n── Per-Instance Total Completion Tokens ──")
print_stats("Thunder", fmt(t_inst["total_comp"]))
print()
print_stats("Baseline", fmt(b_inst["total_comp"]))

print("\n── Per-Instance Steps ──")
print_stats("Thunder", fmt(t_inst["steps"]))
print()
print_stats("Baseline", fmt(b_inst["steps"]))

# Thunder time breakdown
print("\n── Per-Instance Time Breakdown (Thunder) ──")
t_inst["total_s"] = t_inst["total_prefill"] + t_inst["total_decode"] + t_inst["total_pause"]
t_inst["prefill_frac"] = t_inst["total_prefill"] / t_inst["total_s"]
t_inst["decode_frac"] = t_inst["total_decode"] / t_inst["total_s"]
t_inst["pause_frac"] = t_inst["total_pause"] / t_inst["total_s"]

print_stats("total_s per task", fmt(t_inst["total_s"]))
print()
print_stats("total_prefill_s per task", fmt(t_inst["total_prefill"]))
print()
print_stats("total_decode_s per task", fmt(t_inst["total_decode"]))
print()
print_stats("total_pause_s per task", fmt(t_inst["total_pause"]))
print()
print(f"  Mean prefill fraction: {t_inst['prefill_frac'].mean():.3f}")
print(f"  Mean decode fraction:  {t_inst['decode_frac'].mean():.3f}")
print(f"  Mean pause fraction:   {t_inst['pause_frac'].mean():.3f}")


# ══════════════════════════════════════════════════════════════════════
# PART 5: Preemption / Pause Analysis (Thunder)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 5: PREEMPTION / PAUSE ANALYSIS (Thunder)")
print("=" * 80)

# A "preemption" step is one where pause_s is significant
# The user mentioned 109 preemptions; let's see how pause correlates
print(f"\n  Total steps: {len(df_thunder)}")
print(f"  Steps with pause_s > 0.001s: {(pause > 0.001).sum()}")
print(f"  Steps with pause_s > 1s:     {(pause > 1).sum()}")
print(f"  Steps with pause_s > 10s:    {(pause > 10).sum()}")
print(f"  Steps with pause_s > 60s:    {(pause > 60).sum()}")
print(f"  Steps with pause_s > 300s:   {(pause > 300).sum()}")

# Per-instance preemption count
df_thunder["is_preempted"] = df_thunder["pause_s"] > 1.0
preempt_per_task = df_thunder.groupby("program_id")["is_preempted"].sum()
print(f"\n  Per-instance preemption count (pause > 1s):")
print_stats("preemptions per task", fmt(preempt_per_task))

# Does preemption cause low KV hit rate?
print("\n  KV hit rate for preempted vs non-preempted steps:")
preempted_steps = df_thunder[df_thunder["is_preempted"] & df_thunder["kv_hit_rate"].notna()]
non_preempted_steps = df_thunder[~df_thunder["is_preempted"] & df_thunder["kv_hit_rate"].notna()]
print(f"    Preempted steps (pause > 1s):   n={len(preempted_steps)}, mean kv_hit_rate={preempted_steps['kv_hit_rate'].mean():.4f}")
print(f"    Non-preempted steps (pause <=1s): n={len(non_preempted_steps)}, mean kv_hit_rate={non_preempted_steps['kv_hit_rate'].mean():.4f}")

# Check if step after a long pause has low KV hit rate
df_thunder_sorted = df_thunder.sort_values(["program_id", "completed_at"]).reset_index(drop=True)
df_thunder_sorted["prev_pause"] = df_thunder_sorted.groupby("program_id")["pause_s"].shift(1)
df_thunder_sorted["after_preemption"] = df_thunder_sorted["prev_pause"] > 1.0
after_preempt = df_thunder_sorted[df_thunder_sorted["after_preemption"] & df_thunder_sorted["kv_hit_rate"].notna()]
not_after_preempt = df_thunder_sorted[~df_thunder_sorted["after_preemption"] & df_thunder_sorted["kv_hit_rate"].notna()]
print(f"\n  KV hit rate for steps AFTER a preemption vs not:")
print(f"    After preemption:     n={len(after_preempt)}, mean={after_preempt['kv_hit_rate'].mean():.4f}, median={after_preempt['kv_hit_rate'].median():.4f}")
print(f"    Not after preemption: n={len(not_after_preempt)}, mean={not_after_preempt['kv_hit_rate'].mean():.4f}, median={not_after_preempt['kv_hit_rate'].median():.4f}")


# ══════════════════════════════════════════════════════════════════════
# PART 6: Summary Table
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 6: SUMMARY TABLE")
print("=" * 80)
print()
print(f"  {'Metric':<40s}  {'Thunder':>14s}  {'Baseline':>14s}  {'Delta':>14s}")
print(f"  {'-'*40}  {'-'*14}  {'-'*14}  {'-'*14}")

def row(label, tv, bv, delta=None):
    """Print a comparison row. tv/bv can be ints or pre-formatted strings."""
    if isinstance(tv, (int, np.integer)):
        if delta is None:
            delta = int(tv) - int(bv)
        print(f"  {label:<40s}  {int(tv):>14,}  {int(bv):>14,}  {int(delta):>+14,}")
    else:
        # tv and bv are strings; delta should be provided or computed
        if delta is None:
            try:
                delta = float(tv.replace(",","")) - float(bv.replace(",",""))
                delta = f"{delta:+.2f}"
            except:
                delta = "N/A"
        print(f"  {label:<40s}  {str(tv):>14s}  {str(bv):>14s}  {str(delta):>14s}")

row("Total steps", len(df_thunder), len(df_baseline))
row("Unique tasks", df_thunder["program_id"].nunique(), df_baseline["instance_id"].nunique())
row("Mean steps/task", f"{steps_per_task.mean():.1f}", f"{steps_per_task_b.mean():.1f}", f"{steps_per_task.mean()-steps_per_task_b.mean():+.1f}")
row("Total prompt tokens (M)", f"{total_prompt_t/1e6:.2f}", f"{total_prompt_b/1e6:.2f}", f"{(total_prompt_t-total_prompt_b)/1e6:+.2f}")
row("Total gen tokens (M)", f"{total_comp_t/1e6:.2f}", f"{total_comp_b/1e6:.2f}", f"{(total_comp_t-total_comp_b)/1e6:+.2f}")
row("Total cached tokens (M)", f"{total_cached_t/1e6:.2f}", f"{total_cached_b/1e6:.2f}", f"{(total_cached_t-total_cached_b)/1e6:+.2f}")
row("Cache-compute tokens (M)", f"{compute_t/1e6:.2f}", f"{compute_b/1e6:.2f}", f"{(compute_t-compute_b)/1e6:+.2f}")
row("Overall cache hit rate (%)", f"{overall_t:.2f}", f"{overall_b:.2f}", f"{overall_t-overall_b:+.2f}")
row("Mean per-step kv_hit_rate", f"{kvhr.mean():.4f}", f"{kvhr_b.mean():.4f}", f"{kvhr.mean()-kvhr_b.mean():+.4f}")
row("Median per-step kv_hit_rate", f"{kvhr.median():.4f}", f"{kvhr_b.median():.4f}", f"{kvhr.median()-kvhr_b.median():+.4f}")
row("P90 per-step kv_hit_rate", f"{kvhr.quantile(0.9):.4f}", f"{kvhr_b.quantile(0.9):.4f}", f"{kvhr.quantile(0.9)-kvhr_b.quantile(0.9):+.4f}")
row(f"Catastrophic steps (<{CAT_THRESHOLD})", cat_steps_t, cat_steps_b, cat_steps_t - cat_steps_b)
pct_cat_t = 100 * cat_steps_t / len(kvhr)
pct_cat_b = 100 * cat_steps_b / len(kvhr_b)
row("Catastrophic step %", f"{pct_cat_t:.1f}%", f"{pct_cat_b:.1f}%", f"{pct_cat_t-pct_cat_b:+.1f}%")

# Known aggregate metrics
print()
print("  ── Known Aggregate Metrics (from experiment logs) ──")
print(f"  {'Metric':<40s}  {'Thunder':>14s}  {'Baseline':>14s}")
print(f"  {'-'*40}  {'-'*14}  {'-'*14}")
print(f"  {'Preemptions':<40s}  {'109':>14s}  {'246':>14s}")
print(f"  {'Prompt tokens (M) [known]':<40s}  {'16.25':>14s}  {'20.59':>14s}")
print(f"  {'Gen tokens (M) [known]':<40s}  {'2.64':>14s}  {'2.87':>14s}")
print(f"  {'Cache hit (M) [known]':<40s}  {'11.14':>14s}  {'12.14':>14s}")
print(f"  {'Cache compute (M) [known]':<40s}  {'5.11':>14s}  {'8.45':>14s}")


# ══════════════════════════════════════════════════════════════════════
# PART 7: ThunderAgent Timing Deep-Dive
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 7: THUNDERAGENT TIMING DEEP-DIVE")
print("=" * 80)

print("\n── Prefill time vs prompt tokens correlation ──")
# Bin by prompt_tokens ranges
df_t_valid = df_thunder[df_thunder["kv_hit_rate"].notna()].copy()
pt_bins = [0, 1500, 2000, 3000, 5000, 10000, 50000]
pt_labels = ["<=1500", "1500-2k", "2k-3k", "3k-5k", "5k-10k", "10k-50k"]
df_t_valid["prompt_bin"] = pd.cut(df_t_valid["prompt_tokens"], bins=pt_bins, labels=pt_labels, right=True)
print(f"  {'Prompt range':>12s}  {'n':>6s}  {'mean_prefill':>13s}  {'mean_decode':>12s}  {'mean_pause':>11s}  {'mean_kv_hit':>12s}")
for label in pt_labels:
    subset = df_t_valid[df_t_valid["prompt_bin"] == label]
    if len(subset) == 0:
        continue
    print(f"  {label:>12s}  {len(subset):>6d}  {subset['prefill_s'].mean():>13.3f}  {subset['decode_s'].mean():>12.3f}  {subset['pause_s'].mean():>11.3f}  {subset['kv_hit_rate'].mean():>12.4f}")

# Per-step timing breakdown (Thunder only)
print("\n── Aggregate Time Budget (Thunder) ──")
total_prefill = df_thunder["prefill_s"].sum()
total_decode = df_thunder["decode_s"].sum()
total_pause = df_thunder["pause_s"].sum()
total_tool = df_thunder["tool_call_s"].sum()
total_all = total_prefill + total_decode + total_pause + total_tool
print(f"  Total prefill_s:   {total_prefill:>12.1f}s  ({100*total_prefill/total_all:.1f}%)")
print(f"  Total decode_s:    {total_decode:>12.1f}s  ({100*total_decode/total_all:.1f}%)")
print(f"  Total pause_s:     {total_pause:>12.1f}s  ({100*total_pause/total_all:.1f}%)")
print(f"  Total tool_call_s: {total_tool:>12.1f}s  ({100*total_tool/total_all:.1f}%)")
print(f"  Total:             {total_all:>12.1f}s")

# Average per-step
print(f"\n  Avg prefill per step:   {total_prefill/len(df_thunder):.3f}s")
print(f"  Avg decode per step:    {total_decode/len(df_thunder):.3f}s")
print(f"  Avg pause per step:     {total_pause/len(df_thunder):.3f}s")

# Throughput
print(f"\n── Throughput (Thunder) ──")
active_s = total_prefill + total_decode  # exclude pause
print(f"  Active inference time: {active_s:.1f}s = {active_s/3600:.2f} hours")
print(f"  Prompt tokens/sec (active): {total_prompt_t/active_s:.1f}")
print(f"  Completion tokens/sec (active): {total_comp_t/active_s:.1f}")
print(f"  Total tokens/sec (active): {(total_prompt_t+total_comp_t)/active_s:.1f}")
print(f"  Prompt tokens/sec (wall): {total_prompt_t/wall_s:.1f}")
print(f"  Completion tokens/sec (wall): {total_comp_t/wall_s:.1f}")


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
