# Phase 5 ERAWAN Search-Distillation Runbook

This runbook prepares the Phase 5 vertical slice for large-scale ERAWAN training:
generate search-improved decision data with bounded one-turn root search, merge the
array shards, then train the Torch behavior-cloning/distillation model from the
merged dataset.

Phase 5 builds on the Phase 4 package and uses the existing 9-deck by 4-benchmark
matchup grid. It does not require integrating the full 13-deck pool before the first
large run.

## 1. Verify The Command Surface

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
"$PY" -m ptcg_abc --help | grep -E "rl-generate-search-data|rl-merge-search-data|rl-train-bc"
```

## 2. Login-Node Smoke

Keep this tiny. It verifies the Search API, hidden-state sampler, search traces,
search-improved `DecisionFrame` output, and safe bounded rollout termination.

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
"$PY" -m ptcg_abc rl-generate-search-data \
  --games 1 \
  --max-steps 60 \
  --top-k 4 \
  --rollout-steps 18 \
  --require-changed \
  --output data/datasets/rl/phase5_search_smoke.jsonl \
  --trace-output experiments/rl/phase5_search_smoke_traces.jsonl
```

Pass gate:

- `errors` is `0`.
- `probe_errors` is `0`.
- `truncated_rollouts` is `0` or low enough to inspect manually.
- `changed_decisions` is greater than `0`.

## 3. Small SLURM Array Smoke

Run two shards with two games each before spending a full allocation.

```bash
cd ~/ptcg_abc
JOB=$(
  GAMES_PER_SHARD=2 \
  MAX_STEPS=80 \
  TOP_K=4 \
  ROLLOUT_STEPS=18 \
  sbatch --parsable --array=0-1 --time=00:30:00 scripts/slurm/phase5_search_data_array.sbatch
)
mkdir -p experiments/rl/phase5_search
echo "$JOB" | tee experiments/rl/phase5_search/latest_search_smoke_job.txt
```

Inspect:

```bash
cd ~/ptcg_abc
JOB=$(cat experiments/rl/phase5_search/latest_search_smoke_job.txt)
squeue -j "$JOB"
tail -n 80 experiments/rl/slurm-${JOB}-0-phase5-search.out
tail -n 80 experiments/rl/slurm-${JOB}-1-phase5-search.out
cat experiments/rl/phase5_search/summaries/phase5_search_summary_shard-0.json
cat experiments/rl/phase5_search/summaries/phase5_search_summary_shard-1.json
```

## 4. Merge Smoke Shards

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
"$PY" -m ptcg_abc rl-merge-search-data \
  --input 'data/datasets/rl/phase5_search/shards/phase5_search_decisions_shard-*.jsonl' \
  --trace-input 'experiments/rl/phase5_search/traces/phase5_search_traces_shard-*.jsonl' \
  --output data/datasets/rl/phase5_search_decisions_merged.jsonl \
  --trace-output experiments/rl/phase5_search_traces_merged.jsonl \
  --manifest experiments/rl/phase5_search_merge_manifest.json
```

Pass gate: the manifest shows every expected shard and nonzero `decision_records`.

## 5. Large Search-Data Array

Recommended first large run:

- 32 shards.
- 1,000 games per shard.
- `max_steps=600`.
- `top_k=4`.
- `rollout_steps=18`.

```bash
cd ~/ptcg_abc
JOB=$(
  GAMES_PER_SHARD=1000 \
  MAX_STEPS=600 \
  TOP_K=4 \
  ROLLOUT_STEPS=18 \
  sbatch --parsable --array=0-31 scripts/slurm/phase5_search_data_array.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_search_large_job.txt
```

Monitor:

```bash
cd ~/ptcg_abc
JOB=$(cat experiments/rl/phase5_search/latest_search_large_job.txt)
squeue -j "$JOB"
tail -n 80 experiments/rl/slurm-${JOB}-0-phase5-search.out
```

## 6. Merge And Train

After all shards finish, merge and run Torch BC/search distillation:

```bash
cd ~/ptcg_abc
JOB=$(
  BC_EPOCHS=2 \
  BC_LEARNING_RATE=0.02 \
  sbatch --parsable --gres=gpu:1 scripts/slurm/phase5_merge_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_train_job.txt
```

Outputs:

- `data/datasets/rl/phase5_search_decisions_merged.jsonl`
- `experiments/rl/phase5_search_traces_merged.jsonl`
- `experiments/rl/phase5_search_merge_manifest.json`
- `models/rl/phase5_search_distill.pt`
- `models/rl/phase5_search_distill.json`
- `experiments/rl/phase5_search_distill_report.json`

## 7. Ready-To-Train Checklist

- Login-node smoke passes with changed decisions and zero probe errors.
- Two-shard SLURM smoke produces two nonempty decision shards and two nonempty trace shards.
- Merge manifest reports the expected shard count.
- Torch import/CUDA check inside the merge/train SLURM output reports the expected device.
- `rl-train-bc --backend torch` starts from the merged dataset and writes both checkpoint and exported JSON model.
