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

ERAWAN may reject a 32-task array with `QOSMaxSubmitJobPerUserLimit`. The safe
large-run pattern is:

- Keep the logical dataset at 32 shards by setting `SHARD_COUNT=32`.
- Submit only two array tasks at a time, because this has been accepted by the
  current QOS.
- Inspect the first two large shards before submitting the remaining waves.
- Do not run the full merge/train step until all shard files `0` through `31`
  exist, unless intentionally training on a partial dataset.

Submit only the first two large shards:

```bash
cd ~/ptcg_abc
JOB=$(
  SHARD_COUNT=32 \
  GAMES_PER_SHARD=1000 \
  MAX_STEPS=600 \
  TOP_K=4 \
  ROLLOUT_STEPS=18 \
  sbatch --parsable --array=0-1 scripts/slurm/phase5_search_data_array.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_search_large_job_0_1.txt
```

Monitor the first wave:

```bash
cd ~/ptcg_abc
JOB=$(cat experiments/rl/phase5_search/latest_search_large_job_0_1.txt)
squeue -j "$JOB"
tail -n 80 experiments/rl/slurm-${JOB}-0-phase5-search.out
tail -n 80 experiments/rl/slurm-${JOB}-1-phase5-search.out
tail -n 80 experiments/rl/slurm-${JOB}-0-phase5-search.err
tail -n 80 experiments/rl/slurm-${JOB}-1-phase5-search.err
```

After the first wave finishes, inspect the summaries and shard sizes:

```bash
cat experiments/rl/phase5_search/summaries/phase5_search_summary_shard-0.json
cat experiments/rl/phase5_search/summaries/phase5_search_summary_shard-1.json
ls -lh data/datasets/rl/phase5_search/shards/phase5_search_decisions_shard-0.jsonl
ls -lh data/datasets/rl/phase5_search/shards/phase5_search_decisions_shard-1.jsonl
```

Pass gate for each large shard:

- `games_started` is `1000`.
- `errors` is `0`.
- `probe_errors` is `0` or low enough to inspect manually.
- `decisions` is nonzero.
- `searched_decisions` is nonzero.
- `changed_decisions` is nonzero.
- The decision and trace JSONL files are nonempty.

If the first wave passes, submit the remaining waves two shards at a time. Keep
`SHARD_COUNT=32` every time:

```bash
cd ~/ptcg_abc

JOB=$(
  SHARD_COUNT=32 GAMES_PER_SHARD=1000 MAX_STEPS=600 TOP_K=4 ROLLOUT_STEPS=18 \
  sbatch --parsable --array=2-3 scripts/slurm/phase5_search_data_array.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_search_large_job_2_3.txt
```

Repeat for:

```text
--array=4-5
--array=6-7
--array=8-9
--array=10-11
--array=12-13
--array=14-15
--array=16-17
--array=18-19
--array=20-21
--array=22-23
--array=24-25
--array=26-27
--array=28-29
--array=30-31
```

Check completion before merging:

```bash
ls data/datasets/rl/phase5_search/shards/phase5_search_decisions_shard-*.jsonl | wc -l
ls experiments/rl/phase5_search/traces/phase5_search_traces_shard-*.jsonl | wc -l
```

Both counts should be `32`.

## 6. Merge Large Shards

After all 32 shards finish, merge the decision and trace JSONL files:

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
"$PY" -m ptcg_abc rl-merge-search-data \
  --input 'data/datasets/rl/phase5_search/shards/phase5_search_decisions_shard-*.jsonl' \
  --trace-input 'experiments/rl/phase5_search/traces/phase5_search_traces_shard-*.jsonl' \
  --output data/datasets/rl/phase5_search_decisions_merged.jsonl \
  --trace-output experiments/rl/phase5_search_traces_merged.jsonl \
  --manifest experiments/rl/phase5_search_merge_manifest.json
cat experiments/rl/phase5_search_merge_manifest.json
```

Pass gate:

- `decision_files` is `32`.
- `trace_files` is `32`.
- `decision_records` is nonzero.
- `trace_records` is nonzero.

## 7. Train Search-Distillation Model

After the large merge passes, run Torch BC/search distillation:

```bash
cd ~/ptcg_abc
JOB=$(
  BC_EPOCHS=2 \
  BC_LEARNING_RATE=0.02 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_merge_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_train_job.txt
```

If ERAWAN still rejects this with `QOSMaxCpuPerJobLimit`, retry with two CPUs:

```bash
JOB=$(
  BC_EPOCHS=2 \
  BC_LEARNING_RATE=0.02 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 --mem=32G scripts/slurm/phase5_merge_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_train_job.txt
```

### Partial 10-Shard Training

If stopping at 10 completed large shards, train a partial model before continuing.
This is useful for checking the training path and getting a first checkpoint. The
job below merges whatever shard files currently match the default shard glob, so
run it when only the intended large shards are present in
`data/datasets/rl/phase5_search/shards/`.

```bash
cd ~/ptcg_abc
JOB=$(
  BC_EPOCHS=2 \
  BC_LEARNING_RATE=0.02 \
  MERGED_DATASET=data/datasets/rl/phase5_search_decisions_10shards.jsonl \
  MERGED_TRACES=experiments/rl/phase5_search_traces_10shards.jsonl \
  MERGE_MANIFEST=experiments/rl/phase5_search_merge_manifest_10shards.json \
  BC_CHECKPOINT=models/rl/phase5_search_distill_10shards.pt \
  BC_MODEL=models/rl/phase5_search_distill_10shards.json \
  BC_REPORT=experiments/rl/phase5_search_distill_report_10shards.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_merge_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_train_job_10shards.txt
```

If the CPU QOS limit appears again, use the same command with
`--cpus-per-task=2 --mem=32G`.

Monitor:

```bash
JOB=$(cat experiments/rl/phase5_search/latest_train_job.txt)
squeue -j "$JOB"
tail -n 120 experiments/rl/slurm-${JOB}-phase5-train.out
tail -n 120 experiments/rl/slurm-${JOB}-phase5-train.err
cat experiments/rl/phase5_search_distill_report.json
```

Outputs:

- `data/datasets/rl/phase5_search_decisions_merged.jsonl`
- `experiments/rl/phase5_search_traces_merged.jsonl`
- `experiments/rl/phase5_search_merge_manifest.json`
- `models/rl/phase5_search_distill.pt`
- `models/rl/phase5_search_distill.json`
- `experiments/rl/phase5_search_distill_report.json`

## 8. Ready-To-Train Checklist

- Login-node smoke passes with changed decisions and zero probe errors.
- Two-shard SLURM smoke produces two nonempty decision shards and two nonempty trace shards.
- First large two-shard wave passes the large-shard summary gates.
- All 32 large decision shards and all 32 large trace shards exist before full merge/train.
- Large merge manifest reports `decision_files: 32` and `trace_files: 32`.
- Torch import/CUDA check inside the merge/train SLURM output reports the expected device.
- `rl-train-bc --backend torch` starts from the merged dataset and writes both checkpoint and exported JSON model.
