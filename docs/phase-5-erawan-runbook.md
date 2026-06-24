# Phase 5 ERAWAN Search-Distillation Runbook

This is the operating runbook for the current Phase 5 vertical slice. The full
Phase 5 architecture, training, and evaluation map is in
`docs/phase-5-master-plan.md`.

This runbook prepares the search-distillation slice for large-scale ERAWAN
training: generate search-improved decision data with bounded one-turn root
search, merge the array shards, then train the Torch behavior-cloning/distillation
model from the merged dataset.

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
# Full run:
JOB=$(cat experiments/rl/phase5_search/latest_train_job.txt)

# Partial 10-shard run:
# JOB=$(cat experiments/rl/phase5_search/latest_train_job_10shards.txt)

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

## 8. Diagnose Search Distillation

After training a partial or full search-distilled model, run offline diagnostics
before spending more compute. This checks whether model accuracy is coming from
easy unchanged decisions or whether the model actually learned the
search-changed labels.

Always run diagnostics as a SLURM job. By default the job diagnoses the exported
JSON fallback model. For the 10-shard partial model:

```bash
cd ~/ptcg_abc
JOB=$(
  DATASET=data/datasets/rl/phase5_search_decisions_10shards.jsonl \
  MODEL=models/rl/phase5_search_distill_10shards.json \
  TRACE_INPUT=experiments/rl/phase5_search_traces_10shards.jsonl \
  REPORT_JSON=reports/phase5_search_distill_10shards_diagnostics.json \
  REPORT_MD=reports/phase5_search_distill_10shards_diagnostics.md \
  sbatch --parsable scripts/slurm/phase5_diagnose_search_distill_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_diag_10shards_job.txt
```

To diagnose the torch checkpoint from the same run, set `CHECKPOINT` and submit
the diagnostic job with a GPU:

```bash
JOB=$(
  DATASET=data/datasets/rl/phase5_search_decisions_10shards.jsonl \
  MODEL=models/rl/phase5_search_distill_10shards.json \
  CHECKPOINT=models/rl/phase5_search_distill_10shards.pt \
  TRACE_INPUT=experiments/rl/phase5_search_traces_10shards.jsonl \
  REPORT_JSON=reports/phase5_search_distill_10shards_checkpoint_diagnostics.json \
  REPORT_MD=reports/phase5_search_distill_10shards_checkpoint_diagnostics.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_diagnose_search_distill_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_diag_10shards_checkpoint_job.txt
```

For a full 32-shard model, use the merged full-run paths:

```bash
JOB=$(
  DATASET=data/datasets/rl/phase5_search_decisions_merged.jsonl \
  MODEL=models/rl/phase5_search_distill.json \
  TRACE_INPUT=experiments/rl/phase5_search_traces_merged.jsonl \
  REPORT_JSON=reports/phase5_search_distill_diagnostics.json \
  REPORT_MD=reports/phase5_search_distill_diagnostics.md \
  sbatch --parsable scripts/slurm/phase5_diagnose_search_distill_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_diag_full_job.txt
```

Monitor any diagnostics job:

```bash
JOB=$(cat experiments/rl/phase5_diag_10shards_job.txt)
squeue -j "$JOB"
tail -n 120 experiments/rl/slurm-${JOB}-phase5-diag.out
tail -n 120 experiments/rl/slurm-${JOB}-phase5-diag.err
```

Key fields to inspect:

- `search_changed.frames`: number of changed decisions available for learning.
- `search_changed.search_hit_rate`: how often the model selects a search label
  on changed decisions.
- `search_changed.baseline_hit_rate`: how often the model falls back toward the
  original rule choice on changed decisions.
- `search_changed.mean_model_search_minus_baseline_score`: positive means the
  model scores search labels above rule baseline labels.
- `trace.mean_search_minus_baseline_combined_score`: positive means the search
  traces themselves preferred search choices over baseline choices.

Do not start PPO from a checkpoint whose changed-decision diagnostics are poor.

## 9. Reweighted Search-Distillation Retrain

If diagnostics show that `search_changed.search_hit_rate` is poor and the model
still prefers baseline actions on changed decisions, retrain from the same shards
with changed decisions upweighted. Exclude direct rule-score features so the
exported JSON model cannot solve the majority class by simply copying the rule
baseline.

For the 10-shard partial dataset:

```bash
cd ~/ptcg_abc
JOB=$(
  BC_EPOCHS=2 \
  BC_LEARNING_RATE=0.02 \
  BC_CHANGED_WEIGHT=12 \
  BC_UNCHANGED_WEIGHT=1 \
  BC_EXCLUDE_FEATURES="rule_score rule_rank_inv" \
  MERGED_DATASET=data/datasets/rl/phase5_search_decisions_10shards_reweighted.jsonl \
  MERGED_TRACES=experiments/rl/phase5_search_traces_10shards_reweighted.jsonl \
  MERGE_MANIFEST=experiments/rl/phase5_search_merge_manifest_10shards_reweighted.json \
  BC_CHECKPOINT=models/rl/phase5_search_distill_10shards_changedw.pt \
  BC_MODEL=models/rl/phase5_search_distill_10shards_changedw.json \
  BC_REPORT=experiments/rl/phase5_search_distill_report_10shards_changedw.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_merge_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_train_job_10shards_changedw.txt
```

After it finishes, submit diagnostics against the reweighted model:

```bash
JOB=$(
  DATASET=data/datasets/rl/phase5_search_decisions_10shards_reweighted.jsonl \
  MODEL=models/rl/phase5_search_distill_10shards_changedw.json \
  TRACE_INPUT=experiments/rl/phase5_search_traces_10shards_reweighted.jsonl \
  REPORT_JSON=reports/phase5_search_distill_10shards_changedw_diagnostics.json \
  REPORT_MD=reports/phase5_search_distill_10shards_changedw_diagnostics.md \
  sbatch --parsable scripts/slurm/phase5_diagnose_search_distill_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_diag_10shards_changedw_job.txt
```

Then run the same 360-game `rl` and `hybrid` benchmark comparison before
generating more shards.

## 10. Pairwise All-Negative Changed-Decision Retrain

If the reweighted binary model starts learning changed decisions but often picks
third actions that are neither baseline nor search, train changed frames with a
pairwise all-negative objective:

```text
score(search_action) > score(every other legal action)
```

This is stronger than the earlier search-over-baseline pairwise run. Use it when
diagnostics show third-action drift, for example when the model scores the search
action above the baseline on average but still predicts a retreat, end turn, or
another non-search action.

Submit a separate 10-shard pairwise-all model:

```bash
cd ~/ptcg_abc
JOB=$(
  BC_EPOCHS=2 \
  BC_LEARNING_RATE=0.02 \
  BC_CHANGED_WEIGHT=6 \
  BC_UNCHANGED_WEIGHT=0.25 \
  BC_EXCLUDE_FEATURES="rule_score rule_rank_inv" \
  BC_PAIRWISE_CHANGED=1 \
  BC_PAIRWISE_MARGIN=1.0 \
  BC_PAIRWISE_NEGATIVES=all \
  MERGED_DATASET=data/datasets/rl/phase5_search_decisions_10shards_pairwise_all.jsonl \
  MERGED_TRACES=experiments/rl/phase5_search_traces_10shards_pairwise_all.jsonl \
  MERGE_MANIFEST=experiments/rl/phase5_search_merge_manifest_10shards_pairwise_all.json \
  BC_CHECKPOINT=models/rl/phase5_search_distill_10shards_pairwise_all.pt \
  BC_MODEL=models/rl/phase5_search_distill_10shards_pairwise_all.json \
  BC_REPORT=experiments/rl/phase5_search_distill_report_10shards_pairwise_all.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_merge_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_train_job_10shards_pairwise_all.txt
```

Submit diagnostics before battle evaluation:

```bash
JOB=$(
  DATASET=data/datasets/rl/phase5_search_decisions_10shards_pairwise_all.jsonl \
  MODEL=models/rl/phase5_search_distill_10shards_pairwise_all.json \
  TRACE_INPUT=experiments/rl/phase5_search_traces_10shards_pairwise_all.jsonl \
  REPORT_JSON=reports/phase5_search_distill_10shards_pairwise_all_diagnostics.json \
  REPORT_MD=reports/phase5_search_distill_10shards_pairwise_all_diagnostics.md \
  sbatch --parsable scripts/slurm/phase5_diagnose_search_distill_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_diag_10shards_pairwise_all_job.txt
```

If the exported JSON diagnostics are still poor, diagnose the torch checkpoint
before doing more linear-export retraining:

```bash
JOB=$(
  DATASET=data/datasets/rl/phase5_search_decisions_10shards_pairwise_all.jsonl \
  MODEL=models/rl/phase5_search_distill_10shards_pairwise_all.json \
  CHECKPOINT=models/rl/phase5_search_distill_10shards_pairwise_all.pt \
  TRACE_INPUT=experiments/rl/phase5_search_traces_10shards_pairwise_all.jsonl \
  REPORT_JSON=reports/phase5_search_distill_10shards_pairwise_all_checkpoint_diagnostics.json \
  REPORT_MD=reports/phase5_search_distill_10shards_pairwise_all_checkpoint_diagnostics.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_diagnose_search_distill_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_diag_10shards_pairwise_all_checkpoint_job.txt
```

Only run battle smoke if either JSON or checkpoint diagnostics show
`search_changed.search_hit_rate` improves without a large rise in third-action
drift. If only the checkpoint passes, the next implementation step is
torch-checkpoint inference for `rl-evaluate`; do not promote the JSON fallback.

If checkpoint diagnostics show near-constant model scores, the old mitigation
was to retrain with the action-residual checkpoint format. That format adds a
direct action-feature scoring path to the torch actor so it cannot collapse into
board-only tie scores as easily:

```bash
cd ~/ptcg_abc
JOB=$(
  BC_EPOCHS=2 \
  BC_LEARNING_RATE=0.005 \
  BC_CHANGED_WEIGHT=6 \
  BC_UNCHANGED_WEIGHT=0.25 \
  BC_EXCLUDE_FEATURES="rule_score rule_rank_inv" \
  BC_PAIRWISE_CHANGED=1 \
  BC_PAIRWISE_MARGIN=1.0 \
  BC_PAIRWISE_NEGATIVES=all \
  MERGED_DATASET=data/datasets/rl/phase5_search_decisions_10shards_residual.jsonl \
  MERGED_TRACES=experiments/rl/phase5_search_traces_10shards_residual.jsonl \
  MERGE_MANIFEST=experiments/rl/phase5_search_merge_manifest_10shards_residual.json \
  BC_CHECKPOINT=models/rl/phase5_search_distill_10shards_residual.pt \
  BC_MODEL=models/rl/phase5_search_distill_10shards_residual.json \
  BC_REPORT=experiments/rl/phase5_search_distill_report_10shards_residual.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_merge_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search/latest_train_job_10shards_residual.txt
```

Then diagnose the residual checkpoint as a job:

```bash
JOB=$(
  DATASET=data/datasets/rl/phase5_search_decisions_10shards_residual.jsonl \
  MODEL=models/rl/phase5_search_distill_10shards_residual.json \
  CHECKPOINT=models/rl/phase5_search_distill_10shards_residual.pt \
  TRACE_INPUT=experiments/rl/phase5_search_traces_10shards_residual.jsonl \
  REPORT_JSON=reports/phase5_search_distill_10shards_residual_checkpoint_diagnostics.json \
  REPORT_MD=reports/phase5_search_distill_10shards_residual_checkpoint_diagnostics.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_diagnose_search_distill_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_diag_10shards_residual_checkpoint_job.txt
```

As of the adapter/encoder pivot, do not run another large residual retrain before
the symbolic Phase 5 input stack is wired into training. Keep the commands above
for reproducibility, but treat them as superseded by the real Phase 5
adapter/encoder track below.

## 11. Real Phase 5 Adapter/Encoder Track

The next implementation track follows `docs/ptcg_rl_strategy_recommendation.md`
more directly:

- Canonical `StateAdapter` and `LegalOptionAdapter`.
- Minimal `GameMemory` and `BeliefState`.
- Symbolic global/entity/legal-action tensors.
- AlphaStar-style policy model with a transformer entity/state core and
  autoregressive previous-action context for turn-level action sequences.

Do not submit more large search-distillation jobs until this symbolic model has
a dataset conversion and a small supervised smoke train.

Local smoke after pulling the adapter/encoder slice:

```bash
"$PY" -m unittest tests.test_rl_phase5_adapters
```

ERAWAN smoke after pulling the adapter/encoder slice:

```bash
cd ~/ptcg_abc
export PYTHONPATH="$PWD/src"
PY="$PWD/.conda_ptcg/bin/python"
"$PY" -m unittest tests.test_rl_phase5_adapters
```

The next missing command is a symbolic dataset builder/trainer. Add that before
resuming large-scale training.

## 12. Ready-To-Train Checklist

- Adapter smoke proves raw observations become canonical `GameState`,
  `LegalAction`, symbolic tensors, and AlphaStar-style model inputs.
- Symbolic dataset builder converts search records or raw observation traces into
  global/entity/legal-action tensors and action-sequence labels.
- A small supervised AlphaStar-style policy smoke train completes on a bounded
  sample and writes a torch checkpoint.
- Offline evaluation compares the symbolic direct policy against the rule agent
  and the old Phase 4-style distilled policy.
- One-turn root search is wired to the symbolic policy/value path before more
  large-scale shard generation.
- After the symbolic path exists, diagnostics still run as SLURM jobs, never as
  large login-node workloads.
