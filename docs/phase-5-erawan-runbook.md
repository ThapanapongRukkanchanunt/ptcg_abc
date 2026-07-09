# Phase 5 ERAWAN Search-Distillation Runbook

This is the operating runbook for the current Phase 5 vertical slice. The full
Phase 5 architecture, training, and evaluation map is in
`docs/phase-5-master-plan.md`.

Use `docs/phase-5-changelog.md` as the report-ready history of implementation
work, ERAWAN results, diagnostics, conclusions, and artifact decisions.

This runbook prepares the search-distillation slice for large-scale ERAWAN
training: generate search-improved decision data with bounded one-turn root
search, merge the array shards, then train the Torch behavior-cloning/distillation
model from the merged dataset.

Phase 5 builds on the Phase 4 package and uses the existing 9-deck by 4-benchmark
matchup grid. It does not require integrating the full 13-deck pool before the first
large run.

## Storage Convention

For future game-data generation on ERAWAN, keep large generated datasets under:

```bash
GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
```

Use this root for generated decision/trajectory JSONL shards and merged datasets.
Keep `reports/`, `models/`, and `experiments/` in the repository as before.
Trace JSONL files are treated as experiment/debug artifacts and stay under
`experiments/rl/...` unless explicitly overridden.

Report artifact ownership:

- Reports produced on ERAWAN should be committed and pushed from ERAWAN before
  local inspection whenever practical.
- The local Codex workspace should then pull those report commits and analyze
  the tracked artifacts.
- Avoid committing uploaded copies of reports from the local workspace when the
  same paths may already exist as untracked ERAWAN outputs. This prevents
  `git pull --ff-only` from stopping with "untracked working tree files would be
  overwritten by merge".

The Phase 5 SLURM scripts now default to this convention:

- `scripts/slurm/phase5_search_data_array.sbatch`
- `scripts/slurm/phase5_merge_train_conda.sbatch`
- `scripts/slurm/phase5_symbolic_train_conda.sbatch`
- `scripts/slurm/phase5_symbolic_diagnose_conda.sbatch`

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
GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
"$PY" -m ptcg_abc rl-generate-search-data \
  --games 1 \
  --max-steps 60 \
  --top-k 4 \
  --rollout-steps 30 \
  --require-changed \
  --output "$GAME_DATA_ROOT/phase5_search_smoke.jsonl" \
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
  GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th \
  GAMES_PER_SHARD=2 \
  MAX_STEPS=80 \
  TOP_K=4 \
  ROLLOUT_STEPS=30 \
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
GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
"$PY" -m ptcg_abc rl-merge-search-data \
  --input "$GAME_DATA_ROOT/phase5_search/shards/phase5_search_decisions_shard-*.jsonl" \
  --trace-input 'experiments/rl/phase5_search/traces/phase5_search_traces_shard-*.jsonl' \
  --output "$GAME_DATA_ROOT/phase5_search_decisions_merged.jsonl" \
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
  GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th \
  SHARD_COUNT=32 \
  GAMES_PER_SHARD=1000 \
  MAX_STEPS=600 \
  TOP_K=4 \
  ROLLOUT_STEPS=30 \
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
ls -lh /project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search/shards/phase5_search_decisions_shard-0.jsonl
ls -lh /project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search/shards/phase5_search_decisions_shard-1.jsonl
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
  GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th \
  SHARD_COUNT=32 GAMES_PER_SHARD=1000 MAX_STEPS=600 TOP_K=4 ROLLOUT_STEPS=30 \
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
ls /project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search/shards/phase5_search_decisions_shard-*.jsonl | wc -l
ls experiments/rl/phase5_search/traces/phase5_search_traces_shard-*.jsonl | wc -l
```

Both counts should be `32`.

## 6. Merge Large Shards

After all 32 shards finish, merge the decision and trace JSONL files:

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
"$PY" -m ptcg_abc rl-merge-search-data \
  --input "$GAME_DATA_ROOT/phase5_search/shards/phase5_search_decisions_shard-*.jsonl" \
  --trace-input 'experiments/rl/phase5_search/traces/phase5_search_traces_shard-*.jsonl' \
  --output "$GAME_DATA_ROOT/phase5_search_decisions_merged.jsonl" \
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
`/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search/shards/`.

```bash
cd ~/ptcg_abc
JOB=$(
  GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th \
  BC_EPOCHS=2 \
  BC_LEARNING_RATE=0.02 \
  MERGED_DATASET=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards.jsonl \
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

- `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_merged.jsonl`
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
GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
JOB=$(
  DATASET="$GAME_DATA_ROOT/phase5_search_decisions_10shards.jsonl" \
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
  DATASET="$GAME_DATA_ROOT/phase5_search_decisions_10shards.jsonl" \
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
  DATASET="$GAME_DATA_ROOT/phase5_search_decisions_merged.jsonl" \
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
  MERGED_DATASET=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards_reweighted.jsonl \
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
  DATASET=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards_reweighted.jsonl \
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
  MERGED_DATASET=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards_pairwise_all.jsonl \
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
  DATASET=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards_pairwise_all.jsonl \
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
  DATASET=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards_pairwise_all.jsonl \
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
  MERGED_DATASET=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards_residual.jsonl \
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
  DATASET=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards_residual.jsonl \
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

## 11. Real Phase 5 Adapter/Encoder And Symbolic Trainer Track

The next implementation track follows `docs/ptcg_rl_strategy_recommendation.md`
more directly:

- Canonical `StateAdapter` and `LegalOptionAdapter`.
- Minimal `GameMemory` and `BeliefState`.
- Symbolic global/entity/legal-action tensors.
- AlphaStar-style policy model with a transformer entity/state core and
  autoregressive previous-action context for turn-level action sequences.
- Direct symbolic supervised trainer that reads existing Phase 5 `DecisionFrame`
  JSONL records and writes a torch checkpoint.

Do not submit more large search-distillation jobs until this symbolic model has a
small supervised smoke train and an offline/battle evaluation path.

Local smoke after pulling the symbolic trainer slice:

```bash
python -m unittest tests.test_rl_phase5_adapters tests.test_rl_phase5_symbolic_training
```

ERAWAN smoke after pulling the symbolic trainer slice:

```bash
cd ~/ptcg_abc
git pull --ff-only
export PYTHONPATH="$PWD/src"
PY="$PWD/.conda_ptcg/bin/python"
"$PY" -m unittest tests.test_rl_phase5_adapters tests.test_rl_phase5_symbolic_training
```

Use the merged 10-shard decision JSONL as the first trainer input. The individual
per-shard decision files are not needed for this trainer after a successful
merge, as long as the merged decision JSONL, merged trace JSONL, and merge
manifest are present.

Set the dataset path to the merged 10-shard file you actually kept:

```bash
GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
DATASET="$GAME_DATA_ROOT/phase5_search_decisions_merged.jsonl"
ls -lh "$DATASET" \
  experiments/rl/phase5_search_traces_merged.jsonl \
  experiments/rl/phase5_search_merge_manifest.json
```

Build a small symbolic JSONL only for inspection. Do not build the full symbolic
JSONL unless you intentionally want a much larger expanded tensor file:

```bash
"$PY" -m ptcg_abc rl-build-phase5-symbolic-dataset \
  --dataset "$DATASET" \
  --limit 1000 \
  --output "$GAME_DATA_ROOT/phase5_symbolic_decisions_smoke.jsonl"
```

Submit the first bounded symbolic trainer smoke as a job:

```bash
JOB=$(
  DATASET="$DATASET" \
  LIMIT=2000 \
  EPOCHS=1 \
  BATCH_SIZE=32 \
  D_MODEL=64 \
  CHANGED_WEIGHT=4.0 \
  UNCHANGED_WEIGHT=0.5 \
  CHECKPOINT=models/rl/phase5_symbolic_policy_smoke.pt \
  REPORT_JSON=experiments/rl/phase5_symbolic_train_report_smoke.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_symbolic_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_symbolic_smoke_job.txt
squeue -j "$JOB"

# After the job starts:
tail -f "experiments/rl/slurm-${JOB}-phase5-symbolic-train.out"
```

If the smoke report has nonzero examples, a finite loss, and a written
checkpoint, submit the 10-shard pass:

```bash
JOB=$(
  DATASET="$DATASET" \
  LIMIT=0 \
  EPOCHS=1 \
  BATCH_SIZE=64 \
  D_MODEL=128 \
  CHANGED_WEIGHT=4.0 \
  UNCHANGED_WEIGHT=0.5 \
  CHECKPOINT=models/rl/phase5_symbolic_policy_10shards.pt \
  REPORT_JSON=experiments/rl/phase5_symbolic_train_report_10shards.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_symbolic_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_symbolic_10shards_job.txt
```

After the 10-shard symbolic checkpoint trains, evaluate it as a job. Start with
one game per matchup:

```bash
JOB=$(
  MODEL=models/rl/phase5_symbolic_policy_10shards.pt \
  GAMES_PER_MATCHUP=1 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_symbolic_10shards_smoke.json \
  REPORT_MD=reports/phase5_symbolic_10shards_smoke.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_symbolic_eval_smoke_job.txt
squeue -j "$JOB"

# After the job starts:
tail -f "experiments/rl/slurm-${JOB}-phase5-symbolic-eval.out"
```

If the smoke has `errors: 0`, run the 10-game benchmark:

```bash
JOB=$(
  MODEL=models/rl/phase5_symbolic_policy_10shards.pt \
  GAMES_PER_MATCHUP=10 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_symbolic_10shards_10g.json \
  REPORT_MD=reports/phase5_symbolic_10shards_10g.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_symbolic_eval_10g_job.txt
```

If the 10-game benchmark is below the rule baseline, diagnose the symbolic
checkpoint against the merged search-decision dataset as a job:

```bash
JOB=$(
  DATASET="$DATASET" \
  CHECKPOINT=models/rl/phase5_symbolic_policy_10shards.pt \
  LIMIT=0 \
  BATCH_SIZE=128 \
  REPORT_JSON=reports/phase5_symbolic_10shards_diagnostics.json \
  REPORT_MD=reports/phase5_symbolic_10shards_diagnostics.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_diagnose_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_symbolic_diag_10shards_job.txt
squeue -j "$JOB"

# After the job starts:
tail -f "experiments/rl/slurm-${JOB}-phase5-symbolic-diag.out"
```

Read the report:

```bash
cat reports/phase5_symbolic_10shards_diagnostics.md
```

Your first 10-shard symbolic diagnostic showed the expected Stage 1 failure:
overall search agreement was reasonable, but search-changed agreement was only
`0.356`, the model still scored baseline actions above search actions on
changed frames, and the changed-frame third-action rate was `0.292`. Respond by
training a changed-frame pairwise/all-negative symbolic checkpoint:

```bash
JOB=$(
  DATASET="$DATASET" \
  LIMIT=0 \
  EPOCHS=1 \
  BATCH_SIZE=64 \
  D_MODEL=128 \
  CHANGED_WEIGHT=6.0 \
  UNCHANGED_WEIGHT=0.25 \
  PAIRWISE_CHANGED=1 \
  PAIRWISE_WEIGHT=1.0 \
  PAIRWISE_MARGIN=1.0 \
  PAIRWISE_NEGATIVES=all \
  CHECKPOINT=models/rl/phase5_symbolic_policy_10shards_pairwise_all.pt \
  REPORT_JSON=experiments/rl/phase5_symbolic_train_report_10shards_pairwise_all.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_symbolic_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_symbolic_pairwise_all_job.txt
squeue -j "$JOB"

# After the job starts:
tail -f "experiments/rl/slurm-${JOB}-phase5-symbolic-train.out"
```

Evaluate the pairwise/all-negative symbolic checkpoint as a job:

```bash
JOB=$(
  MODEL=models/rl/phase5_symbolic_policy_10shards_pairwise_all.pt \
  GAMES_PER_MATCHUP=10 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_symbolic_10shards_pairwise_all_10g.json \
  REPORT_MD=reports/phase5_symbolic_10shards_pairwise_all_10g.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_symbolic_pairwise_all_eval_10g_job.txt
squeue -j "$JOB"
```

Run diagnostics on the pairwise checkpoint as a job, regardless of the battle
score:

```bash
JOB=$(
  DATASET="$DATASET" \
  CHECKPOINT=models/rl/phase5_symbolic_policy_10shards_pairwise_all.pt \
  LIMIT=0 \
  BATCH_SIZE=128 \
  REPORT_JSON=reports/phase5_symbolic_10shards_pairwise_all_diagnostics.json \
  REPORT_MD=reports/phase5_symbolic_10shards_pairwise_all_diagnostics.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_diagnose_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_symbolic_pairwise_all_diag_job.txt
squeue -j "$JOB"
```

Promotion gate for this slice: the pairwise checkpoint should raise
search-changed hit rate, make the mean model search-minus-baseline margin
positive, reduce changed third-action rate, and improve the 10-game benchmark
toward or above the rule baseline.

After confirming the merged files exist, you may remove the large per-shard
inputs to reclaim space:

```bash
du -sh "$GAME_DATA_ROOT/phase5_search/shards" experiments/rl/phase5_search/traces
ls -lh "$DATASET" \
  experiments/rl/phase5_search_traces_merged.jsonl \
  experiments/rl/phase5_search_merge_manifest.json

# Remove only after the merged files above are present and readable.
rm -i "$GAME_DATA_ROOT"/phase5_search/shards/phase5_search_decisions_shard-*.jsonl
rm -i experiments/rl/phase5_search/traces/phase5_search_traces_shard-*.jsonl
```

## 12. Online Phase 5 Search Evaluation

The supervised symbolic sweep is complete when no direct policy clears the
offline gate. The next Phase 5 slice evaluates the actual intended inference
shape: symbolic policy prior plus one-turn root search.

Start with the plain symbolic checkpoint as the conservative policy prior:

```bash
MODEL=models/rl/phase5_symbolic_policy_10shards.pt
```

Submit a one-game-per-matchup search smoke:

```bash
JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=1 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_search_agent_plain_smoke.json \
  REPORT_MD=reports/phase5_search_agent_plain_smoke.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_agent_plain_smoke_job.txt
squeue -j "$JOB"

# After the job starts:
tail -f "experiments/rl/slurm-${JOB}-phase5-symbolic-eval.out"
```

If the smoke has `errors: 0`, run the 10-game benchmark:

```bash
JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=10 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_search_agent_plain_10g.json \
  REPORT_MD=reports/phase5_search_agent_plain_10g.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_agent_plain_10g_job.txt
```

The generated JSON and Markdown reports now include a `Search Telemetry`
section for `phase5-search`. Inspect it after the job completes:

```bash
grep -E "Search Telemetry|Searched decisions|Search-changed decisions|Search errors|Candidate errors|Truncated candidates|Average search seconds|Max search seconds" \
  reports/phase5_search_agent_plain_10g.md

"$PY" - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("reports/phase5_search_agent_plain_10g.json").read_text())
print(json.dumps(report.get("search_telemetry", {}), indent=2, sort_keys=True))
PY
```

Compare against direct symbolic, rule, and the mid pairwise checkpoint. Only
promote this path if search improves battle win rate without increasing errors
or timeouts.

If the 10-game benchmark remains above the rule baseline and telemetry is clean,
run a larger confirmation benchmark:

```bash
JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=30 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_search_agent_plain_30g.json \
  REPORT_MD=reports/phase5_search_agent_plain_30g.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_agent_plain_30g_job.txt
```

To inspect one-turn search truncations and disagreements, run a smaller trace
capture job. Keep this at 1-3 games per matchup unless you explicitly need a
large trace file:

```bash
JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=3 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_search_agent_plain_trace_3g.json \
  REPORT_MD=reports/phase5_search_agent_plain_trace_3g.md \
  SEARCH_TRACE_OUTPUT=experiments/rl/phase5_search_agent_plain_trace_3g.jsonl \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_agent_plain_trace_3g_job.txt
```

After the trace job finishes, inspect truncation and search-change examples:

```bash
"$PY" -m ptcg_abc rl-diagnose-search-traces \
  --trace-input experiments/rl/phase5_search_agent_plain_trace_3g.jsonl \
  --report-json reports/phase5_search_agent_plain_trace_3g_diagnostics.json \
  --report-md reports/phase5_search_agent_plain_trace_3g_diagnostics.md

"$PY" - <<'PY'
import json
from pathlib import Path

path = Path("experiments/rl/phase5_search_agent_plain_trace_3g.jsonl")
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
changed = [record for record in records if record.get("changed")]
truncated = [
    record for record in records
    if any(candidate.get("truncated") for candidate in record.get("candidates", []))
]

print("records", len(records))
print("changed", len(changed))
print("truncated_records", len(truncated))

for label, bucket in (("changed", changed), ("truncated", truncated)):
    print("\\n==", label, "examples ==")
    for record in bucket[:5]:
        print(json.dumps({
            "game_index": record.get("game_index"),
            "deck_index": record.get("deck_index"),
            "deck_label": record.get("deck_label"),
            "opponent": record.get("opponent"),
            "turn": record.get("turn"),
            "baseline": record.get("baseline_indices"),
            "search": record.get("search_indices"),
            "candidate_summary": [
                {
                    "indices": candidate.get("indices"),
                    "option_type": candidate.get("option_type"),
                    "card_name": candidate.get("card_name"),
                    "combined_score": candidate.get("combined_score"),
                    "tactical_score": candidate.get("tactical_score"),
                    "truncated": candidate.get("truncated"),
                    "error": candidate.get("error"),
                }
                for candidate in record.get("candidates", [])
            ],
        }, indent=2))
PY
```

If selected-truncated changed decisions remain a concern, compare the default
rollout cap against a higher cap with another small trace run:

```bash
JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=3 \
  MAX_STEPS=600 \
  SEARCH_ROLLOUT_STEPS=30 \
  REPORT_JSON=reports/phase5_search_agent_plain_trace_3g_cap30.json \
  REPORT_MD=reports/phase5_search_agent_plain_trace_3g_cap30.md \
  SEARCH_TRACE_OUTPUT=experiments/rl/phase5_search_agent_plain_trace_3g_cap30.jsonl \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_agent_plain_trace_3g_cap30_job.txt
```

Then run the trace diagnostic:

```bash
"$PY" -m ptcg_abc rl-diagnose-search-traces \
  --trace-input experiments/rl/phase5_search_agent_plain_trace_3g_cap30.jsonl \
  --report-json reports/phase5_search_agent_plain_trace_3g_cap30_diagnostics.json \
  --report-md reports/phase5_search_agent_plain_trace_3g_cap30_diagnostics.md
```

Compare default cap 18 versus cap 30 on win rate, average search seconds,
truncated candidate rate, selected-truncated rate, and changed
selected-truncated rate. Promote the higher cap only if it reduces dangerous
truncation without hurting win rate or timing.

If the 3-game cap-30 trace looks clean, run the 10-game cap-30 benchmark with a
trace so the benchmark and truncation diagnostics come from the same job:

```bash
JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=10 \
  MAX_STEPS=600 \
  SEARCH_ROLLOUT_STEPS=30 \
  REPORT_JSON=reports/phase5_search_agent_plain_10g_cap30.json \
  REPORT_MD=reports/phase5_search_agent_plain_10g_cap30.md \
  SEARCH_TRACE_OUTPUT=experiments/rl/phase5_search_agent_plain_10g_cap30.jsonl \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_agent_plain_10g_cap30_job.txt
```

After it finishes, diagnose the 10-game trace:

```bash
"$PY" -m ptcg_abc rl-diagnose-search-traces \
  --trace-input experiments/rl/phase5_search_agent_plain_10g_cap30.jsonl \
  --report-json reports/phase5_search_agent_plain_10g_cap30_diagnostics.json \
  --report-md reports/phase5_search_agent_plain_10g_cap30_diagnostics.md
```

Cap 30 passed this promotion gate: 148 / 360 wins, 0.411 win rate, 1 timeout,
0 errors, average search 0.0631 seconds, max search 3.4057 seconds. The default
`RootSearchConfig.max_rollout_steps` is now 30. Continue to watch max latency in
larger runs because the cap-30 max was higher than cap 18's 1.4773 seconds.

To summarize the benchmark side of the same cap-30 job, inspect:

```bash
cat reports/phase5_search_agent_plain_10g_cap30.json
grep -E "Games:|Wins:|Losses:|Draws:|Timeouts:|Errors:|Win rate:|Searched decisions:|Search-changed decisions:|Average search seconds:|Max search seconds:" \
  reports/phase5_search_agent_plain_10g_cap30.md
```

Both the benchmark summary and trace diagnostic are recorded in
`docs/phase-5-changelog.md`. The first 10-game cap-30 trace diagnostic had
14,680 records, 3,218 changed records, 0 search/candidate errors, 353 truncated
candidates, 64 selected-truncated records, and 15 changed selected-truncated
records.

For the next confirmation run, use the default cap 30 and omit
`SEARCH_ROLLOUT_STEPS`:

```bash
JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=30 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_search_agent_plain_30g_cap30_default.json \
  REPORT_MD=reports/phase5_search_agent_plain_30g_cap30_default.md \
  SEARCH_TRACE_OUTPUT=experiments/rl/phase5_search_agent_plain_30g_cap30_default.jsonl \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_agent_plain_30g_cap30_default_job.txt
```

After it finishes, diagnose the 30-game trace:

```bash
"$PY" -m ptcg_abc rl-diagnose-search-traces \
  --trace-input experiments/rl/phase5_search_agent_plain_30g_cap30_default.jsonl \
  --report-json reports/phase5_search_agent_plain_30g_cap30_default_diagnostics.json \
  --report-md reports/phase5_search_agent_plain_30g_cap30_default_diagnostics.md
```

Summarize the benchmark side too:

```bash
cat reports/phase5_search_agent_plain_30g_cap30_default.json
grep -E "Games:|Wins:|Losses:|Draws:|Timeouts:|Errors:|Win rate:|Searched decisions:|Search-changed decisions:|Average search seconds:|Max search seconds:" \
  reports/phase5_search_agent_plain_30g_cap30_default.md
```

## 13. Next Phase 5 Training Track

Historical note: this section records the previous single-generalist training
track. It is retained for artifact reproducibility, but it is no longer the
active next-action plan after the July 2, 2026 switch to the full-agent
AlphaStar-like league track in section 17.

The previous order was:

1. Generate `phase5-search` self-play data.
2. Add value, Q/action-value, and tactical heads.
3. Train a generalist model from rule demonstrations, search-improved decisions,
   and self-play outcomes.
4. Evaluate on the current 9-deck required benchmark.
5. Expand to more decks if the 9-deck gate is stable.
6. Start larger PPO/self-play only after the supervised/value generalist model
   is stable.

Next implementation slice:

- Use the Phase 5 search self-play collector command:
  - `rl-generate-phase5-search-selfplay`
- Use the SLURM wrapper:
  - `scripts/slurm/phase5_search_selfplay_conda.sbatch`
- Output trajectory JSONL with final outcome targets and per-player metadata
  under `$GAME_DATA_ROOT`.
- Preserve search telemetry in the JSON report and optionally write sampled
  search traces under `experiments/rl/phase5_search_selfplay/traces`.
- Write a report with games, steps, timeouts, errors, deck matchup counts,
  search decisions, search changes, truncation, and output paths.

Do not use old PPO/update commands as the next main step. They remain useful for
reference, but the next model needs value/Q/tactical targets before larger
policy optimization.

### Phase 5 Search Self-Play Smoke

After pulling the self-play collector slice, run a small job first:

```bash
cd ~/ptcg_abc
git pull origin main
export PYTHONPATH="$PWD/src"
GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
MODEL=models/rl/phase5_symbolic_policy_10shards.pt

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  MODEL="$MODEL" \
  GAMES_PER_SHARD=2 \
  MAX_STEPS=600 \
  SELFPLAY_DECK_INDICES="1 2" \
  SEARCH_TRACE_GAMES=2 \
  sbatch --parsable --array=0-0 --gres=gpu:1 --cpus-per-task=2 --time=00:30:00 \
    scripts/slurm/phase5_search_selfplay_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_selfplay/latest_smoke_job.txt
```

Inspect:

```bash
JOB=$(cat experiments/rl/phase5_search_selfplay/latest_smoke_job.txt)
squeue -j "$JOB"
tail -n 120 "experiments/rl/slurm-${JOB}-0-phase5-search-selfplay.out"
tail -n 120 "experiments/rl/slurm-${JOB}-0-phase5-search-selfplay.err"
cat experiments/rl/phase5_search_selfplay/summaries/phase5_search_selfplay_summary_shard-0.json
ls -lh "$GAME_DATA_ROOT/phase5_search_selfplay/shards/phase5_search_selfplay_shard-0.jsonl"
ls -lh experiments/rl/phase5_search_selfplay/traces/phase5_search_selfplay_traces_shard-0.jsonl
```

Pass gate:

- `errors` is `0`.
- `steps` is nonzero.
- `search_telemetry.searched_decisions` is nonzero.
- `search_telemetry.search_errors` is `0`.
- `search_telemetry.candidate_errors` is `0`.
- The trajectory JSONL exists under `$GAME_DATA_ROOT`.

If the smoke passes, run a bounded two-shard job:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  MODEL="$MODEL" \
  SHARD_COUNT=2 \
  GAMES_PER_SHARD=25 \
  MAX_STEPS=600 \
  SELFPLAY_DECK_INDICES="1 2 3 4 5 6 7 8 9" \
  SEARCH_TRACE_GAMES=1 \
  sbatch --parsable --array=0-1 --gres=gpu:1 --cpus-per-task=2 \
    scripts/slurm/phase5_search_selfplay_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_selfplay/latest_2shard_job.txt
```

This writes game trajectories to:

```bash
$GAME_DATA_ROOT/phase5_search_selfplay/shards/
```

Reports and optional traces stay in:

```bash
experiments/rl/phase5_search_selfplay/
```

Bounded two-shard result recorded on June 26, 2026:

- Games started: 50 / 50.
- Steps written: 8,424.
- Errors: 0.
- Timeouts: 0.
- Draws: 0.
- Search decisions: 4,468.
- Search-changed decisions: 942.
- Search-change rate: 0.211.
- Candidate probes: 16,344.
- Search errors: 0.
- Candidate errors: 0.
- Truncated candidates: 63.
- Truncated-candidate rate: 0.00385.
- Average search seconds: 0.0813.
- Max search seconds: 3.8083.
- Trace records: 165.

After this gate passes, launch the larger two-shard self-play dataset. This
defaults to about 10,000 total games, split as 5,000 games per shard, and writes
trajectory data outside the repo under `$GAME_DATA_ROOT`.

```bash
cd ~/ptcg_abc
export PYTHONPATH="$PWD/src"
GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
MODEL=models/rl/phase5_symbolic_policy_10shards.pt
mkdir -p experiments/rl/phase5_search_selfplay_10k

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  MODEL="$MODEL" \
  TOTAL_GAMES=10000 \
  MAX_STEPS=600 \
  SELFPLAY_DECK_INDICES="1 2 3 4 5 6 7 8 9" \
  SEARCH_TRACE_GAMES=1 \
  sbatch --parsable scripts/slurm/phase5_search_selfplay_2shard_10k.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_selfplay_10k/latest_job.txt
```

If the intended total is 1,000 rather than 10,000, use the same command with
`TOTAL_GAMES=1000`.

Inspect the large job:

```bash
JOB=$(cat experiments/rl/phase5_search_selfplay_10k/latest_job.txt)
squeue -j "$JOB"
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-0-phase5-search-selfplay-10k.out"
tail -n 120 "experiments/rl/slurm-${JOB}-1-phase5-search-selfplay-10k.out"
tail -n 120 "experiments/rl/slurm-${JOB}-0-phase5-search-selfplay-10k.err"
tail -n 120 "experiments/rl/slurm-${JOB}-1-phase5-search-selfplay-10k.err"
cat experiments/rl/phase5_search_selfplay_10k/summaries/phase5_search_selfplay_summary_shard-0.json
cat experiments/rl/phase5_search_selfplay_10k/summaries/phase5_search_selfplay_summary_shard-1.json
wc -l "$GAME_DATA_ROOT"/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-*.jsonl
ls -lh "$GAME_DATA_ROOT"/phase5_search_selfplay_10k/shards/
ls -lh experiments/rl/phase5_search_selfplay_10k/traces/
```

Large-run pass gate:

- Both array tasks complete with `ExitCode` `0:0`.
- Combined `games_started` is close to `TOTAL_GAMES`.
- Combined `errors` and `timeouts` are `0` or small enough to explain.
- Combined `search_telemetry.search_errors` is `0`.
- Combined `search_telemetry.candidate_errors` is `0`.
- Trajectory JSONL row count is nonzero for both shards.
- Traces exist only for the sampled games; do not set `SEARCH_TRACE_GAMES=0`
  for large runs because the CLI treats `0` as trace all games.

Recorded 10,000-game result on June 27, 2026:

- Games started: 10,000 / 10,000.
- Trajectory rows: 1,597,717.
- Shard rows: 796,919 and 800,798.
- Shard storage: 61G total, about 31G per shard.
- Errors: 0.
- Timeouts: 50.
- Draws: 20.
- Deck A wins: 4,919.
- Deck B wins: 5,061.
- Search decisions: 827,784.
- Search-changed decisions: 176,728.
- Search-change rate: 0.2135.
- Candidate probes: 3,011,687.
- Search errors: 0.
- Candidate errors: 0.
- Truncated candidates: 12,338.
- Truncated-candidate rate: 0.00410.
- Average search seconds: 0.0808.
- Max search seconds: 4.8117.
- Trace records: 146.

## 14. Phase 5 Generalist Multi-Head Training

The next model uses:

- rule demonstrations from baseline/rule labels in the 10-shard decision data,
- search-improved decisions from the same 10-shard decision data,
- self-play outcomes from the 10,000-game `phase5-search` trajectory data.

It trains the shared AlphaStar-style core with:

- autoregressive legal-action policy head,
- state-value head,
- selected-action Q head,
- first tactical scalar head that predicts the normalized rule/tactical prior
  for each legal action.

Submit a bounded smoke first:

```bash
cd ~/ptcg_abc
export PYTHONPATH="$PWD/src"
GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
mkdir -p experiments/rl/phase5_generalist

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  DECISION_DATASET="$GAME_DATA_ROOT/phase5_search_decisions_10shards.jsonl" \
  SELFPLAY_DATASETS="$GAME_DATA_ROOT/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-0.jsonl $GAME_DATA_ROOT/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-1.jsonl" \
  CHECKPOINT=models/rl/phase5_generalist_policy_smoke.pt \
  REPORT_JSON=experiments/rl/phase5_generalist_train_report_smoke.json \
  DECISION_LIMIT=2000 \
  SELFPLAY_LIMIT=2000 \
  D_MODEL=64 \
  BATCH_SIZE=32 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_generalist_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_generalist_smoke_job.txt
```

Inspect the smoke:

```bash
JOB=$(cat experiments/rl/phase5_generalist_smoke_job.txt)
squeue -j "$JOB"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-generalist-train.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-generalist-train.err"
cat experiments/rl/phase5_generalist_train_report_smoke.json
ls -lh models/rl/phase5_generalist_policy_smoke.pt
```

Smoke pass gate:

- The job exits with `0:0`.
- `decision_examples`, `rule_examples`, and `selfplay_examples` are nonzero.
- `value_examples` and `action_value_examples` are nonzero.
- `tactical_examples` is nonzero.
- `final_loss` is finite.
- A checkpoint is written.

If the smoke passes, submit the full mixed generalist train:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  DECISION_DATASET="$GAME_DATA_ROOT/phase5_search_decisions_10shards.jsonl" \
  SELFPLAY_DATASETS="$GAME_DATA_ROOT/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-0.jsonl $GAME_DATA_ROOT/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-1.jsonl" \
  CHECKPOINT=models/rl/phase5_generalist_policy_10k.pt \
  REPORT_JSON=experiments/rl/phase5_generalist_train_report_10k.json \
  EPOCHS=1 \
  BATCH_SIZE=64 \
  D_MODEL=128 \
  DECISION_LIMIT=0 \
  SELFPLAY_LIMIT=0 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_generalist_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_generalist_10k_job.txt
```

Inspect the full train:

```bash
JOB=$(cat experiments/rl/phase5_generalist_10k_job.txt)
squeue -j "$JOB"
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-generalist-train.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-generalist-train.err"
cat experiments/rl/phase5_generalist_train_report_10k.json
ls -lh models/rl/phase5_generalist_policy_10k.pt
```

After the full train finishes, run a one-game smoke benchmark for the direct
generalist policy and for `phase5-search` using the generalist policy as prior:

```bash
MODEL=models/rl/phase5_generalist_policy_10k.pt

JOB=$(
  AGENT=phase5-symbolic \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=1 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_generalist_symbolic_smoke.json \
  REPORT_MD=reports/phase5_generalist_symbolic_smoke.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_generalist_symbolic_smoke_eval_job.txt

JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=1 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_generalist_search_smoke.json \
  REPORT_MD=reports/phase5_generalist_search_smoke.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_generalist_search_smoke_eval_job.txt
```

Only move to the 10-game and 30-game benchmark if both smoke evaluations finish
with `errors=0`.

Recorded generalist benchmark result on June 29, 2026:

- Direct `phase5-symbolic`, `models/rl/phase5_generalist_policy_10k.pt`:
  - 10-game benchmark: 117 / 360 wins, 0.325 win rate, 4 timeouts, 0 errors.
  - 30-game benchmark: 361 / 1,080 wins, 0.334 win rate, 12 timeouts, 0 errors.
- `phase5-search`, `models/rl/phase5_generalist_policy_10k.pt` as prior:
  - 10-game benchmark: 138 / 360 wins, 0.383 win rate, 0 timeouts, 0 errors.
  - 30-game benchmark: 414 / 1,080 wins, 0.383 win rate, 5 timeouts, 0 errors.
  - 30-game search telemetry: 44,267 searched decisions, 8,584 changed
    decisions, 0 search errors, 0 candidate errors, 677 truncated candidates,
    0.0514 average search seconds, 2.4492 max search seconds.

Interpretation:

- Keep `phase5-search` with `models/rl/phase5_generalist_policy_10k.pt` as the
  current best 9-deck inference path.
- Do not promote the direct `phase5-symbolic` generalist as a standalone policy;
  it improved over the first symbolic model but remains below the rule baseline.
- The 30-game `phase5-search` generalist-prior result slightly beats the prior
  30-game search benchmark, 414 / 1,080 vs. 408 / 1,080, and reduces truncation
  from 6,395 to 677 candidates.
- Deck 1, Alakazam Dudunsparce, remains the main weakness at only 4 / 120 wins
  in the 30-game `phase5-search` generalist-prior benchmark.

Next operational step:

- Start the Phase 5 deck-expansion slice while preserving the current 9-deck
  required benchmark.
- Use `models/rl/phase5_generalist_policy_10k.pt` as the default
  `phase5-search` prior for new data generation and comparison.
- Add targeted data/model treatment for Alakazam Dudunsparce before larger PPO.

## 15. Phase 5 13-Deck League Self-Play Smoke

This slice expands data generation from the current 9 Tournament 559 decks to a
13-deck Phase 5 league pool:

- Decks 1-9: the current Tournament 559 required evaluation decks.
- Decks 10-13: the four required sample/benchmark decks: Crustle, Mega Lucario
  ex, Mega Abomasnow ex, and Iono's Bellibolt ex.

The existing required benchmark remains unchanged. Use this deck pool only when
generating expanded self-play data.

After pulling the latest code on ERAWAN, set the environment:

```bash
export PYTHONPATH="$PWD/src"
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
MODEL=models/rl/phase5_generalist_policy_10k.pt

test -f "$MODEL"
```

Submit the bounded 13-deck smoke. This requests 338 games: 13 x 13 ordered deck
pairs with the two player-order passes distributed across two shards.

```bash
mkdir -p experiments/rl/phase5_search_selfplay_13deck_338

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  PHASE5_SELFPLAY_RUN_NAME=phase5_search_selfplay_13deck_338 \
  TOTAL_GAMES=338 \
  MODEL="$MODEL" \
  DECK_POOL=league-13 \
  SELFPLAY_DECK_INDICES="1 2 3 4 5 6 7 8 9 10 11 12 13" \
  SEARCH_TRACE_GAMES=2 \
  sbatch --parsable scripts/slurm/phase5_search_selfplay_2shard_10k.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_selfplay_13deck_338/latest_job.txt
```

Inspect it as a normal job:

```bash
JOB=$(cat experiments/rl/phase5_search_selfplay_13deck_338/latest_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES

tail -n 120 "experiments/rl/slurm-${JOB}-0-phase5-search-selfplay-10k.out"
tail -n 120 "experiments/rl/slurm-${JOB}-1-phase5-search-selfplay-10k.out"

cat experiments/rl/phase5_search_selfplay_13deck_338/summaries/phase5_search_selfplay_summary_shard-0.json
cat experiments/rl/phase5_search_selfplay_13deck_338/summaries/phase5_search_selfplay_summary_shard-1.json
wc -l "$GAME_DATA_ROOT"/phase5_search_selfplay_13deck_338/shards/phase5_search_selfplay_shard-*.jsonl
ls -lh "$GAME_DATA_ROOT"/phase5_search_selfplay_13deck_338/shards/
ls -lh experiments/rl/phase5_search_selfplay_13deck_338/traces/
```

Expected pass criteria:

- Both shards complete with `errors=0`.
- Each summary reports `"deck_pool": "league-13"`, `pair_count=169`, and deck
  indices 1 through 13.
- Search telemetry has `search_errors=0` and `candidate_errors=0`.
- Timeouts are low enough to treat as simulator/game-length noise, not a failed
  run.
- Dataset shards are stored under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_13deck_338/shards`.

Recorded 338-game smoke result on June 29, 2026:

- Games requested / started: 338 / 338.
- Deck pool: `league-13`.
- Pair count: 169 in each shard.
- Steps written: 51,945.
- Draws: 4.
- Timeouts: 3.
- Errors: 0.
- Searched decisions: 26,920.
- Search-changed decisions: 5,286.
- Change rate: 0.196.
- Candidate probes: 97,661.
- Search errors: 0.
- Candidate errors: 0.
- Truncated candidates: 641.
- Truncated-candidate rate: 0.00656.
- Average search seconds: 0.0539.
- Max search seconds: 1.6210.

If the 338-game smoke passes, submit the larger 13-deck data run:

```bash
mkdir -p experiments/rl/phase5_search_selfplay_13deck_10k

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  PHASE5_SELFPLAY_RUN_NAME=phase5_search_selfplay_13deck_10k \
  TOTAL_GAMES=10000 \
  MODEL="$MODEL" \
  DECK_POOL=league-13 \
  SELFPLAY_DECK_INDICES="1 2 3 4 5 6 7 8 9 10 11 12 13" \
  SEARCH_TRACE_GAMES=3 \
  sbatch --parsable scripts/slurm/phase5_search_selfplay_2shard_10k.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_search_selfplay_13deck_10k/latest_job.txt
```

This submits one SLURM array job with two array tasks. Seeing two running tasks,
for example `${JOB}_0` and `${JOB}_1`, is expected. If `sbatch --parsable`
times out after the scheduler accepted the job and `latest_job.txt` was not
written, do not resubmit immediately. Recover the base job ID from `squeue`:

```bash
squeue -u "$USER" --name=ptcg-p5-search-selfplay-10k --format="%i %j %T %M %R"

# Example: if squeue shows 72810_0 and 72810_1, persist the base job ID.
printf "%s\n" 72810 | tee experiments/rl/phase5_search_selfplay_13deck_10k/latest_job.txt
```

Do not remove the existing 9-deck `phase5_search_selfplay_10k` shards yet. They
are the completed source for `models/rl/phase5_generalist_policy_10k.pt` and are
still useful for comparisons.

## 16. Phase 5 13-Deck Mixed Generalist Train

Historical note: this section records the completed 13-deck mixed-generalist
artifact path. The resulting checkpoint was clean but not promoted on the
required 9x4 30-game gate, so do not use this as the active mainline path or
the mainline PPO seed. The active replacement track is section 17.

The generalist trainer already accepts repeated `--selfplay-dataset` arguments,
and `scripts/slurm/phase5_generalist_train_conda.sbatch` expands a
space-separated `SELFPLAY_DATASETS` list into those repeated arguments. Use the
13-deck train as a new artifact family, not an overwrite of the current 9-deck
checkpoint.

Training inputs:

- Search-decision data:
  `$GAME_DATA_ROOT/phase5_search_decisions_10shards.jsonl`.
- Existing 9-deck self-play:
  `$GAME_DATA_ROOT/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-0.jsonl`
  and
  `$GAME_DATA_ROOT/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-1.jsonl`.
- New 13-deck self-play, after the run completes:
  `$GAME_DATA_ROOT/phase5_search_selfplay_13deck_10k/shards/phase5_search_selfplay_shard-0.jsonl`
  and
  `$GAME_DATA_ROOT/phase5_search_selfplay_13deck_10k/shards/phase5_search_selfplay_shard-1.jsonl`.

Before training, verify all four self-play shards exist:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th

ls -lh "$GAME_DATA_ROOT"/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-*.jsonl
ls -lh "$GAME_DATA_ROOT"/phase5_search_selfplay_13deck_10k/shards/phase5_search_selfplay_shard-*.jsonl
wc -l "$GAME_DATA_ROOT"/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-*.jsonl
wc -l "$GAME_DATA_ROOT"/phase5_search_selfplay_13deck_10k/shards/phase5_search_selfplay_shard-*.jsonl
```

Submit a bounded mixed-train smoke first:

```bash
mkdir -p experiments/rl models/rl

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  CHECKPOINT=models/rl/phase5_generalist_policy_13deck_smoke.pt \
  REPORT_JSON=experiments/rl/phase5_generalist_train_report_13deck_smoke.json \
  DECISION_LIMIT=5000 \
  SELFPLAY_LIMIT=5000 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_generalist_train_13deck_10k.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_generalist_13deck_smoke_job.txt
```

Inspect the smoke:

```bash
JOB=$(cat experiments/rl/phase5_generalist_13deck_smoke_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-generalist-13deck-train.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-generalist-13deck-train.err"
cat experiments/rl/phase5_generalist_train_report_13deck_smoke.json
ls -lh models/rl/phase5_generalist_policy_13deck_smoke.pt
```

Pass criteria:

- The job exits successfully.
- `decision_examples`, `rule_examples`, and `selfplay_examples` are nonzero.
- The report lists all four self-play dataset paths.
- No artifact path uses the existing `phase5_generalist_policy_10k.pt` or
  `phase5_generalist_train_report_10k.json` names.

If the smoke passes, submit the full mixed train:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_generalist_train_13deck_10k.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_generalist_13deck_10k_job.txt
```

Inspect the full train:

```bash
JOB=$(cat experiments/rl/phase5_generalist_13deck_10k_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-generalist-13deck-train.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-generalist-13deck-train.err"
cat experiments/rl/phase5_generalist_train_report_13deck_10k.json
ls -lh models/rl/phase5_generalist_policy_13deck_10k.pt
```

The 13-deck league data is a training expansion, not a replacement evaluation
gate. After the full train, evaluate the new checkpoint on the existing required
9x4 benchmark first:

```bash
MODEL=models/rl/phase5_generalist_policy_13deck_10k.pt

JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=1 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_generalist_13deck_search_smoke.json \
  REPORT_MD=reports/phase5_generalist_13deck_search_smoke.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_generalist_13deck_search_smoke_eval_job.txt
```

Only move to 10-game and 30-game required benchmarks if the smoke has
`errors=0`.

After the new checkpoint passes the required 9x4 gate, run the 13-deck league
benchmark. This is intentionally separate from the required gate: each Phase 5
league deck is controlled by the selected agent against each Phase 5 league deck
controlled by the rule agent, with player-order balance across games per
matchup.

```bash
MODEL=models/rl/phase5_generalist_policy_13deck_10k.pt

JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=2 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_generalist_13deck_league_search_2g.json \
  REPORT_MD=reports/phase5_generalist_13deck_league_search_2g.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_league_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_generalist_13deck_league_eval_job.txt
```

To test neural value/Q/tactical priors inside root search, keep this as a
separate comparison run:

```bash
JOB=$(
  AGENT=phase5-search \
  MODEL="$MODEL" \
  GAMES_PER_MATCHUP=1 \
  MAX_STEPS=600 \
  POLICY_PRIOR_WEIGHT=0.05 \
  NEURAL_ACTION_VALUE_WEIGHT=0.05 \
  NEURAL_TACTICAL_WEIGHT=0.05 \
  REPORT_JSON=reports/phase5_generalist_13deck_search_neural_prior_smoke.json \
  REPORT_MD=reports/phase5_generalist_13deck_search_neural_prior_smoke.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_symbolic_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_generalist_13deck_neural_prior_smoke_eval_job.txt
```

Compare any old/new benchmark reports locally or on ERAWAN:

```bash
export PYTHONPATH="$PWD/src"

"$PY" -m ptcg_abc phase5-compare-benchmarks \
  --baseline reports/phase5_generalist_search_30g.json \
  --candidate reports/phase5_generalist_13deck_search_30g.json \
  --report-json reports/phase5_generalist_13deck_vs_10k_comparison.json \
  --report-md reports/phase5_generalist_13deck_vs_10k_comparison.md
```

If this report-only command fails with `No module named 'lxml'`, pull the
latest `main`. The CLI now imports the Limitless scraper lazily so
`phase5-compare-benchmarks` does not require scraper dependencies.

Deprecated follow-up: this PPO smoke command is retained only for
reproducibility. Do not run it as the mainline next step unless a later
promotable checkpoint is selected and the decision is recorded in
`docs/phase-5-changelog.md`.

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  CHECKPOINT=models/rl/phase5_generalist_policy_13deck_10k.pt \
  OUTPUT_CHECKPOINT=models/rl/phase5_generalist_policy_13deck_ppo_smoke.pt \
  REPORT_JSON=experiments/rl/phase5_ppo_train_report_13deck_smoke.json \
  SELFPLAY_LIMIT=5000 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_ppo_train_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_ppo_13deck_smoke_job.txt
```

Policy-pool self-play is now available for future data generation:

```bash
POLICY_POOL_MODELS="models/rl/phase5_generalist_policy_10k.pt models/rl/phase5_generalist_policy_13deck_10k.pt" \
  DECK_POOL=league-13 \
  PHASE5_SELFPLAY_RUN_NAME=phase5_search_selfplay_policy_pool_13deck \
  sbatch --parsable scripts/slurm/phase5_search_selfplay_2shard_10k.sbatch
```

## 17. AlphaStar-Like League Replacement Track

The single-generalist promotion track is no longer the active pacing plan. The
new track is to implement the full agent, pretrain all non-gameplay models, then
run a compact AlphaStar-like 13-deck league.

Training schedule:

- Bootstrap: generate rule-based 13-deck gameplay for all 13 x 13 matchups with
  balanced player order, then train one specialist policy per deck.
- League iteration: each deck specialist plays 100 training games.
- Update point: after 13 x 100 = 1,300 training games, update all 13 deck
  specialists and write one checkpoint family.
- Evaluation point: after each update, run full agent vs rule-based for all
  13 x 13 matchups, 30 games per matchup, balanced player order. This is 5,070
  evaluation games per iteration.

Storage and cleanup are mandatory:

- ERAWAN generated league data goes under:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-XXXX`.
- Raw training gameplay for an iteration goes under that iteration's
  `raw_train/` directory.
- Raw evaluation trajectories should not be written by default. Evaluation
  should write aggregate reports and a tiny sampled trace only.
- After the model/policy update succeeds, remove that iteration's `raw_train/`
  data. Keep only:
  - model checkpoints and optimizer states under `models/rl/phase5_league_alpha/`,
  - train reports under `experiments/rl/phase5_league_alpha/`,
  - evaluation reports under `reports/`,
  - manifests/checksums/row counts that prove what was consumed,
  - bounded sampled traces for diagnostics.
- Do not launch the next league iteration while the previous iteration's raw
  gameplay still exists, unless a written diagnostic reason is added to
  `docs/phase-5-changelog.md`.

Reason:

- The project folder has about 400 GB of practical capacity.
- Previous Phase 5 search self-play shards were about 30 GB each.
- A league that keeps raw gameplay for every iteration will fill the project
  space quickly and make the experiment state hard to audit.

This replacement track supersedes the older instruction to start PPO from
`models/rl/phase5_generalist_policy_13deck_10k.pt`. That checkpoint is retained
as a non-promoted artifact, but the next active work is full-agent runtime,
offline pretraining, per-deck specialists, and clean league iteration.

### Alpha League Bootstrap Commands

After pulling the implementation slice, start with a small bootstrap smoke:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  ITERATION=0 \
  GAMES_PER_PAIR=1 \
  MAX_STEPS=300 \
  sbatch --parsable scripts/slurm/phase5_alpha_rule_bootstrap.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0000_rule_bootstrap_job.txt
```

Inspect:

```bash
JOB=$(cat experiments/rl/phase5_league_alpha/iter-0000_rule_bootstrap_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-alpha-bootstrap.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-alpha-bootstrap.err"
cat experiments/rl/phase5_league_alpha/iter-0000_rule_bootstrap_report.json
ls -lh "$GAME_DATA_ROOT"/phase5_league_alpha/iterations/iter-0000/raw_train/
```

If the bootstrap smoke is clean, verify that specialist training can consume
the raw trajectory data. Use limits for this trainer smoke:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  ITERATION=0 \
  DECISION_LIMIT=2000 \
  SELFPLAY_LIMIT=2000 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_deck_specialists_train.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_job.txt
```

Inspect:

```bash
JOB=$(cat experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-deck-specialists.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-deck-specialists.err"
cat experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_report.json
ls -lh models/rl/phase5_league_alpha/iter-0000/specialists/
```

After both smokes pass, remove the smoke raw gameplay before generating the
fuller iteration-0 bootstrap:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  ITERATION=0 \
  sbatch --parsable scripts/slurm/phase5_alpha_cleanup_iteration.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0000_cleanup_job.txt
```

Inspect cleanup:

```bash
JOB=$(cat experiments/rl/phase5_league_alpha/iter-0000_cleanup_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
cat experiments/rl/phase5_league_alpha/iter-0000_cleanup_report.json
test ! -d "$GAME_DATA_ROOT"/phase5_league_alpha/iterations/iter-0000/raw_train
```

Then run the default bootstrap with two games per ordered pair:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  ITERATION=0 \
  GAMES_PER_PAIR=2 \
  MAX_STEPS=600 \
  sbatch --parsable scripts/slurm/phase5_alpha_rule_bootstrap.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0000_rule_bootstrap_job.txt
```

Train deck specialists from the fuller bootstrap trajectory data and existing
search-decision data without smoke limits:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  ITERATION=0 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_deck_specialists_train.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_job.txt
```

Inspect the no-limit update:

```bash
JOB=$(cat experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-deck-specialists.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-deck-specialists.err"
cat experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_report.json
ls -lh models/rl/phase5_league_alpha/iter-0000/specialists/
```

After the no-limit update succeeds and its report/checkpoints are preserved,
remove the fuller bootstrap raw gameplay:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  ITERATION=0 \
  sbatch --parsable scripts/slurm/phase5_alpha_cleanup_iteration.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0000_cleanup_job.txt
```

Inspect cleanup:

```bash
JOB=$(cat experiments/rl/phase5_league_alpha/iter-0000_cleanup_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
cat experiments/rl/phase5_league_alpha/iter-0000_cleanup_report.json
test ! -d "$GAME_DATA_ROOT"/phase5_league_alpha/iterations/iter-0000/raw_train
```

Evaluation after a specialist update uses the full-agent alias and aggregate
reports only:

```bash
JOB=$(
  AGENT=phase5-full \
  SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0000/specialists \
  GAMES_PER_MATCHUP=30 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_alpha_iter0000_full_vs_rule_30g.json \
  REPORT_MD=reports/phase5_alpha_iter0000_full_vs_rule_30g.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_league_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0000_eval_job.txt
```

`SPECIALIST_MODEL_DIR` makes the evaluator load `deck-01.pt` through
`deck-13.pt` for the matching controlled deck. Leave `MODEL` unset in this
mode; the script still has a single-model fallback for older diagnostics.

Package the best iteration-0 specialist Kaggle candidates after the specialist
eval is preserved. Based on the first true specialist eval, use deck 11 Mega
Lucario ex and deck 12 Mega Abomasnow ex. Deck 11 is the clear best against the
four required rule-based specialist opponents, while deck 12 is the better
general second pick after considering its much stronger full 13 x 13 rule eval:

```bash
export PYTHONPATH="$PWD/src"
python -m ptcg_abc phase5-package \
  --deck-pool league-13 \
  --deck-index 11 \
  --deck-index 12 \
  --model-dir models/rl/phase5_league_alpha/iter-0000/specialists \
  --output-dir submissions/phase5_alpha_iter0000_specialists_top2
```

Expected direct Kaggle zip outputs:

```bash
ls -lh submissions/phase5_alpha_iter0000_specialists_top2/*submission.zip
```

Historical note: the first learned-agent window, `iter-0001`, was generated
before the true online-RL pivot and then trained with the mixed
deck-specialist trainer. Keep it as a baseline. For future training windows,
use the stochastic neural `phase5-rl` collector and the PPO specialist update
below.

Start the next online-RL league iteration. This writes a fresh raw training
window under `iter-0002` using the iteration-1 specialist checkpoint family.
Each deck is scheduled for 100 games:

```bash
JOB=$(
  AGENT=phase5-rl \
  ITERATION=2 \
  SOURCE_ITERATION=1 \
  SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0001/specialists \
  GAMES_PER_DECK=100 \
  MAX_STEPS=600 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_alpha_league_iteration.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0002_league_iteration_job.txt
```

Inspect:

```bash
JOB=$(cat experiments/rl/phase5_league_alpha/iter-0002_league_iteration_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-alpha-league.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-alpha-league.err"
cat experiments/rl/phase5_league_alpha/iter-0002_league_iteration_report.json
ls -lh "$GAME_DATA_ROOT"/phase5_league_alpha/iterations/iter-0002/raw_train/
```

Then update the iteration-2 specialist family with PPO from that on-policy raw
window:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  ITERATION=2 \
  SOURCE_ITERATION=1 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_alpha_ppo_specialists_train.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0002_ppo_specialists_job.txt
```

Inspect:

```bash
JOB=$(cat experiments/rl/phase5_league_alpha/iter-0002_ppo_specialists_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-alpha-ppo-specialists.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-alpha-ppo-specialists.err"
cat experiments/rl/phase5_league_alpha/iter-0002_ppo_specialists_report.json
```

After the PPO update report and checkpoint family exist, evaluate the updated
specialists with the same full-agent-vs-rule gate:

```bash
JOB=$(
  AGENT=phase5-full \
  SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0002/specialists \
  GAMES_PER_MATCHUP=30 \
  MAX_STEPS=600 \
  REPORT_JSON=reports/phase5_alpha_iter0002_specialists_full_vs_rule_30g.json \
  REPORT_MD=reports/phase5_alpha_iter0002_specialists_full_vs_rule_30g.md \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_league_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_league_alpha/iter-0002_full_vs_rule_eval_job.txt
```

Clean `iter-0002/raw_train` with
`scripts/slurm/phase5_alpha_cleanup_iteration.sbatch` only after the PPO update
report/checkpoints are preserved and the next action has been recorded.

## 18. Specialized Public-Agent Rule-Opponent Curriculum

The generic rule-agent league did not produce reliable online-RL improvement:
iteration 5 remained the best checkpoint, later self-play updates regressed or
plateaued, and Kaggle replay inspection showed concrete tactical failures such
as Mega Abomasnow ex missing obvious energy attachments and attacks. The active
training target is therefore to replace the generic rule-opponent gate with
available specialized public/sample rule agents and train until the full agent
clears a 50% aggregate win-rate gate against every available specialized
opponent and every controlled deck.

The roster source is the public-20-plus-sample-4 notebook. That notebook is a
metadata roster, not a complete source bundle, so the loader uses the built-in
24-source roster and discovers whichever exported `.py` or `.ipynb` agents are
available locally. Missing public notebooks are recorded as unavailable and
skipped. The repo currently has a built-in adapter for the sample Dragapult
agent; other sample/public agents should be exported into the public-agent root.

Recommended ERAWAN layout:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
export PUBLIC_AGENT_ROOTS=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agents
```

Place exported agents in any of these shapes under one of the roots:

```text
<root>/<agent-key>/submission.py
<root>/<agent-key>.py
<root>/<agent-key>.ipynb
<root>/<kaggle-source-slug>/submission.py
```

Roster discovery is intentionally a SLURM job so availability diagnostics are
preserved:

```bash
JOB=$(
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  REPORT_JSON=reports/phase5_public_agent_roster.json \
  sbatch --parsable scripts/slurm/phase5_public_agent_roster.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_roster_job.txt
```

Inspect:

```bash
JOB=$(cat experiments/rl/phase5_public_agent_roster_job.txt)
sacct -j "$JOB" --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-public-roster.out"
tail -n 120 "experiments/rl/slurm-${JOB}-phase5-public-roster.err"
cat reports/phase5_public_agent_roster.json
```

Evaluate the current best specialist checkpoint family against all available
specialized public/sample rule agents. Keep iteration 5 as the starting
checkpoint unless a later checkpoint has already been explicitly promoted:

```bash
JOB=$(
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0005/specialists \
  AGENT=phase5-full \
  GAMES_PER_MATCHUP=30 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  MIN_OPPONENT_WIN_RATE=0.5 \
  REPORT_JSON=reports/phase5_public_agent_eval_iter0005_30g.json \
  REPORT_MD=reports/phase5_public_agent_eval_iter0005_30g.md \
  STATUS_JSON=reports/phase5_public_agent_status_iter0005.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_public_agent_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_eval_iter0005_job.txt
```

The JSON and Markdown reports include `public_agent_gate`, with:

- `passed`: every public-opponent aggregate and every controlled-deck aggregate
  is at least `MIN_OPPONENT_WIN_RATE` and has zero errors.
- `strict_matchup_passed`: every individual controlled-deck-vs-public-agent row
  clears the threshold. Treat this as a diagnostic, not the first promotion
  gate.
- `failing_opponents`, `failing_controlled_decks`, and `failing_matchups`: the
  curriculum targets for the next data window.

Generate public-agent rule-opponent trajectories from the current checkpoint
family. These are our-agent trajectories against fixed specialized opponents,
not generic rule-agent self-play:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0005/specialists \
  AGENT=phase5-rl \
  GAMES_PER_MATCHUP=10 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  OUTPUT="$GAME_DATA_ROOT"/phase5_public_agent_rule_train/iter-0006_public_agent_trajectories.jsonl \
  REPORT_JSON=experiments/rl/phase5_public_agent_rule_train_iter0006_report.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_public_agent_trajectories.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_rule_train_iter0006_job.txt
```

Update the next checkpoint family with the existing on-policy PPO specialist
trainer by pointing `TRAJECTORY_DATASET` at the public-agent trajectory file.
Use a separate public-agent curriculum checkpoint root so this targeted update
does not overwrite the historical generic Alpha league `iter-0006` artifacts:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  ITERATION=6 \
  SOURCE_ITERATION=5 \
  TRAJECTORY_DATASET="$GAME_DATA_ROOT"/phase5_public_agent_rule_train/iter-0006_public_agent_trajectories.jsonl \
  SOURCE_CHECKPOINT_DIR=models/rl/phase5_league_alpha/iter-0005/specialists \
  OUTPUT_CHECKPOINT_DIR=models/rl/phase5_public_agent_curriculum/iter-0006/specialists \
  REPORT_JSON=experiments/rl/phase5_public_agent_curriculum/iter-0006_ppo_specialists_report.json \
  REPORT_DIR=experiments/rl/phase5_public_agent_curriculum/iter-0006_ppo_specialists \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_alpha_ppo_specialists_train.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_curriculum/iter-0006_ppo_job.txt
```

Evaluate the candidate with the same specialized public-agent gate:

```bash
JOB=$(
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  SPECIALIST_MODEL_DIR=models/rl/phase5_public_agent_curriculum/iter-0006/specialists \
  AGENT=phase5-full \
  GAMES_PER_MATCHUP=30 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  MIN_OPPONENT_WIN_RATE=0.5 \
  REPORT_JSON=reports/phase5_public_agent_eval_iter0006_30g.json \
  REPORT_MD=reports/phase5_public_agent_eval_iter0006_30g.md \
  STATUS_JSON=reports/phase5_public_agent_status_iter0006.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_public_agent_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_eval_iter0006_job.txt
```

Repeat public-agent trajectory generation, PPO update, and evaluation until the
gate clears or the failing rows show a consistent tactical pattern that needs an
agent/runtime fix. Keep raw public-agent trajectory windows under
`$GAME_DATA_ROOT/phase5_public_agent_rule_train/` and delete old raw windows
after their PPO reports/checkpoints and evaluation reports are preserved.

Post-result note from the first targeted public-agent PPO signal check:
iteration 6 regressed from the iteration-5 public-agent baseline, falling from
50 / 390 wins to 38 / 390 wins against the built-in sample Dragapult opponent.
Do not repeat the same loss-heavy on-policy PPO loop unchanged. The next public
agent curriculum implementation should first change the training signal or add
diagnostics, such as successful-trajectory filtering, denser tactical rewards,
opponent demonstrations where applicable, or decision-level traces for missed
setup, attack, and energy-attachment actions.

### One-Deck Public-Agent Micro Experiment

The `phase5-rl` collector samples from the neural policy distribution with a
temperature, not epsilon-greedy random legal-action injection. Before another
large public-agent curriculum pass, use a one-deck, one-opponent PPO signal
check to see whether 100 on-policy games can improve a targeted specialist.

Recommended first target: deck 12 Mega Abomasnow ex against the built-in
`sample_dragapult` opponent, starting from the current best package candidate,
`models/rl/phase5_league_alpha/iter-0005/specialists`.

Baseline eval for the single matchup:

```bash
export PUBLIC_AGENT_ROOTS=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agents

JOB=$(
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  PUBLIC_AGENT_KEYS=sample_dragapult \
  DECK_INDICES=12 \
  SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0005/specialists \
  AGENT=phase5-full \
  GAMES_PER_MATCHUP=30 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  REPORT_JSON=reports/phase5_public_agent_deck12_dragapult_baseline_30g.json \
  REPORT_MD=reports/phase5_public_agent_deck12_dragapult_baseline_30g.md \
  STATUS_JSON=reports/phase5_public_agent_deck12_dragapult_baseline_status.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_public_agent_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_deck12_dragapult_baseline_job.txt
```

Collect 100 games of on-policy data for only that matchup:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  PUBLIC_AGENT_KEYS=sample_dragapult \
  DECK_INDICES=12 \
  SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0005/specialists \
  AGENT=phase5-rl \
  GAMES_PER_MATCHUP=100 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  OUTPUT="$GAME_DATA_ROOT"/phase5_public_agent_rule_train/deck12_vs_sample_dragapult_100.jsonl \
  REPORT_JSON=experiments/rl/phase5_public_agent_deck12_dragapult_100_trajectories_report.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_public_agent_trajectories.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_deck12_dragapult_100_trajectories_job.txt
```

Update only deck 12 into an isolated checkpoint root:

```bash
JOB=$(
  ITERATION=12 \
  DECK_INDICES=12 \
  TRAJECTORY_DATASET="$GAME_DATA_ROOT"/phase5_public_agent_rule_train/deck12_vs_sample_dragapult_100.jsonl \
  SOURCE_CHECKPOINT_DIR=models/rl/phase5_league_alpha/iter-0005/specialists \
  OUTPUT_CHECKPOINT_DIR=models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100/specialists \
  REPORT_JSON=experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_ppo_report.json \
  REPORT_DIR=experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_ppo \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_alpha_ppo_specialists_train.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_ppo_job.txt
```

Evaluate only deck 12 from that isolated checkpoint root:

```bash
JOB=$(
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  PUBLIC_AGENT_KEYS=sample_dragapult \
  DECK_INDICES=12 \
  SPECIALIST_MODEL_DIR=models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100/specialists \
  AGENT=phase5-full \
  GAMES_PER_MATCHUP=30 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  REPORT_JSON=reports/phase5_public_agent_deck12_dragapult_100_update_30g.json \
  REPORT_MD=reports/phase5_public_agent_deck12_dragapult_100_update_30g.md \
  STATUS_JSON=reports/phase5_public_agent_deck12_dragapult_100_update_status.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_public_agent_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_deck12_dragapult_100_update_eval_job.txt
```

Only promote the idea if the post-update single-matchup eval beats the baseline
by a clear margin with zero errors/timeouts. Otherwise, stop and change the
training signal before spending more compute.

### One-Deck Tactical-Reward Micro Experiment

After the first deck-12 PPO micro experiment, the post-update eval moved only
from 6 / 30 to 8 / 30 wins and the 100-game data window itself was 8 / 100
wins. The next diagnostic should keep the same narrow matchup but change the
training signal so correct attack/energy decisions are not drowned out by
repeated terminal-loss rewards.

Generate a shaped trajectory window. This keeps the same deck/opponent filters,
scales terminal outcome reward to 25%, and adds small per-decision rewards for
taking attack/attach actions plus penalties for ending/attacking while obvious
attack/attach actions are available:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
export PUBLIC_AGENT_ROOTS=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agents

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  PUBLIC_AGENT_KEYS=sample_dragapult \
  DECK_INDICES=12 \
  SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0005/specialists \
  AGENT=phase5-rl \
  GAMES_PER_MATCHUP=100 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  OUTCOME_REWARD_SCALE=0.25 \
  TACTICAL_REWARD_MODE=basic \
  TACTICAL_ATTACK_BONUS=0.10 \
  TACTICAL_ATTACH_BONUS=0.06 \
  TACTICAL_MISSED_ATTACK_PENALTY=-0.10 \
  TACTICAL_MISSED_ATTACH_PENALTY=-0.06 \
  OUTPUT="$GAME_DATA_ROOT"/phase5_public_agent_rule_train/deck12_vs_sample_dragapult_100_tactical.jsonl \
  REPORT_JSON=experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical_trajectories_report.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_public_agent_trajectories.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical_trajectories_job.txt
```

Update only deck 12 from the shaped trajectory file:

```bash
JOB=$(
  ITERATION=13 \
  DECK_INDICES=12 \
  TRAJECTORY_DATASET="$GAME_DATA_ROOT"/phase5_public_agent_rule_train/deck12_vs_sample_dragapult_100_tactical.jsonl \
  SOURCE_CHECKPOINT_DIR=models/rl/phase5_league_alpha/iter-0005/specialists \
  OUTPUT_CHECKPOINT_DIR=models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical/specialists \
  REPORT_JSON=experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical_ppo_report.json \
  REPORT_DIR=experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical_ppo \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_alpha_ppo_specialists_train.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical_ppo_job.txt
```

Evaluate only deck 12 from the shaped checkpoint:

```bash
JOB=$(
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  PUBLIC_AGENT_KEYS=sample_dragapult \
  DECK_INDICES=12 \
  SPECIALIST_MODEL_DIR=models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical/specialists \
  AGENT=phase5-full \
  GAMES_PER_MATCHUP=30 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  REPORT_JSON=reports/phase5_public_agent_deck12_dragapult_100_tactical_update_30g.json \
  REPORT_MD=reports/phase5_public_agent_deck12_dragapult_100_tactical_update_30g.md \
  STATUS_JSON=reports/phase5_public_agent_deck12_dragapult_100_tactical_update_status.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_public_agent_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical_eval_job.txt
```

Inspect the trajectory report's `tactical_reward_summary` before interpreting
the PPO result. If attack/attach opportunities are rare, this diagnostic is
measuring the wrong failure mode; if missed attack/attach counts are high, use
those counts to tune the reward weights or add a supervised rule-specialist
target.

### Search Score-Component Diagnostics

Use this when deciding whether to change the `phase5-full` action equation
weights. Normal public-agent eval reports do not store discarded candidate
scores, so first run a traced eval, then summarize the trace by win/loss
outcome.

Run a targeted deck-12 vs built-in `sample_dragapult` eval with full
root-search traces:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
export PUBLIC_AGENT_ROOTS=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agents

JOB=$(
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  PUBLIC_AGENT_KEYS=sample_dragapult \
  DECK_INDICES=12 \
  SPECIALIST_MODEL_DIR=models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical/specialists \
  AGENT=phase5-full \
  GAMES_PER_MATCHUP=100 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  SEARCH_TRACE_OUTPUT=experiments/rl/phase5_public_agent_micro/deck12_dragapult_tactical_full_100g_score_traces.jsonl \
  SEARCH_TRACE_GAMES=0 \
  REPORT_JSON=reports/phase5_public_agent_deck12_dragapult_tactical_full_trace_100g.json \
  REPORT_MD=reports/phase5_public_agent_deck12_dragapult_tactical_full_trace_100g.md \
  STATUS_JSON=reports/phase5_public_agent_deck12_dragapult_tactical_full_trace_status.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_public_agent_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_dragapult_tactical_full_trace_eval_job.txt
```

After the traced eval finishes, summarize score components by outcome:

```bash
JOB=$(
  TRACE_INPUT=experiments/rl/phase5_public_agent_micro/deck12_dragapult_tactical_full_100g_score_traces.jsonl \
  REPORT_JSON=reports/phase5_public_agent_deck12_dragapult_tactical_score_components.json \
  REPORT_MD=reports/phase5_public_agent_deck12_dragapult_tactical_score_components.md \
  sbatch --parsable --cpus-per-task=1 scripts/slurm/phase5_search_score_components_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_dragapult_tactical_score_components_job.txt
```

Interpretation checklist:

- Compare selected-candidate `tactical_score` range against `prior_score`
  range. If tactical is much wider, neural/rule prior weights are too small to
  affect most decisions.
- Compare selected-minus-baseline `combined_score`, `tactical_score`, and
  `prior_score` margins for wins vs losses. If losses have higher combined
  margins than wins, the current score function is confidently selecting the
  wrong short-horizon signals.
- Only tune `RootSearchConfig` weights after this report; otherwise the change
  is guesswork.

### One-Deck Leaf State-Value Search Experiment

This is the narrow test for replacing/mixing handcrafted rollout leaf scoring
with the neural `state_value` head. It uses only deck 12 Mega Abomasnow ex
against built-in `sample_dragapult`, and starts from rule-vs-rule bootstrap
trajectories instead of the failed loss-heavy PPO loop.

Generate rule-vs-rule bootstrap trajectories for the matchup:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
export PUBLIC_AGENT_ROOTS=/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agents

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  PUBLIC_AGENT_KEYS=sample_dragapult \
  DECK_INDICES=12 \
  AGENT=rule \
  GAMES_PER_MATCHUP=500 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  OUTPUT="$GAME_DATA_ROOT"/phase5_public_agent_rule_train/deck12_vs_sample_dragapult_rule_500.jsonl \
  REPORT_JSON=experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_rule_500_trajectories_report.json \
  sbatch --parsable --cpus-per-task=4 scripts/slurm/phase5_public_agent_trajectories.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_rule_500_trajectories_job.txt
```

Train a deck-12 bootstrap/value specialist only from that rule-vs-rule window:

```bash
JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  ITERATION=21 \
  DECK_INDICES=12 \
  SELFPLAY_DATASET="$GAME_DATA_ROOT"/phase5_public_agent_rule_train/deck12_vs_sample_dragapult_rule_500.jsonl \
  NO_DECISION_DATASET=1 \
  CHECKPOINT_DIR=models/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value/specialists \
  REPORT_DIR=experiments/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value/specialists \
  REPORT_JSON=experiments/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value_train_report.json \
  EPOCHS=3 \
  PAIRWISE_CHANGED=0 \
  VALUE_LOSS_WEIGHT=1.0 \
  ACTION_VALUE_LOSS_WEIGHT=0.25 \
  TACTICAL_LOSS_WEIGHT=0.0 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 scripts/slurm/phase5_deck_specialists_train.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value_train_job.txt
```

Evaluate the normalized tactical baseline from the new checkpoint:

```bash
JOB=$(
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  PUBLIC_AGENT_KEYS=sample_dragapult \
  DECK_INDICES=12 \
  SPECIALIST_MODEL_DIR=models/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value/specialists \
  AGENT=phase5-full \
  GAMES_PER_MATCHUP=100 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  NORMALIZE_TACTICAL_SCORE=1 \
  TACTICAL_SCORE_WEIGHT=1.0 \
  POLICY_PRIOR_WEIGHT=0.25 \
  LEAF_STATE_VALUE_WEIGHT=0.0 \
  SEARCH_TRACE_OUTPUT=experiments/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value_norm_tactical_traces.jsonl \
  SEARCH_TRACE_GAMES=0 \
  REPORT_JSON=reports/phase5_public_agent_deck12_rule_bootstrap_norm_tactical_100g.json \
  REPORT_MD=reports/phase5_public_agent_deck12_rule_bootstrap_norm_tactical_100g.md \
  STATUS_JSON=reports/phase5_public_agent_deck12_rule_bootstrap_norm_tactical_status.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_public_agent_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_rule_bootstrap_norm_tactical_eval_job.txt
```

Evaluate the leaf-value variant from the same checkpoint:

```bash
JOB=$(
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  PUBLIC_AGENT_KEYS=sample_dragapult \
  DECK_INDICES=12 \
  SPECIALIST_MODEL_DIR=models/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value/specialists \
  AGENT=phase5-full \
  GAMES_PER_MATCHUP=100 \
  MAX_STEPS=600 \
  REQUIRE_MIN_OPPONENTS=1 \
  NORMALIZE_TACTICAL_SCORE=1 \
  TACTICAL_SCORE_WEIGHT=0.5 \
  POLICY_PRIOR_WEIGHT=0.25 \
  LEAF_STATE_VALUE_WEIGHT=0.5 \
  SEARCH_TRACE_OUTPUT=experiments/rl/phase5_public_agent_micro/deck12_rule_bootstrap_leaf_value_traces.jsonl \
  SEARCH_TRACE_GAMES=0 \
  REPORT_JSON=reports/phase5_public_agent_deck12_rule_bootstrap_leaf_value_100g.json \
  REPORT_MD=reports/phase5_public_agent_deck12_rule_bootstrap_leaf_value_100g.md \
  STATUS_JSON=reports/phase5_public_agent_deck12_rule_bootstrap_leaf_value_status.json \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=2 scripts/slurm/phase5_public_agent_eval_conda.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_public_agent_micro/deck12_rule_bootstrap_leaf_value_eval_job.txt
```

After either eval, summarize its trace with
`scripts/slurm/phase5_search_score_components_conda.sbatch` to inspect
`leaf_state_value`, `leaf_state_value_prior`, and `leaf_state_value_score` by
win/loss outcome.

## 18.5 One-Deck Epsilon Curriculum: Dragapult vs Lucario

This experiment trains a fresh model-controlled official sample Dragapult ex
deck against a fixed rule-based official sample Mega Lucario ex deck. It is
separate from the 13-deck Alpha league artifacts.

The run uses:

- controlled learner deck: built-in `sample_dragapult`;
- opponent: built-in `sample_lucario` rule-agent fallback;
- synthetic controlled deck index: `101`;
- generation 0: scratch random Phase 5 policy checkpoint;
- generations 1-10: 1000 epsilon-greedy training games, PPO update, 100-game
  zero-exploration eval;
- epsilon schedule: linearly decays from `1.0` on generation 1 to `0.10` on
  generation 10;
- raw trajectory JSONL: deleted after the PPO update succeeds;
- retained eval replay artifacts: at most one win and one loss per generation,
  saved as compact JSON plus static HTML web views.

Submit the full 10-generation job:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
export PUBLIC_AGENT_ROOTS="$GAME_DATA_ROOT/phase5_public_agents"

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  RUN_NAME=phase5_dragapult_vs_lucario_epsilon \
  CONTROLLED_PUBLIC_AGENT_KEY=sample_dragapult \
  OPPONENT_PUBLIC_AGENT_KEYS=sample_lucario \
  CONTROLLED_DECK_INDEX=101 \
  GENERATIONS=10 \
  TRAIN_GAMES_PER_GENERATION=1000 \
  EVAL_GAMES_PER_GENERATION=100 \
  EPSILON_START=1.0 \
  EPSILON_END=0.10 \
  MAX_STEPS=600 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 \
    scripts/slurm/phase5_one_deck_public_epsilon_curriculum.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_one_deck_public_epsilon_dragapult_lucario_job.txt
```

Key repo artifacts:

- checkpoints:
  `models/rl/phase5_one_deck_public_epsilon/phase5_dragapult_vs_lucario_epsilon/gen-000*/specialists/deck-101.pt`;
- per-generation reports and replay views:
  `experiments/rl/phase5_one_deck_public_epsilon/phase5_dragapult_vs_lucario_epsilon/gen-000*/`;
- eval reports:
  `reports/phase5_dragapult_vs_lucario_epsilon_gen-000*_eval_100g.json`;
  `reports/phase5_dragapult_vs_lucario_epsilon_gen-000*_eval_100g.md`.

Raw training windows are written under:

`$GAME_DATA_ROOT/phase5_one_deck_public_epsilon/phase5_dragapult_vs_lucario_epsilon/generations/gen-000*/raw_train/`

The script removes each generation's trajectory JSONL immediately after the
successful PPO update. Keep checkpoints, reports, and the small replay HTML/JSON
files.

Recovery note for job 73798:

- Job 73798 completed generation-1 trajectory collection, then failed before
  PPO update because the original script called the 13-deck league specialist
  wrapper with custom deck index `101`.
- The script now uses the single-checkpoint Phase 5 PPO updater for the
  controlled public deck and supports `REUSE_EXISTING_TRAJECTORIES=1`.
- If the generation-1 raw JSONL still exists, rerun with
  `REUSE_EXISTING_TRAJECTORIES=1` to avoid replaying those 1000 games:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
export PUBLIC_AGENT_ROOTS="$GAME_DATA_ROOT/phase5_public_agents"

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  RUN_NAME=phase5_dragapult_vs_lucario_epsilon \
  CONTROLLED_PUBLIC_AGENT_KEY=sample_dragapult \
  OPPONENT_PUBLIC_AGENT_KEYS=sample_lucario \
  CONTROLLED_DECK_INDEX=101 \
  GENERATIONS=10 \
  TRAIN_GAMES_PER_GENERATION=1000 \
  EVAL_GAMES_PER_GENERATION=100 \
  EPSILON_START=1.0 \
  EPSILON_END=0.10 \
  MAX_STEPS=600 \
  REUSE_EXISTING_TRAJECTORIES=1 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 \
    scripts/slurm/phase5_one_deck_public_epsilon_curriculum.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_one_deck_public_epsilon_dragapult_lucario_retry_job.txt
```

### One-Deck Mixed Rule/Epsilon Curriculum

Use this after the sparse epsilon-only run. It keeps the same controlled deck
pair, but generation 0 first creates a retained rule-vs-rule Dragapult vs
Lucario trajectory window. Each model generation then trains on two trajectory
datasets:

- the retained generation-0 rule-vs-rule bootstrap window;
- a newly generated epsilon-model-vs-rule window from the previous checkpoint.

The script deletes only the per-generation epsilon window after a successful PPO
update. The rule bootstrap JSONL is kept and reused for every generation.
Because rule-vs-rule frames are off-policy demonstrations, the mixed script does
not pass `--require-on-policy` by default.

Submit the full 10-generation mixed job:

```bash
export GAME_DATA_ROOT=/project/SIGGI/thapanapong.r@cmu.ac.th
export PUBLIC_AGENT_ROOTS="$GAME_DATA_ROOT/phase5_public_agents"

JOB=$(
  GAME_DATA_ROOT="$GAME_DATA_ROOT" \
  PUBLIC_AGENT_ROOTS="$PUBLIC_AGENT_ROOTS" \
  RUN_NAME=phase5_dragapult_vs_lucario_mixed \
  CONTROLLED_PUBLIC_AGENT_KEY=sample_dragapult \
  OPPONENT_PUBLIC_AGENT_KEYS=sample_lucario \
  CONTROLLED_DECK_INDEX=101 \
  GENERATIONS=10 \
  RULE_BOOTSTRAP_GAMES=1000 \
  TRAIN_GAMES_PER_GENERATION=1000 \
  EVAL_GAMES_PER_GENERATION=100 \
  EPSILON_START=1.0 \
  EPSILON_END=0.10 \
  MAX_STEPS=600 \
  sbatch --parsable --gres=gpu:1 --cpus-per-task=4 \
    scripts/slurm/phase5_one_deck_public_mixed_curriculum.sbatch
)
echo "$JOB" | tee experiments/rl/phase5_one_deck_public_mixed_dragapult_lucario_job.txt
```

Key repo artifacts:

- checkpoints:
  `models/rl/phase5_one_deck_public_mixed/phase5_dragapult_vs_lucario_mixed/gen-000*/specialists/deck-101.pt`;
- per-generation reports and replay views:
  `experiments/rl/phase5_one_deck_public_mixed/phase5_dragapult_vs_lucario_mixed/gen-000*/`;
- eval reports:
  `reports/phase5_dragapult_vs_lucario_mixed_gen-000*_eval_100g.json`;
  `reports/phase5_dragapult_vs_lucario_mixed_gen-000*_eval_100g.md`.

Retained rule bootstrap data:

`$GAME_DATA_ROOT/phase5_one_deck_public_mixed/phase5_dragapult_vs_lucario_mixed/rule_bootstrap/phase5_public_rule_bootstrap_gen-0000.jsonl`

Per-generation epsilon raw windows are temporary and are deleted after the PPO
update succeeds:

`$GAME_DATA_ROOT/phase5_one_deck_public_mixed/phase5_dragapult_vs_lucario_mixed/generations/gen-000*/raw_train/`

## 19. Ready-To-Train Checklist

- Adapter smoke proves raw observations become canonical `GameState`,
  `LegalAction`, symbolic tensors, and AlphaStar-style model inputs.
- Symbolic dataset builder converts search records or raw observation traces into
  global/entity/legal-action tensors and action-sequence labels.
- A small supervised AlphaStar-style policy smoke train completes on a bounded
  sample and writes a torch checkpoint.
- The direct symbolic trainer can read the merged 10-shard `DecisionFrame` JSONL
  without writing a full expanded symbolic dataset.
- Phase 5 search self-play records exist with final outcome/value targets.
- Multi-head symbolic trainer can consume rule demonstrations, search-improved
  decisions, and self-play trajectory records in one mixed run.
- Offline evaluation compares the symbolic direct policy against the rule agent
  and the old Phase 4-style distilled policy using `rl-evaluate --agent
  phase5-symbolic`.
- One-turn root search is wired to the symbolic policy/value path before more
  large-scale shard generation.
- After the symbolic path exists, diagnostics still run as SLURM jobs, never as
  large login-node workloads.
