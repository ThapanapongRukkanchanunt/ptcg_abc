# Phase 5 Changelog

This file records Phase 5 work, implementation decisions, ERAWAN operating
results, diagnostics, and conclusions. Keep it factual and append-only enough
that a report can be reconstructed without reading chat history.

## Current Status

- Project phase: Phase 5, advanced RL/search-distillation track.
- Current implementation base: Phase 4 package and workflow.
- Current active gate: online `phase5-search` evaluation, which combines the
  symbolic torch policy with bounded one-turn root search.
- Latest pushed implementation commit: `bd75835 Add Phase 5 online search
  evaluation agent`.
- Large-scale data state: 10 completed Phase 5 search shards were enough for the
  first adapter/encoder and symbolic-policy experiments. More shards are not the
  next priority until the online search gate is measured.

## Standing Decisions

- Treat `docs/ptcg_rl_strategy_recommendation.md`,
  `docs/ptcg_rl_advanced_training_plan.md`, and
  `docs/ptcg_rl_evaluation_plan.md` as the authoritative Phase 5 design
  documents.
- Keep Phase 4 as the implementation base while replacing the Phase 5 plan.
- Build a vertical slice first, not the entire architecture at once.
- Use the already wired 9-deck required benchmark for the first Phase 5 slice;
  do not block on integrating all 13 decks.
- Training-side code may require PyTorch. Submission-side code should remain
  Kaggle-compatible, and PyTorch is acceptable only if competition/runtime
  constraints permit it.
- Generate valid search-improved training data from local smoke games and
  ERAWAN shards, with trace logs proving one-turn root search changes decisions
  and terminates safely.
- Start with one-turn root search before deeper or multi-turn search.
- Diagnostics should run as SLURM jobs on ERAWAN, not as large login-node jobs.

## Phase 5 Plan Recording

- Created `docs/phase-5-master-plan.md` as the umbrella Phase 5 plan.
- Connected the master plan to:
  - `docs/ptcg_rl_strategy_recommendation.md`
  - `docs/ptcg_rl_advanced_training_plan.md`
  - `docs/ptcg_rl_evaluation_plan.md`
  - `docs/phase-5-erawan-runbook.md`
- Recorded the target architecture:
  - raw observation
  - `StateAdapter`
  - `LegalOptionAdapter`
  - `GameMemory` / `BeliefState`
  - symbolic state and legal-action encoders
  - policy/value/action scorer
  - optional one-turn root search at selected decisions
  - direct policy mode for normal decisions
- Recorded that the advanced training plan's internal phase numbers are
  substages inside project Phase 5.

## Search-Improved Data Generation Slice

### Implementation

- Added a Phase 5 one-turn root-search data generator.
- Main command:
  - `python -m ptcg_abc rl-generate-search-data`
- Main implementation:
  - `src/ptcg_abc/rl/phase5_search.py`
- Added support for:
  - deterministic shard indices
  - shard counts
  - game offsets
  - append mode
  - trace output
  - safe rollout caps
  - probe-error accounting
  - truncated-rollout accounting
- Added merge command:
  - `python -m ptcg_abc rl-merge-search-data`
- Added ERAWAN scripts:
  - `scripts/slurm/phase5_search_data_array.sbatch`
  - `scripts/slurm/phase5_merge_train_conda.sbatch`
- Added operating instructions:
  - `docs/phase-5-erawan-runbook.md`

### Local Smoke Result

Command class:

```bash
python -m ptcg_abc rl-generate-search-data \
  --games 1 \
  --max-steps 60 \
  --top-k 4 \
  --rollout-steps 18 \
  --require-changed \
  --output data/datasets/rl/phase5_search_smoke.jsonl \
  --trace-output experiments/rl/phase5_search_smoke_traces.jsonl
```

Observed result:

- Games started: 1.
- Training decisions written: 26.
- Root decisions searched: 12.
- Search-changed decisions: 2.
- Candidate probes: 41.
- Probe errors: 0.
- Truncated search rollouts: 0.

Conclusion: the local smoke proved valid records, changed search decisions, trace
logging, and bounded safe termination.

### Early ERAWAN Merge Smoke

Two-shard merge manifest:

```json
{
  "decision_files": 2,
  "decision_records": 123,
  "trace_files": 2,
  "trace_records": 65,
  "output_path": "data/datasets/rl/phase5_search_decisions_merged.jsonl",
  "trace_path": "experiments/rl/phase5_search_traces_merged.jsonl"
}
```

Conclusion: shard generation and merge worked on small ERAWAN output.

### ERAWAN Operating Notes

- Submitting a larger job array hit:
  - `QOSMaxSubmitJobPerUserLimit`
- Two shards worked, while four shards did not under the tested submission
  pattern.
- User preference: submit a single wave and wait for the result rather than
  actively managing many overlapping waves.
- One shard was about 3 GB.
- ERAWAN provides about 1 TB of available storage.
- Merge can temporarily increase disk usage because merged outputs coexist with
  per-shard files. The runbook therefore includes cleanup instructions only
  after merged files are verified.
- User stopped at 10 shards because more shards would take a long time.

## First Phase 4-Style Search Distillation

### Training Result

Dataset and training result:

```json
{
  "frames": 791974,
  "actions": 9280556,
  "positives": 1684328,
  "negatives": 7596228,
  "epochs": 2,
  "checkpoint_path": "models/rl/phase5_search_distill_10shards.pt",
  "export_model_path": "models/rl/phase5_search_distill_10shards.json",
  "accuracy": 0.9039141457203597,
  "final_loss": 0.012990633957087994,
  "device": "cuda"
}
```

Conclusion: data generation, shard merge, CUDA training, and JSON export worked.
The high training accuracy did not imply better battle play.

### Required Benchmark Results

Rule baseline, 10 games per matchup:

- Games: 360.
- Wins: 126.
- Losses: 233.
- Draws: 1.
- Timeouts: 5.
- Errors: 0.
- Win rate: 0.350.

Trained `rl` model only, 10 games per matchup:

- Games: 360.
- Wins: 79.
- Losses: 280.
- Draws: 1.
- Timeouts: 2.
- Errors: 0.
- Win rate: 0.21944444444444444.

Hybrid model plus rule blend, 10 games per matchup:

- Games: 360.
- Wins: 81.
- Losses: 278.
- Draws: 1.
- Timeouts: 4.
- Errors: 0.
- Win rate: 0.225.

Conclusion: the first 10-shard Phase 4-style distillation checkpoint was valid
but not promotable. It underperformed the rule baseline.

### First Search-Distillation Diagnostic

Overall:

- Frames: 791,974.
- Search-changed frames: 76,068.
- Changed-decision share: about 9.6%.
- Overall search-hit rate: 0.9035637533555394.
- Overall baseline-hit rate: 0.9996123610118514.
- Mean model search-minus-baseline score: -0.5915803368262146.

Search-changed split:

- Frames: 76,068.
- Search-hit rate: 0.0.
- Baseline-hit rate: 1.0.
- Mean model search-minus-baseline score: -6.159176600904512.
- Mean rule search-minus-baseline score: -15889.199860650997.

Trace summary:

- Trace records: 418,900.
- Changed records: 76,068.
- Search errors: 0.
- Candidate errors: 0.
- Truncated candidates: 55,650.
- Mean search-minus-baseline combined score: 0.5960886340580955.
- Mean search-minus-baseline tactical score: 0.6037441871567197.

Conclusion: traces said root search improved changed decisions, but the exported
model learned the baseline/rule action on changed frames.

## Reweighted And Pairwise Phase 4-Style Experiments

### Changed-Weighted Retrain

Diagnostic:

- Overall search-hit rate: 0.5146646228285272.
- Overall baseline-hit rate: 0.5100344203218793.
- Overall mean model search-minus-baseline score: -0.036087918666830775.
- Search-changed search-hit rate: 0.24215175895251617.
- Search-changed baseline-hit rate: 0.19394489141294632.
- Search-changed mean model search-minus-baseline score: -0.37572557840674975.
- Search-unchanged search-hit rate: 0.5436202518207698.

Smoke benchmark:

- Changed-weight `rl` smoke: 7 wins / 36 games, 0.194 win rate, 1 timeout.
- Changed-weight hybrid smoke: 8 wins / 36 games, 0.222 win rate, 1 timeout.

Conclusion: reweighting reduced baseline-copying on changed decisions, but it
did not improve battle performance and introduced off-target action drift.

### Pairwise Search-Over-Baseline Retrain

Diagnostic:

- Overall search-hit rate: 0.4358463787952635.
- Overall baseline-hit rate: 0.42910373320336276.
- Overall mean model search-minus-baseline score: 0.040455066914619205.
- Search-changed search-hit rate: 0.16331440290266602.
- Search-changed baseline-hit rate: 0.09311405584477046.
- Search-changed mean model search-minus-baseline score: 0.421193684133126.
- Search errors: 0.
- Candidate errors: 0.
- Truncated candidates: 55,650.

Conclusion: pairwise training made the model score search above baseline on
average for changed frames, but it still often selected a third action rather
than search or baseline.

### Pairwise-All JSON/Torch Fallback Diagnostics

Pairwise-all JSON fallback diagnostic:

- Overall search-hit rate: about 0.421.
- Overall baseline-hit rate: about 0.418.
- Search-changed search-hit rate: about 0.150.
- Search-changed baseline-hit rate: about 0.121.
- Search-changed mean model search-minus-baseline score: about -0.153218.

Pairwise-all torch checkpoint diagnostic:

- Overall search-hit rate: about 0.904.
- Overall baseline-hit rate: about 1.0.
- Search-changed search-hit rate: 0.0.
- Search-changed baseline-hit rate: 1.0.
- Search-changed model search-minus-baseline score: 0.0.
- Model scores were effectively flat, so ranking fell back to rule-score
  tie-breaking.

Conclusion: the Phase 4-style ranker path was no longer worth scaling. Move to
the real Phase 5 adapter/encoder/model foundation.

## Real Phase 5 Adapter/Encoder Slice

### Implementation

Implemented the real Phase 5 symbolic input path:

- Canonical `StateAdapter`.
- Canonical `LegalOptionAdapter`.
- `GameMemory`.
- `BeliefState`.
- Symbolic global/entity/legal-action encoder.
- AlphaStar-inspired torch policy core:
  - transformer entity/state core
  - previous-action context for turn-level action sequences
  - legal-action scoring head
- Symbolic DecisionFrame bridge.
- Bounded symbolic dataset writer.
- Direct symbolic supervised trainer.
- SLURM job script for ERAWAN symbolic training.
- Direct `phase5-symbolic` evaluation agent for the required benchmark.

Key files:

- `src/ptcg_abc/rl/phase5_adapters.py`
- `src/ptcg_abc/rl/phase5_encoder.py`
- `src/ptcg_abc/rl/phase5_policy.py`
- `src/ptcg_abc/rl/phase5_symbolic_training.py`
- `src/ptcg_abc/agent/phase5_symbolic.py`
- `scripts/slurm/phase5_symbolic_train_conda.sbatch`
- `scripts/slurm/phase5_symbolic_eval_conda.sbatch`

### First Symbolic Policy Training Result

Checkpoint:

- `models/rl/phase5_symbolic_policy_10shards.pt`

Training report:

```json
{
  "frames_seen": 791974,
  "examples": 791667,
  "skipped_no_target": 307,
  "changed_examples": 76068,
  "actions": 4639871,
  "epochs": 1,
  "checkpoint_path": "models/rl/phase5_symbolic_policy_10shards.pt",
  "accuracy": 0.780060303132504,
  "final_loss": 0.5189497470855713,
  "device": "cuda",
  "target_source": "search",
  "changed_weight": 4.0,
  "unchanged_weight": 0.5,
  "max_entities": 96,
  "max_actions": 128,
  "max_previous_actions": 16,
  "config": {
    "global_dim": 17,
    "entity_dim": 28,
    "action_dim": 53,
    "d_model": 128,
    "nhead": 4,
    "transformer_layers": 2,
    "feedforward_dim": 256,
    "turn_hidden_dim": 128,
    "dropout": 0.05
  }
}
```

Model structure:

- Global input dimension: 17.
- Entity input dimension: 28.
- Legal-action input dimension: 53.
- Transformer model width: 128.
- Attention heads: 4.
- Transformer layers: 2.
- Feedforward dimension: 256.
- Turn context hidden dimension: 128.
- Dropout: 0.05.

### First Symbolic Benchmark Results

One game per matchup:

- Games: 36.
- Wins: 12.
- Losses: 24.
- Draws: 0.
- Timeouts: 0.
- Errors: 0.
- Win rate: 0.333.

Ten games per matchup:

- Games: 360.
- Wins: 109.
- Losses: 251.
- Draws: 0.
- Timeouts: 3.
- Errors: 0.
- Win rate: 0.303.

Conclusion: the real symbolic path was better than the old 10-shard `rl` and
hybrid checkpoints, but still below the rule baseline at about 0.350.

## Symbolic Pairwise Sweep

### Pairwise-All Training

Checkpoint:

- `models/rl/phase5_symbolic_policy_10shards_pairwise_all.pt`

Training report:

```json
{
  "frames_seen": 791974,
  "examples": 791667,
  "skipped_no_target": 307,
  "changed_examples": 76068,
  "actions": 4639871,
  "epochs": 1,
  "checkpoint_path": "models/rl/phase5_symbolic_policy_10shards_pairwise_all.pt",
  "accuracy": 0.5517003992840424,
  "final_loss": 6.1702423095703125,
  "device": "cuda",
  "target_source": "search",
  "changed_weight": 6.0,
  "unchanged_weight": 0.25,
  "pairwise_changed": true,
  "pairwise_weight": 1.0,
  "pairwise_margin": 1.0,
  "pairwise_negatives": "all"
}
```

Diagnostic summary:

- Overall search-hit rate: about 0.572.
- Search-changed search-hit rate: about 0.517.
- Search-unchanged search-hit rate: about 0.578.
- Search-changed mean model search-minus-baseline margin: about +3.516.
- Changed third-action rate: about 0.481.

Conclusion: pairwise-all strongly improved changed-decision preference but caused
too much third-action drift.

### Pairwise Baseline Soft Training

Checkpoint:

- `models/rl/phase5_symbolic_policy_10shards_pairwise_baseline_soft.pt`

Training report:

```json
{
  "accuracy": 0.579742492740003,
  "actions": 4639871,
  "changed_examples": 76068,
  "changed_weight": 4.0,
  "checkpoint_path": "models/rl/phase5_symbolic_policy_10shards_pairwise_baseline_soft.pt",
  "epochs": 1,
  "examples": 791667,
  "final_loss": 1.7716162204742432,
  "pairwise_changed": true,
  "pairwise_margin": 1.0,
  "pairwise_negatives": "baseline",
  "pairwise_weight": 0.25,
  "target_source": "search",
  "unchanged_weight": 0.75
}
```

Diagnostic summary:

- Overall search-hit rate: about 0.611.
- Search-changed search-hit rate: about 0.516.
- Search-unchanged search-hit rate: about 0.621.
- Search-changed margin: about +2.318.
- Changed third-action rate: about 0.473.

Conclusion: baseline-soft still improved changed-decision preference, but
third-action drift remained too high.

### Pairwise Baseline Tiny Training

Checkpoint:

- `models/rl/phase5_symbolic_policy_10shards_pairwise_baseline_tiny.pt`

Training report:

```json
{
  "accuracy": 0.8756358418375403,
  "actions": 4639871,
  "changed_examples": 76068,
  "changed_weight": 2.0,
  "checkpoint_path": "models/rl/phase5_symbolic_policy_10shards_pairwise_baseline_tiny.pt",
  "epochs": 1,
  "examples": 791667,
  "final_loss": 0.6731644868850708,
  "pairwise_changed": true,
  "pairwise_margin": 1.0,
  "pairwise_negatives": "baseline",
  "pairwise_weight": 0.05,
  "target_source": "search",
  "unchanged_weight": 1.0
}
```

Diagnostic summary:

- Overall search-hit rate: about 0.882.
- Search-changed search-hit rate: about 0.176.
- Search-unchanged search-hit rate: about 0.957.
- Search-changed margin: about -0.762.
- Changed third-action rate: about 0.134.

Conclusion: baseline-tiny preserved imitation and avoided drift, but it mostly
copied baseline choices on changed frames.

### Pairwise Baseline Mid Diagnostic

Diagnostic grep result:

- Overall: frames=791667, search_hit=0.760, baseline_hit=0.738,
  changed=0.096, mean_margin=0.042063.
- Search-changed: frames=76068, search_hit=0.423, baseline_hit=0.193,
  changed=1.000, mean_margin=0.437762.
- Search-unchanged: frames=715599, search_hit=0.796, baseline_hit=0.796,
  changed=0.000, mean_margin=0.000000.
- Model search-minus-baseline score: 0.437762.
- Changed third-action rate: 0.384.

Conclusion: baseline-mid was the best compromise among the supervised symbolic
loss variants, but it still did not clear the gate.

### Pairwise Sweep Conclusion

No direct supervised symbolic policy cleared the Phase 5 gate:

- Strong pairwise losses improved changed decisions but drifted to third actions.
- Tiny pairwise regularization preserved imitation but copied baseline too often.
- Mid pairwise was better balanced but still not obviously promotable.

Decision: stop spending compute on direct supervised-loss sweeps and evaluate the
intended inference shape, policy plus one-turn root search.

## Online One-Turn Root Search Evaluation Slice

### Implementation

Commit:

- `bd75835 Add Phase 5 online search evaluation agent`

Pushed to:

- `origin/main`

New agent:

- `src/ptcg_abc/agent/phase5_search.py`
- Class: `Phase5SearchPolicyAgent`

CLI additions:

- `rl-evaluate --agent phase5-search`
- `rl-rollout --agent phase5-search`

Workflow integration:

- `src/ptcg_abc/rl/workflow.py`
- `_make_agent(..., agent_kind="phase5-search")`
- Required benchmark passes the opponent deck into the search agent so hidden
  state sampling can construct opponent deck/prize/hand candidates.

Runbook update:

- `docs/phase-5-erawan-runbook.md`, section 12:
  - one-game-per-matchup `phase5-search` smoke
  - ten-game-per-matchup `phase5-search` benchmark

SLURM update:

- `scripts/slurm/phase5_symbolic_eval_conda.sbatch` now accepts:
  - `AGENT=phase5-search`

Tests:

- Added parser coverage for `phase5-search`.
- Added missing-checkpoint failure coverage.
- Added opponent-deck validation coverage.
- Added fail-closed unit coverage: if root search setup fails, keep the policy
  baseline action instead of selecting a different candidate from unscored
  priors.

Verification:

- Focused Phase 5 tests: 17 tests passed, 2 skipped.
- Phase 4/diagnostic tests: 28 tests passed, 2 skipped.
- `py_compile` passed for modified Python files.
- CLI help shows `phase5-search`.

### One-Turn Search Behavior

At each legal decision:

1. Parse raw observation through the Phase 5 state and legal-action adapters.
2. Update memory and construct the current belief state.
3. Encode global, entity, legal-action, and previous-action features.
4. Score legal options with the trained symbolic torch policy.
5. Choose the direct policy top action as the baseline.
6. If the root is eligible, build a candidate set from the baseline plus the
   highest-scoring policy actions up to `top_k`.
7. Sample hidden zones from public information, own deck, and opponent deck.
8. Start the Search API state.
9. For each root candidate, apply the candidate action.
10. Roll forward with existing rule-based choices until the turn ends, the game
    ends, or the rollout step cap is hit.
11. Score each candidate with the existing tactical scorer.
12. Add a small normalized policy-prior bonus.
13. Select the highest combined-score candidate.
14. If root search setup fails, keep the policy baseline action.

Current search gate:

- `select_type == MAIN`.
- context is `MAIN`.
- target count is exactly 1.
- legal option count is at least `min_candidates`.
- `top_k > 0`.

This is not full MCTS and not multi-turn search. It is the first online
one-turn root-search evaluation path.

### Current Next ERAWAN Command

After pulling `origin/main` on ERAWAN:

```bash
git pull origin main
MODEL=models/rl/phase5_symbolic_policy_10shards.pt

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
tail -f "experiments/rl/slurm-${JOB}-phase5-symbolic-eval.out"
```

If smoke has `errors: 0`, run the 10-game benchmark:

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

Promotion condition for this slice:

- `phase5-search` improves battle win rate over direct `phase5-symbolic`.
- It does not increase errors or timeouts materially.
- It is preferably competitive with or better than the rule baseline.

## Artifact Notes

Important model artifacts:

- `models/rl/phase5_search_distill_10shards.pt`
- `models/rl/phase5_search_distill_10shards.json`
- `models/rl/phase5_symbolic_policy_10shards.pt`
- `models/rl/phase5_symbolic_policy_10shards_pairwise_all.pt`
- `models/rl/phase5_symbolic_policy_10shards_pairwise_baseline_soft.pt`
- `models/rl/phase5_symbolic_policy_10shards_pairwise_baseline_tiny.pt`

Important dataset artifacts:

- `data/datasets/rl/phase5_search_decisions_10shards.jsonl`
- `data/datasets/rl/phase5_search_decisions_10shards_pairwise.jsonl`
- `data/datasets/rl/phase5_search_decisions_10shards_pairwise_all.jsonl`
- `data/datasets/rl/phase5_search_decisions_10shards_reweighted.jsonl`

File-retention decision:

- Keep the 791,974-record 10-shard datasets that match completed experiments.
- Do not use the 637,227-record `phase5_search_decisions_10shards_remerged.jsonl`
  as the canonical 10-shard dataset because it has fewer rows than expected.
- Remove large shard files only after canonical merged files and reports are
  verified.

## Open Questions And Next Work

- Run the online `phase5-search` smoke and 10-game benchmark on ERAWAN.
- Compare online search against:
  - direct `phase5-symbolic`
  - rule baseline
  - best supervised pairwise checkpoint
- Decide whether one-turn search should use:
  - plain symbolic checkpoint
  - baseline-mid checkpoint
  - a new value/Q/tactical scorer once implemented
- Refactor reusable Search API code out of `phase5_search.py` into a stable
  wrapper used by both data generation and online evaluation.
- Add report fields or a separate search-eval trace report for:
  - search attempt count
  - changed-decision count
  - search errors
  - probe errors
  - truncation count
  - time per searched decision
- Implement value, Q, and auxiliary tactical heads after the online search gate.
- Continue toward the full Phase 5 plan only after the current online search
  slice produces measurable battle evidence.
