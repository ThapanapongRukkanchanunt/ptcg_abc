# Phase 5 Changelog

This file records Phase 5 work, implementation decisions, ERAWAN operating
results, diagnostics, and conclusions. Keep it factual and append-only enough
that a report can be reconstructed without reading chat history.

## Current Status

- Project phase: Phase 5, advanced RL/search-distillation track.
- Current implementation base: Phase 4 package and workflow.
- Current active gate: confirm and instrument online `phase5-search`, which
  combines the symbolic torch policy with bounded one-turn root search.
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
- Future project updates and new Codex chats must keep this changelog current.
  Record meaningful implementation changes, ERAWAN results, diagnostics,
  conclusions, artifact-retention decisions, and next steps here so reports can
  be written from the repository instead of chat history.

## Changelog Maintenance

### 2026-06-26

- Added a workspace-level `AGENTS.md` instruction requiring future Phase 5
  updates to record report-relevant changes in this changelog.
- Added a repo-level `AGENTS.md` so the changelog discipline travels with the
  GitHub project and is visible to future project-local Codex chats.
- Linked this changelog from the master plan, ERAWAN runbook, and project-state
  document in the previous changelog commit.

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

### 10-Game Required Benchmark Result

Model:

- `models/rl/phase5_symbolic_policy_10shards.pt`

Command shape:

- `rl-evaluate --agent phase5-search`
- `--games-per-matchup 10`
- `--max-steps 600`

Overall result:

- Games: 360.
- Wins: 139.
- Losses: 220.
- Draws: 1.
- Timeouts: 1.
- Errors: 0.
- Win rate: 0.386.

Comparison against prior gates:

- Direct `phase5-symbolic` on the same 10-game benchmark: 109 / 360 wins,
  0.303 win rate, 3 timeouts, 0 errors.
- Rule baseline: 126 / 360 wins, 0.350 win rate, 5 timeouts, 0 errors.
- Old Phase 4-style trained `rl` model: 79 / 360 wins, 0.219 win rate.
- Old Phase 4-style hybrid model: 81 / 360 wins, 0.225 win rate.

Conclusion:

- This is the first Phase 5 agent in this run to beat the rule baseline on the
  required 10-game benchmark.
- Online one-turn root search improved the plain symbolic policy by 30 wins
  over 360 games, from 0.303 to 0.386 win rate.
- It also beat the rule baseline by 13 wins over 360 games, from 0.350 to 0.386
  win rate, with fewer timeouts and no errors.
- Treat this as a promotable signal for the online-search path, but not yet a
  final claim. The next step is to confirm robustness with a larger benchmark or
  repeated seeds and to add search telemetry to the report.

Strong matchups in this run:

- Crustle as our deck: 24 / 40 wins.
- Dragapult: 23 / 40 wins.
- Ogerpon Box: 20 / 40 wins.
- Several Iono's Bellibolt ex benchmark matchups were favorable, including
  Ogerpon Box at 8 / 10 and multiple decks at 6-7 / 10.

Weak matchups in this run:

- Alakazam Dudunsparce remained poor: 2 / 40 wins.
- Dragapult Dusknoir remained poor: 6 / 40 wins.
- Mega Lucario ex benchmark matchups were still difficult for several decks.

### Search Telemetry Implementation

Implementation update:

- Added `Phase5SearchPolicyAgent.search_telemetry()`.
- The agent now tracks:
  - searched decisions
  - Search API start count
  - search-changed decisions
  - search setup errors
  - candidate probes
  - candidate probe errors
  - truncated candidates
  - total, average, and maximum search seconds
- `run_phase4_required_benchmark` now aggregates telemetry per matchup row and
  across the whole report when `agent_kind == "phase5-search"`.
- `write_phase4_benchmark_report` now writes a `Search Telemetry` summary and a
  `Search Telemetry By Matchup` table in Markdown.
- The JSON report now includes `search_telemetry` at top level and per row when
  telemetry exists.
- Added unit coverage for telemetry rates and benchmark report rendering.

Verification:

- `tests.test_rl_phase5_symbolic_agent` and `tests.test_rl_phase4` passed:
  33 tests, 2 skipped.
- `py_compile` passed for:
  - `src/ptcg_abc/agent/phase5_search.py`
  - `src/ptcg_abc/rl/workflow.py`
  - `src/ptcg_abc/evaluation.py`

Next ERAWAN use:

- Pull the telemetry implementation.
- Re-run or continue confirmation with `AGENT=phase5-search`.
- Inspect the new report telemetry before promoting the result beyond the
  first positive 10-game benchmark.

### 30-Game Required Benchmark Confirmation

Model:

- `models/rl/phase5_symbolic_policy_10shards.pt`

Command shape:

- `rl-evaluate --agent phase5-search`
- `--games-per-matchup 30`
- `--max-steps 600`

Overall result:

- Games: 1,080.
- Wins: 408.
- Losses: 670.
- Draws: 2.
- Timeouts: 1.
- Errors: 0.
- Win rate: 0.378.

Search telemetry:

- Searched decisions: 44,114.
- Search-started decisions: 44,114.
- Search-changed decisions: 9,685.
- Search change rate: 0.21954481570476492.
- Search errors: 0.
- Search error rate: 0.0.
- Candidate probes: 161,386.
- Candidate errors: 0.
- Candidate error rate: 0.0.
- Truncated candidates: 6,395.
- Truncated candidate rate: 0.03962549415686615.
- Total search seconds: 2,592.746139023453.
- Average search seconds: 0.05877377111627721.
- Max search seconds: 2.4530896823853254.

Conclusion:

- The 30-game confirmation stayed above the rule baseline: `0.378` versus the
  previous rule baseline of `0.350`.
- The result is slightly below the first 10-game `phase5-search` benchmark
  (`0.386`), but still supports the same conclusion: online one-turn root search
  is better than the direct symbolic policy and current rule baseline.
- Telemetry is clean: no search errors, no candidate errors, low timeout count,
  and average search cost under 60 ms per searched decision.
- Truncation is nonzero at about 3.96% of candidate probes, so truncation
  examples should be inspected before treating the search scorer as final.

### Pairwise-Mid Search-Prior Probe

Model:

- `models/rl/phase5_symbolic_policy_10shards_pairwise_baseline_mid.pt`

Command shape:

- `rl-evaluate --agent phase5-search`
- `--games-per-matchup 10`
- `--max-steps 600`

Overall result:

- Games: 360.
- Wins: 138.
- Losses: 222.
- Draws: 0.
- Timeouts: 0.
- Errors: 0.
- Win rate: 0.383.

Search telemetry:

- Searched decisions: 14,684.
- Search-changed decisions: 3,389.
- Search change rate: about 0.231.
- Search errors: 0.
- Candidate errors: 0.
- Truncated candidates: 2,196.
- Average search seconds: 0.0628.
- Max search seconds: 1.4773.

Comparison:

- Plain symbolic search prior at 10 games per matchup: 139 / 360 wins, 0.386
  win rate, 1 timeout, 0 errors.
- Pairwise-mid search prior at 10 games per matchup: 138 / 360 wins, 0.383 win
  rate, 0 timeouts, 0 errors.
- Plain symbolic search prior at 30 games per matchup: 408 / 1,080 wins, 0.378
  win rate, 1 timeout, 0 errors.

Conclusion:

- Pairwise-mid is a clean and competitive search prior, but it did not beat the
  plain symbolic prior in the 10-game benchmark.
- Do not spend a 30-game confirmation on pairwise-mid yet unless we specifically
  need a variance check. Keep the plain symbolic checkpoint as the current
  `phase5-search` prior.
- The next higher-value work is search telemetry analysis, truncation inspection,
  stable Search API wrapper refactor, and value/Q/tactical heads.

### Search Evaluation Trace Output

Implementation update:

- Added optional `rl-evaluate --search-trace-output PATH`.
- `run_phase4_required_benchmark` now writes `phase5-search` root-search traces
  to JSONL when a trace output path is provided.
- The trace writer enriches in-memory search traces with benchmark metadata:
  game index, deck index, deck label, and benchmark opponent.
- `scripts/slurm/phase5_symbolic_eval_conda.sbatch` now accepts optional
  `SEARCH_TRACE_OUTPUT=...`.
- Added tests for the CLI option and trace metadata enrichment.

Verification:

- `tests.test_rl_phase5_symbolic_agent` and `tests.test_rl_phase4` passed:
  34 tests, 2 skipped.
- `py_compile` passed for:
  - `src/ptcg_abc/cli.py`
  - `src/ptcg_abc/rl/workflow.py`
  - `src/ptcg_abc/agent/phase5_search.py`
  - `src/ptcg_abc/evaluation.py`

Next ERAWAN use:

- Run a small `SEARCH_TRACE_OUTPUT` job, usually 1-3 games per matchup, and
  inspect changed decisions plus truncated candidates before changing the search
  scorer.

### 3-Game Trace-Capture Result

Model:

- `models/rl/phase5_symbolic_policy_10shards.pt`

Command shape:

- `rl-evaluate --agent phase5-search`
- `--games-per-matchup 3`
- `--max-steps 600`
- `--search-trace-output experiments/rl/phase5_search_agent_plain_trace_3g.jsonl`

Overall result:

- Games: 108.
- Wins: 39.
- Losses: 69.
- Draws: 0.
- Timeouts: 0.
- Errors: 0.
- Win rate: 0.361.

Search telemetry:

- Searched decisions: 4,513.
- Search-started decisions: 4,513.
- Search-changed decisions: 1,040.
- Search change rate: 0.230.
- Search errors: 0.
- Search error rate: 0.000.
- Candidate probes: 16,459.
- Candidate errors: 0.
- Truncated candidates: 699.
- Truncated candidate rate: about 0.0425.
- Average search seconds: 0.0613.
- Max search seconds: 1.9257.

Notable truncation concentrations:

- Deck 3 vs Mega Abomasnow ex: 67 truncated candidates from 485 probes.
- Deck 8 vs Mega Lucario ex: 58 from 542.
- Deck 8 vs Mega Abomasnow ex: 57 from 610.
- Deck 3 vs Iono's Bellibolt ex: 44 from 476.
- Deck 1 vs Mega Abomasnow ex: 41 from 602.
- Deck 5 vs Iono's Bellibolt ex: 41 from 489.

Conclusion:

- The trace-capture run is clean: 0 search errors, 0 candidate errors, 0
  timeouts, and average search time around 61 ms.
- The 3-game win rate is lower than the 10-game and 30-game confirmations, but
  it is only a trace-capture smoke and should not override the larger benchmark
  conclusions.
- The next action is to inspect the JSONL trace examples, especially truncated
  candidates in the concentrated deck/opponent pairs above.

### Trace Example Inspection

Sample inspected:

- First 10 truncated-record examples from
  `experiments/rl/phase5_search_agent_plain_trace_3g.jsonl`.

Observed in the pasted sample:

- Example objects inspected: 10.
- Changed examples: 1.
- Records where all candidates were truncated: 5.
- Records where the selected candidate was truncated: 5.
- Changed records where the selected candidate was truncated: 0.

Interpretation:

- The shown changed decision selected a non-truncated Ability candidate over a
  baseline Evolve candidate, with no search/candidate errors.
- In the examples where every candidate truncated, search usually returned the
  policy baseline. Those rows are less dangerous for action selection, but they
  do not provide meaningful tactical discrimination; the policy prior dominates.
- Truncated candidates often appear among Evolve/Ability/Attach sequences in
  mid-turn states. This suggests the current rollout cap can cut off long
  intra-turn chains, especially for evolution/setup turns.
- The pasted examples do not show a changed decision caused by a truncated
  selected candidate. Before changing the scorer, run a full-trace summary for:
  selected-candidate truncation rate, changed-selected truncation rate, all-
  candidates-truncated rate, and action-type/deck/opponent concentration.

### Full Trace Truncation Summary

Trace:

- `experiments/rl/phase5_search_agent_plain_trace_3g.jsonl`

Observed full-trace counts:

- Records: 4,513.
- Changed records: 1,040.
- All-candidates-truncated records: 72.
- Selected-truncated records: 97.
- Changed selected-truncated records: 43.
- Selected-truncated by type:
  - `PLAY`: 35.
  - `ABILITY`: 25.
  - `EVOLVE`: 20.
  - `ATTACH`: 16.
  - `RETREAT`: 1.
- Top selected-truncated matchups:
  - Deck 3 vs Mega Abomasnow ex: 12.
  - Deck 8 vs Mega Lucario ex: 11.
  - Deck 1 vs Mega Abomasnow ex: 7.
  - Deck 8 vs Mega Abomasnow ex: 7.
  - Deck 9 vs Iono's Bellibolt ex: 6.
  - Deck 1 vs Mega Lucario ex: 5.
  - Deck 3 vs Iono's Bellibolt ex: 5.
  - Deck 8 vs Iono's Bellibolt ex: 5.

Interpretation:

- Selected-truncated decisions are uncommon overall: 97 / 4,513, about 2.15%.
- Changed selected-truncated decisions are more relevant: 43 / 1,040 changed
  records, about 4.13%.
- This is not large enough to explain the whole search behavior, but it is
  significant enough to track before tuning the rollout cap or tactical scorer.
- Truncation affects several action types, especially `PLAY`, `ABILITY`,
  `EVOLVE`, and `ATTACH`, so the issue is not isolated to one action family.

Implementation update:

- Extended `diagnose_search_traces` with selected-truncation statistics,
  all-candidates-truncated records, rates, selected-truncated action-type
  counts, and selected-truncated matchup concentrations.
- Added standalone CLI:
  - `rl-diagnose-search-traces`
- Added Markdown/JSON report output for trace-only diagnostics.
- Updated the ERAWAN runbook to use the new command before manual example
  inspection.

Verification:

- `tests.test_rl_phase4` and `tests.test_rl_phase5_symbolic_agent` passed:
  34 tests, 2 skipped.
- `py_compile` passed for:
  - `src/ptcg_abc/cli.py`
  - `src/ptcg_abc/rl/phase5_diagnostics.py`

### Search Rollout-Cap Experiment Support

Implementation update:

- Added online `phase5-search` evaluation overrides:
  - `rl-evaluate --search-top-k N`
  - `rl-evaluate --search-rollout-steps N`
- `run_phase4_required_benchmark` now passes an optional `RootSearchConfig` into
  `Phase5SearchPolicyAgent`.
- `scripts/slurm/phase5_symbolic_eval_conda.sbatch` now accepts:
  - `SEARCH_TOP_K`
  - `SEARCH_ROLLOUT_STEPS`
- Updated the runbook with a small cap-30 trace/eval comparison against the
  default cap-18 behavior.

Verification:

- `tests.test_rl_phase5_symbolic_agent` and `tests.test_rl_phase4` passed:
  34 tests, 2 skipped.
- `py_compile` passed for:
  - `src/ptcg_abc/cli.py`
  - `src/ptcg_abc/rl/workflow.py`

Next ERAWAN use:

- Run a small `SEARCH_ROLLOUT_STEPS=30` trace job and compare against the default
  cap-18 trace diagnostics on selected-truncated and changed-selected-truncated
  rates before changing the default search config.

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

- Treat online `phase5-search` with the plain symbolic checkpoint as the current
  best Phase 5 agent path, pending broader/deeper evaluation.
- Optionally repeat the 30-game benchmark with a different seed once seed control
  is exposed or if variance remains a concern.
- Compare online search against:
  - direct `phase5-symbolic`
  - rule baseline
  - best supervised pairwise checkpoint; pairwise-mid 10g is complete and did
    not beat the plain symbolic prior.
- Decide whether one-turn search should use:
  - plain symbolic checkpoint
  - baseline-mid checkpoint, only if later evidence overturns the first probe
  - a new value/Q/tactical scorer once implemented
- Refactor reusable Search API code out of `phase5_search.py` into a stable
  wrapper used by both data generation and online evaluation.
- Inspect the captured trace JSONL for truncation examples and search
  disagreements before tuning search scoring.
- Implement value, Q, and auxiliary tactical heads after the online search gate.
- Continue toward the full Phase 5 plan only after the current online search
  slice produces measurable battle evidence.
