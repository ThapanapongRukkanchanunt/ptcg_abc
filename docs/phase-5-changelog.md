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

### Search Rollout-Cap 30 Trace Result

Trace diagnostic:

- Trace: `experiments/rl/phase5_search_agent_plain_trace_3g_cap30.jsonl`.
- Report: `reports/phase5_search_agent_plain_trace_3g_cap30_diagnostics.md`.

Observed cap-30 counts:

- Records: 4,239.
- Comparable records: 4,239.
- Changed records: 927.
- Candidate probes: 15,534.
- Search errors: 0.
- Candidate errors: 0.
- All-candidates-truncated records: 0.
- Truncated candidates: 14.
- Truncated candidate rate: 0.000901.
- Selected-truncated records: 1.
- Selected-truncated rate: 0.000236.
- Changed selected-truncated records: 1.
- Changed selected-truncated rate: 0.001079.
- Mean search-minus-baseline combined score: 0.657376.
- Mean search-minus-baseline tactical score: 0.664480.

Comparison with the default cap-18 3-game trace:

- All-candidates-truncated records improved from 72 / 4,513 to 0 / 4,239.
- Selected-truncated records improved from 97 / 4,513 to 1 / 4,239.
- Changed selected-truncated records improved from 43 / 1,040 changed records to
  1 / 927 changed records.
- The remaining selected-truncated case was one `ATTACH` action in Deck 3 vs
  Mega Abomasnow ex.

Interpretation:

- Cap 30 substantially reduces the trace-risk introduced by rollouts ending
  before candidate sequences finish.
- This is strong enough to proceed to a required benchmark comparison with
  `SEARCH_ROLLOUT_STEPS=30`.
- The following 10-game benchmark resolved the promotion gate and made cap 30
  the current default.

### Search Rollout-Cap 30 10-Game Trace Diagnostic

Trace diagnostic:

- Trace: `experiments/rl/phase5_search_agent_plain_10g_cap30.jsonl`.
- Report: `reports/phase5_search_agent_plain_10g_cap30_diagnostics.md`.

Observed cap-30 10-game trace counts:

- Records: 14,680.
- Comparable records: 14,680.
- Changed records: 3,218.
- Candidate probes: 53,890.
- Search errors: 0.
- Candidate errors: 0.
- All-candidates-truncated records: 63.
- All-candidates-truncated rate: 0.004292.
- Truncated candidates: 353.
- Truncated candidate rate: 0.006550.
- Selected-truncated records: 64.
- Selected-truncated rate: 0.004360.
- Changed selected-truncated records: 15.
- Changed selected-truncated rate: 0.004661.
- Mean search-minus-baseline combined score: 0.628433.
- Mean search-minus-baseline tactical score: 0.635204.

Selected-truncated concentration:

- By action type:
  - `PLAY`: 31.
  - `EVOLVE`: 15.
  - `ATTACH`: 10.
  - `ABILITY`: 7.
  - `ATTACK`: 1.
- By matchup:
  - Deck 1 vs Mega Lucario ex: 33.
  - Deck 1 vs Iono's Bellibolt ex: 26.
  - Deck 1 vs Crustle: 3.
  - Deck 3 vs Iono's Bellibolt ex: 2.

Interpretation:

- Cap 30 remains much cleaner than the prior cap-18 10-game telemetry headline:
  truncated candidates dropped from 2,196 to 353.
- The remaining selected-truncated risk is low: 64 / 14,680 records overall and
  15 / 3,218 changed records.
- The remaining truncation is concentrated in Deck 1 matchups, especially vs
  Mega Lucario ex and Iono's Bellibolt ex, so future inspection should start
  there if the cap-30 benchmark underperforms.
- The benchmark gate below resolves whether this cleaner cap should become the
  default.

### Search Rollout-Cap 30 Promotion

Benchmark:

- Agent: `phase5-search`.
- Model: `models/rl/phase5_symbolic_policy_10shards.pt`.
- Games per matchup: 10.
- Max selections per game: 600.
- Search rollout cap: 30.

Observed benchmark summary:

- Games: 360.
- Wins: 148.
- Losses: 211.
- Draws: 1.
- Timeouts: 1.
- Errors: 0.
- Win rate: 0.411.
- Searched decisions: 14,680.
- Search-changed decisions: 3,218.
- Average search seconds: 0.0631.
- Max search seconds: 3.4057.

Comparison against the cap-18 10-game result:

- Win rate improved from 0.386 to 0.411.
- Wins improved from 139 / 360 to 148 / 360.
- Timeouts stayed at 1.
- Errors stayed at 0.
- Average search seconds remained effectively flat: 0.0628 to 0.0631.
- Max search seconds increased from 1.4773 to 3.4057, so max latency should be
  monitored in the next 30-game confirmation run.
- Trace safety improved substantially:
  - Truncated candidates: 2,196 to 353.
  - Selected-truncated records: 64 / 14,680 under cap 30.
  - Changed selected-truncated records: 15 / 3,218 under cap 30.

Implementation update:

- Promoted `RootSearchConfig.max_rollout_steps` from 18 to 30.
- Kept CLI and SLURM overrides available:
  - `rl-evaluate --search-rollout-steps N`
  - `SEARCH_ROLLOUT_STEPS=N`
- Added a regression test asserting the promoted default cap is 30.

Decision:

- Treat cap 30 as the current default for Phase 5 online one-turn root search.
- Next benchmark should be a 30-game required run with the default cap 30,
  including trace output, to confirm win rate and monitor the higher max latency.

### Search Rollout-Cap 30 30-Game Trace Diagnostic

Trace diagnostic:

- Trace: `experiments/rl/phase5_search_agent_plain_30g_cap30_default.jsonl`.
- Report: `reports/phase5_search_agent_plain_30g_cap30_default_diagnostics.md`.

Observed cap-30 default 30-game trace counts:

- Records: 44,811.
- Comparable records: 44,811.
- Changed records: 9,693.
- Candidate probes: 163,873.
- Search errors: 0.
- Candidate errors: 0.
- All-candidates-truncated records: 133.
- All-candidates-truncated rate: 0.002968.
- Truncated candidates: 777.
- Truncated candidate rate: 0.004741.
- Selected-truncated records: 140.
- Selected-truncated rate: 0.003124.
- Changed selected-truncated records: 32.
- Changed selected-truncated rate: 0.003301.
- Mean search-minus-baseline combined score: 0.605419.
- Mean search-minus-baseline tactical score: 0.612222.

Selected-truncated concentration:

- By action type:
  - `PLAY`: 61.
  - `EVOLVE`: 36.
  - `ATTACH`: 27.
  - `ABILITY`: 14.
  - `ATTACK`: 1.
  - `RETREAT`: 1.
- By matchup:
  - Deck 1 vs Mega Lucario ex: 63.
  - Deck 1 vs Iono's Bellibolt ex: 35.
  - Deck 1 vs Crustle: 22.
  - Deck 1 vs Mega Abomasnow ex: 11.
  - Deck 3 vs Mega Abomasnow ex: 4.
  - Deck 3 vs Mega Lucario ex: 2.
  - Deck 5 vs Mega Lucario ex: 1.
  - Deck 5 vs Iono's Bellibolt ex: 1.
  - Deck 8 vs Mega Lucario ex: 1.

Interpretation:

- The default cap-30 search remains operationally clean at 30-game scale:
  no search errors and no candidate errors.
- Truncation remains low: 777 / 163,873 candidate probes, 140 / 44,811 selected
  records, and 32 / 9,693 changed records.
- Remaining selected truncation is strongly concentrated in Deck 1 matchups, so
  future trace inspection should prioritize Alakazam Dudunsparce lines.
- The benchmark win/loss/timing summary for the 30-game default-cap run is still
  needed to complete the confirmation record.

## Next Phase 5 Training Track

Historical note: this was the active plan before the July 2, 2026
AlphaStar-like league replacement plan below. It is retained for context but is
no longer the current next-action track.

The next plan follows this order:

1. Generate `phase5-search` self-play data.
2. Add value, Q/action-value, and tactical heads to the symbolic model.
3. Train a generalist model from a mixture of:
   - rule demonstrations,
   - search-improved decisions,
   - search/self-play outcome trajectories.
4. Evaluate the resulting direct and search agents on the current 9-deck
   required benchmark.
5. Expand to more decks only after the 9-deck gate is stable. If the broader
   deck set exposes capacity limits, increase the model size before scaling
   reinforcement learning.
6. Start larger PPO/self-play only after the generalist supervised/value model
   clears the benchmark gate.

Implementation implications:

- The immediate next code slice should be a Phase 5 self-play collector, not PPO.
- The collector should run `phase5-search` versus `phase5-search`, write
  trajectory records with final outcome targets, preserve per-player metadata,
  and optionally write search traces for sampled games.
- The multi-head trainer should consume three data families through one symbolic
  adapter/encoder path:
  - imitation/action labels from rule demonstrations,
  - search labels and candidate scores from search-improved data,
  - value/outcome and tactical targets from self-play trajectories.
- PPO remains explicitly downstream; it should not start until the supervised
  generalist model beats or matches the current search baseline on the required
  benchmark.

### ERAWAN Game-Data Storage Policy

User directive:

- Future game-data generation should write large generated datasets under
  `/project/SIGGI/thapanapong.r@cmu.ac.th`.
- `reports/`, `models/`, and `experiments/` should stay in the repository.

Implementation update:

- Updated `scripts/slurm/phase5_search_data_array.sbatch`:
  - Adds `GAME_DATA_ROOT`, defaulting to
    `/project/SIGGI/thapanapong.r@cmu.ac.th`.
  - Writes future search decision shards to
    `$GAME_DATA_ROOT/phase5_search/shards`.
  - Keeps search traces and summaries under `experiments/rl/phase5_search`.
  - Defaults `ROLLOUT_STEPS` to 30, matching the promoted cap-30 search default.
- Updated `scripts/slurm/phase5_merge_train_conda.sbatch`:
  - Reads future decision shards from `$GAME_DATA_ROOT/phase5_search/shards`.
  - Writes merged decision datasets to `$GAME_DATA_ROOT`.
  - Keeps merged traces, merge manifests, training reports, model checkpoints,
    and exported models in their existing `experiments/`, `reports/`, and
    `models/` locations.
- Updated symbolic train/diagnostic SLURM defaults to read merged datasets from
  `$GAME_DATA_ROOT`.
- Updated `rl-generate-search-data` CLI default rollout cap to the promoted
  `RootSearchConfig` value, currently 30.
- Updated the ERAWAN runbook and project-state with the storage convention.

### Phase 5 Search Self-Play Collector

Implementation update:

- Added dedicated CLI:
  - `rl-generate-phase5-search-selfplay`
- Added SLURM wrapper:
  - `scripts/slurm/phase5_search_selfplay_conda.sbatch`
- Extended `rollout_selfplay_games` to support:
  - `phase5-search` cap-30 self-play,
  - deterministic `game_offset` for array shards,
  - optional `RootSearchConfig` overrides,
  - aggregate search telemetry,
  - optional sampled search trace JSONL output.
- Extended `SelfPlaySummary` with:
  - `search_telemetry`,
  - `trace_path`,
  - `trace_records`.
- Future self-play trajectory shards default to:
  - `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay/shards`.
- Reports and optional search traces stay under:
  - `experiments/rl/phase5_search_selfplay`.
- Updated the ERAWAN runbook with:
  - a 2-game smoke job on decks 1 and 2,
  - a bounded two-shard 25-game-per-shard job over the current 9-deck pool,
  - inspection and pass-gate commands.

Purpose:

- This is the first step of the revised next Phase 5 plan:
  generate `phase5-search` self-play data before adding value/Q/tactical heads.
- The collector writes trajectory records with final outcome targets that can
  train value and action-value heads in the next slice.

Verification:

- `tests.test_rl_phase4` and `tests.test_rl_phase5_symbolic_agent` passed:
  35 tests, 2 skipped.
- `py_compile` passed for:
  - `src/ptcg_abc/cli.py`
  - `src/ptcg_abc/rl/workflow.py`
- Parser smoke confirmed `rl-generate-phase5-search-selfplay` accepts game
  offset, deck subset, search overrides, and trace sampling flags.

### Phase 5 Search Self-Play Smoke Result

Run:

- Agent: `phase5-search`.
- Deck subset: decks 1 and 2.
- Games requested: 2.
- Max steps: 600.
- Trace games: 2.
- Output:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay/shards/phase5_search_selfplay_shard-0.jsonl`.
- Trace:
  `experiments/rl/phase5_search_selfplay/traces/phase5_search_selfplay_traces_shard-0.jsonl`.

Observed summary:

- Games started: 2 / 2.
- Steps written: 298.
- Errors: 0.
- Timeouts: 0.
- Draws: 0.
- Trace records: 153.
- Search decisions: 153.
- Search-changed decisions: 24.
- Change rate: 0.156863.
- Candidate probes: 520.
- Search errors: 0.
- Candidate errors: 0.
- Truncated candidates: 1.
- Average search seconds: 0.019357.
- Max search seconds: 0.075529.

Interpretation:

- The Phase 5 search self-play collector is working on ERAWAN.
- It produced valid trajectory records with final outcome targets and sampled
  search traces.
- Search telemetry is clean enough to proceed to the bounded two-shard
  self-play job over the current 9-deck pool.

### Phase 5 Search Self-Play Bounded Two-Shard Result

Run:

- Agent: `phase5-search`.
- Model: `models/rl/phase5_symbolic_policy_10shards.pt`.
- Deck pool: current 9 prepared decks.
- Shards: 2.
- Games per shard: 25.
- Max steps: 600.
- Trace games per shard: 1.
- Data root:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay/shards`.
- Reports and traces:
  `experiments/rl/phase5_search_selfplay`.

Combined observed summary:

- Games started: 50 / 50.
- Steps written: 8,424.
- Errors: 0.
- Timeouts: 0.
- Draws: 0.
- Trace records: 165.
- Search decisions: 4,468.
- Search-changed decisions: 942.
- Search-change rate: 0.210833.
- Candidate probes: 16,344.
- Search errors: 0.
- Candidate errors: 0.
- Truncated candidates: 63.
- Truncated-candidate rate: 0.003855.
- Average search seconds: 0.081298.
- Max search seconds: 3.808336.
- Deck A wins: 17.
- Deck B wins: 33.

Shard 0:

- Games started: 25 / 25.
- Steps written: 3,825.
- Search decisions: 2,099.
- Search-changed decisions: 436.
- Candidate probes: 7,606.
- Truncated candidates: 58.
- Average search seconds: 0.060574.
- Max search seconds: 3.808336.
- Trace records: 75.

Shard 1:

- Games started: 25 / 25.
- Steps written: 4,599.
- Search decisions: 2,369.
- Search-changed decisions: 506.
- Candidate probes: 8,738.
- Truncated candidates: 5.
- Average search seconds: 0.099660.
- Max search seconds: 1.682883.
- Trace records: 90.

Interpretation:

- The bounded two-shard self-play gate passed: no errors, no timeouts, nonzero
  trajectory rows, clean search telemetry, and a stable roughly 21% search-change
  rate.
- Truncation is present but low at this scale, so it should not block a larger
  self-play dataset.
- The next dataset should be generated by a dedicated two-shard large launcher,
  not by overwriting the bounded shard directory.

### Phase 5 Search Self-Play Large Launcher

Implementation:

- Added:
  - `scripts/slurm/phase5_search_selfplay_2shard_10k.sbatch`.
- The launcher submits as a two-task SLURM array with:
  - `TOTAL_GAMES=10000` by default,
  - `GAMES_PER_SHARD=5000` by default,
  - the current 9-deck self-play pool,
  - `models/rl/phase5_symbolic_policy_10shards.pt` as the default
    `phase5-search` prior,
  - `MAX_STEPS=600` by default.
- The launcher writes trajectory shards to:
  - `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_10k/shards`.
- It keeps run summaries and optional sampled traces in:
  - `experiments/rl/phase5_search_selfplay_10k`.
- It refuses `SEARCH_TRACE_GAMES=0` because the CLI interprets zero as
  "trace all games", which is unsafe for a large run.
- The ERAWAN runbook now includes launch, inspection, pass-gate, and
  `TOTAL_GAMES=1000` override guidance.

### Phase 5 10,000-Game Search Self-Play Result

Run:

- Agent: `phase5-search`.
- Model: `models/rl/phase5_symbolic_policy_10shards.pt`.
- Deck pool: current 9 prepared decks.
- Shards: 2.
- Games per shard: 5,000.
- Max steps: 600.
- Data root:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_10k/shards`.
- Reports and sampled traces:
  `experiments/rl/phase5_search_selfplay_10k`.

Combined observed summary:

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
- Search-change rate: 0.213495.
- Candidate probes: 3,011,687.
- Search errors: 0.
- Candidate errors: 0.
- Truncated candidates: 12,338.
- Truncated-candidate rate: 0.004097.
- Average search seconds: 0.080765.
- Max search seconds: 4.811727.
- Trace records: 146.

Interpretation:

- Stage 1 self-play data generation is complete for the current 9-deck Phase 5
  pool.
- The dataset is large enough for the first value/Q/tactical generalist
  training slice.
- Search/candidate errors remained zero at 10,000 games.
- Timeout rate is low enough to keep the dataset, but timeout matchups should be
  watched in later deck expansion.
- Truncation stayed low at about 0.41% of candidate probes.

### Phase 5 Generalist Multi-Head Training Slice

Implementation:

- Extended `AlphaStarTurnPolicy` with action-conditioned auxiliary heads:
  - `action_q`,
  - `tactical_score`.
- Kept the existing checkpoint format and made symbolic inference/diagnostics
  load checkpoints with `strict=False` so older checkpoints without the new heads
  still run.
- Extended `Phase5SymbolicDecisionRecord` with optional:
  - `value_target`,
  - `action_value_targets`,
  - `action_value_mask`,
  - `tactical_targets`,
  - `tactical_mask`.
- Added streaming trajectory reader support for `TrajectoryStep` self-play
  JSONL.
- Added `phase5_symbolic_record_from_trajectory`, which uses the recorded
  `chosen_indices` as policy/Q positives and the stored self-play reward as the
  state-value and selected-action Q target.
- Added `train_phase5_generalist_policy`, a streaming mixed trainer for:
  - search-improved decisions,
  - baseline/rule demonstration targets,
  - phase5-search self-play outcomes.
- Added CLI:
  - `rl-train-phase5-generalist`.
- Added SLURM wrapper:
  - `scripts/slurm/phase5_generalist_train_conda.sbatch`.
- Updated the ERAWAN runbook with:
  - bounded smoke-train command,
  - full 10k mixed generalist-train command,
  - post-train direct symbolic and `phase5-search` smoke evaluation commands.

Current head targets:

- Policy target:
  - search-improved target for decision records,
  - baseline/rule target for rule-demo duplicates,
  - actual `phase5-search` chosen action for self-play records.
- State-value target:
  - terminal/shaped self-play reward.
- Action-Q target:
  - terminal/shaped self-play reward on selected actions only.
- Tactical target:
  - normalized rule/tactical prior for every legal action. This is a first
    auxiliary target and can later be augmented with root-search trace tactical
    scores.

Next operational step:

- Submit the bounded `phase5_generalist_policy_smoke.pt` training job from the
  runbook.
- If nonzero decision/rule/self-play/value/Q/tactical counts and finite loss are
  reported, submit the full `phase5_generalist_policy_10k.pt` job.

### Phase 5 Generalist Evaluation Result

Artifacts evaluated:

- Model:
  - `models/rl/phase5_generalist_policy_10k.pt`.
- Direct policy reports:
  - `reports/phase5_generalist_symbolic_10g.json`
  - `reports/phase5_generalist_symbolic_10g.md`
  - `reports/phase5_generalist_symbolic_30g.json`
  - `reports/phase5_generalist_symbolic_30g.md`
- Search policy reports:
  - `reports/phase5_generalist_search_10g.json`
  - `reports/phase5_generalist_search_10g.md`
  - `reports/phase5_generalist_search_30g.json`
  - `reports/phase5_generalist_search_30g.md`

Direct `phase5-symbolic` with the generalist checkpoint:

- 10-game benchmark:
  - Games: 360.
  - Wins: 117.
  - Losses: 242.
  - Draws: 1.
  - Timeouts: 4.
  - Errors: 0.
  - Win rate: 0.325.
- 30-game benchmark:
  - Games: 1,080.
  - Wins: 361.
  - Losses: 716.
  - Draws: 3.
  - Timeouts: 12.
  - Errors: 0.
  - Win rate: 0.334.

`phase5-search` with the generalist checkpoint as prior:

- 10-game benchmark:
  - Games: 360.
  - Wins: 138.
  - Losses: 222.
  - Draws: 0.
  - Timeouts: 0.
  - Errors: 0.
  - Win rate: 0.383.
  - Searched decisions: 14,661.
  - Search-changed decisions: 3,002.
  - Change rate: 0.205.
  - Search errors: 0.
  - Candidate errors: 0.
  - Truncated candidates: 261.
  - Truncated-candidate rate: 0.00487.
  - Average search seconds: 0.0536.
  - Max search seconds: 1.4181.
- 30-game benchmark:
  - Games: 1,080.
  - Wins: 414.
  - Losses: 665.
  - Draws: 1.
  - Timeouts: 5.
  - Errors: 0.
  - Win rate: 0.383.
  - Searched decisions: 44,267.
  - Search-changed decisions: 8,584.
  - Change rate: 0.194.
  - Search errors: 0.
  - Candidate errors: 0.
  - Truncated candidates: 677.
  - Truncated-candidate rate: 0.00419.
  - Average search seconds: 0.0514.
  - Max search seconds: 2.4492.

Comparison to earlier gates:

- Direct generalist improved over the first direct symbolic 10-game benchmark:
  - 117 / 360, 0.325 vs. 109 / 360, 0.303.
- Direct generalist is still below the rule baseline:
  - 117 / 360, 0.325 vs. rule baseline 126 / 360, 0.350.
- `phase5-search` with the generalist prior is slightly above the previous
  30-game search benchmark:
  - 414 / 1,080, 0.383 vs. 408 / 1,080, 0.378.
- The generalist prior materially reduced truncation in the 30-game search run:
  - 677 truncated candidates vs. the prior default-cap 30-game run's 6,395.
- Average search latency also improved:
  - 0.0514 seconds vs. 0.0588 seconds.

Interpretation:

- The multi-head generalist is useful as a search prior, but not yet strong
  enough as a standalone direct policy.
- The new best 9-deck inference path is `phase5-search` with
  `models/rl/phase5_generalist_policy_10k.pt`.
- The 30-game result is a small but real improvement over the previous
  `phase5-search` 30-game benchmark, with much cleaner truncation telemetry.
- Deck 1, Alakazam Dudunsparce, remains the largest weakness. It produced only
  4 / 120 wins under the 30-game `phase5-search` generalist-prior benchmark.
- The 9-deck generalist/search gate is stable enough to move to the next plan:
  broaden deck coverage and add targeted data/model capacity before PPO.

## 2026-06-29 - Phase 5 13-Deck Self-Play Pool Wired

Implementation:

- Added `phase5_league_prepared_decks()` in `src/ptcg_abc/evaluation.py`.
- The new pool is opt-in and preserves existing benchmark semantics:
  - deck indices 1-9 are the current Tournament 559 decks,
  - deck indices 10-13 are the four required sample decks: Crustle,
    Mega Lucario ex, Mega Abomasnow ex, and Iono's Bellibolt ex.
- Added `--deck-pool` to `rl-generate-phase5-search-selfplay`.
- Added `DECK_POOL` support to:
  - `scripts/slurm/phase5_search_selfplay_conda.sbatch`,
  - `scripts/slurm/phase5_search_selfplay_2shard_10k.sbatch`.
- Self-play summaries now include `deck_pool`.
- Self-play trajectory metadata now includes `selfplay_deck_pool`, so mixed
  9-deck and 13-deck training datasets can be separated later.
- Added unit coverage for the 13-deck pool count, indices, card counts, first
  nine tournament ranks, and four required sample deck names.
- Updated `docs/phase-5-erawan-runbook.md` with a bounded 338-game 13-deck
  smoke and the follow-up 13-deck 10k data command.
- Updated `docs/project-state.md` with the new state and next ERAWAN action.

Operational plan:

- Keep `models/rl/phase5_generalist_policy_10k.pt` as the default
  `phase5-search` prior for the expanded self-play data.
- First run `phase5_search_selfplay_13deck_338`, a 338-game two-shard smoke
  covering the 13 x 13 ordered pair grid with player-order alternation.
- If both shards finish with `errors=0`, `search_errors=0`, and
  `candidate_errors=0`, submit `phase5_search_selfplay_13deck_10k`.
- Do not remove the existing 9-deck `phase5_search_selfplay_10k` shards yet;
  they remain the source for the current generalist checkpoint and comparisons.

## 2026-06-29 - Phase 5 13-Deck Self-Play Smoke Passed

Artifacts:

- `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_13deck_338/shards/phase5_search_selfplay_shard-0.jsonl`
- `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_13deck_338/shards/phase5_search_selfplay_shard-1.jsonl`
- `experiments/rl/phase5_search_selfplay_13deck_338/summaries/phase5_search_selfplay_summary_shard-0.json`
- `experiments/rl/phase5_search_selfplay_13deck_338/summaries/phase5_search_selfplay_summary_shard-1.json`
- `experiments/rl/phase5_search_selfplay_13deck_338/traces/phase5_search_selfplay_traces_shard-0.jsonl`
- `experiments/rl/phase5_search_selfplay_13deck_338/traces/phase5_search_selfplay_traces_shard-1.jsonl`

Aggregate result:

- Games requested / started: 338 / 338.
- Deck pool: `league-13`.
- Pair count: 169 in each shard.
- Deck indices: 1-13 in each shard.
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

Conclusion:

- The expanded 13-deck self-play data path is valid.
- The low timeout count is acceptable for the smoke and does not indicate a
  search failure.
- Proceed to `phase5_search_selfplay_13deck_10k` using
  `models/rl/phase5_generalist_policy_10k.pt` as the `phase5-search` prior.

## 2026-06-29 - Phase 5 Kaggle Package Candidates

Decision:

- Use the current best recorded inference path for Kaggle package candidates:
  `phase5-search` with `models/rl/phase5_generalist_policy_10k.pt`.
- Package two Tournament 559 decks based on the 30-game per-matchup
  `phase5-search` generalist-prior per-deck totals:
  - Deck 2, Crustle, 68 / 120 wins, 0.567 win rate, 0 errors.
  - Deck 8, Dragapult Blaziken, 56 / 120 wins, 0.467 win rate, 0 errors.
- Hydrapple tied Dragapult Blaziken at 56 / 120, but Dragapult Blaziken was
  selected as the second package for deck-profile diversity.

Implementation:

- Added `phase5-package`, a dedicated packaging command for Phase 5
  `phase5-search` submission bundles.
- The package builder includes:
  - `main.py`,
  - `deck.csv`,
  - `model.pt`,
  - the Kaggle `cg` package,
  - the full local `ptcg_abc` package needed by Phase 5 symbolic/search
    inference.
- Submission `main.py` resolves bundled files relative to itself and chooses a
  submission-safe opponent prior from the Phase 5 13-deck pool using visible
  opponent card IDs. If no opponent cards are visible yet, it defaults to the
  required-sample Crustle prior.

Local artifacts:

- `submissions/phase5_search_generalist_10k/deck-02-crustle/submission.tar.gz`
- `submissions/phase5_search_generalist_10k/deck-08-dragapult-blaziken/submission.tar.gz`
- `submissions/phase5_search_generalist_10k/deck-02-crustle-phase5-search-submission.zip`
- `submissions/phase5_search_generalist_10k/deck-08-dragapult-blaziken-phase5-search-submission.zip`

Packaging correction:

- The zip artifacts were rewritten as direct Kaggle archives after an upload
  failure showed that wrapping `submission.tar.gz` inside a zip does not expose
  root-level `main.py`.
- The corrected zip archive roots contain `main.py`, `deck.csv`, `model.pt`,
  `cg/`, and `ptcg_abc/`.
- A later submit attempt failed because Kaggle raw-exec did not define
  `__file__`, while generated `main.py` used `__file__` during module import.
  The package template now discovers the agent root from safe candidates:
  generated-file directory when available, current working directory, and
  `/kaggle_simulations/agent`.
- The corrected zips were rebuilt and smoke-tested by executing `main.py`
  without `__file__` in globals. Both packages loaded, exposed callable
  `agent`, resolved their package root, and read 60-card `deck.csv` files.

Verification:

- `py_compile` passed for modified packaging files.
- `phase5-package --help` is available through the CLI.
- Both generated packages contain `main.py`, `deck.csv`, `model.pt`, `cg`,
  `ptcg_abc/agent/phase5_search.py`, `ptcg_abc/agent/phase5_symbolic.py`,
  `ptcg_abc/rl/phase5_policy.py`, and `ptcg_abc/rl/phase5_search.py`.
- Both `deck.csv` files contain exactly 60 cards.
- Local Python does not have PyTorch installed, so full checkpoint inference
  and battle validation must run on ERAWAN or Kaggle.

## 2026-06-29 - Phase 5 13-Deck Generalist Train Prep

Inspection:

- The generalist trainer already consumes multiple self-play trajectory shards:
  the CLI repeats `--selfplay-dataset`, and
  `scripts/slurm/phase5_generalist_train_conda.sbatch` expands the
  space-separated `SELFPLAY_DATASETS` environment variable.
- No trainer code change is required to mix:
  - the existing 10-shard search-decision data,
  - the completed 9-deck `phase5_search_selfplay_10k` shards,
  - the pending 13-deck `phase5_search_selfplay_13deck_10k` shards.

Implementation:

- Added `scripts/slurm/phase5_generalist_train_13deck_10k.sbatch`, a thin
  wrapper around the existing generalist trainer with explicit 13-deck artifact
  defaults:
  - `models/rl/phase5_generalist_policy_13deck_10k.pt`,
  - `experiments/rl/phase5_generalist_train_report_13deck_10k.json`.
- Extended `docs/phase-5-erawan-runbook.md` with:
  - recovery guidance for the 13-deck self-play `sbatch --parsable` socket
    timeout case,
  - shard existence checks before training,
  - bounded 13-deck mixed-train smoke commands,
  - full 13-deck mixed-train commands,
  - required-benchmark smoke evaluation commands for the new checkpoint.

Decision:

- Preserve the existing `models/rl/phase5_generalist_policy_10k.pt` and
  `experiments/rl/phase5_generalist_train_report_10k.json` artifacts.
- Treat the 13-deck league self-play as a training expansion, not a replacement
  evaluation gate.
- Keep the existing required 9x4 benchmark as the first promotion gate for
  `models/rl/phase5_generalist_policy_13deck_10k.pt`.
- Define a separate 13x13 evaluation only after choosing the intended target
  shape, such as agent-vs-rule or head-to-head self-play with player-order
  balance.

## 2026-06-30 - Phase 5 Full-Agent Scaffolds

Implementation:

- Added reusable opponent-prior inference in
  `src/ptcg_abc/rl/phase5_belief.py`.
  - It scores Phase 5 league deck priors from visible opponent active, bench,
    discard, and lost-zone cards.
  - The Kaggle package template now uses this shared inference path instead of
    duplicating visible-card matching logic.
- Hardened Phase 5 package generation:
  - `phase5-package` now builds direct root-level Kaggle zip artifacts as part
    of the package command.
  - Generated `main.py` supports Kaggle raw execution without `__file__` by
    probing generated-file directory, current working directory, and
    `/kaggle_simulations/agent`.
- Added optional neural search priors:
  - `RootSearchConfig.policy_prior_weight`,
  - `RootSearchConfig.neural_action_value_weight`,
  - `RootSearchConfig.neural_tactical_weight`.
  Defaults are zero, preserving existing benchmark behavior.
  `phase5-search` traces now retain policy logits, action-Q, tactical-head
  scores, normalized priors, and combined prior contribution per candidate.
- Added Phase 5 policy-pool self-play support:
  - CLI: repeat `--policy-pool-model` on
    `rl-generate-phase5-search-selfplay`.
  - SLURM: set `POLICY_POOL_MODELS="old.pt new.pt ..."`.
  - Trajectory metadata records the selected `policy_pool_model` for each
    player side.
- Added a Phase 5 legal-action PPO-style update:
  - CLI: `rl-train-phase5-ppo`.
  - SLURM: `scripts/slurm/phase5_ppo_train_conda.sbatch`.
  - It loads a Phase 5 checkpoint, streams trajectory JSONL shards, applies a
    clipped legal-action policy objective with value and entropy terms, and
    writes a new Phase 5 checkpoint.
- Added a concrete 13-deck league evaluation:
  - CLI: `rl-evaluate-phase5-league`.
  - SLURM: `scripts/slurm/phase5_league_eval_conda.sbatch`.
  - The default target is our selected agent on each of the 13 league decks
    against a rule-agent opponent on each of the 13 league decks, with
    player-order balance across games per matchup.
- Added automated promotion comparison reports:
  - CLI: `phase5-compare-benchmarks`.
  - Report helper: `src/ptcg_abc/rl/phase5_reports.py`.
  - Outputs overall, per-deck, and per-matchup deltas from two benchmark JSON
    reports.

Verification:

- `py_compile` passed for modified Phase 5 modules using an isolated pycache.
- Unit tests passed:
  - `tests.test_phase5_full_agent_scaffolds`,
  - `tests.test_rl_phase5_symbolic_agent`,
  - `tests.test_rl_phase5_symbolic_training`,
  - `tests.test_rl_phase4`,
  - `tests.test_evaluation`.
- Git Bash `bash -n` passed for:
  - `scripts/slurm/phase5_ppo_train_conda.sbatch`,
  - `scripts/slurm/phase5_league_eval_conda.sbatch`,
  - `scripts/slurm/phase5_search_selfplay_conda.sbatch`,
  - `scripts/slurm/phase5_search_selfplay_2shard_10k.sbatch`.
- CLI help checks passed for:
  - `rl-train-phase5-ppo`,
  - `rl-evaluate-phase5-league`,
  - `phase5-compare-benchmarks`,
  - `rl-generate-phase5-search-selfplay`.

Decision:

- Existing `phase5-search` defaults remain unchanged so current benchmark
  comparisons stay valid.
- Neural value/Q/tactical priors are opt-in until an ERAWAN benchmark proves
  they improve the required 9x4 gate.
- PPO should start only from a checkpoint that is stable on the required 9x4
  gate; the later 13-deck comparison below supersedes the original assumption
  that `models/rl/phase5_generalist_policy_13deck_10k.pt` would become that
  seed.

## 2026-07-01 - ERAWAN 13-Deck Data Sync Unblock

Context:

- The larger `phase5_search_selfplay_13deck_10k` self-play run was reported
  complete after ERAWAN came back online.
- The first ERAWAN `git pull --ff-only origin main` stopped while updating from
  `1411cb3` to `586cedc` because these local untracked report artifacts would
  be overwritten by newly tracked files:
  - `reports/phase5_generalist_search_10g.json`,
  - `reports/phase5_generalist_search_10g.md`,
  - `reports/phase5_generalist_search_30g.json`,
  - `reports/phase5_generalist_search_30g.md`.

Decision:

- Do not delete or overwrite the ERAWAN copies blindly. Archive or move those
  four untracked files, then rerun the fast-forward pull.
- After ERAWAN reaches `586cedc`, verify both 13-deck shard summaries and shard
  line counts before starting the bounded
  `phase5_generalist_policy_13deck_smoke.pt` train.

## 2026-07-02 - ERAWAN Compare Import-Path Note

Diagnostic:

- Running `python -m ptcg_abc phase5-compare-benchmarks` on ERAWAN failed with
  `No module named ptcg_abc`.
- Cause: the active conda environment had not installed the repo package, and
  the command was missing `PYTHONPATH="$PWD/src"`.
- After adding `PYTHONPATH`, the same command failed during CLI import with
  `ModuleNotFoundError: No module named 'lxml'`.
- Cause: `ptcg_abc.cli` imported the Limitless scraper at module import time,
  even though `phase5-compare-benchmarks` only needs local JSON reports.

Decision:

- Run lightweight repo CLI report tools from the repository root with
  `export PYTHONPATH="$PWD/src"` or inline `PYTHONPATH="$PWD/src" python -m ...`.
- Updated the ERAWAN runbook comparison block to export `PYTHONPATH` before
  invoking `phase5-compare-benchmarks`.
- Moved the `ptcg_abc.limitless` imports into the two scraper-dependent
  commands, `missing-limitless` and `collect-corpus`, so lightweight report
  commands no longer require `lxml`.
- Added a regression test that blocks `lxml`, imports the CLI, and parses
  `phase5-compare-benchmarks`.

## 2026-07-02 - Phase 5 13-Deck Generalist Promotion Comparison

Artifacts:

- `reports/phase5_generalist_13deck_vs_10k_comparison.json`
- `reports/phase5_generalist_13deck_vs_10k_comparison.md`

Comparison:

- Baseline: `reports/phase5_generalist_search_30g.json`, the current
  `phase5-search` path using `models/rl/phase5_generalist_policy_10k.pt`.
- Candidate: `reports/phase5_generalist_13deck_search_30g.json`, the
  `phase5-search` path using `models/rl/phase5_generalist_policy_13deck_10k.pt`.
- Baseline: 414 / 1,080 wins, 0.383 win rate, 1 draw, 5 timeouts, 0 errors.
- Candidate: 399 / 1,080 wins, 0.369 win rate, 5 draws, 6 timeouts, 0 errors.
- Overall delta: -15 wins, -0.014 win rate, +4 draws, +1 timeout, +0 errors.

Deck deltas:

- Improved:
  - Dragapult Dusknoir: +8 wins, +0.067 win rate.
  - Dragapult Blaziken: +4 wins, +0.033 win rate.
- Flat:
  - Alakazam Dudunsparce: +0 wins, +0.000 win rate, still 4 / 120.
- Regressed:
  - Ogerpon Box: -7 wins, -0.058 win rate.
  - Hydrapple: -5 wins, -0.042 win rate.
  - Dragapult: -5 wins, -0.042 win rate.
  - Raging Bolt Ogerpon: -4 wins, -0.033 win rate.
  - Crustle: -3 wins, -0.025 win rate.
  - Dragapult Dudunsparce: -3 wins, -0.025 win rate.

Largest matchup swings:

- Positive:
  - Crustle vs Iono's Bellibolt ex: +8 wins.
  - Dragapult Dusknoir vs Mega Abomasnow ex: +7 wins.
  - Dragapult Blaziken vs Crustle: +4 wins.
  - Dragapult Blaziken vs Mega Lucario ex: +4 wins.
- Negative:
  - Crustle vs Mega Lucario ex: -7 wins.
  - Ogerpon Box vs Mega Lucario ex: -6 wins.
  - Raging Bolt Ogerpon vs Crustle: -6 wins.
  - Crustle mirror: -5 wins.

Conclusion:

- `models/rl/phase5_generalist_policy_13deck_10k.pt` is clean enough to keep as
  a training artifact but is not promotable as the default required-gate model.
- Keep `phase5-search` with `models/rl/phase5_generalist_policy_10k.pt` as the
  current best inference path.
- Do not start PPO from the 13-deck checkpoint as the mainline path until a
  targeted follow-up recovers the required 9x4 benchmark.
- The next training slice should add targeted retention/weighting for the
  required 9x4 gate and the known weak Alakazam Dudunsparce line, rather than
  treating the 13-deck expansion as a replacement objective.

## 2026-07-02 - Phase 5 AlphaStar-Like League Replacement Plan

Decision:

- Replace the slow single-generalist promotion plan with a full-agent,
  league-first schedule.
- Implement the full Phase 5 runtime as the single inference surface:
  per-deck model selection, belief/opponent-prior inference, legal-action
  encoder, neural value/Q/tactical priors, one-turn root search, rule fallback,
  telemetry, and Kaggle-safe packaging.
- Train every model that does not require fresh learned-agent gameplay before
  launching the league:
  - shared foundation encoder,
  - 13 per-deck behavior-cloning/specialist policies,
  - value, selected-action Q, tactical, and belief heads from existing labels.
- Bootstrap with rule-based gameplay over all 13 x 13 league matchups, balanced
  by player order, then train one policy/model per deck.
- Run an AlphaStar-like league:
  - 13 main deck specialists,
  - historical snapshots,
  - fixed rule-agent anchors,
  - later exploiters if diagnostics justify them.
- Update schedule:
  - each deck plays 100 league-training games,
  - one global iteration is 13 x 100 = 1,300 training games,
  - update all 13 deck specialists once per global iteration.
- Evaluation schedule:
  - after every iteration, evaluate full agent vs rule-based across all
    13 x 13 matchups,
  - 30 games per matchup,
  - balanced player order,
  - 5,070 evaluation games per iteration.

Data-retention rule:

- Keep league data clean and bounded. The project folder has about 400 GB of
  practical capacity, and previous Phase 5 search self-play shards were about
  30 GB each.
- Generated league gameplay on ERAWAN goes under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/`.
- Raw league training data is ephemeral: remove each iteration's raw gameplay
  after the corresponding model/policy update succeeds and reports/checkpoints
  are written.
- Keep only model checkpoints, optimizer states, manifests/checksums, row
  counts, train reports, aggregate evaluation reports, comparison reports, and
  bounded sampled traces.
- Do not write full raw evaluation trajectories by default. Evaluation should
  emit reports and tiny sampled traces only.
- Do not launch a new league iteration while the previous iteration's raw data
  remains, unless a written diagnostic-retention reason is recorded here.

Updated docs:

- `docs/phase-5-master-plan.md` now marks the old single-generalist gate as
  superseded by the full-agent league plan.
- `docs/phase-5-erawan-runbook.md` now includes the AlphaStar-like league
  replacement track and mandatory cleanup rules.

## 2026-07-02 - Phase 5 Alpha League First Implementation Slice

Implementation:

- Added `src/ptcg_abc/rl/phase5_alpha_league.py`, the first orchestration layer
  for the AlphaStar-like league track.
  - `generate_phase5_alpha_rule_bootstrap` wraps the existing trajectory
    recorder with `agent_kind="rule"` over the 13-deck league pool.
  - `train_phase5_deck_specialists` trains one Phase 5 checkpoint per selected
    league deck from shared input datasets without copying large filtered data.
  - `cleanup_phase5_alpha_raw_train` removes an iteration's `raw_train/`
    directory only after an update report exists, and writes a cleanup report
    with file and byte counts.
- Added `phase5-full` as the public full-agent runtime alias. It currently maps
  to the Phase 5 policy-plus-root-search implementation while giving the league
  plan a stable agent name.
- Added optional `deck_index_filter` support to the mixed Phase 5 trainer so
  deck specialists can train directly from shared JSONL inputs.
- Added CLI commands:
  - `rl-generate-phase5-alpha-bootstrap`,
  - `rl-train-phase5-deck-specialists`,
  - `rl-clean-phase5-alpha-iteration`.
- Added SLURM scripts:
  - `scripts/slurm/phase5_alpha_rule_bootstrap.sbatch`,
  - `scripts/slurm/phase5_deck_specialists_train.sbatch`,
  - `scripts/slurm/phase5_alpha_cleanup_iteration.sbatch`.
- Added `tests/test_phase5_alpha_league.py` for CLI exposure, the
  `phase5-full` agent alias, default deck indices, and guarded raw-data cleanup.

Verification:

- `py_compile` passed for:
  - `src/ptcg_abc/cli.py`,
  - `src/ptcg_abc/rl/workflow.py`,
  - `src/ptcg_abc/rl/phase5_alpha_league.py`,
  - `src/ptcg_abc/rl/phase5_symbolic_training.py`.
- Unit tests passed:
  - `tests.test_phase5_alpha_league`,
  - `tests.test_phase5_full_agent_scaffolds`,
  - `tests.test_rl_phase5_symbolic_training`.
- Git Bash `bash -n` passed for:
  - `scripts/slurm/phase5_alpha_rule_bootstrap.sbatch`,
  - `scripts/slurm/phase5_deck_specialists_train.sbatch`,
  - `scripts/slurm/phase5_alpha_cleanup_iteration.sbatch`,
  - `scripts/slurm/phase5_league_eval_conda.sbatch`.

Resolved limitation:

- The 13 x 13 evaluation command now supports per-deck specialist dispatch via
  `--specialist-model-dir`, so deck 1 loads `deck-01.pt`, deck 2 loads
  `deck-02.pt`, and so on during full-agent league evaluation.

## 2026-07-02 - Alpha League Rule-Bootstrap Smoke Result

ERAWAN result:

- Recorded report artifact:
  `experiments/rl/phase5_league_alpha/iter-0000_rule_bootstrap_report.json`.
- Command path: `scripts/slurm/phase5_alpha_rule_bootstrap.sbatch` with
  `ITERATION=0`, `GAMES_PER_PAIR=1`, `MAX_STEPS=300`, `DECK_POOL=league-13`,
  and rule-agent gameplay.
- Output trajectory path:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0000/raw_train/phase5_alpha_rule_bootstrap.jsonl`.
- Aggregate: 169 / 169 games started, 25,349 trajectory steps, 169 ordered
  deck pairs, 1 draw, 5 timeouts, 0 errors, and no error records.
- Deck coverage was balanced for the smoke: every one of the 13 league decks
  appeared in 26 games.
- The report marks `cleanup_required=true` and keeps the data policy explicit:
  one active raw window, 400 GB practical capacity, raw training data deleted
  after successful model update, and full raw evaluation trajectories disabled
  by default.

Conclusion:

- The rule-based 13 x 13 bootstrap path is valid for smoke scale.
- Because this was `GAMES_PER_PAIR=1`, use it first for the bounded specialist
  train smoke (`DECISION_LIMIT=2000`, `SELFPLAY_LIMIT=2000`) before deciding
  whether to run the default `GAMES_PER_PAIR=2` bootstrap or advance directly
  to the first full specialist update.
- Do not clean
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0000/raw_train/`
  until the deck-specialist update report and checkpoints exist.

## 2026-07-02 - Alpha League Deck-Specialist Smoke Result

Implementation:

- Added `iteration` metadata to future `rl-train-phase5-deck-specialists`
  aggregate reports, and wired `scripts/slurm/phase5_deck_specialists_train.sbatch`
  to pass `--iteration "$ITERATION"`.

Verification:

- `py_compile` passed for `src/ptcg_abc/cli.py` and
  `src/ptcg_abc/rl/phase5_alpha_league.py`.
- `tests.test_phase5_alpha_league` passed with `PYTHONPATH=src`.
- CLI help for `rl-train-phase5-deck-specialists` exposes `--iteration`.
- Git Bash `bash -n` passed for
  `scripts/slurm/phase5_deck_specialists_train.sbatch`.

ERAWAN result:

- Recorded report artifact:
  `experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_report.json`.
- Command path: `scripts/slurm/phase5_deck_specialists_train.sbatch` with
  `ITERATION=0`, `DECISION_LIMIT=2000`, `SELFPLAY_LIMIT=2000`, CUDA, one epoch,
  and all 13 league decks.
- Inputs:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards.jsonl`
  plus
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0000/raw_train/phase5_alpha_rule_bootstrap.jsonl`.
- Output checkpoint family:
  `models/rl/phase5_league_alpha/iter-0000/specialists/deck-01.pt` through
  `deck-13.pt`.
- Aggregate: 13 / 13 specialist summaries with checkpoint paths, 1,998
  decision examples, 1,998 rule-demo examples, 2,000 self-play examples, 5,996
  value examples, 2,069 action-value examples, 33,923 tactical examples, 358
  changed examples, and 4 skipped no-target records.
- Accuracy ranged from 0.183 to 0.568, and final loss ranged from 1.013 to
  3.012. These are smoke diagnostics, not promotion metrics.
- Decks 10-13 had zero search-decision examples because the canonical
  search-decision dataset covers the original 9 tournament decks; those four
  sample decks trained from bootstrap/self-play signals in this smoke.
- Decks 3, 6, and 9 had fewer than 20 self-play examples under the bounded
  `SELFPLAY_LIMIT=2000` smoke cap, so the next full train should remove the
  smoke limits.

Conclusion:

- The per-deck specialist training path is valid for smoke scale.
- The smoke update report now permits cleanup of the smoke raw directory, but
  only after preserving this report and checkpoint/report paths.
- Next ERAWAN sequence: clean the smoke `raw_train/`, rerun the fuller
  `GAMES_PER_PAIR=2`, `MAX_STEPS=600` bootstrap for `ITERATION=0`, then train
  the 13 specialists without `DECISION_LIMIT` or `SELFPLAY_LIMIT`.

## 2026-07-02 - Alpha League Full Rule-Bootstrap Result

ERAWAN result:

- Replaced the current report artifact
  `experiments/rl/phase5_league_alpha/iter-0000_rule_bootstrap_report.json`
  with the fuller iteration-0 bootstrap report. The earlier smoke report at
  the same path remains recoverable from git history and documented above.
- Command path: `scripts/slurm/phase5_alpha_rule_bootstrap.sbatch` with
  `ITERATION=0`, `GAMES_PER_PAIR=2`, expected `MAX_STEPS=600`,
  `DECK_POOL=league-13`, and rule-agent gameplay.
- Output trajectory path:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0000/raw_train/phase5_alpha_rule_bootstrap.jsonl`.
- Aggregate: 338 / 338 games started, 53,443 trajectory steps, 169 ordered
  deck pairs, 3 draws, 7 timeouts, 0 errors, and no error records.
- Deck coverage was balanced for the full bootstrap: every one of the 13 league
  decks appeared in 52 games.

Conclusion:

- This is the correct report for the fuller iteration-0 rule bootstrap.
- The no-limit deck-specialist job can consume this raw trajectory. Await
  `experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_report.json`
  from that job before cleaning the raw directory again.

## 2026-07-02 - Alpha League No-Limit Specialist Update

Implementation:

- Added per-deck specialist dispatch to `rl-evaluate-phase5-league`.
  `--specialist-model-dir` points at a directory containing `deck-01.pt`
  through `deck-13.pt`; the evaluator loads the matching checkpoint for the
  controlled league deck.
- Updated `scripts/slurm/phase5_league_eval_conda.sbatch` to accept
  `SPECIALIST_MODEL_DIR` and echo the eval configuration in the SLURM log.
- Updated `docs/phase-5-erawan-runbook.md` so the iteration-0 full-agent eval
  uses `SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0000/specialists`.

Verification:

- `py_compile` passed for `src/ptcg_abc/cli.py` and
  `src/ptcg_abc/rl/workflow.py`.
- Unit tests passed: `tests.test_phase5_full_agent_scaffolds` and
  `tests.test_phase5_alpha_league`.
- CLI help for `rl-evaluate-phase5-league` exposes `--specialist-model-dir`.
- Git Bash `bash -n` passed for
  `scripts/slurm/phase5_league_eval_conda.sbatch`.

ERAWAN result:

- Replaced the current report artifact
  `experiments/rl/phase5_league_alpha/iter-0000_deck_specialists_report.json`
  with the no-limit iteration-0 specialist update report. The earlier bounded
  smoke report at the same path remains recoverable from git history and
  documented above.
- ERAWAN job: `73248`, submitted via
  `scripts/slurm/phase5_deck_specialists_train.sbatch` with `ITERATION=0`, all
  13 league decks, CUDA, one epoch, and no `DECISION_LIMIT` or
  `SELFPLAY_LIMIT`.
- Inputs:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_decisions_10shards.jsonl`
  plus the full rule-bootstrap trajectory
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0000/raw_train/phase5_alpha_rule_bootstrap.jsonl`.
- Output checkpoint family:
  `models/rl/phase5_league_alpha/iter-0000/specialists/deck-01.pt` through
  `deck-13.pt`.
- Aggregate: 13 / 13 specialist summaries with checkpoint paths, 791,974
  decision frames scanned per deck, 53,443 self-play steps scanned per deck,
  791,667 decision examples, 791,667 rule-demo examples, 53,421 self-play
  examples, 1,636,755 value examples, 56,902 action-value examples, 9,582,634
  tactical examples, 152,136 changed examples, and 636 skipped no-target
  records.
- Accuracy ranged from 0.518 to 0.950, and final loss ranged from 0.038 to
  2.001. These remain training diagnostics, not promotion metrics.
- Decks 10-13 still have zero search-decision examples because the canonical
  search-decision dataset covers the original 9 tournament decks. They now have
  full-bootstrap self-play coverage: deck 10 has 2,975 examples, deck 11 has
  3,120, deck 12 has 2,063, and deck 13 has 5,862.

Conclusion:

- The no-limit iteration-0 specialist update succeeded and is the checkpoint
  family to evaluate.
- The iteration-0 raw training directory can be cleaned after this report and
  the checkpoint family are preserved on ERAWAN.
- Next ERAWAN sequence: clean `iter-0000/raw_train/`, then run the 13 x 13 x
  30 `phase5-full` vs rule eval with `SPECIALIST_MODEL_DIR`.

## Artifact Notes

Important model artifacts:

- `models/rl/phase5_search_distill_10shards.pt`
- `models/rl/phase5_search_distill_10shards.json`
- `models/rl/phase5_symbolic_policy_10shards.pt`
- `models/rl/phase5_symbolic_policy_10shards_pairwise_all.pt`
- `models/rl/phase5_symbolic_policy_10shards_pairwise_baseline_soft.pt`
- `models/rl/phase5_symbolic_policy_10shards_pairwise_baseline_tiny.pt`
- `models/rl/phase5_generalist_policy_smoke.pt`
- `models/rl/phase5_generalist_policy_10k.pt`
- Non-promoted 13-deck expansion checkpoint:
  `models/rl/phase5_generalist_policy_13deck_10k.pt`
- PPO should wait for a promotable checkpoint; do not use
  `models/rl/phase5_generalist_policy_13deck_10k.pt` as the mainline PPO seed.

Important dataset artifacts:

- `data/datasets/rl/phase5_search_decisions_10shards.jsonl`
- `data/datasets/rl/phase5_search_decisions_10shards_pairwise.jsonl`
- `data/datasets/rl/phase5_search_decisions_10shards_pairwise_all.jsonl`
- `data/datasets/rl/phase5_search_decisions_10shards_reweighted.jsonl`
- `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-0.jsonl`
- `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_10k/shards/phase5_search_selfplay_shard-1.jsonl`
- `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_13deck_338/shards/phase5_search_selfplay_shard-0.jsonl`
- `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_13deck_338/shards/phase5_search_selfplay_shard-1.jsonl`
- User reported complete; pending summary verification:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_search_selfplay_13deck_10k/shards/phase5_search_selfplay_shard-*.jsonl`

File-retention decision:

- Keep the 791,974-record 10-shard datasets that match completed experiments.
- Do not use the 637,227-record `phase5_search_decisions_10shards_remerged.jsonl`
  as the canonical 10-shard dataset because it has fewer rows than expected.
- Remove large shard files only after canonical merged files and reports are
  verified.

## Open Questions And Next Work

- Record the full `phase5_generalist_train_report_10k.json` if needed for the
  final report.
- On ERAWAN, archive the four untracked
  `reports/phase5_generalist_search_*` artifacts that block pulling `586cedc`,
  then rerun `git pull --ff-only origin main`.
- Record the completed 13-deck 10k shard summaries and 13-deck generalist train
  report if those ERAWAN artifacts are available.
- Inspect the full candidate benchmark report telemetry if available, especially
  search truncation and timeout changes.
- Keep optional 13-deck league evaluation as a breadth diagnostic, not a
  promotion gate, because the required 9x4 comparison regressed.
- Start Phase 5 PPO only after a follow-up checkpoint recovers or improves the
  required 9x4 benchmark.
- Keep `phase5-search` with `models/rl/phase5_generalist_policy_10k.pt` as the
  current best 9-deck inference path.
- Clean `iter-0000/raw_train/` now that the no-limit specialist update report
  and checkpoint family exist.
- Run the learned-agent league-iteration runner for `ITERATION=1`, then train
  iteration-1 specialists from that new raw window.

## 2026-07-03 - Alpha League Single-Model Eval Diagnostic

ERAWAN result:

- Preserved uploaded reports as explicit single-model diagnostics:
  - `reports/phase5_alpha_iter0000_full_vs_rule_30g_single_model.json`
  - `reports/phase5_alpha_iter0000_full_vs_rule_30g_single_model.md`
- The report header says:
  `Model: models/rl/phase5_generalist_policy_13deck_10k.pt`.
  Therefore this was not the intended per-deck specialist evaluation, even
  though the agent label was `phase5-league:phase5-full`.
- Aggregate: 2,662 / 5,070 wins, 2,390 losses, 18 draws, 50 timeouts, 0 errors,
  and 0.525 win rate.
- Search telemetry: 196,748 searched decisions, 37,167 search-changed decisions,
  0 search errors, 0 candidate errors, 3,476 truncated candidates, 0.0629
  average search seconds, and 3.3680 max search seconds.
- Per-deck win rates:
  - Deck 1 Alakazam Dudunsparce: 49 / 390, 0.126.
  - Deck 2 Crustle: 258 / 390, 0.662.
  - Deck 3 Dragapult Dusknoir: 160 / 390, 0.410.
  - Deck 4 Dragapult: 177 / 390, 0.454.
  - Deck 5 Dragapult Dudunsparce: 180 / 390, 0.462.
  - Deck 6 Hydrapple: 218 / 390, 0.559.
  - Deck 7 Raging Bolt Ogerpon: 209 / 390, 0.536.
  - Deck 8 Dragapult Blaziken: 195 / 390, 0.500.
  - Deck 9 Ogerpon Box: 200 / 390, 0.513.
  - Deck 10 Crustle sample: 222 / 390, 0.569.
  - Deck 11 Mega Lucario ex: 295 / 390, 0.756.
  - Deck 12 Mega Abomasnow ex: 294 / 390, 0.754.
  - Deck 13 Iono's Bellibolt ex: 205 / 390, 0.526.

Conclusion:

- This is a useful 13 x 13 breadth diagnostic for the single 13-deck generalist
  checkpoint, but it is not accepted as the iteration-0 specialist eval.
- Rerun the 13 x 13 x 30 eval after pulling commit `b28013b` or later, using
  `SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0000/specialists`.

## 2026-07-03 - Alpha League Learned-Iteration Runner

Implementation:

- Added per-deck specialist dispatch to `rollout_selfplay_games` via
  `specialist_model_dir`, with trajectory metadata recording
  `specialist_model_dir` and `specialist_model_path`.
- Added `generate_phase5_alpha_league_iteration`, which validates
  `deck-01.pt` through `deck-13.pt`, writes learned-agent raw gameplay under
  an iteration `raw_train/` directory, records the checkpoint family that
  generated the data, and preserves the strict cleanup policy.
- Added CLI command `rl-generate-phase5-alpha-league-iteration`.
- Added SLURM script `scripts/slurm/phase5_alpha_league_iteration.sbatch`.
  Defaults: `ITERATION=1`, `SOURCE_ITERATION=0`, `GAMES_PER_DECK=100`,
  `AGENT=phase5-full`, source checkpoints from
  `models/rl/phase5_league_alpha/iter-0000/specialists`, and output raw data to
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0001/raw_train/phase5_alpha_league_selfplay.jsonl`.
- Updated `scripts/slurm/phase5_deck_specialists_train.sbatch` so
  `SELFPLAY_DATASET` can point at either rule-bootstrap raw data or learned
  league self-play raw data.
- Updated `docs/phase-5-erawan-runbook.md` with the iteration-1 generation and
  update commands.

Verification:

- `py_compile` passed for `src/ptcg_abc/cli.py`,
  `src/ptcg_abc/rl/workflow.py`, and
  `src/ptcg_abc/rl/phase5_alpha_league.py`.
- Unit tests passed: `tests.test_phase5_alpha_league` and
  `tests.test_phase5_full_agent_scaffolds`.
- CLI help for `rl-generate-phase5-alpha-league-iteration` exposes
  `--specialist-model-dir`, `--games-per-deck`, and `--search-trace-games`.
- Git Bash `bash -n` passed for
  `scripts/slurm/phase5_alpha_league_iteration.sbatch` and
  `scripts/slurm/phase5_deck_specialists_train.sbatch`.

Next step:

- The true iteration-0 specialist eval is now recorded. Launch `ITERATION=1`
  learned-agent league data generation with the new SLURM script.

## 2026-07-03 - Alpha League Iteration-0 Specialist Eval

ERAWAN result:

- Preserved uploaded true per-deck specialist eval reports:
  - `reports/phase5_alpha_iter0000_specialists_full_vs_rule_30g.json`
  - `reports/phase5_alpha_iter0000_specialists_full_vs_rule_30g.md`
- The report header confirms:
  `Model: models/rl/phase5_league_alpha/iter-0000/specialists`.
  This is accepted as the intended iteration-0 specialist evaluation.
- Aggregate over the 13 x 13 x 30 full-agent-vs-rule benchmark: 2,651 / 5,070
  wins, 2,404 losses, 15 draws, 48 timeouts, 0 errors, and 0.5229 win rate.
- Search telemetry: 202,460 searched decisions, 31,122 search-changed
  decisions, 0 search errors, 0 candidate errors, 2,819 truncated candidates,
  0.0582 average search seconds, and 3.1895 max search seconds.
- Against the 4 required sample rule-agent opponents only: 659 / 1,560 wins,
  0.4224 win rate, 0 draws, 0 timeouts, and 0 errors.
- By required sample opponent:
  - Crustle sample: 175 / 390, 0.4487.
  - Mega Lucario ex sample: 119 / 390, 0.3051.
  - Mega Abomasnow ex sample: 137 / 390, 0.3513.
  - Iono's Bellibolt ex sample: 228 / 390, 0.5846.
- Controlled tournament decks 1-9: 1,670 / 3,510 wins, 0.4758 win rate.
- Controlled sample decks 10-13: 981 / 1,560 wins, 0.6288 win rate.
- Tournament rule-agent opponents 1-9: 1,992 / 3,510 wins, 0.5675 win rate.
- Required sample rule-agent opponents 10-13: 659 / 1,560 wins, 0.4224 win
  rate.
- Per-deck win rates:
  - Deck 1 Alakazam Dudunsparce: 69 / 390, 0.1769.
  - Deck 2 Crustle: 223 / 390, 0.5718.
  - Deck 3 Dragapult Dusknoir: 150 / 390, 0.3846.
  - Deck 4 Dragapult: 215 / 390, 0.5513.
  - Deck 5 Dragapult Dudunsparce: 200 / 390, 0.5128.
  - Deck 6 Hydrapple: 200 / 390, 0.5128.
  - Deck 7 Raging Bolt Ogerpon: 197 / 390, 0.5051.
  - Deck 8 Dragapult Blaziken: 208 / 390, 0.5333.
  - Deck 9 Ogerpon Box: 208 / 390, 0.5333.
  - Deck 10 Crustle sample: 224 / 390, 0.5744.
  - Deck 11 Mega Lucario ex: 297 / 390, 0.7615.
  - Deck 12 Mega Abomasnow ex: 257 / 390, 0.6590.
  - Deck 13 Iono's Bellibolt ex: 203 / 390, 0.5205.

Comparison to the single-model diagnostic:

- Overall moved from 2,662 / 5,070 to 2,651 / 5,070, down 11 wins and 0.22
  percentage points. This is effectively flat for this noisy 30-game-per-matchup
  breadth eval.
- Improved controlled-deck results: Alakazam Dudunsparce +20 wins, Dragapult
  +38, Dragapult Dudunsparce +20, Dragapult Blaziken +13, Ogerpon Box +8,
  Crustle sample +2, Mega Lucario ex +2.
- Regressed controlled-deck results: Crustle -35 wins, Dragapult Dusknoir -10,
  Hydrapple -18, Raging Bolt Ogerpon -12, Mega Abomasnow ex -37, Iono's
  Bellibolt ex -2.
- Against required sample opponents improved slightly from 654 / 1,560 to
  659 / 1,560, but remains the main pressure point. Mega Lucario ex and Mega
  Abomasnow ex are the hardest rule-agent opponents.

Conclusion:

- The iteration-0 specialist family is operationally clean: per-deck dispatch
  works, search telemetry is stable, and there are 0 benchmark errors.
- It is not a clear strength improvement over the single 13-deck generalist yet,
  but it is good enough as the first AlphaStar-like league baseline.
- Proceed to `ITERATION=1` learned-agent league data generation, then train
  iteration-1 specialists and evaluate the next checkpoint family with the same
  13 x 13 x 30 full-agent-vs-rule benchmark.

## 2026-07-03 - ERAWAN Report Artifact Ownership

Artifact decision:

- Future reports generated on ERAWAN should be committed and pushed from ERAWAN
  first whenever practical.
- The local Codex workspace should pull those report commits and inspect the
  tracked files locally.
- Avoid preserving uploaded report copies directly into the local repo when the
  same paths may still exist as untracked ERAWAN outputs. This prevents
  `git pull --ff-only` from failing because tracked report paths would overwrite
  local untracked SLURM outputs.
- The current `df61d24` specialist-eval commit already contains the two uploaded
  reports, so ERAWAN still needs to move, remove, or commit its untracked local
  copies of those same two paths before pulling that commit.

## 2026-07-03 - Iteration-0 Specialist Kaggle Package Candidates

Selection:

- Picked deck 11 Mega Lucario ex as the first specialist package candidate:
  85 / 120 wins against the four required rule-based specialist opponents,
  0.7083 sample-specialist win rate, and 297 / 390 full 13 x 13 rule-eval wins.
- Picked deck 12 Mega Abomasnow ex as the second specialist package candidate:
  65 / 120 wins against the four required rule-based specialist opponents,
  0.5417 sample-specialist win rate, and 257 / 390 full 13 x 13 rule-eval wins.
- Deck 4 Dragapult had 66 / 120 against the four sample specialists, one win
  ahead of deck 12, but deck 12 was selected as the more general package
  candidate because it scored 42 more full-eval wins and had the stronger
  overall rule-agent profile.

Implementation:

- Extended `phase5-package` with `--deck-pool {tournament-9,league-13}` so
  Kaggle packages can be built for the full 13-deck Phase 5 league pool.
- Added `--model-dir` to `phase5-package`; when set, each packaged deck uses
  its matching per-deck specialist checkpoint named `deck-XX.pt`.
- Updated the ERAWAN runbook with the package command for the selected two
  iteration-0 specialist candidates:
  `submissions/phase5_alpha_iter0000_specialists_top2`.

Validation:

- `py_compile` passed for `src/ptcg_abc/cli.py`.
- `phase5-package --help` exposes `--deck-pool` and `--model-dir`.
- A temporary local package build succeeded for league decks 11 and 12 using
  stand-in per-deck checkpoint filenames. The real specialist checkpoint files
  are not present in this local checkout, so the final specialist zips should be
  generated on ERAWAN from
  `models/rl/phase5_league_alpha/iter-0000/specialists/deck-11.pt` and
  `deck-12.pt`.

## 2026-07-04 - Alpha League Iteration-1 Specialist Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0001_deck_specialists_report.json`
  - `slurm-73372-phase5-deck-specialists.out`
- ERAWAN job: `73372`.
- Iteration: `1`.
- Input learned-agent self-play dataset:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0001/raw_train/phase5_alpha_league_selfplay.jsonl`.
- Wrote all 13 specialist checkpoints under
  `models/rl/phase5_league_alpha/iter-0001/specialists/`.
- Aggregate examples across the 13 specialists: 791,667 decision, 791,667
  rule-demo, 193,598 self-play, 1,776,932 value, 206,777 action-value,
  10,442,154 tactical, 152,136 changed, and 614 skipped no-target records.
- Each specialist scanned 791,974 decision frames and 193,598 self-play steps.
- Decks 10-13 still have zero search-decision examples because the canonical
  search-decision dataset covers only the original 9 tournament decks; they are
  now trained from the learned-agent self-play window for iteration 1.
- Per-deck self-play examples:
  - Deck 1: 18,340.
  - Deck 2: 11,478.
  - Deck 3: 15,614.
  - Deck 4: 17,863.
  - Deck 5: 17,344.
  - Deck 6: 13,600.
  - Deck 7: 13,250.
  - Deck 8: 20,036.
  - Deck 9: 12,890.
  - Deck 10: 12,218.
  - Deck 11: 12,854.
  - Deck 12: 6,183.
  - Deck 13: 21,928.
- Accuracy range across specialists: 0.5375 to 0.9360. Final loss range:
  0.0271 to 2.1235.

Conclusion:

- The iteration-1 specialist update succeeded and produced a complete 13-deck
  checkpoint family.
- Proceed to 13 x 13 x 30 full-agent-vs-rule evaluation using
  `SPECIALIST_MODEL_DIR=models/rl/phase5_league_alpha/iter-0001/specialists`.
- The iteration-1 raw training window can now be cleaned according to the
  Phase 5 data policy after preserving the update report/checkpoints.

## 2026-07-04 - True Online RL Alpha Loop

Implementation:

- Added `Phase5SamplingPolicyAgent`, exposed as `phase5-rl`, for stochastic
  neural self-play from Phase 5 checkpoints. It samples legal actions from the
  policy logits with temperature 1.0, records the action log-probability,
  predicted state value, and `policy_on_policy=true` metadata for PPO.
- Updated trajectory recording so `TrajectoryStep.logprob` and
  `TrajectoryStep.value` are populated from the acting policy instead of
  defaulting to zero.
- Updated Phase 5 PPO training to support `deck_index_filter` and
  `require_on_policy`; the Alpha path can now refuse legacy/search trajectories
  that were not generated by the live stochastic policy.
- Fixed PPO action probability accounting for multi-select decisions by summing
  the selected legal-position log-probabilities.
- Added `rl-train-phase5-alpha-ppo-specialists` and
  `scripts/slurm/phase5_alpha_ppo_specialists_train.sbatch`. The command loads
  `deck-XX.pt` from a source specialist directory, trains only that deck's
  on-policy examples, and writes updated `deck-XX.pt` files under the target
  iteration.
- `scripts/slurm/phase5_alpha_league_iteration.sbatch` now defaults to
  `AGENT=phase5-rl` for future learned-agent data windows. Explicit
  `AGENT=phase5-full` remains available for search/full-agent diagnostics, but
  it is no longer the default training-data collector.

Operational conclusion:

- Iteration 0 remains rule-bootstrap supervised/mixed training.
- Iteration 1 was already produced with the earlier mixed specialist path.
- From iteration 2 onward, the intended loop is:
  1. generate 100 games per deck using `AGENT=phase5-rl` from the previous
     specialist checkpoint family;
  2. update each specialist with PPO from the fresh raw window only;
  3. evaluate the updated checkpoint family against rule agents;
  4. clean the raw window after the update report/checkpoints are preserved.

Validation:

- `py_compile` passed for the touched agent, workflow, trainer, Alpha-league,
  and CLI modules.
- Unit tests passed:
  - `tests.test_phase5_alpha_league`;
  - `tests.test_rl_phase5_symbolic_training`;
  - `tests.test_phase5_full_agent_scaffolds`;
  - `tests.test_rl_phase5_symbolic_agent`.

## 2026-07-04 - Alpha League Iteration-2 Online PPO Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0002_league_iteration_report.json`;
  - `iter-0002_ppo_specialists_report.json`;
  - `slurm-73394-phase5-alpha-ppo-specialists.out`.
- PPO specialist update job: `73394`.
- Online collector: `AGENT=phase5-rl`.
- Source checkpoint family:
  `models/rl/phase5_league_alpha/iter-0001/specialists`.
- Output checkpoint family:
  `models/rl/phase5_league_alpha/iter-0002/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0002/raw_train/phase5_alpha_league_selfplay.jsonl`.

League collection:

- Games: 1,300 / 1,300 started, 13 decks x 100 scheduled games.
- Steps / trajectory rows: 215,597.
- Draws: 8. Timeouts: 26. Errors: 0.
- Pair side balance: deck A 643 wins, deck B 649 wins.
- Strongest online self-play records by deck win rate:
  - Deck 11 Mega Lucario ex: 137 / 204, 67.2%.
  - Deck 2 Crustle: 121 / 191, 63.4%.
  - Deck 10 Crustle sample: 124 / 204, 60.8%.
  - Deck 12 Mega Abomasnow ex: 118 / 204, 57.8%.
- Main weak point remains deck 1 Alakazam Dudunsparce: 28 / 204, 13.7%.

PPO update:

- All 13 specialist checkpoints were written under
  `models/rl/phase5_league_alpha/iter-0002/specialists`.
- Total PPO examples: 215,597, exactly matching the trajectory row count.
- `require_on_policy=true` for every deck.
- Skipped off-policy examples: 0. Skipped no-target examples: 0.
- Per-deck PPO examples ranged from 7,432 to 29,477:
  - Deck 1: 29,477.
  - Deck 2: 11,952.
  - Deck 3: 15,236.
  - Deck 4: 17,555.
  - Deck 5: 16,898.
  - Deck 6: 15,452.
  - Deck 7: 14,662.
  - Deck 8: 21,955.
  - Deck 9: 15,070.
  - Deck 10: 13,224.
  - Deck 11: 12,855.
  - Deck 12: 7,432.
  - Deck 13: 23,829.
- Mean advantage ranged from -0.2640 to 0.3917; deck 6 was highest and deck 12
  lowest. Final loss ranged from -0.0073 to 0.4495.

Conclusion and next step:

- The first true online PPO specialist update succeeded: no off-policy leakage,
  no missing target records, and a complete iteration-2 checkpoint family.
- Do not launch iteration 3 yet. First evaluate
  `models/rl/phase5_league_alpha/iter-0002/specialists` with the 13 x 13 x 30
  full-agent-vs-rule benchmark.
- Keep `iter-0002/raw_train/` until the eval result is inspected and the
  iteration-2 PPO update report/checkpoints are preserved; clean it afterward
  according to the raw-data retention policy.

## 2026-07-04 - Alpha League Iteration-2 Full-Agent Evaluation

ERAWAN result:

- Uploaded and inspected:
  - `phase5_alpha_iter0002_specialists_full_vs_rule_30g.json`;
  - `phase5_alpha_iter0002_specialists_full_vs_rule_30g.md`;
  - `slurm-73396-phase5-league-eval.out`;
  - `slurm-73396-phase5-league-eval.err`.
- ERAWAN job: `73396`.
- Agent: `phase5-full`.
- Specialist model directory:
  `models/rl/phase5_league_alpha/iter-0002/specialists`.
- Benchmark: 13 x 13 league agent-vs-rule, 30 games per matchup.

Aggregate:

- Games: 5,070.
- Wins: 2,710.
- Losses: 2,340.
- Draws: 20.
- Timeouts: 55.
- Errors: 0.
- Win rate: 0.5345.
- Search telemetry: 200,132 searched decisions, 33,554 search-changed decisions,
  0 search errors, 0 candidate errors, 2,666 truncated candidates, average
  search 0.0536s, max search 3.5554s.

Comparison:

- Versus the recorded iteration-0 specialist eval, iteration 2 is up 59 wins:
  2,710 / 5,070 vs 2,651 / 5,070.
- Against the four required sample rule-agent opponents, iteration 2 is up
  35 wins: 694 / 1,560 vs 659 / 1,560.

Deck totals:

- Deck 1 Alakazam Dudunsparce: 65 / 390, 16.7%.
- Deck 2 Crustle: 249 / 390, 63.8%.
- Deck 3 Dragapult Dusknoir: 140 / 390, 35.9%.
- Deck 4 Dragapult: 220 / 390, 56.4%.
- Deck 5 Dragapult Dudunsparce: 186 / 390, 47.7%.
- Deck 6 Hydrapple: 213 / 390, 54.6%.
- Deck 7 Raging Bolt Ogerpon: 214 / 390, 54.9%.
- Deck 8 Dragapult Blaziken: 197 / 390, 50.5%.
- Deck 9 Ogerpon Box: 228 / 390, 58.5%.
- Deck 10 Crustle sample: 227 / 390, 58.2%.
- Deck 11 Mega Lucario ex: 292 / 390, 74.9%.
- Deck 12 Mega Abomasnow ex: 256 / 390, 65.6%.
- Deck 13 Iono's Bellibolt ex: 223 / 390, 57.2%.

Conclusion and next step:

- The first online PPO iteration produced a measurable full-agent improvement
  over the recorded iteration-0 specialist baseline.
- Deck 11 remains the strongest overall specialist. Deck 12 remains strong but
  is down one win from iteration 0. Deck 1 remains the main weakness and lost
  four wins versus iteration 0.
- Proceed with the online loop for iteration 3 from
  `models/rl/phase5_league_alpha/iter-0002/specialists`.
- Clean `iter-0002/raw_train/` before launching the next raw window, now that
  the PPO update and eval reports have been inspected.

## 2026-07-05 - Alpha League Iteration-4 Self-Play Window

ERAWAN result:

- Uploaded and inspected:
  - `iter-0004_league_iteration_report.json`;
  - `slurm-73448-phase5-alpha-league.out`;
  - `slurm-73448-phase5-alpha-league.err`.
- ERAWAN job: `73448`.
- Online collector: `AGENT=phase5-rl`.
- Source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0003/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0004/raw_train/phase5_alpha_league_selfplay.jsonl`.

League collection:

- Games: 1,300 / 1,300 started, 13 decks x 100 scheduled games.
- Steps / trajectory rows: 211,968.
- Draws: 11. Timeouts: 27. Errors: 0.
- Pair side balance: deck A 657 wins, deck B 632 wins.
- Per-deck online self-play records:
  - Deck 1 Alakazam Dudunsparce: 24 / 204, 11.8%.
  - Deck 2 Crustle: 123 / 204, 60.3%.
  - Deck 3 Dragapult Dusknoir: 75 / 204, 36.8%.
  - Deck 4 Dragapult: 111 / 204, 54.4%.
  - Deck 5 Dragapult Dudunsparce: 79 / 204, 38.7%.
  - Deck 6 Hydrapple: 101 / 204, 49.5%.
  - Deck 7 Raging Bolt Ogerpon: 99 / 191, 51.8%.
  - Deck 8 Dragapult Blaziken: 91 / 191, 47.6%.
  - Deck 9 Ogerpon Box: 112 / 191, 58.6%.
  - Deck 10 Crustle sample: 116 / 191, 60.7%.
  - Deck 11 Mega Lucario ex: 142 / 204, 69.6%.
  - Deck 12 Mega Abomasnow ex: 113 / 204, 55.4%.
  - Deck 13 Iono's Bellibolt ex: 103 / 204, 50.5%.

Conclusion and next step:

- The iteration-4 raw self-play window is valid: zero errors and enough
  on-policy trajectory rows for the next per-deck PPO update.
- Start the iteration-4 PPO specialist update from source iteration 3 while the
  iteration-3 full-agent evaluation is still running.
- Keep `iter-0004/raw_train/` until the iteration-4 PPO update report and eval
  result are inspected.

## 2026-07-05 - Alpha League Iteration-4 Online PPO Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0004_ppo_specialists_report.json`;
  - `slurm-73451-phase5-alpha-ppo-specialists.out`.
- ERAWAN job: `73451`.
- PPO source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0003/specialists`.
- PPO output specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0004/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0004/raw_train/phase5_alpha_league_selfplay.jsonl`.

PPO update:

- Deck checkpoints written: 13 / 13.
- On-policy trajectory rows consumed: 211,968.
- Skipped no-target rows: 0.
- Skipped off-policy rows: 0.
- `require_on_policy`: true for every deck update.
- Per-deck PPO examples:
  - Deck 1: 28,241 examples, mean advantage -0.1991, final loss 0.7449.
  - Deck 2: 12,180 examples, mean advantage 0.0464, final loss 0.1648.
  - Deck 3: 15,613 examples, mean advantage 0.2386, final loss 0.6290.
  - Deck 4: 18,730 examples, mean advantage 0.0667, final loss 0.1562.
  - Deck 5: 17,964 examples, mean advantage -0.0902, final loss 0.5568.
  - Deck 6: 14,536 examples, mean advantage -0.1183, final loss 0.0095.
  - Deck 7: 14,050 examples, mean advantage 0.0364, final loss 0.2890.
  - Deck 8: 19,020 examples, mean advantage 0.1105, final loss 0.1572.
  - Deck 9: 13,672 examples, mean advantage -0.1072, final loss 0.1228.
  - Deck 10: 12,189 examples, mean advantage -0.0175, final loss 0.1942.
  - Deck 11: 13,368 examples, mean advantage 0.2036, final loss 0.2252.
  - Deck 12: 7,488 examples, mean advantage -0.2524, final loss 0.5555.
  - Deck 13: 24,917 examples, mean advantage -0.1710, final loss 0.0314.

Conclusion and next step:

- The iteration-4 online PPO update is valid: it used the full iter-4 raw
  self-play window and produced a complete 13-checkpoint specialist family with
  no off-policy leakage.
- Queue the iteration-4 13 x 13 x 30 full-agent-vs-rule evaluation from
  `models/rl/phase5_league_alpha/iter-0004/specialists` while the iteration-3
  eval continues to run, respecting the two-job ERAWAN concurrency limit.
- Keep `iter-0004/raw_train/` until the iteration-4 eval result is inspected.

## 2026-07-05 - Alpha League Iteration-3 Full-Agent Evaluation

ERAWAN result:

- Uploaded and inspected:
  - `phase5_alpha_iter0003_specialists_full_vs_rule_30g.json`;
  - `phase5_alpha_iter0003_specialists_full_vs_rule_30g.md`;
  - `slurm-73454-phase5-league-eval.err`.
- ERAWAN job: `73454`.
- Agent: `phase5-full`.
- Specialist model directory:
  `models/rl/phase5_league_alpha/iter-0003/specialists`.
- Benchmark: 13 x 13 league agent-vs-rule, 30 games per matchup.

Aggregate:

- Games: 5,070.
- Wins: 2,697.
- Losses: 2,353.
- Draws: 20.
- Timeouts: 38.
- Errors: 0.
- Win rate: 0.5320.
- Search telemetry: 199,064 searched decisions, 33,954 search-changed
  decisions, 0 search errors, 0 candidate errors, 1,820 truncated candidates,
  average search 0.0526s, max search 3.1836s.
- `slurm-73454-phase5-league-eval.err` only contained the PyTorch nested-tensor
  prototype warning.

Comparison:

- Versus iteration 2, iteration 3 is down 13 wins: 2,697 / 5,070 vs
  2,710 / 5,070.
- Versus the recorded iteration-0 specialist eval, iteration 3 is up 46 wins:
  2,697 / 5,070 vs 2,651 / 5,070.
- Against the four required sample rule-agent opponents, iteration 3 scored
  677 / 1,560, down 17 wins from iteration 2's 694 / 1,560 and up 18 wins from
  iteration 0's 659 / 1,560.

Deck totals:

- Deck 1 Alakazam Dudunsparce: 55 / 390, 14.1%.
- Deck 2 Crustle: 266 / 390, 68.2%.
- Deck 3 Dragapult Dusknoir: 143 / 390, 36.7%.
- Deck 4 Dragapult: 228 / 390, 58.5%.
- Deck 5 Dragapult Dudunsparce: 194 / 390, 49.7%.
- Deck 6 Hydrapple: 200 / 390, 51.3%.
- Deck 7 Raging Bolt Ogerpon: 191 / 390, 49.0%.
- Deck 8 Dragapult Blaziken: 197 / 390, 50.5%.
- Deck 9 Ogerpon Box: 234 / 390, 60.0%.
- Deck 10 Crustle sample: 230 / 390, 59.0%.
- Deck 11 Mega Lucario ex: 301 / 390, 77.2%.
- Deck 12 Mega Abomasnow ex: 243 / 390, 62.3%.
- Deck 13 Iono's Bellibolt ex: 215 / 390, 55.1%.

Required-sample opponent slice by agent deck:

- Deck 11 Mega Lucario ex: 87 / 120, 72.5%.
- Deck 4 Dragapult: 67 / 120, 55.8%.
- Deck 2 Crustle: 65 / 120, 54.2%.
- Deck 12 Mega Abomasnow ex: 60 / 120, 50.0%.
- Deck 9 Ogerpon Box: 58 / 120, 48.3%.
- Deck 10 Crustle sample: 57 / 120, 47.5%.
- Deck 6 Hydrapple: 52 / 120, 43.3%.
- Deck 13 Iono's Bellibolt ex: 51 / 120, 42.5%.
- Deck 8 Dragapult Blaziken: 49 / 120, 40.8%.
- Deck 5 Dragapult Dudunsparce: 48 / 120, 40.0%.
- Deck 7 Raging Bolt Ogerpon: 44 / 120, 36.7%.
- Deck 3 Dragapult Dusknoir: 35 / 120, 29.2%.
- Deck 1 Alakazam Dudunsparce: 4 / 120, 3.3%.

Conclusion and next step:

- Iteration 3 is a small regression from iteration 2 overall and on the
  required-sample opponent slice, but remains ahead of the iteration-0
  specialist baseline and has clean eval telemetry.
- Deck 11 is now the strongest full-agent specialist both overall and against
  the four required sample opponents. Deck 1 remains the main weakness and
  regressed further versus iteration 2.
- Since iteration-4 PPO has already completed and iteration-4 eval is running,
  use the open ERAWAN slot for iteration-5 online self-play from
  `models/rl/phase5_league_alpha/iter-0004/specialists`. Do not start
  iteration-5 PPO until the iteration-5 self-play report is inspected.

## 2026-07-05 - Alpha League Iteration-5 Self-Play Window

ERAWAN result:

- Uploaded and inspected:
  - `iter-0005_league_iteration_report.json`;
  - `slurm-73456-phase5-alpha-league.out`.
- ERAWAN job: `73456`.
- Online collector: `AGENT=phase5-rl`.
- Source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0004/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0005/raw_train/phase5_alpha_league_selfplay.jsonl`.

League collection:

- Games: 1,300 / 1,300 started, 13 decks x 100 scheduled games.
- Steps / trajectory rows: 214,943.
- Draws: 14. Timeouts: 27. Errors: 0.
- Pair side balance: deck A 628 wins, deck B 658 wins.
- Per-deck online self-play records:
  - Deck 1 Alakazam Dudunsparce: 24 / 204, 11.8%.
  - Deck 2 Crustle: 124 / 204, 60.8%.
  - Deck 3 Dragapult Dusknoir: 52 / 191, 27.2%.
  - Deck 4 Dragapult: 105 / 191, 55.0%.
  - Deck 5 Dragapult Dudunsparce: 83 / 191, 43.5%.
  - Deck 6 Hydrapple: 95 / 191, 49.7%.
  - Deck 7 Raging Bolt Ogerpon: 110 / 204, 53.9%.
  - Deck 8 Dragapult Blaziken: 94 / 204, 46.1%.
  - Deck 9 Ogerpon Box: 126 / 204, 61.8%.
  - Deck 10 Crustle sample: 114 / 204, 55.9%.
  - Deck 11 Mega Lucario ex: 140 / 204, 68.6%.
  - Deck 12 Mega Abomasnow ex: 119 / 204, 58.3%.
  - Deck 13 Iono's Bellibolt ex: 100 / 204, 49.0%.

Comparison to iteration 4 self-play:

- Deck 11 remains the strongest self-play deck, though down 2 wins from
  iteration 4's 142 / 204.
- Deck 9 improved from 112 / 191 to 126 / 204 and is now the second-strongest
  self-play deck in the window.
- Deck 3 regressed most sharply, from 75 / 204 to 52 / 191.
- Deck 1 is unchanged at 24 / 204 and remains the main self-play weakness.

Conclusion and next step:

- The iteration-5 raw self-play window is valid: zero errors and enough
  on-policy trajectory rows for the next per-deck PPO update.
- While iteration-4 eval is still running, use the open ERAWAN slot for the
  iteration-5 PPO update from source iteration 4 to target iteration 5.
- Keep `iter-0005/raw_train/` until the iteration-5 PPO report and later eval
  result are inspected.

## 2026-07-05 - Alpha League Iteration-5 Online PPO Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0005_ppo_specialists_report.json`;
  - `slurm-73460-phase5-alpha-ppo-specialists.out`.
- ERAWAN job: `73460`.
- PPO source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0004/specialists`.
- PPO output specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0005/raw_train/phase5_alpha_league_selfplay.jsonl`.

PPO update:

- Deck checkpoints written: 13 / 13.
- On-policy trajectory rows consumed: 214,943.
- Skipped no-target rows: 0.
- Skipped off-policy rows: 0.
- `require_on_policy`: true for every deck update.
- Per-deck PPO examples:
  - Deck 1: 28,938 examples, mean advantage -0.2215, final loss -0.0005.
  - Deck 2: 12,256 examples, mean advantage 0.0141, final loss 0.0911.
  - Deck 3: 14,716 examples, mean advantage 0.0175, final loss 0.5249.
  - Deck 4: 17,344 examples, mean advantage 0.0283, final loss 0.1994.
  - Deck 5: 16,727 examples, mean advantage -0.0692, final loss 0.0261.
  - Deck 6: 14,303 examples, mean advantage 0.0548, final loss 1.2442.
  - Deck 7: 15,042 examples, mean advantage 0.0336, final loss 0.0871.
  - Deck 8: 21,584 examples, mean advantage -0.1186, final loss 0.0184.
  - Deck 9: 14,538 examples, mean advantage -0.1238, final loss 0.0549.
  - Deck 10: 12,992 examples, mean advantage -0.0364, final loss 0.6766.
  - Deck 11: 13,355 examples, mean advantage 0.0530, final loss 0.4402.
  - Deck 12: 7,702 examples, mean advantage 0.0182, final loss 0.7552.
  - Deck 13: 25,446 examples, mean advantage 0.0787, final loss 0.4327.

Conclusion and next step:

- The iteration-5 online PPO update is valid: it used the full iter-5 raw
  self-play window and produced a complete 13-checkpoint specialist family with
  no off-policy leakage.
- Queue iteration-6 online self-play from
  `models/rl/phase5_league_alpha/iter-0005/specialists` while the iteration-4
  eval continues to run. Do not start iteration-6 PPO until the iteration-6
  self-play report is inspected.
- Keep `iter-0005/raw_train/` until the iteration-5 eval result is inspected.

## 2026-07-05 - Alpha League Iteration-4 Full-Agent Evaluation

ERAWAN result:

- Uploaded and inspected:
  - `phase5_alpha_iter0004_specialists_full_vs_rule_30g.json`;
  - `phase5_alpha_iter0004_specialists_full_vs_rule_30g.md`.
- The uploaded `slurm-73454-phase5-league-eval.out` corresponds to the earlier
  iteration-3 eval, so the iteration-4 eval job id was not confirmed from the
  uploaded SLURM output.
- Agent: `phase5-full`.
- Specialist model directory:
  `models/rl/phase5_league_alpha/iter-0004/specialists`.
- Benchmark: 13 x 13 league agent-vs-rule, 30 games per matchup.

Aggregate:

- Games: 5,070.
- Wins: 2,692.
- Losses: 2,364.
- Draws: 14.
- Timeouts: 39.
- Errors: 0.
- Win rate: 0.5310.
- Search telemetry: 199,346 searched decisions, 34,148 search-changed
  decisions, 0 search errors, 0 candidate errors, 2,113 truncated candidates,
  average search 0.0531s, max search 3.2708s.

Comparison:

- Versus iteration 3, iteration 4 is down 5 wins: 2,692 / 5,070 vs
  2,697 / 5,070.
- Versus iteration 2, iteration 4 is down 18 wins: 2,692 / 5,070 vs
  2,710 / 5,070.
- Versus the recorded iteration-0 specialist eval, iteration 4 is up 41 wins:
  2,692 / 5,070 vs 2,651 / 5,070.
- Against the four required sample rule-agent opponents, iteration 4 scored
  686 / 1,560, up 9 wins from iteration 3's 677 / 1,560, down 8 wins from
  iteration 2's 694 / 1,560, and up 27 wins from iteration 0's 659 / 1,560.

Deck totals:

- Deck 1 Alakazam Dudunsparce: 53 / 390, 13.6%.
- Deck 2 Crustle: 254 / 390, 65.1%.
- Deck 3 Dragapult Dusknoir: 127 / 390, 32.6%.
- Deck 4 Dragapult: 203 / 390, 52.1%.
- Deck 5 Dragapult Dudunsparce: 187 / 390, 47.9%.
- Deck 6 Hydrapple: 219 / 390, 56.2%.
- Deck 7 Raging Bolt Ogerpon: 204 / 390, 52.3%.
- Deck 8 Dragapult Blaziken: 208 / 390, 53.3%.
- Deck 9 Ogerpon Box: 226 / 390, 57.9%.
- Deck 10 Crustle sample: 238 / 390, 61.0%.
- Deck 11 Mega Lucario ex: 303 / 390, 77.7%.
- Deck 12 Mega Abomasnow ex: 252 / 390, 64.6%.
- Deck 13 Iono's Bellibolt ex: 218 / 390, 55.9%.

Required-sample opponent slice by agent deck:

- Deck 11 Mega Lucario ex: 81 / 120, 67.5%.
- Deck 10 Crustle sample: 71 / 120, 59.2%.
- Deck 2 Crustle: 69 / 120, 57.5%.
- Deck 12 Mega Abomasnow ex: 65 / 120, 54.2%.
- Deck 13 Iono's Bellibolt ex: 62 / 120, 51.7%.
- Deck 9 Ogerpon Box: 58 / 120, 48.3%.
- Deck 6 Hydrapple: 55 / 120, 45.8%.
- Deck 7 Raging Bolt Ogerpon: 49 / 120, 40.8%.
- Deck 4 Dragapult: 48 / 120, 40.0%.
- Deck 8 Dragapult Blaziken: 46 / 120, 38.3%.
- Deck 5 Dragapult Dudunsparce: 44 / 120, 36.7%.
- Deck 3 Dragapult Dusknoir: 34 / 120, 28.3%.
- Deck 1 Alakazam Dudunsparce: 4 / 120, 3.3%.

Conclusion and next step:

- Iteration 4 continues the small overall regression from iteration 2, but the
  required-sample opponent slice rebounded versus iteration 3.
- Deck 11 is still the strongest full-agent specialist overall and on the
  required-sample opponent slice. Deck 1 remains the main weakness.
- Do not promote iteration 4 over iteration 2 on the overall gate. Continue the
  online loop and evaluate iteration 5 next.

## 2026-07-05 - Alpha League Iteration-6 Self-Play Window

ERAWAN result:

- Uploaded and inspected:
  - `iter-0006_league_iteration_report.json`;
  - `slurm-73470-phase5-alpha-league.out`.
- ERAWAN job: `73470`.
- Online collector: `AGENT=phase5-rl`.
- Source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0006/raw_train/phase5_alpha_league_selfplay.jsonl`.

League collection:

- Games: 1,300 / 1,300 started, 13 decks x 100 scheduled games.
- Steps / trajectory rows: 216,080.
- Draws: 9. Timeouts: 30. Errors: 0.
- Per-deck online self-play records:
  - Deck 1 Alakazam Dudunsparce: 28 / 191, 14.7%.
  - Deck 2 Crustle: 120 / 191, 62.8%.
  - Deck 3 Dragapult Dusknoir: 63 / 204, 30.9%.
  - Deck 4 Dragapult: 112 / 204, 54.9%.
  - Deck 5 Dragapult Dudunsparce: 77 / 204, 37.7%.
  - Deck 6 Hydrapple: 107 / 204, 52.5%.
  - Deck 7 Raging Bolt Ogerpon: 113 / 204, 55.4%.
  - Deck 8 Dragapult Blaziken: 106 / 204, 52.0%.
  - Deck 9 Ogerpon Box: 115 / 204, 56.4%.
  - Deck 10 Crustle sample: 121 / 204, 59.3%.
  - Deck 11 Mega Lucario ex: 126 / 204, 61.8%.
  - Deck 12 Mega Abomasnow ex: 106 / 191, 55.5%.
  - Deck 13 Iono's Bellibolt ex: 97 / 191, 50.8%.

Conclusion and next step:

- The iteration-6 raw self-play window is valid: zero errors and enough
  on-policy trajectory rows for the next per-deck PPO update.
- Fill the two ERAWAN slots with:
  - iteration-6 PPO update from source iteration 5 to target iteration 6;
  - iteration-5 full-agent-vs-rule eval from
    `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Keep `iter-0006/raw_train/` until the iteration-6 PPO report and later eval
  result are inspected.

## 2026-07-05 - Alpha League Iteration-6 Online PPO Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0006_ppo_specialists_report.json`;
  - `slurm-73489-phase5-alpha-ppo-specialists.out`.
- ERAWAN job: `73489`.
- PPO source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- PPO output specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0006/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0006/raw_train/phase5_alpha_league_selfplay.jsonl`.

PPO update:

- Deck checkpoints written: 13 / 13.
- On-policy trajectory rows consumed: 216,080.
- Skipped no-target rows: 0.
- Skipped off-policy rows: 0.
- `require_on_policy`: true for every deck update.
- Per-deck PPO examples:
  - Deck 1: 28,825 examples, mean advantage -0.2528, final loss 0.0045.
  - Deck 2: 11,378 examples, mean advantage 0.1509, final loss 0.3234.
  - Deck 3: 15,485 examples, mean advantage 0.0944, final loss 0.1368.
  - Deck 4: 18,746 examples, mean advantage 0.0709, final loss 0.4524.
  - Deck 5: 17,665 examples, mean advantage -0.1940, final loss 0.3399.
  - Deck 6: 14,924 examples, mean advantage 0.0846, final loss 0.0678.
  - Deck 7: 15,073 examples, mean advantage 0.2259, final loss 0.2757.
  - Deck 8: 21,525 examples, mean advantage -0.0019, final loss 0.2881.
  - Deck 9: 14,948 examples, mean advantage -0.3128, final loss 0.1864.
  - Deck 10: 13,723 examples, mean advantage 0.1223, final loss 0.3526.
  - Deck 11: 12,930 examples, mean advantage 0.0248, final loss 0.1434.
  - Deck 12: 7,564 examples, mean advantage -0.2018, final loss 0.4155.
  - Deck 13: 23,294 examples, mean advantage 0.2550, final loss 0.7832.

Conclusion and next step:

- The iteration-6 online PPO update is valid: it used the full iter-6 raw
  self-play window and produced a complete 13-checkpoint specialist family with
  no off-policy leakage.
- Since iteration-5 eval is still running, use the open ERAWAN slot for
  iteration-7 online self-play from
  `models/rl/phase5_league_alpha/iter-0006/specialists`.
- Do not start iteration-7 PPO until the iteration-7 self-play report is
  inspected, and keep `iter-0006/raw_train/` until the iteration-6 eval result
  is inspected.

## 2026-07-05 - Alpha League Iteration-7 Self-Play Window

ERAWAN result:

- Uploaded and inspected:
  - `iter-0007_league_iteration_report.json`;
  - `slurm-73496-phase5-alpha-league.out`.
- ERAWAN job: `73496`.
- Online collector: `AGENT=phase5-rl`.
- Source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0006/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0007/raw_train/phase5_alpha_league_selfplay.jsonl`.

League collection:

- Games: 1,300 / 1,300 started, 13 decks x 100 scheduled games.
- Steps / trajectory rows: 211,335.
- Draws: 6. Timeouts: 21. Errors: 0.
- Per-deck online self-play records:
  - Deck 1 Alakazam Dudunsparce: 20 / 204, 9.8%.
  - Deck 2 Crustle: 133 / 204, 65.2%.
  - Deck 3 Dragapult Dusknoir: 63 / 204, 30.9%.
  - Deck 4 Dragapult: 108 / 204, 52.9%.
  - Deck 5 Dragapult Dudunsparce: 75 / 204, 36.8%.
  - Deck 6 Hydrapple: 99 / 204, 48.5%.
  - Deck 7 Raging Bolt Ogerpon: 115 / 204, 56.4%.
  - Deck 8 Dragapult Blaziken: 107 / 191, 56.0%.
  - Deck 9 Ogerpon Box: 106 / 191, 55.5%.
  - Deck 10 Crustle sample: 122 / 191, 63.9%.
  - Deck 11 Mega Lucario ex: 133 / 191, 69.6%.
  - Deck 12 Mega Abomasnow ex: 120 / 204, 58.8%.
  - Deck 13 Iono's Bellibolt ex: 93 / 204, 45.6%.

Conclusion and next step:

- The iteration-7 raw self-play window is valid: zero errors and enough
  on-policy trajectory rows for the next per-deck PPO update.
- Because iteration-5 eval is still running and the user wants both evals ready
  by morning, queue iteration-6 full-agent-vs-rule eval from
  `models/rl/phase5_league_alpha/iter-0006/specialists` in the open ERAWAN
  slot.
- After either eval slot frees, the next training job is iteration-7 PPO from
  source iteration 6 to target iteration 7.
- Keep `iter-0007/raw_train/` until the iteration-7 PPO report and later eval
  result are inspected.

## 2026-07-06 - Alpha League Iteration-5 Full-Agent Evaluation

ERAWAN result:

- Uploaded and inspected:
  - `phase5_alpha_iter0005_specialists_full_vs_rule_30g.json`;
  - `phase5_alpha_iter0005_specialists_full_vs_rule_30g.md`;
  - `slurm-73490-phase5-league-eval.out`.
- ERAWAN job: `73490`.
- Agent: `phase5-full`.
- Specialist model directory:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Benchmark: 13 x 13 league agent-vs-rule, 30 games per matchup.

Aggregate:

- Games: 5,070.
- Wins: 2,714.
- Losses: 2,341.
- Draws: 15.
- Timeouts: 46.
- Errors: 0.
- Win rate: 0.5353.
- Search telemetry: 199,116 searched decisions, 35,598 search-changed
  decisions, 0 search errors, 0 candidate errors, 1,980 truncated candidates,
  average search 0.0626s, max search 3.9439s.

Comparison:

- Versus iteration 2, iteration 5 is up 4 wins: 2,714 / 5,070 vs
  2,710 / 5,070.
- Versus iteration 4, iteration 5 is up 22 wins: 2,714 / 5,070 vs
  2,692 / 5,070.
- Versus the recorded iteration-0 specialist eval, iteration 5 is up 63 wins:
  2,714 / 5,070 vs 2,651 / 5,070.
- Against the four required sample rule-agent opponents, iteration 5 scored
  704 / 1,560, up 10 wins from iteration 2's 694 / 1,560 and up 45 wins from
  iteration 0's 659 / 1,560.

Deck totals:

- Deck 1 Alakazam Dudunsparce: 55 / 390, 14.1%.
- Deck 2 Crustle: 248 / 390, 63.6%.
- Deck 3 Dragapult Dusknoir: 138 / 390, 35.4%.
- Deck 4 Dragapult: 215 / 390, 55.1%.
- Deck 5 Dragapult Dudunsparce: 207 / 390, 53.1%.
- Deck 6 Hydrapple: 217 / 390, 55.6%.
- Deck 7 Raging Bolt Ogerpon: 203 / 390, 52.1%.
- Deck 8 Dragapult Blaziken: 203 / 390, 52.1%.
- Deck 9 Ogerpon Box: 219 / 390, 56.2%.
- Deck 10 Crustle sample: 234 / 390, 60.0%.
- Deck 11 Mega Lucario ex: 303 / 390, 77.7%.
- Deck 12 Mega Abomasnow ex: 257 / 390, 65.9%.
- Deck 13 Iono's Bellibolt ex: 215 / 390, 55.1%.

Required-sample opponent slice:

- Aggregate: 704 / 1,560, 45.1%.
- Best decks: deck 11 Mega Lucario ex at 86 / 120 and deck 12 Mega Abomasnow ex
  at 75 / 120.
- Deck 1 remains the weakest at 7 / 120.

Conclusion and next step:

- Iteration 5 is the new best full-agent checkpoint on the overall 13 x 13 x 30
  gate and on the four-required-sample-opponent slice.
- Treat `models/rl/phase5_league_alpha/iter-0005/specialists` as the current
  promotion/package candidate unless a later evaluation beats it.
- Continue the online loop from the already collected iteration-7 raw window by
  running iteration-7 PPO when a slot is available.

## 2026-07-06 - Alpha League Iteration-6 Full-Agent Evaluation

ERAWAN result:

- Uploaded and inspected:
  - `phase5_alpha_iter0006_specialists_full_vs_rule_30g.json`;
  - `phase5_alpha_iter0006_specialists_full_vs_rule_30g.md`;
  - `slurm-73503-phase5-league-eval.out`.
- ERAWAN job: `73503`.
- Agent: `phase5-full`.
- Specialist model directory:
  `models/rl/phase5_league_alpha/iter-0006/specialists`.
- Benchmark: 13 x 13 league agent-vs-rule, 30 games per matchup.

Aggregate:

- Games: 5,070.
- Wins: 2,654.
- Losses: 2,400.
- Draws: 16.
- Timeouts: 50.
- Errors: 0.
- Win rate: 0.5235.
- Search telemetry: 199,609 searched decisions, 35,698 search-changed
  decisions, 0 search errors, 0 candidate errors, 2,226 truncated candidates,
  average search 0.0624s, max search 4.5432s.

Comparison:

- Versus iteration 5, iteration 6 is down 60 wins: 2,654 / 5,070 vs
  2,714 / 5,070.
- Versus iteration 2, iteration 6 is down 56 wins: 2,654 / 5,070 vs
  2,710 / 5,070.
- Versus the recorded iteration-0 specialist eval, iteration 6 is up 3 wins:
  2,654 / 5,070 vs 2,651 / 5,070.
- Against the four required sample rule-agent opponents, iteration 6 scored
  684 / 1,560, down 20 wins from iteration 5's 704 / 1,560.

Deck totals:

- Deck 1 Alakazam Dudunsparce: 58 / 390, 14.9%.
- Deck 2 Crustle: 238 / 390, 61.0%.
- Deck 3 Dragapult Dusknoir: 126 / 390, 32.3%.
- Deck 4 Dragapult: 195 / 390, 50.0%.
- Deck 5 Dragapult Dudunsparce: 192 / 390, 49.2%.
- Deck 6 Hydrapple: 213 / 390, 54.6%.
- Deck 7 Raging Bolt Ogerpon: 201 / 390, 51.5%.
- Deck 8 Dragapult Blaziken: 216 / 390, 55.4%.
- Deck 9 Ogerpon Box: 222 / 390, 56.9%.
- Deck 10 Crustle sample: 235 / 390, 60.3%.
- Deck 11 Mega Lucario ex: 310 / 390, 79.5%.
- Deck 12 Mega Abomasnow ex: 254 / 390, 65.1%.
- Deck 13 Iono's Bellibolt ex: 194 / 390, 49.7%.

Required-sample opponent slice:

- Aggregate: 684 / 1,560, 43.8%.
- Best decks: deck 11 Mega Lucario ex at 78 / 120 and deck 12 Mega Abomasnow ex
  at 70 / 120.
- Deck 1 remains the weakest at 5 / 120.

Conclusion and next step:

- Iteration 6 is a regression from iteration 5 and iteration 2 on the overall
  gate and the required-sample opponent slice. Do not promote iteration 6.
- Continue exploration with iteration-7 PPO from the already inspected
  iteration-7 raw self-play window, but keep iteration 5 as the current best
  promotion/package candidate.

## 2026-07-06 - Alpha League Iteration-7 Online PPO Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0007_ppo_specialists_report.json`;
  - `slurm-73508-phase5-alpha-ppo-specialists.out`.
- ERAWAN job: `73508`.
- PPO source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0006/specialists`.
- PPO output specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0007/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0007/raw_train/phase5_alpha_league_selfplay.jsonl`.

PPO update:

- Deck checkpoints written: 13 / 13.
- On-policy trajectory rows consumed: 211,335.
- Skipped no-target rows: 0.
- Skipped off-policy rows: 0.
- `require_on_policy`: true for every deck update.
- Per-deck PPO examples:
  - Deck 1: 26,376 examples, mean advantage -0.4014, final loss 0.0017.
  - Deck 2: 12,430 examples, mean advantage 0.1471, final loss 0.0868.
  - Deck 3: 15,046 examples, mean advantage -0.0068, final loss -0.0084.
  - Deck 4: 18,770 examples, mean advantage 0.0736, final loss 1.4361.
  - Deck 5: 17,395 examples, mean advantage -0.0980, final loss 0.0245.
  - Deck 6: 15,070 examples, mean advantage -0.0468, final loss 0.0628.
  - Deck 7: 14,535 examples, mean advantage 0.0869, final loss 0.0728.
  - Deck 8: 19,761 examples, mean advantage -0.1632, final loss 0.1624.
  - Deck 9: 14,496 examples, mean advantage -0.2006, final loss 0.0639.
  - Deck 10: 12,754 examples, mean advantage -0.0279, final loss 0.0815.
  - Deck 11: 11,986 examples, mean advantage 0.3172, final loss 0.0753.
  - Deck 12: 7,616 examples, mean advantage -0.0772, final loss 1.0839.
  - Deck 13: 25,100 examples, mean advantage -0.0476, final loss 0.0835.

Conclusion and next step:

- The iteration-7 online PPO update is valid: it used the full iter-7 raw
  self-play window and produced a complete 13-checkpoint specialist family with
  no off-policy leakage.
- Keep iteration 5 as the current best promotion/package candidate until a
  later evaluation beats it.
- Fill the two ERAWAN slots with:
  - iteration-7 full-agent-vs-rule eval from
    `models/rl/phase5_league_alpha/iter-0007/specialists`;
  - iteration-8 online self-play from
    `models/rl/phase5_league_alpha/iter-0007/specialists`.

## 2026-07-06 - Alpha League Evaluation Time-Series Plot

Artifact:

- Generated presentation-ready SVG plots and a CSV summary from the available
  full-agent-vs-rule evaluation reports:
  - `reports/phase5_alpha_eval_winrate_timeseries_combined.svg`;
  - `reports/phase5_alpha_eval_winrate_timeseries_overall.svg`;
  - `reports/phase5_alpha_eval_winrate_timeseries_per_deck.svg`;
  - `reports/phase5_alpha_eval_winrate_timeseries.html`;
  - `reports/phase5_alpha_eval_winrate_timeseries_summary.csv`.
- Included evaluated iterations: 0, 2, 3, 4, 5, and 6. Iteration 1 is omitted
  because no local iteration-1 full-agent-vs-rule JSON report was available.
- The plot preserves the iteration gap instead of interpolating or inventing
  missing iteration-1 data.

Conclusion:

- The overall time series shows iteration 5 as the current best checkpoint at
  2,714 / 5,070 wins, 0.5353 win rate.
- The per-deck plot shows the persistent deck-1 weakness and the stable strength
  of decks 11 and 12 across evaluated checkpoints.

## 2026-07-06 - Alpha League Iteration-8 Self-Play Window

ERAWAN result:

- Uploaded and inspected:
  - `iter-0008_league_iteration_report.json`;
  - `slurm-73526-phase5-alpha-league.out`.
- ERAWAN job: `73526`.
- Online collector: `AGENT=phase5-rl`.
- Source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0007/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0008/raw_train/phase5_alpha_league_selfplay.jsonl`.

League collection:

- Games: 1,300 / 1,300 started, 13 decks x 100 scheduled games.
- Steps / trajectory rows: 208,449.
- Draws: 9. Timeouts: 22. Errors: 0.
- Per-deck online self-play records:
  - Deck 1 Alakazam Dudunsparce: 29 / 204, 14.2%.
  - Deck 2 Crustle: 138 / 204, 67.6%.
  - Deck 3 Dragapult Dusknoir: 65 / 204, 31.9%.
  - Deck 4 Dragapult: 96 / 191, 50.3%.
  - Deck 5 Dragapult Dudunsparce: 70 / 191, 36.6%.
  - Deck 6 Hydrapple: 95 / 191, 49.7%.
  - Deck 7 Raging Bolt Ogerpon: 102 / 191, 53.4%.
  - Deck 8 Dragapult Blaziken: 115 / 204, 56.4%.
  - Deck 9 Ogerpon Box: 108 / 204, 52.9%.
  - Deck 10 Crustle sample: 119 / 204, 58.3%.
  - Deck 11 Mega Lucario ex: 129 / 204, 63.2%.
  - Deck 12 Mega Abomasnow ex: 120 / 204, 58.8%.
  - Deck 13 Iono's Bellibolt ex: 105 / 204, 51.5%.

Conclusion and next step:

- The iteration-8 raw self-play window is valid: zero errors and enough
  on-policy trajectory rows for the next per-deck PPO update.
- Since iteration-7 eval is still running, use the open ERAWAN slot for
  iteration-8 PPO from source iteration 7 to target iteration 8.
- Keep iteration 5 as the current best promotion/package candidate until a
  later evaluation beats it.
- Keep `iter-0008/raw_train/` until the iteration-8 PPO report and later eval
  result are inspected.

## 2026-07-06 - Alpha League Iteration-8 Online PPO Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0008_ppo_specialists_report.json`;
  - `slurm-73542-phase5-alpha-ppo-specialists.out`.
- ERAWAN job: `73542`.
- PPO source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0007/specialists`.
- PPO output specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0008/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0008/raw_train/phase5_alpha_league_selfplay.jsonl`.

PPO update:

- Deck checkpoints written: 13 / 13.
- On-policy trajectory rows consumed: 208,449.
- Skipped no-target rows: 0.
- Skipped off-policy rows: 0.
- `require_on_policy`: true for every deck update.
- Per-deck PPO examples:
  - Deck 1: 25,202 examples, mean advantage -0.2065, final loss 0.0165.
  - Deck 2: 12,894 examples, mean advantage -0.0289, final loss 0.2209.
  - Deck 3: 14,392 examples, mean advantage 0.1752, final loss 0.0789.
  - Deck 4: 17,800 examples, mean advantage 0.0226, final loss 0.2430.
  - Deck 5: 15,315 examples, mean advantage -0.2064, final loss -0.0081.
  - Deck 6: 13,699 examples, mean advantage 0.0257, final loss 0.0032.
  - Deck 7: 13,814 examples, mean advantage -0.0051, final loss 0.5931.
  - Deck 8: 20,522 examples, mean advantage -0.0976, final loss 0.7084.
  - Deck 9: 14,819 examples, mean advantage -0.3629, final loss 0.0544.
  - Deck 10: 13,307 examples, mean advantage 0.1222, final loss 0.0833.
  - Deck 11: 13,549 examples, mean advantage -0.0017, final loss 0.2165.
  - Deck 12: 7,794 examples, mean advantage -0.0619, final loss 0.5862.
  - Deck 13: 25,342 examples, mean advantage -0.1366, final loss 0.1498.

Conclusion and next step:

- The iteration-8 online PPO update is valid: it used the full iter-8 raw
  self-play window and produced a complete 13-checkpoint specialist family with
  no off-policy leakage.
- Since iteration-7 eval is still running, use the open ERAWAN slot for
  iteration-9 online self-play from
  `models/rl/phase5_league_alpha/iter-0008/specialists`.
- Keep iteration 5 as the current best promotion/package candidate until a
  later evaluation beats it.

## 2026-07-06 - Alpha League Iteration-9 Self-Play Window

ERAWAN result:

- Uploaded and inspected:
  - `iter-0009_league_iteration_report.json`;
  - `slurm-73550-phase5-alpha-league.out`.
- ERAWAN job: `73550`.
- Online collector: `AGENT=phase5-rl`.
- Source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0008/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0009/raw_train/phase5_alpha_league_selfplay.jsonl`.

League collection:

- Games: 1,300 / 1,300 started, 13 decks x 100 scheduled games.
- Steps / trajectory rows: 208,155.
- Draws: 11. Timeouts: 28. Errors: 0.
- Per-deck online self-play records:
  - Deck 1 Alakazam Dudunsparce: 24 / 191, 12.6%.
  - Deck 2 Crustle: 131 / 191, 68.6%.
  - Deck 3 Dragapult Dusknoir: 46 / 191, 24.1%.
  - Deck 4 Dragapult: 106 / 204, 52.0%.
  - Deck 5 Dragapult Dudunsparce: 53 / 204, 26.0%.
  - Deck 6 Hydrapple: 99 / 204, 48.5%.
  - Deck 7 Raging Bolt Ogerpon: 117 / 204, 57.4%.
  - Deck 8 Dragapult Blaziken: 101 / 204, 49.5%.
  - Deck 9 Ogerpon Box: 116 / 204, 56.9%.
  - Deck 10 Crustle sample: 125 / 204, 61.3%.
  - Deck 11 Mega Lucario ex: 145 / 204, 71.1%.
  - Deck 12 Mega Abomasnow ex: 130 / 204, 63.7%.
  - Deck 13 Iono's Bellibolt ex: 96 / 191, 50.3%.

Conclusion and next step:

- The iteration-9 raw self-play window is valid: zero errors and enough
  on-policy trajectory rows for the next per-deck PPO update.
- Since iteration-7 eval is still running, use the open ERAWAN slot for
  iteration-9 PPO from source iteration 8 to target iteration 9.
- Keep iteration 5 as the current best promotion/package candidate until a
  later evaluation beats it.
- Keep `iter-0009/raw_train/` until the iteration-9 PPO report and later eval
  result are inspected.

## 2026-07-06 - Alpha League Iteration-7 Full-Agent Evaluation

ERAWAN result:

- Uploaded and inspected:
  - `phase5_alpha_iter0007_specialists_full_vs_rule_30g.json`;
  - `phase5_alpha_iter0007_specialists_full_vs_rule_30g.md`;
  - `slurm-73525-phase5-league-eval.out`.
- ERAWAN job: `73525`.
- Agent: `phase5-full`.
- Generalist prior:
  `models/rl/phase5_generalist_policy_13deck_10k.pt`.
- Specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0007/specialists`.
- Gate: 13 x 13 agent-vs-rule league evaluation, 30 games per matchup,
  max 600 selections per game.

Evaluation:

- Overall: 2,641 / 5,070 wins, 0.5209 win rate, 18 draws, 52 timeouts, 0
  errors.
- This is down 73 wins from the current best iteration 5 and down 13 wins from
  iteration 6, so iteration 7 is not a promotion candidate.
- Against the four required sample rule-agent opponents: 639 / 1,560 wins,
  0.4096 win rate, 3 draws, 0 timeouts, 0 errors.
- Top overall decks:
  - Deck 11 Mega Lucario ex: 298 / 390, 76.4%.
  - Deck 12 Mega Abomasnow ex: 248 / 390, 63.6%.
  - Deck 10 Crustle sample: 236 / 390, 60.5%.
  - Deck 2 Crustle: 235 / 390, 60.3%.
  - Deck 6 Hydrapple: 213 / 390, 54.6%.
- Required-sample slice leaders:
  - Deck 11 Mega Lucario ex: 80 / 120, 66.7%.
  - Deck 12 Mega Abomasnow ex: 63 / 120, 52.5%.
  - Deck 4 Dragapult: 60 / 120, 50.0%.
  - Deck 2 Crustle: 58 / 120, 48.3%.
- Persistent weaknesses:
  - Deck 1 Alakazam Dudunsparce: 60 / 390 overall and 0 / 120 against the
    required sample rule-agent opponents.
  - Deck 3 Dragapult Dusknoir: 147 / 390 overall and 29 / 120 against the
    required sample rule-agent opponents.

Search telemetry:

- Searched decisions: 197,493.
- Search-changed decisions: 36,252, 0.1836 change rate.
- Search errors: 0. Candidate errors: 0.
- Candidate probes: 715,257.
- Truncated candidates: 2,125.
- Average search seconds: 0.0510. Max search seconds: 2.7742.

Conclusion:

- Do not promote iteration 7. Keep
  `models/rl/phase5_league_alpha/iter-0005/specialists` as the current best
  promotion/package candidate.
- Next evaluation target is iteration 8; next online training target after the
  completed iteration-9 PPO update is iteration 10 self-play.

## 2026-07-06 - Alpha League Iteration-9 Online PPO Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0009_ppo_specialists_report.json`;
  - `slurm-73571-phase5-alpha-ppo-specialists.out`.
- ERAWAN job: `73571`.
- PPO source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0008/specialists`.
- PPO output specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0009/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0009/raw_train/phase5_alpha_league_selfplay.jsonl`.

PPO update:

- Deck checkpoints written: 13 / 13.
- On-policy trajectory rows consumed: 208,155.
- Skipped no-target rows: 0.
- Skipped off-policy rows: 0.
- `require_on_policy`: true for every deck update.
- Per-deck PPO examples:
  - Deck 1: 26,948 examples, mean advantage -0.1441, final loss 1.7519.
  - Deck 2: 11,738 examples, mean advantage 0.0349, final loss 0.3033.
  - Deck 3: 12,737 examples, mean advantage -0.0362, final loss 0.1003.
  - Deck 4: 19,381 examples, mean advantage -0.0180, final loss 0.0635.
  - Deck 5: 15,517 examples, mean advantage -0.3806, final loss 0.0577.
  - Deck 6: 14,244 examples, mean advantage -0.0981, final loss -0.0091.
  - Deck 7: 14,339 examples, mean advantage 0.1749, final loss 0.4947.
  - Deck 8: 21,005 examples, mean advantage -0.2664, final loss 0.4725.
  - Deck 9: 15,031 examples, mean advantage -0.2433, final loss 0.3353.
  - Deck 10: 13,754 examples, mean advantage -0.0241, final loss 0.2730.
  - Deck 11: 12,577 examples, mean advantage 0.2783, final loss 0.8019.
  - Deck 12: 8,288 examples, mean advantage -0.0211, final loss 1.1182.
  - Deck 13: 22,596 examples, mean advantage 0.2464, final loss 0.5052.

Conclusion and next step:

- The iteration-9 online PPO update is valid: it used the full iter-9 raw
  self-play window and produced a complete 13-checkpoint specialist family with
  no off-policy leakage.
- The iteration-9 raw training window can be deleted after confirming the
  report and checkpoints are retained, matching the Phase 5 data policy.
- Fill the next two ERAWAN slots with:
  - iteration-8 full-agent-vs-rule eval from
    `models/rl/phase5_league_alpha/iter-0008/specialists`;
  - iteration-10 online self-play from
    `models/rl/phase5_league_alpha/iter-0009/specialists`.
- Keep iteration 5 as the current best promotion/package candidate until a
  later evaluation beats it.

## 2026-07-06 - Alpha League Iteration-10 Self-Play Window

ERAWAN result:

- Uploaded and inspected:
  - `iter-0010_league_iteration_report.json`;
  - `slurm-73578-phase5-alpha-league.out`.
- ERAWAN job: `73578`.
- Online collector: `AGENT=phase5-rl`.
- Source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0009/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0010/raw_train/phase5_alpha_league_selfplay.jsonl`.

League collection:

- Games: 1,300 / 1,300 started, 13 decks x 100 scheduled games.
- Steps / trajectory rows: 211,241.
- Draws: 9. Timeouts: 22. Errors: 0.
- Per-deck online self-play records:
  - Deck 1 Alakazam Dudunsparce: 30 / 204, 14.7%.
  - Deck 2 Crustle: 134 / 204, 65.7%.
  - Deck 3 Dragapult Dusknoir: 54 / 204, 26.5%.
  - Deck 4 Dragapult: 106 / 204, 52.0%.
  - Deck 5 Dragapult Dudunsparce: 69 / 204, 33.8%.
  - Deck 6 Hydrapple: 99 / 204, 48.5%.
  - Deck 7 Raging Bolt Ogerpon: 111 / 204, 54.4%.
  - Deck 8 Dragapult Blaziken: 120 / 204, 58.8%.
  - Deck 9 Ogerpon Box: 108 / 191, 56.5%.
  - Deck 10 Crustle sample: 111 / 191, 58.1%.
  - Deck 11 Mega Lucario ex: 120 / 191, 62.8%.
  - Deck 12 Mega Abomasnow ex: 124 / 191, 64.9%.
  - Deck 13 Iono's Bellibolt ex: 105 / 204, 51.5%.

Conclusion and next step:

- The iteration-10 raw self-play window is valid: zero errors and enough
  on-policy trajectory rows for the next per-deck PPO update.
- Since iteration-8 eval is still running, use the open ERAWAN slot for
  iteration-10 PPO from source iteration 9 to target iteration 10.
- Keep iteration 5 as the current best promotion/package candidate until a
  later evaluation beats it.
- Keep `iter-0010/raw_train/` until the iteration-10 PPO report and later eval
  result are inspected.

## 2026-07-06 - Alpha League Iteration-10 Online PPO Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0010_ppo_specialists_report.json`;
  - `slurm-73587-phase5-alpha-ppo-specialists.out`.
- ERAWAN job: `73587`.
- PPO source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0009/specialists`.
- PPO output specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0010/specialists`.
- Raw online window:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_league_alpha/iterations/iter-0010/raw_train/phase5_alpha_league_selfplay.jsonl`.

PPO update:

- Deck checkpoints written: 13 / 13.
- On-policy trajectory rows consumed: 211,241.
- Skipped no-target rows: 0.
- Skipped off-policy rows: 0.
- `require_on_policy`: true for every deck update.
- Per-deck PPO examples:
  - Deck 1: 26,282 examples, mean advantage -0.0872, final loss 0.0074.
  - Deck 2: 12,505 examples, mean advantage 0.1388, final loss 0.0564.
  - Deck 3: 14,234 examples, mean advantage 0.0361, final loss 0.0784.
  - Deck 4: 18,881 examples, mean advantage 0.1324, final loss 2.3183.
  - Deck 5: 16,028 examples, mean advantage -0.2248, final loss 0.0718.
  - Deck 6: 14,846 examples, mean advantage -0.0927, final loss 0.2639.
  - Deck 7: 14,880 examples, mean advantage -0.1499, final loss 0.0241.
  - Deck 8: 21,780 examples, mean advantage 0.2588, final loss 0.1679.
  - Deck 9: 14,193 examples, mean advantage -0.1528, final loss 0.8347.
  - Deck 10: 12,997 examples, mean advantage -0.1341, final loss 1.2447.
  - Deck 11: 12,177 examples, mean advantage -0.1347, final loss 0.1060.
  - Deck 12: 7,664 examples, mean advantage -0.1245, final loss 0.1842.
  - Deck 13: 24,774 examples, mean advantage 0.1762, final loss 0.1818.

Conclusion and next step:

- The iteration-10 online PPO update is valid: it used the full iter-10 raw
  self-play window and produced a complete 13-checkpoint specialist family with
  no off-policy leakage.
- Per user direction, stop online RL collection and PPO updates at iteration
  10. Do not queue iteration-11 self-play or PPO unless the plan changes.
- The iteration-10 raw training window can be deleted after confirming the
  report and checkpoints are retained, matching the Phase 5 data policy.
- Finish the evaluation backlog only: iteration 8, iteration 9, and iteration
  10 full-agent-vs-rule evaluations. Keep iteration 5 as the current best
  promotion/package candidate until one of those evaluations beats it.

## 2026-07-07 - Alpha League Final Evaluation Backlog and Dashboard

ERAWAN results:

- Uploaded and inspected:
  - `phase5_alpha_iter0008_specialists_full_vs_rule_30g.json`;
  - `phase5_alpha_iter0008_specialists_full_vs_rule_30g.md`;
  - `slurm-73577-phase5-league-eval.out`;
  - `phase5_alpha_iter0009_specialists_full_vs_rule_30g.json`;
  - `phase5_alpha_iter0009_specialists_full_vs_rule_30g.md`;
  - `slurm-73593-phase5-league-eval.out`;
  - `phase5_alpha_iter0010_specialists_full_vs_rule_30g.json`;
  - `phase5_alpha_iter0010_specialists_full_vs_rule_30g.md`;
  - `slurm-73597-phase5-league-eval.out`.
- Agent: `phase5-full`.
- Generalist prior:
  `models/rl/phase5_generalist_policy_13deck_10k.pt`.
- Gate: 13 x 13 agent-vs-rule league evaluation, 30 games per matchup,
  max 600 selections per game.

Evaluation backlog:

- Iteration 8, ERAWAN job `73577`, specialist checkpoint family
  `models/rl/phase5_league_alpha/iter-0008/specialists`:
  2,699 / 5,070 wins, 0.5323 win rate, 13 draws, 45 timeouts, 0 errors.
  Against the four required sample rule-agent opponents: 671 / 1,560 wins,
  0.4301 win rate. This is down 15 overall wins and 33 required-sample wins
  from iteration 5. Top overall decks were deck 11 Mega Lucario ex at
  299 / 390, deck 12 Mega Abomasnow ex at 260 / 390, deck 2 Crustle at
  251 / 390, and deck 10 Crustle sample at 241 / 390. Deck 1 Alakazam
  Dudunsparce remained weakest at 61 / 390 overall and 2 / 120 against the
  required sample opponents.
- Iteration 9, ERAWAN job `73593`, specialist checkpoint family
  `models/rl/phase5_league_alpha/iter-0009/specialists`:
  2,631 / 5,070 wins, 0.5189 win rate, 23 draws, 57 timeouts, 0 errors.
  Against the four required sample rule-agent opponents: 636 / 1,560 wins,
  0.4077 win rate. This is down 83 overall wins and 68 required-sample wins
  from iteration 5. Deck 11 Mega Lucario ex remained strong at 304 / 390
  overall and 89 / 120 against required sample opponents, but the aggregate
  regressed because several other decks fell, especially deck 1 at 56 / 390
  and deck 3 Dragapult Dusknoir at 134 / 390.
- Iteration 10, ERAWAN job `73597`, specialist checkpoint family
  `models/rl/phase5_league_alpha/iter-0010/specialists`:
  2,646 / 5,070 wins, 0.5219 win rate, 17 draws, 39 timeouts, 0 errors.
  Against the four required sample rule-agent opponents: 690 / 1,560 wins,
  0.4423 win rate. This is down 68 overall wins and 14 required-sample wins
  from iteration 5. Deck 12 Mega Abomasnow ex improved to 278 / 390 overall
  and 75 / 120 against required sample opponents, but deck 1 remained weakest
  at 55 / 390 overall and 1 / 120 against required sample opponents.

Search telemetry:

- Iteration 8: 198,090 searched decisions, 36,940 search-changed decisions,
  0 search errors, 0 candidate errors, 1,825 truncated candidates,
  average search 0.0526 seconds, max 3.2174 seconds.
- Iteration 9: 196,764 searched decisions, 36,888 search-changed decisions,
  0 search errors, 0 candidate errors, 2,055 truncated candidates,
  average search 0.0531 seconds, max 2.5953 seconds.
- Iteration 10: 197,105 searched decisions, 37,599 search-changed decisions,
  0 search errors, 0 candidate errors, 2,159 truncated candidates,
  average search 0.0533 seconds, max 2.3020 seconds.

Dashboard artifacts:

- Added reusable dashboard generator:
  `scripts/analysis/phase5_alpha_eval_dashboard.py`.
- Generated dashboard artifacts from the uploaded eval JSONs:
  - `reports/phase5_alpha_eval_dashboard.html`;
  - `reports/phase5_alpha_eval_dashboard_summary.csv`;
  - `reports/phase5_alpha_eval_dashboard_per_deck.csv`;
  - `reports/phase5_alpha_eval_dashboard_matchups.csv`.
- Dashboard coverage: iterations 0, 2, 3, 4, 5, 6, 7, 8, 9, and 10. Iteration
  1 is omitted because no iteration-1 full-agent-vs-rule eval report is
  available.
- Raw uploaded eval reports were used as inputs but intentionally not copied
  into `reports/` for this commit, to avoid future ERAWAN `git pull` conflicts
  with untracked report files already present on the cluster.

Conclusion:

- No later checkpoint beat iteration 5. Keep
  `models/rl/phase5_league_alpha/iter-0005/specialists` as the current best
  promotion/package candidate at 2,714 / 5,070 wins and 704 / 1,560 wins
  against the four required sample rule-agent opponents.
- The online RL loop is stopped at iteration 10 per user direction. Do not
  queue iteration-11 self-play or PPO unless the plan changes.
- Main persistent weakness remains deck 1 Alakazam Dudunsparce. Deck 11 Mega
  Lucario ex remains the strongest specialist, and deck 12 Mega Abomasnow ex
  is the clearest late-iteration improvement, but neither changes the promotion
  decision.

## 2026-07-07 - Specialized Public-Agent Rule-Opponent Pivot

Implementation update:

- Added a Phase 5 public/specialized Kaggle-agent roster loader using the
  public-20-plus-sample-4 roster. The uploaded roster notebook is metadata-only,
  so the implementation keeps the 20 public agents plus four sample agents as a
  built-in source list and discovers whichever exported `.py` or `.ipynb`
  agents are present locally.
- Added a public-agent adapter path that can load local `submission.py` files,
  local notebooks, and the repo-bundled sample Dragapult adapter. Missing or
  unloadable public agents are reported as unavailable and skipped rather than
  blocking the whole run.
- Added `phase5-public-agent-roster` for availability reports.
- Added `rl-evaluate-phase5-public-agents`, which evaluates the selected Phase
  5 full/search/RL agent against available specialized public/sample rule
  agents using each public agent's own deck and policy.
- Added `public_agent_gate` to the specialized public-agent evaluation JSON and
  Markdown reports. The gate records the minimum aggregate win-rate threshold,
  per-public-opponent aggregates, per-controlled-deck aggregates, worst
  opponent/deck, failing aggregates, and strict failing matchups.
- Added `rl-generate-phase5-public-agent-trajectories`, which records our
  agent's on-policy trajectories against fixed specialized public/sample
  opponents for the existing PPO specialist updater.
- Added SLURM scripts:
  - `scripts/slurm/phase5_public_agent_roster.sbatch`;
  - `scripts/slurm/phase5_public_agent_eval_conda.sbatch`;
  - `scripts/slurm/phase5_public_agent_trajectories.sbatch`.
- Updated the runbook with the new specialized public-agent curriculum:
  discover available agents, evaluate iteration 5 against the 50% public-agent
  gate, generate trajectories against specialized opponents, run PPO with
  `TRAJECTORY_DATASET` pointing at that window, then repeat until the gate
  clears or failing rows indicate a tactical/runtime bug.

Conclusion:

- The final Alpha league results showed that behavior cloning/bootstrap was
  useful but the generic self-play PPO loop did not improve beyond iteration 5.
  Kaggle replay inspection also showed concrete deck behavior failures, notably
  Mega Abomasnow ex missing energy attachments and attacks despite enough
  resources.
- The active training plan is changed: stop generic learned-agent self-play at
  iteration 10 and do not queue iteration 11. Replace the generic rule-opponent
  objective with specialized public/sample rule agents and train/evaluate until
  every available specialized opponent and every controlled deck aggregate is at
  least 50% win rate.
- The existing generic 13 x 13 full-agent-vs-rule reports remain historical
  artifacts. The new specialized public-agent gate is the next training
  decision surface.

Validation:

- The new public-agent roster was parsed locally from the uploaded notebook:
  24 sources total, 20 public agents and four sample agents.
- Focused tests cover roster counts, local Python-agent loading, CLI exposure,
  and the public-agent gate aggregation.

## 2026-07-07 - Iteration-5 Specialized Public-Agent Eval

ERAWAN results:

- Uploaded and inspected:
  - `phase5_public_agent_roster.json`;
  - `phase5_public_agent_status_iter0005.json`;
  - `phase5_public_agent_eval_iter0005_30g.json`;
  - `phase5_public_agent_eval_iter0005_30g.md`;
  - `slurm-73639-phase5-public-roster.out`;
  - `slurm-73657-phase5-public-eval.out`.
- Roster job: `73639`.
- Eval job: `73657`.
- Agent: `phase5-full`.
- Specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Available specialized opponents during eval: one built-in sample opponent,
  `Official sample Dragapult ex`.
- Unavailable roster entries during eval: 23 / 24, because no exported public
  agent files were present under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agents`.

Specialized-opponent evaluation:

- Overall vs sample Dragapult: 50 / 390 wins, 0.1282 win rate, 0 draws,
  0 timeouts, 0 errors.
- Public-agent gate: failed.
- Strict per-matchup gate: failed.
- Worst controlled decks:
  - Deck 3 Dragapult Dusknoir: 0 / 30 wins.
  - Deck 5 Dragapult Dudunsparce: 0 / 30 wins.
  - Deck 8 Dragapult Blaziken: 1 / 30 wins.
- Best controlled decks:
  - Deck 11 Mega Lucario ex: 10 / 30 wins, 0.3333.
  - Deck 12 Mega Abomasnow ex: 7 / 30 wins, 0.2333.
  - Deck 1 Alakazam Dudunsparce: 6 / 30 wins, 0.2000.

Search telemetry:

- Searched decisions: 11,880.
- Search-changed decisions: 2,009, 0.1691 change rate.
- Search errors: 0. Candidate errors: 0.
- Candidate probes: 42,187.
- Truncated candidates: 62.
- Average search seconds: 0.0468. Max search seconds: 2.3205.

Artifact note:

- The standalone roster report from job `73639` recorded 0 available agents,
  while the eval status from job `73657` recorded the built-in sample Dragapult
  adapter as available. Treat the eval status as the canonical availability
  artifact for this result. Re-run the roster job after the next code sync if
  availability counts need to be audited independently.

Conclusion and next step:

- The current best generic-league checkpoint is not remotely sufficient against
  the specialized sample Dragapult rule agent. This validates the pivot away
  from generic full-agent-vs-generic-rule self-play.
- Before spending compute on a broad curriculum, export more public/sample
  agents into the public-agent root when available. However, even one available
  specialized opponent is enough to start a targeted training window because
  the current win rate is only 12.8%.
- Next ERAWAN action: generate `phase5-rl` trajectories from iteration 5
  specialists against the available specialized public-agent roster, then run a
  PPO specialist update with `TRAJECTORY_DATASET` pointing at that public-agent
  trajectory file. Write the targeted checkpoint family under
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists` instead of
  overwriting the historical generic Alpha league
  `models/rl/phase5_league_alpha/iter-0006/specialists` artifacts. Evaluate the
  candidate against the same specialized public-agent gate before considering
  any promotion.

## 2026-07-07 - Public-Agent Curriculum Iteration-6 Trajectory Window

ERAWAN result:

- Uploaded and inspected:
  - `phase5_public_agent_rule_train_iter0006_report.json`;
  - `slurm-73671-phase5-public-traj.out`.
- ERAWAN job: `73671`.
- Collector: `AGENT=phase5-rl`.
- Source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Available specialized opponents: one built-in sample opponent,
  `sample_dragapult`.
- Raw trajectory output:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agent_rule_train/iter-0006_public_agent_trajectories.jsonl`.

Trajectory generation:

- Games requested/started: 130 / 130, with 10 games for each of the 13
  controlled decks against sample Dragapult.
- Trajectory rows: 6,729.
- Wins/losses/draws: 10 / 120 / 0.
- Timeouts: 0. Errors: 0.
- Per-deck wins:
  - Deck 10 Crustle sample: 3 / 10.
  - Decks 2 and 8: 2 / 10 each.
  - Decks 1, 11, and 13: 1 / 10 each.
  - Decks 3, 4, 5, 6, 7, 9, and 12: 0 / 10.

Conclusion and next step:

- The targeted public-agent trajectory window is valid and has no simulation
  errors. It is intentionally small and very loss-heavy, so treat it as the
  first targeted PPO smoke/update window rather than as sufficient curriculum
  scale.
- Next ERAWAN action: run the PPO specialist update from this JSONL into the
  separate public-agent curriculum checkpoint family
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists`, then
  evaluate that candidate against the same specialized public-agent gate.

## 2026-07-07 - Public-Agent Curriculum Iteration-6 PPO Update

ERAWAN result:

- Uploaded and inspected:
  - `iter-0006_ppo_specialists_report (1).json`;
  - `slurm-73706-phase5-alpha-ppo-specialists.out`.
- ERAWAN job: `73706`.
- Source specialist checkpoint family:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- PPO trajectory dataset:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agent_rule_train/iter-0006_public_agent_trajectories.jsonl`.
- Output specialist checkpoint family:
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists`.

PPO update:

- Deck checkpoints written: 13 / 13.
- Total PPO examples: 6,729.
- Skipped no-target rows: 0.
- Skipped off-policy rows: 0.
- `require_on_policy`: true for every deck update.
- Per-deck examples:
  - Deck 1: 641 examples, mean advantage -0.3051, final loss -0.5609.
  - Deck 2: 492 examples, mean advantage -0.5558, final loss 0.3318.
  - Deck 3: 413 examples, mean advantage -0.5276, final loss 0.1093.
  - Deck 4: 531 examples, mean advantage -0.9504, final loss 0.0809.
  - Deck 5: 535 examples, mean advantage -0.8975, final loss 0.0053.
  - Deck 6: 531 examples, mean advantage -0.9442, final loss -0.0267.
  - Deck 7: 601 examples, mean advantage -0.8195, final loss 0.1310.
  - Deck 8: 614 examples, mean advantage -0.8020, final loss 0.6057.
  - Deck 9: 524 examples, mean advantage -1.3415, final loss 0.1321.
  - Deck 10: 546 examples, mean advantage -0.4633, final loss 0.3194.
  - Deck 11: 421 examples, mean advantage -1.0960, final loss 0.5580.
  - Deck 12: 246 examples, mean advantage -1.3495, final loss 0.4991.
  - Deck 13: 634 examples, mean advantage -0.6787, final loss 0.0457.

Conclusion and next step:

- The first targeted public-agent PPO update is valid and did not overwrite the
  historical generic Alpha league iteration-6 checkpoint family.
- The update was based on a small, loss-heavy sample-Dragapult window. Treat
  the next evaluation as a signal check for whether this targeted PPO direction
  improves the specialized-opponent gate, not as a final curriculum result.
- Next ERAWAN action: evaluate
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists` with the
  same specialized public-agent gate against the available public-agent roster.

## 2026-07-08 - Public-Agent Curriculum Iteration-6 Eval

ERAWAN result:

- Uploaded and inspected:
  - `phase5_public_agent_eval_iter0006_30g.json`;
  - `phase5_public_agent_eval_iter0006_30g.md`;
  - `phase5_public_agent_status_iter0006.json`;
  - `slurm-73711-phase5-public-eval.out`;
  - `slurm-73711-phase5-public-eval.err`.
- ERAWAN job: `73711`.
- Agent: `phase5-full`.
- Specialist checkpoint family:
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists`.
- Available specialized opponents: one built-in sample opponent,
  `Official sample Dragapult ex`. The other 23 roster entries were unavailable
  because no exported public-agent files were present under
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agents`.

Evaluation:

- Overall vs sample Dragapult: 38 / 390 wins, 0.0974 win rate, 0 draws,
  0 timeouts, 0 errors.
- Public-agent gate: failed.
- Strict per-matchup gate: failed.
- Delta from iteration-5 public-agent baseline: -12 wins, from 50 / 390
  (0.1282) to 38 / 390 (0.0974).
- Per-deck win deltas vs iteration 5:
  - Deck 1: 6 -> 2, -4.
  - Deck 2: 4 -> 4, +0.
  - Deck 3: 0 -> 1, +1.
  - Deck 4: 4 -> 4, +0.
  - Deck 5: 0 -> 2, +2.
  - Deck 6: 3 -> 0, -3.
  - Deck 7: 4 -> 1, -3.
  - Deck 8: 1 -> 2, +1.
  - Deck 9: 4 -> 2, -2.
  - Deck 10: 3 -> 4, +1.
  - Deck 11: 10 -> 7, -3.
  - Deck 12: 7 -> 8, +1.
  - Deck 13: 4 -> 1, -3.

Search telemetry:

- Searched decisions: 12,197.
- Search-changed decisions: 2,016, 0.1653 change rate.
- Search errors: 0. Candidate errors: 0.
- Candidate probes: 43,421.
- Truncated candidates: 136.
- Average search seconds: 0.0410. Max search seconds: 1.2752.

Conclusion and next step:

- The first targeted public-agent PPO update made performance worse against
  sample Dragapult. Do not continue by simply repeating the same loss-heavy
  on-policy PPO loop.
- The current signal suggests the training pipeline needs a change before more
  public-agent curriculum compute: for example, collect/weight successful
  trajectories, add denser tactical rewards, collect opponent demonstrations
  where applicable, or run decision-level diagnostics to identify missed
  setup/attack/energy-attachment actions.
- Keep `models/rl/phase5_league_alpha/iter-0005/specialists` as the current
  best packaged checkpoint family. Treat
  `models/rl/phase5_public_agent_curriculum/iter-0006/specialists` as a failed
  targeted PPO experiment unless later analysis identifies a narrow deck-specific
  use.

## 2026-07-08 - Public-Agent Micro-Experiment Filters

Implementation:

- Confirmed the current `phase5-rl` collector uses stochastic neural-policy
  sampling with a temperature and records `policy_on_policy=true` frames; it is
  not epsilon-greedy random legal-action exploration.
- Added optional public-agent key filtering to public-agent roster discovery,
  public-agent evaluation, and public-agent trajectory generation via
  `--public-agent-key`.
- Exposed `PUBLIC_AGENT_KEYS` in the public-agent roster/eval/trajectory SLURM
  wrappers.
- Exposed `DECK_INDICES` in the public-agent eval and trajectory SLURM wrappers,
  matching the existing CLI and trainer deck filter behavior.
- Added parser coverage for the new targeted public-agent/deck filter path.

Experiment decision:

- Before another broad public-agent curriculum pass, run a one-deck,
  one-opponent signal check: deck 12 Mega Abomasnow ex vs built-in
  `sample_dragapult`, starting from
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Collect 100 on-policy `phase5-rl` games, update only deck 12 into
  `models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100/specialists`,
  then evaluate only deck 12 vs `sample_dragapult`.
- Promote the idea only if the post-update eval beats the single-matchup
  baseline by a clear margin with zero errors/timeouts; otherwise, change the
  training signal before spending more compute.

## 2026-07-08 - Deck 12 vs Sample Dragapult Micro Experiment Result

Uploaded and inspected:

- `phase5_public_agent_deck12_dragapult_baseline_30g.json`;
  `phase5_public_agent_deck12_dragapult_baseline_30g.md`.
- `phase5_public_agent_deck12_dragapult_100_trajectories_report.json`;
  `slurm-73732-phase5-public-traj.out`;
  `slurm-73732-phase5-public-traj.err`.
- `deck12_vs_sample_dragapult_100_ppo_report.json`;
  `deck-12_ppo_report.json`;
  `slurm-73733-phase5-alpha-ppo-specialists.out`.
- `phase5_public_agent_deck12_dragapult_100_update_30g.json`;
  `phase5_public_agent_deck12_dragapult_100_update_30g.md`;
  `slurm-73734-phase5-public-eval.out`;
  `slurm-73734-phase5-public-eval.err`.

Baseline single-matchup eval:

- Checkpoint family:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Deck/opponent: deck 12 Mega Abomasnow ex vs built-in `sample_dragapult`.
- Result: 6 / 30 wins, 24 losses, 0 draws, 0 timeouts, 0 errors,
  0.2000 win rate.
- Search telemetry: 445 searched decisions, 81 changed decisions,
  0 search errors, 0 candidate errors, average search 0.0082s, max 0.0690s.

Trajectory collection:

- ERAWAN job: `73732`.
- Agent: `phase5-rl`.
- Filters: `PUBLIC_AGENT_KEYS=sample_dragapult`, `DECK_INDICES=12`.
- Source specialist root:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Raw output:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agent_rule_train/deck12_vs_sample_dragapult_100.jsonl`.
- Result: 100 / 100 games started, 2,527 trajectory rows, 8 wins,
  92 losses, 0 draws, 0 timeouts, 0 errors.
- The SLURM stderr contained only the known PyTorch nested-tensor prototype
  warning.

PPO update:

- ERAWAN job: `73733`.
- Iteration label: `12`; selected deck indices: `[12]`.
- Source checkpoint:
  `models/rl/phase5_league_alpha/iter-0005/specialists/deck-12.pt`.
- Output checkpoint:
  `models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100/specialists/deck-12.pt`.
- Training consumed 2,527 / 2,527 on-policy examples, skipped 0 no-target and
  0 off-policy rows.
- Mean advantage: -1.1553; final loss: 0.0137.

Post-update single-matchup eval:

- ERAWAN job: `73734`.
- Checkpoint:
  `models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100/specialists/deck-12.pt`.
- Result: 8 / 30 wins, 22 losses, 0 draws, 0 timeouts, 0 errors,
  0.2667 win rate.
- Delta from baseline: +2 wins over 30 games, +0.0667 win rate.
- Search telemetry: 543 searched decisions, 110 changed decisions,
  0 search errors, 0 candidate errors, average search 0.0094s, max 0.0666s.

Conclusion and next step:

- The targeted command path is valid: public-agent key filtering, deck filtering,
  one-deck trajectory collection, one-deck PPO update, and one-deck eval all
  worked without runtime errors.
- The learning result is not strong enough to promote or scale. The post-update
  score improved from 6 / 30 to 8 / 30, but that is a small noisy change and
  remains far below the 50% public-agent gate.
- The training data was heavily loss-dominated: only 8 wins in 100 collection
  games and mean PPO advantage -1.1553. This reinforces the earlier conclusion
  that repeating sparse terminal-reward PPO on mostly losing trajectories is not
  the right next broad pipeline.
- Do not continue by scaling this exact 100-game PPO setup to more decks.
  Next useful work should change the training signal: add denser tactical
  rewards/diagnostics for required actions such as energy attachment and
  attacking, collect or weight successful trajectories, or add a supervised
  rule-specialist target before further online PPO.

## 2026-07-08 - Public-Agent Tactical Reward Shaping

Implementation:

- Added opt-in public-agent trajectory reward shaping. The default remains the
  old behavior: `--outcome-reward-scale 1.0` and
  `--tactical-reward-mode none`.
- Added `PublicAgentTacticalRewardConfig` with a `basic` mode that:
  - rewards selected attack actions;
  - rewards selected energy-attach actions;
  - penalizes ending while attack/attach actions are available;
  - penalizes attacking while an attach action is still available.
- Public-agent trajectory rows now record:
  - original terminal/outcome reward;
  - outcome reward scale;
  - scaled outcome reward;
  - tactical step reward;
  - booleans for attack/attach availability, selected attack/attach, selected
    end, empty selection, missed attack, and missed attach.
- Public-agent trajectory reports now include `reward_shaping` and
  `tactical_reward_summary`.
- Exposed the shaping controls through
  `rl-generate-phase5-public-agent-trajectories` and
  `scripts/slurm/phase5_public_agent_trajectories.sbatch`:
  - `OUTCOME_REWARD_SCALE`;
  - `TACTICAL_REWARD_MODE`;
  - `TACTICAL_ATTACK_BONUS`;
  - `TACTICAL_ATTACH_BONUS`;
  - `TACTICAL_MISSED_ATTACK_PENALTY`;
  - `TACTICAL_MISSED_ATTACH_PENALTY`.
- Added unit coverage for public-agent parser flags and basic tactical reward
  math.

Next ERAWAN diagnostic:

- Rerun the same deck-12 vs built-in `sample_dragapult` micro experiment with
  shaped trajectories:
  - `OUTCOME_REWARD_SCALE=0.25`;
  - `TACTICAL_REWARD_MODE=basic`;
  - attack bonus `0.10`;
  - attach bonus `0.06`;
  - missed attack penalty `-0.10`;
  - missed attach penalty `-0.06`.
- Write shaped raw data to
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agent_rule_train/deck12_vs_sample_dragapult_100_tactical.jsonl`,
  update only deck 12 into
  `models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical/specialists`,
  then evaluate only deck 12 vs `sample_dragapult`.
- Interpret the shaped run using both the eval delta and
  `tactical_reward_summary`; if missed attack/attach counts are low, this is not
  the right failure-mode diagnostic.

## 2026-07-08 - Deck 12 Tactical Trajectory Result

Uploaded and inspected:

- `deck12_vs_sample_dragapult_100_tactical_trajectories_report.json`.
- `slurm-73752-phase5-public-traj.out`.

ERAWAN trajectory job:

- Job: `73752`.
- Agent: `phase5-rl`.
- Source specialist root:
  `models/rl/phase5_league_alpha/iter-0005/specialists`.
- Filters: `PUBLIC_AGENT_KEYS=sample_dragapult`, `DECK_INDICES=12`.
- Reward shaping:
  - `OUTCOME_REWARD_SCALE=0.25`;
  - `TACTICAL_REWARD_MODE=basic`;
  - attack bonus `0.10`;
  - attach bonus `0.06`;
  - missed attack penalty `-0.10`;
  - missed attach penalty `-0.06`.
- Raw output:
  `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_public_agent_rule_train/deck12_vs_sample_dragapult_100_tactical.jsonl`.

Collection result:

- 100 / 100 games started.
- 2,626 trajectory rows.
- 8 wins, 92 losses, 0 draws, 0 timeouts, 0 errors.
- Same game-level win count as the unshaped 100-game deck-12 window, but with
  per-step tactical rewards recorded for PPO.

Tactical reward summary:

- Steps: 2,626.
- Tactical reward sum: 50.3000; average tactical reward per step: 0.0192.
- Attack opportunities: 868.
- Attacks taken: 353, 0.4067 attack-taken rate.
- Conservatively missed attacks: 39, 0.0449 missed-attack rate.
- Attachment opportunities: 976.
- Attachments taken: 449, 0.4600 attach-taken rate.
- Conservatively missed attachments: 134, 0.1373 missed-attach rate.
- End selected: 240, 0.0914 of steps.

Conclusion and next step:

- This is a useful shaped PPO input. Missed attack/attach counts are not zero,
  and missed attachment is the clearer tactical signal.
- Proceed with a deck-12-only PPO update from
  `deck12_vs_sample_dragapult_100_tactical.jsonl`, then evaluate only deck 12
  against `sample_dragapult`.
- This still should not be scaled to all decks unless the shaped update improves
  the held-out 30-game eval by more than the previous noisy +2 / 30 result.

## 2026-07-08 - Deck 12 Tactical PPO Eval Result

Uploaded and inspected:

- `phase5_public_agent_deck12_dragapult_100_tactical_update_30g.json`.
- `phase5_public_agent_deck12_dragapult_100_tactical_update_30g.md`.
- `slurm-73757-phase5-public-eval.out`.
- `deck12_vs_sample_dragapult_100_tactical_ppo_report.json`.

PPO update:

- Iteration label: `13`.
- Selected deck indices: `[12]`.
- Source checkpoint:
  `models/rl/phase5_league_alpha/iter-0005/specialists/deck-12.pt`.
- Output checkpoint:
  `models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical/specialists/deck-12.pt`.
- Training consumed 2,626 / 2,626 on-policy examples, skipped 0 no-target rows
  and 0 off-policy rows.
- Mean advantage: -0.5657, compared with -1.1553 for the unshaped 100-game PPO
  micro update.
- Final loss: -0.0885.

Evaluation:

- ERAWAN eval job: `73757`.
- Agent: `phase5-full`.
- Checkpoint family:
  `models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical/specialists`.
- Filters: `PUBLIC_AGENT_KEYS=sample_dragapult`, `DECK_INDICES=12`.
- Result: 8 / 30 wins, 22 losses, 0 draws, 0 timeouts, 0 errors,
  0.2667 win rate.
- Public-agent gate: failed; strict per-matchup diagnostic failed.
- Search telemetry: 518 searched decisions, 98 search-changed decisions,
  0 search errors, 0 candidate errors, 1,826 candidate probes, 0 truncated
  candidates, average search 0.0094s, max search 0.0885s.

Comparison:

- Iteration-5 baseline deck-12 eval vs `sample_dragapult`: 6 / 30 wins.
- Unshaped 100-game PPO update eval: 8 / 30 wins.
- Tactical-shaped 100-game PPO update eval: 8 / 30 wins.
- Tactical shaping matched the previous noisy +2 / 30 result but did not add
  measurable held-out improvement.

Conclusion and next step:

- Do not scale this shaped PPO setup to more decks. The shaped data produced a
  real diagnostic signal for missed attachment/attack opportunities, but the
  PPO update did not turn that signal into better held-out play.
- The shaped reward improved the training signal, moving mean advantage from
  -1.1553 to -0.5657, but did not improve held-out evaluation beyond the
  previous 8 / 30 result. This strengthens the conclusion that PPO-only updates
  on small, mostly losing windows are not enough for this failure mode.
- Next implementation should pivot to stronger direct supervision or diagnostics:
  train a deck/opponent specialist on rule-selected or search-selected tactical
  targets for the same situations, add richer action-level reports for the
  missed attach/attack rows, or change the agent/action selection logic if the
  eval traces show the policy is still vetoed by search/heuristics.

## 2026-07-08 - Deck 12 Pure-Neural Ablation Eval

Uploaded and inspected:

- `phase5_public_agent_deck12_dragapult_100_tactical_pure_rl_100g.json`.
- `phase5_public_agent_deck12_dragapult_100_tactical_pure_rl_100g.md`.
- `phase5_public_agent_deck12_dragapult_100_tactical_pure_rl_status.json`.

Evaluation:

- Agent: `phase5-rl`, pure stochastic neural policy without root search.
- Checkpoint family:
  `models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical/specialists`.
- Filters: `PUBLIC_AGENT_KEYS=sample_dragapult`, `DECK_INDICES=12`.
- Result: 7 / 100 wins, 93 losses, 0 draws, 0 timeouts, 0 errors,
  0.0700 win rate.
- Public-agent gate: failed; strict per-matchup diagnostic failed.
- Available opponent: built-in `sample_dragapult`.

Comparison:

- Iteration-5 baseline `phase5-full`: 6 / 30 wins.
- Unshaped 100-game PPO update with `phase5-full`: 8 / 30 wins.
- Tactical-shaped 100-game PPO update with `phase5-full`: 8 / 30 wins.
- Tactical-shaped 100-game PPO update with pure `phase5-rl`: 7 / 100 wins.
- The pure-neural eval is close to the shaped training collection result
  (8 / 100 wins), and much worse than the `phase5-full` search/heuristic stack.

Conclusion and next step:

- Do not remove search/heuristics from the final or package agent. This ablation
  shows they are helping, not hiding a better neural policy.
- The current neural policy is still too weak on deck 12 vs sample Dragapult.
  The next implementation should not be another PPO-only loop; use direct
  supervision or diagnostics:
  - generate action-level missed attack/attachment reports from the shaped
    trajectory file;
  - train a deck/opponent specialist from rule-selected or search-selected
    tactical targets in those states;
  - then evaluate both pure neural and `phase5-full` again to check whether the
    learned policy actually moved.

## 2026-07-08 - Search Score-Component Trace Diagnostics

Implementation:

- Added opt-in public-agent eval search trace export for `phase5-full` /
  `Phase5SearchPolicyAgent` runs:
  - CLI args on `rl-evaluate-phase5-public-agents`:
    `--search-trace-output` and `--search-trace-games`;
  - SLURM env vars in `scripts/slurm/phase5_public_agent_eval_conda.sbatch`:
    `SEARCH_TRACE_OUTPUT` and `SEARCH_TRACE_GAMES`;
  - trace rows include `game_outcome` (`win`, `loss`, `draw`, `timeout`,
    `error`), deck/opponent metadata, and all root-search candidate component
    fields.
- Added `rl-diagnose-search-score-components` and
  `scripts/slurm/phase5_search_score_components_conda.sbatch`.
- The diagnostic reads a root-search trace JSONL and summarizes, overall and by
  `game_outcome`:
  - selected, baseline, and all-candidate ranges/means;
  - selected-minus-baseline margins;
  - `tactical_score`, raw neural outputs, normalized priors, `prior_score`,
    and `combined_score`;
  - config signatures for the active search weights.

Purpose:

- Existing public-agent eval JSON/markdown reports do not contain discarded
  candidate score components, so they cannot answer whether tactical, neural,
  or rule-prior terms are mis-scaled in winning vs losing sequences.
- The new trace-and-diagnose path is the next diagnostic before changing
  `RootSearchConfig` weights. It should show whether raw `tactical_score`
  dominates the normalized prior terms, or whether losing sequences have high
  selected-over-baseline `combined_score` margins despite poor outcomes.

Next ERAWAN steps:

1. Rerun a targeted deck-12 vs built-in `sample_dragapult` `phase5-full` eval
   with `SEARCH_TRACE_OUTPUT` enabled.
2. Run `phase5_search_score_components_conda.sbatch` on that trace.
3. Inspect win/loss component ranges and selected-minus-baseline margins before
   changing heuristic/neural weights.

## 2026-07-08 - Deck 12 Search Score-Component Diagnostic Result

Uploaded and inspected:

- `phase5_public_agent_deck12_dragapult_tactical_full_trace_100g.json`.
- `phase5_public_agent_deck12_dragapult_tactical_full_trace_100g.md`.
- `phase5_public_agent_deck12_dragapult_tactical_full_trace_status.json`.
- `phase5_public_agent_deck12_dragapult_tactical_score_components.json`.
- `phase5_public_agent_deck12_dragapult_tactical_score_components.md`.

Targeted traced evaluation:

- Agent/checkpoint: `phase5-full` using
  `models/rl/phase5_public_agent_micro/deck12_vs_sample_dragapult_100_tactical/specialists`.
- Matchup: deck 12 Mega Abomasnow ex vs built-in `sample_dragapult`.
- Result: 23 / 100 wins, 77 losses, 0 draws, 0 timeouts, 0 errors,
  0.2300 win rate.
- Search telemetry: 1,599 searched decisions, 313 search-changed decisions,
  0 search errors, 0 candidate errors, 5,577 candidate probes, 0 truncated
  candidates, average search 0.0085s, max search 0.0804s.

Score-component diagnostic:

- Trace records: 1,599; candidate probes: 5,577; comparable records: 1,599.
- Outcome split: 393 decision traces from winning games, 1,206 from losing
  games.
- Active config signature:
  `rule_prior_weight=0.08`, `policy_prior_weight=0.0`,
  `neural_action_value_weight=0.0`, `neural_tactical_weight=0.0`.
- Selected tactical-score range was 101.5000 versus prior-score range 0.0800,
  or 1,268.75x wider.
- Mean selected-minus-baseline margins:
  - overall combined/tactical/prior: 0.8505 / 0.8592 / -0.0087;
  - wins combined/tactical/prior: 3.0494 / 3.0589 / -0.0095;
  - losses combined/tactical/prior: 0.1340 / 0.1424 / -0.0084.
- Selected candidate means:
  - wins: tactical 14.9622, prior 0.0705, combined 15.0326;
  - losses: tactical 1.3423, prior 0.0716, combined 1.4139.
- Neural priors were not part of the combined score in this run. The recorded
  action-Q and neural-tactical normalized priors also had negative
  selected-minus-baseline margins, especially in winning traces, so simply
  turning those weights on with positive coefficients is not yet justified.
- Damage delta was higher in losing traces than winning traces:
  selected damage-delta mean 21.2852 in losses versus 3.5369 in wins, while
  winning traces had much larger tactical/combined margins. This suggests many
  losing lines chase short-horizon damage without converting to prize/terminal
  progress.

Conclusion and next step:

- `0.08` is effectively only a tie-breaker against raw `tactical_score`. The
  current scale does not make tactical and prior functions comparable.
- However, the diagnostic does not show that search is confidently choosing bad
  moves in losing games: winning traces have much stronger positive
  selected-over-baseline tactical/combined margins than losing traces.
- Do not blindly raise neural action-Q or neural tactical weights. First
  implement an ablation that normalizes the rollout tactical score across root
  candidates before combining, then evaluate a small grid such as normalized
  tactical weight 1.0 with policy-prior weights 0.1, 0.25, and 0.5 while keeping
  action-Q/neural-tactical weights at 0.0.
- If normalized-tactical policy-prior ablations still fail below the 50% gate,
  the next training change should target longer-horizon setup/value rather than
  more sparse PPO on this matchup.

## 2026-07-08 - Leaf State-Value Search Ablation Implementation

Implementation:

- Added opt-in leaf state-value scoring to Phase 5 root search.
- New `RootSearchConfig` fields:
  - `tactical_score_weight` (default `1.0`);
  - `normalize_tactical_score` (default `False`);
  - `leaf_state_value_weight` (default `0.0`).
- Defaults preserve the old search equation:
  `combined_score = tactical_score + prior_score`.
- With `normalize_tactical_score=True`, rollout tactical score is normalized
  across root candidates before weighting.
- With `leaf_state_value_weight > 0`, `Phase5SearchPolicyAgent` evaluates each
  rolled leaf state using the existing Phase 5 `state_value` head from the root
  player's perspective and adds the normalized leaf value to candidate scoring.
- Candidate traces now include:
  - `tactical_score_prior`;
  - `tactical_score_component`;
  - `leaf_state_value`;
  - `leaf_state_value_prior`;
  - `leaf_state_value_score`.
- `rl-generate-phase5-public-agent-trajectories` now accepts `--agent rule` so
  one-deck rule-vs-rule bootstrap trajectories can be generated against a
  specialized public/sample opponent.
- `scripts/slurm/phase5_deck_specialists_train.sbatch` now supports
  `NO_DECISION_DATASET`, `SELFPLAY_WEIGHT`, `VALUE_LOSS_WEIGHT`,
  `ACTION_VALUE_LOSS_WEIGHT`, and `TACTICAL_LOSS_WEIGHT` so a start network can
  be trained from only the narrow rule-vs-rule trajectory window.
- `scripts/slurm/phase5_public_agent_eval_conda.sbatch` now accepts
  `NORMALIZE_TACTICAL_SCORE`, `TACTICAL_SCORE_WEIGHT`, `POLICY_PRIOR_WEIGHT`,
  `NEURAL_ACTION_VALUE_WEIGHT`, `NEURAL_TACTICAL_WEIGHT`, and
  `LEAF_STATE_VALUE_WEIGHT`.

Experiment plan:

- Use deck 12 Mega Abomasnow ex vs built-in `sample_dragapult`.
- Generate rule-agent trajectories for our deck against the specialized
  sample Dragapult rule opponent.
- Train an isolated deck-12 specialist from that rule-vs-rule trajectory file
  with value loss enabled.
- Evaluate two one-deck variants:
  - normalized tactical + policy prior, no leaf value;
  - normalized tactical + policy prior + leaf state value.
- Promote the idea only if the leaf-value variant improves the same-matchup
  win rate without introducing search/candidate errors or obvious END/attack
  regressions.

## 2026-07-08 - Deck 12 Leaf State-Value Search Eval Result

Uploaded and inspected:

- `phase5_public_agent_deck12_rule_bootstrap_norm_tactical_100g.json`.
- `phase5_public_agent_deck12_rule_bootstrap_norm_tactical_100g.md`.
- `phase5_public_agent_deck12_rule_bootstrap_norm_tactical_status.json`.
- `slurm-73774-phase5-public-eval.out`.
- `phase5_public_agent_deck12_rule_bootstrap_leaf_value_100g.json`.
- `phase5_public_agent_deck12_rule_bootstrap_leaf_value_100g.md`.
- `phase5_public_agent_deck12_rule_bootstrap_leaf_value_status.json`.
- `slurm-73775-phase5-public-eval.out`.

Shared setup:

- Agent/checkpoint: `phase5-full` using
  `models/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value/specialists`.
- Matchup: deck 12 Mega Abomasnow ex vs built-in `sample_dragapult`.
- Games per matchup: 100; max steps: 600.
- Opponent availability: built-in `sample_dragapult` available; no missing
  opponents.

Normalized tactical baseline:

- ERAWAN job: `73774`.
- Search config:
  - `NORMALIZE_TACTICAL_SCORE=1`;
  - `TACTICAL_SCORE_WEIGHT=1.0`;
  - `POLICY_PRIOR_WEIGHT=0.25`;
  - `LEAF_STATE_VALUE_WEIGHT=0.0`.
- Result: 20 / 100 wins, 80 losses, 0 draws, 0 timeouts, 0 errors,
  0.2000 win rate.
- Public-agent gate: failed.
- Search telemetry: 1,619 searched decisions, 344 search-changed decisions,
  0 search errors, 0 candidate errors, 5,627 candidate probes, 0 truncated
  candidates, average search 0.0082s, max search 0.0745s.

Leaf state-value variant:

- ERAWAN job: `73775`.
- Search config:
  - `NORMALIZE_TACTICAL_SCORE=1`;
  - `TACTICAL_SCORE_WEIGHT=0.5`;
  - `POLICY_PRIOR_WEIGHT=0.25`;
  - `LEAF_STATE_VALUE_WEIGHT=0.5`.
- Result: 29 / 100 wins, 71 losses, 0 draws, 0 timeouts, 0 errors,
  0.2900 win rate.
- Public-agent gate: failed.
- Search telemetry: 1,654 searched decisions, 290 search-changed decisions,
  0 search errors, 0 candidate errors, 5,702 candidate probes, 0 truncated
  candidates, average search 0.0228s, max search 0.0871s.

Comparison and conclusion:

- Leaf state-value search improved the same-checkpoint normalized-tactical
  baseline by +9 wins over 100 games, with no errors/timeouts.
- The value variant also improved over the earlier tactical-shaped checkpoint
  traced eval at 23 / 100, though that comparison changes both checkpoint and
  scoring mix and should be treated as weaker evidence than the +9 same-run
  ablation.
- The leaf value path costs more search time, about 0.0228s average versus
  0.0082s, but remains operationally acceptable for this narrow eval.
- This is the first positive evidence that learned state value can add useful
  signal beyond handcrafted end-of-turn tactical scoring. It is not yet enough:
  29 / 100 is still far below the 50% specialized-opponent gate.

Next steps:

- Run score-component diagnostics on both trace files to verify that
  `leaf_state_value_score` separates winning and losing sequences:
  - `experiments/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value_norm_tactical_traces.jsonl`;
  - `experiments/rl/phase5_public_agent_micro/deck12_rule_bootstrap_leaf_value_traces.jsonl`.
- Inspect the rule-vs-rule trajectory report and deck-12 training report before
  scaling; they were not included with this upload.
- If diagnostics confirm useful value calibration, run a small leaf-value grid
  on the same checkpoint, for example:
  - tactical/value `0.75/0.25`;
  - tactical/value `0.5/0.5` (current best);
  - tactical/value `0.25/0.75`;
  keeping policy prior at `0.25`.
- Do not scale beyond deck 12 until the single-matchup result approaches the
  50% gate or the trace diagnostics identify a clear next value-training fix.

## 2026-07-09 - Deck 12 Policy/Leaf vs Policy/Tactical 50:50 Result

Uploaded and inspected:

- `phase5_public_agent_deck12_policy50_tactical50_100g.json`.
- `phase5_public_agent_deck12_policy50_tactical50_100g.md`.
- `phase5_public_agent_deck12_policy50_tactical50_status.json`.
- `slurm-73781-phase5-public-eval.out`.
- `phase5_public_agent_deck12_policy50_leaf50_100g.json`.
- `phase5_public_agent_deck12_policy50_leaf50_100g.md`.
- `phase5_public_agent_deck12_policy50_leaf50_status.json`.
- `slurm-73782-phase5-public-eval.out`.

Shared setup:

- Agent/checkpoint: `phase5-full` using
  `models/rl/phase5_public_agent_micro/deck12_rule_bootstrap_value/specialists`.
- Matchup: deck 12 Mega Abomasnow ex vs built-in `sample_dragapult`.
- Games per matchup: 100; max steps: 600.
- Search traces:
  - tactical run:
    `experiments/rl/phase5_public_agent_micro/deck12_policy50_tactical50_traces.jsonl`;
  - leaf run:
    `experiments/rl/phase5_public_agent_micro/deck12_policy50_leaf50_traces.jsonl`.

Policy-prior plus tactical-score 50:50:

- ERAWAN job: `73781`.
- Search config:
  - `NORMALIZE_TACTICAL_SCORE=1`;
  - `TACTICAL_SCORE_WEIGHT=0.5`;
  - `POLICY_PRIOR_WEIGHT=0.5`;
  - `LEAF_STATE_VALUE_WEIGHT=0.0`.
- Result: 23 / 100 wins, 77 losses, 0 draws, 0 timeouts, 0 errors,
  0.2300 win rate.
- Public-agent gate: failed.
- Search telemetry: 1,795 searched decisions, 117 search-changed decisions,
  0 search errors, 0 candidate errors, 6,140 candidate probes, 0 truncated
  candidates, average search 0.0079s, max search 0.0875s.

Policy-prior plus leaf state-value 50:50:

- ERAWAN job: `73782`.
- Search config:
  - `NORMALIZE_TACTICAL_SCORE=1`;
  - `TACTICAL_SCORE_WEIGHT=0.0`;
  - `POLICY_PRIOR_WEIGHT=0.5`;
  - `LEAF_STATE_VALUE_WEIGHT=0.5`.
- Result: 28 / 100 wins, 72 losses, 0 draws, 0 timeouts, 0 errors,
  0.2800 win rate.
- Public-agent gate: failed.
- Search telemetry: 1,747 searched decisions, 144 search-changed decisions,
  0 search errors, 0 candidate errors, 6,019 candidate probes, 0 truncated
  candidates, average search 0.0225s, max search 0.1013s.

Comparison and conclusion:

- The clean 50:50 ablation confirms the earlier pattern: learned leaf state
  value outperformed handcrafted end-of-turn tactical score by +5 wins over
  100 games when both were paired equally with the policy prior.
- The policy-plus-leaf result, 28 / 100, is close to the earlier mixed
  tactical/value result at 29 / 100, so the useful signal appears to come from
  the leaf value rather than from the exact scoring mixture.
- Policy-plus-tactical at 23 / 100 is above the normalized tactical-only
  baseline at 20 / 100 and matches the earlier traced tactical-shaped result at
  23 / 100, but it still does not approach the 50% specialized-opponent gate.
- Leaf scoring costs more inference time, about 0.0225s average search versus
  0.0079s for tactical scoring, but remains operationally acceptable for this
  one-deck diagnostic.
- This is positive evidence for adding a learned state-value term to search,
  but it is not sufficient for deck 12. Next work should inspect the two new
  trace files with the score-component SLURM diagnostic before tuning weights
  further or scaling the method to more decks.

## 2026-07-09 - Phase 5 Policy Parameter Count Diagnostic

Local architecture inspected:

- Model: `AlphaStarTurnPolicy` in `src/ptcg_abc/rl/phase5_policy.py`.
- Active Phase 5 config:
  - `global_dim=17`;
  - `entity_dim=28`;
  - `action_dim=53`;
  - `d_model=128`;
  - `nhead=4`;
  - `transformer_layers=2`;
  - `feedforward_dim=256`;
  - `turn_hidden_dim=128`.
- Exact parameter count by module shape:
  - global encoder: 2,560;
  - entity encoder: 3,968;
  - action encoder: 7,168;
  - 2-layer entity transformer: 264,960;
  - turn GRU: 99,072;
  - policy/action-value/tactical heads: 148,227 total;
  - state-value head: 16,641.
- Total per checkpoint: 542,596 trainable parameters.
- Total if all 13 deck-specialist checkpoints are counted as a stored ensemble:
  7,053,748 parameters. Inference normally uses the relevant deck specialist,
  so per-game active neural capacity remains roughly 0.54M parameters.

AlphaStar comparison source:

- DeepMind's Nature paper does not publish a single total parameter count in
  the main article. Its code-availability note points to Supplementary Data
  `detailed-architecture.txt` for the neural architecture and hyperparameters.
- The inspected supplementary architecture describes a much larger model than
  this Phase 5 TCG implementation:
  - up to 512 entities;
  - a 3-layer entity transformer with 2 attention heads, 128-dimensional
    keys/queries/values, 256-channel outputs, and 1024-hidden MLPs;
  - a 128x128 spatial encoder with convolutional downsampling and 4 residual
    blocks;
  - a 3-layer LSTM core with 384 hidden units per layer;
  - autoregressive action argument heads, including an action-type head with
    16 residual blocks of size 256;
  - value/baseline heads with additional 16-resblock stacks.

Conclusion:

- The current TCG Phase 5 network is intentionally tiny relative to AlphaStar:
  about 0.54M active parameters per specialist, or 7.05M stored parameters for
  all 13 specialists.
- AlphaStar's public supplementary architecture is not directly reducible to a
  verified total parameter count without reimplementing every preprocessing,
  action-space, gating, and baseline module, but its module widths and depth are
  clearly at least an order of magnitude larger than the active TCG specialist.
- For the current deck-12 bottleneck, the latest evidence still points more to
  training signal/search scoring than raw parameter count: leaf state value
  helped, but still failed the 50% gate. Do not scale model size blindly before
  fixing the target signal and score diagnostics.

## 2026-07-09 - One-Deck Epsilon Curriculum Implementation

User-requested experiment:

- Learner deck: official sample Dragapult ex.
- Fixed rule-based opponent: official sample Mega Lucario ex.
- Start from scratch.
- Train for 10 generations against rule-based play.
- Use epsilon-greedy exploration, generation 1 at 100% random legal-action
  exploration and generation 10 at 10%, linearly decayed.
- Generate 1000 training games per generation.
- Evaluate every generation with zero exploration for 100 games.
- Delete each generation's raw replay/trajectory data after it has been used
  for the generation update.
- Retain only small reports, checkpoints, and one win plus one loss visual replay
  view per generation.

Implementation:

- Added `Phase5EpsilonGreedyPolicyAgent`, an on-policy epsilon-greedy Phase 5
  policy agent.
  - `epsilon=1.0` means uniformly random legal-action selection.
  - `epsilon=0.10` means 10% random legal-action exploration and 90% greedy
    neural-policy selection.
  - Recorded trajectory metadata includes `policy_epsilon`,
    `policy_epsilon_random_steps`, `policy_epsilon_greedy_steps`, behavior
    `logprob`, and `policy_on_policy=true`.
- Added `rl-init-phase5-policy-checkpoint` to create a scratch Phase 5 symbolic
  policy checkpoint with the standard 96-entity/128-action encoder and 128-wide
  model.
- Extended public-agent trajectory/eval commands with:
  - `--controlled-public-agent-key`;
  - `--controlled-deck-index`;
  - `--policy-epsilon`.
- Public-agent trajectory/eval can now use a public/sample deck as the
  model-controlled deck instead of only the 13 Phase 5 league decks. The
  Dragapult-vs-Lucario experiment uses synthetic controlled deck index `101`,
  so checkpoints are named `deck-101.pt`.
- Added a built-in `sample_lucario` fallback: official sample Mega Lucario deck
  IDs controlled by the repo's `RuleBasedAgent`. This avoids requiring the
  external Kaggle sample Lucario notebook on ERAWAN.
- Public-agent eval can now save compact JSON and static HTML replay views with:
  - `--replay-output-dir`;
  - `--saved-win-replays`;
  - `--saved-loss-replays`;
  - `--replay-trace-limit`.
- Added `scripts/slurm/phase5_one_deck_public_epsilon_curriculum.sbatch`.
  It initializes generation 0 from scratch, then loops:
  collect 1000 epsilon-greedy games -> PPO update -> delete raw JSONL -> 100-game
  zero-exploration eval -> retain one win and one loss replay view.
- Updated `scripts/slurm/phase5_public_agent_trajectories.sbatch` and
  `scripts/slurm/phase5_public_agent_eval_conda.sbatch` to pass the new
  controlled-deck, epsilon, and replay-retention options.
- Updated `docs/phase-5-erawan-runbook.md` with the one-command ERAWAN
  submission and artifact layout.

Validation:

- `py_compile` passed for:
  - `src/ptcg_abc/agent/phase5_symbolic.py`;
  - `src/ptcg_abc/rl/workflow.py`;
  - `src/ptcg_abc/public_agents.py`;
  - `src/ptcg_abc/rl/public_opponents.py`;
  - `src/ptcg_abc/rl/phase5_symbolic_training.py`;
  - `src/ptcg_abc/cli.py`.
- `unittest tests.test_public_agents tests.test_phase5_alpha_league` passed.
- CLI help checks passed for:
  - `rl-generate-phase5-public-agent-trajectories`;
  - `rl-evaluate-phase5-public-agents`;
  - `rl-init-phase5-policy-checkpoint`.

Next ERAWAN action:

- Submit `scripts/slurm/phase5_one_deck_public_epsilon_curriculum.sbatch` with
  `RUN_NAME=phase5_dragapult_vs_lucario_epsilon`,
  `CONTROLLED_PUBLIC_AGENT_KEY=sample_dragapult`,
  `OPPONENT_PUBLIC_AGENT_KEYS=sample_lucario`,
  `CONTROLLED_DECK_INDEX=101`, `GENERATIONS=10`,
  `TRAIN_GAMES_PER_GENERATION=1000`, and `EVAL_GAMES_PER_GENERATION=100`.

## 2026-07-09 - Official Engine Source Audit

Context:

- Kaggle discussion 717141 announces the official competition game engine source
  on the competition Data page as `ptcg_engine`.
- Current Kaggle data download contains:
  - `ptcg_engine/ptcgProgram 22/` C++ source and README;
  - `sample_submission/sample_submission/cg/` Python ctypes wrapper and compiled
    engine binaries.
- The active repo's `src/ptcg_abc/simulator.py` is not a copied engine. It is a
  thin runner around whatever Kaggle `cg` package is supplied by `sample_dir`.

Comparison:

- Our packaged Phase 5 submission `cg` Python wrappers match the current Kaggle
  sample wrapper for:
  - `api.py`: SHA256
    `593F1298E52A635F90F8F505A52113E9AF114F444C293404E37906F18EE06CED`;
  - `game.py`: SHA256
    `3BD3D4F4A369A11E6D2F5DA9094CF15EBC410A2221835E6417B7CFF4883F1FC2`;
  - `utils.py`: SHA256
    `60F29665CEE0A88525D6F0383BC45959A6262D16FE35EF380AECE1E0EA13C49B`;
  - `__init__.py`: empty-file SHA256
    `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`.
- `sim.py` differs only in platform library selection. Current Kaggle sample
  adds `platform`, Darwin `libcg.dylib`, and arm64/aarch64 `libcg-arm64.so`
  support. The ctypes API declarations are otherwise unchanged.
- The compiled engine binaries do not match:
  - our `cg.dll`: 1,523,200 bytes, SHA256
    `C7C87EB76513784B0089B02DCF9D57466A1B0B2217DF4CFC9AF8C74DEDA3969F`;
  - current Kaggle `cg.dll`: 1,525,248 bytes, SHA256
    `9EA2B0A751029689BFF3DDCCB5F29A98EDD46961DAD264490ED121EF704FB500`;
  - our `libcg.so`: 1,338,304 bytes, SHA256
    `75D7D619B56E5AD4C5CAEADC698E61FAECC650678DCEAB52AD687F08D5676BEB`;
  - current Kaggle `libcg.so`: 1,342,400 bytes, SHA256
    `FFD89BF923525A3E6FEB5E6201E96A866C0F456895499ED5C4A566303CAAE67C`.
- Current Kaggle sample also includes `libcg.dylib` and `libcg-arm64.so`, which
  our packaged submission copies do not include.

Smoke:

- Direct `run_battle_smoke` through current Kaggle sample data loaded and ran
  20 steps with no error.
- The same direct smoke through the older local sample submission also loaded
  and ran 20 steps with no error.

Conclusion:

- The active Python runner can use the official current sample engine package
  when pointed at that `sample_dir`.
- Existing packaged submissions and local historical sample copies do not match
  the current official compiled engine binaries released with `ptcg_engine`.
- Before new Kaggle submissions, refresh copied `cg` directories from the
  current Kaggle sample submission so packaging uses the current official
  binaries and platform loader.

## 2026-07-09 - One-Deck Epsilon Curriculum Job 73798 Recovery

ERAWAN result:

- SLURM job: `73798`, script
  `scripts/slurm/phase5_one_deck_public_epsilon_curriculum.sbatch`.
- Run name: `phase5_dragapult_vs_lucario_epsilon`.
- Controlled learner: built-in `sample_dragapult`, synthetic deck index `101`.
- Opponent: built-in rule-based `sample_lucario`.
- Scratch generation-0 checkpoint was created at
  `models/rl/phase5_one_deck_public_epsilon/phase5_dragapult_vs_lucario_epsilon/gen-0000/specialists/deck-101.pt`.
- Generation 1 trajectory collection completed:
  - epsilon: 1.0;
  - games requested/started: 1000 / 1000;
  - trajectory steps: 41,431;
  - wins/losses/draws: 57 / 943 / 0;
  - errors/timeouts: 0 / 0;
  - attack taken rate: 0.1796;
  - attach taken rate: 0.4063;
  - output raw JSONL:
    `/project/SIGGI/thapanapong.r@cmu.ac.th/phase5_one_deck_public_epsilon/phase5_dragapult_vs_lucario_epsilon/generations/gen-0001/raw_train/phase5_public_epsilon_gen-0001.jsonl`;
  - trajectory report:
    `experiments/rl/phase5_one_deck_public_epsilon/phase5_dragapult_vs_lucario_epsilon/gen-0001/trajectory_report.json`.
- The job failed immediately after trajectory collection with:
  `Unknown Phase 5 league deck index: 101.`

Diagnosis:

- The original one-deck script used
  `rl-train-phase5-alpha-ppo-specialists`, which intentionally validates deck
  indices against the 13-deck Phase 5 league.
- The one-deck public-agent experiment uses a synthetic public deck index
  (`101`), so it should not use the 13-deck league wrapper for its PPO update.

Implementation fix:

- Updated `scripts/slurm/phase5_one_deck_public_epsilon_curriculum.sbatch` to
  train each generation with the generic single-checkpoint
  `rl-train-phase5-ppo` command:
  - source checkpoint:
    `PREVIOUS_DIR/deck-101.pt`;
  - output checkpoint:
    `CURRENT_DIR/deck-101.pt`;
  - filter:
    `--deck-index-filter 101`;
  - on-policy guard:
    `--require-on-policy`.
- Added `REUSE_EXISTING_TRAJECTORIES=1` support to skip trajectory generation
  when a generation raw JSONL already exists. This allows job 73798's completed
  generation-1 trajectory window to be reused if it is still present.
- Updated the ERAWAN runbook with a retry command using
  `REUSE_EXISTING_TRAJECTORIES=1`.

Next ERAWAN action:

- Pull the fix, then resubmit the one-deck epsilon curriculum with
  `REUSE_EXISTING_TRAJECTORIES=1`.
- If the generation-1 raw JSONL was removed manually, omit
  `REUSE_EXISTING_TRAJECTORIES=1` or leave it set; the script will regenerate
  any missing raw window.

## 2026-07-09 - One-Deck Epsilon Curriculum Retry 73820 Result

ERAWAN result:

- SLURM job: `73820`, script
  `scripts/slurm/phase5_one_deck_public_epsilon_curriculum.sbatch`.
- Run name: `phase5_dragapult_vs_lucario_epsilon`.
- Controlled learner: built-in `sample_dragapult`, synthetic deck index `101`.
- Opponent: built-in rule-based `sample_lucario`.
- The retry used `REUSE_EXISTING_TRAJECTORIES=1` and reused the generation-1
  raw JSONL from job `73798`; generations 2-10 generated fresh raw windows.
- The curriculum completed all 10 generations with 1000 training games and 100
  zero-exploration eval games per generation. The stderr contained repeated
  PyTorch nested-tensor warnings only; there were no eval errors or timeouts.

Eval summary:

| Generation | Eval wins | Eval losses | Win rate | Gate |
| ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 98 | 0.020 | fail |
| 2 | 1 | 99 | 0.010 | fail |
| 3 | 2 | 98 | 0.020 | fail |
| 4 | 3 | 97 | 0.030 | fail |
| 5 | 1 | 99 | 0.010 | fail |
| 6 | 5 | 95 | 0.050 | fail |
| 7 | 5 | 95 | 0.050 | fail |
| 8 | 3 | 97 | 0.030 | fail |
| 9 | 2 | 98 | 0.020 | fail |
| 10 | 2 | 98 | 0.020 | fail |

Training-window telemetry from the retry:

| Generation | Epsilon | Train wins | Train losses | Attack taken rate | Attach taken rate | End selected rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 57 | 943 | 0.1796 | 0.4063 | 0.1403 |
| 2 | 0.9 | 64 | 936 | 0.1495 | 0.3532 | 0.1603 |
| 3 | 0.8 | 42 | 958 | 0.1180 | 0.2925 | 0.1967 |
| 4 | 0.7 | 32 | 968 | 0.1014 | 0.2490 | 0.2400 |
| 5 | 0.6 | 30 | 970 | 0.0748 | 0.1974 | 0.2936 |
| 6 | 0.5 | 16 | 984 | 0.0711 | 0.1513 | 0.3403 |
| 7 | 0.4 | 13 | 987 | 0.0476 | 0.1297 | 0.3975 |
| 8 | 0.3 | 8 | 992 | 0.0345 | 0.0839 | 0.4525 |
| 9 | 0.2 | 8 | 992 | 0.0178 | 0.0601 | 0.5045 |
| 10 | 0.1 | 20 | 980 | 0.0084 | 0.0294 | 0.5777 |

Conclusion:

- The one-deck epsilon curriculum pipeline now runs end to end and cleans raw
  training windows after PPO updates, but scratch PPO from sparse terminal
  outcomes did not learn a usable Dragapult-vs-Lucario policy.
- Best eval checkpoints were generations 6 and 7 at only 5 / 100 wins; final
  generation 10 fell back to 2 / 100 wins.
- As exploration decayed, the learner selected `END` more often and attached or
  attacked much less often. This matches the observed Kaggle replay failure mode
  where neural agents sometimes skip available attach/attack lines.
- Do not scale this same sparse-reward curriculum. The next experiment should
  add stronger dense tactical reward or explicit `END` penalties when useful
  attack/attach options are available, or warm-start the learner from
  rule-demonstration/behavior-cloning data before PPO.

Artifact notes:

- Pulled local inspection folder:
  `D:\pokemon_rl\erawan_pull\phase5_dragapult_vs_lucario_epsilon_73820`.
- Gen-0010 replay artifacts include one loss and one win compact JSON/HTML view
  under that folder's `gen-0010-replays` directory.

## 2026-07-09 - One-Deck Mixed Rule/Epsilon Curriculum

Implementation:

- Added `scripts/slurm/phase5_one_deck_public_mixed_curriculum.sbatch`.
- The mixed script keeps the same focused one-deck matchup:
  - controlled learner deck: built-in `sample_dragapult`;
  - opponent: built-in rule-based `sample_lucario`;
  - synthetic controlled deck index: `101`.
- Generation 0 initializes a scratch `deck-101.pt` checkpoint and generates a
  retained 1000-game rule-vs-rule bootstrap dataset:
  `${GAME_DATA_ROOT}/phase5_one_deck_public_mixed/${RUN_NAME}/rule_bootstrap/phase5_public_rule_bootstrap_gen-0000.jsonl`.
- For every model generation, the script generates a fresh epsilon-model-vs-rule
  dataset from the previous checkpoint and trains with both trajectory inputs:
  the retained rule bootstrap dataset and the fresh epsilon dataset.
- This makes the intended mix 50/50 by requested game windows: 1000
  rule-vs-rule bootstrap games plus 1000 newly generated epsilon-vs-rule games
  per generation by default.
- After a successful PPO update, only the fresh epsilon JSONL is deleted. The
  rule bootstrap JSONL is retained and reused across all generations.
- The mixed script intentionally does not pass `--require-on-policy` by default
  because rule-vs-rule frames are off-policy demonstrations with zero
  logprob/value metadata. The epsilon half remains on-policy.
- Updated `docs/phase-5-erawan-runbook.md` with the ERAWAN submit command and
  artifact paths.

Next ERAWAN action:

- Pull the latest `origin/main`, then submit
  `scripts/slurm/phase5_one_deck_public_mixed_curriculum.sbatch` with
  `RUN_NAME=phase5_dragapult_vs_lucario_mixed`.
- Inspect the per-generation eval reports and PPO summaries after completion,
  especially whether attach/attack rates stay near the rule bootstrap behavior
  instead of collapsing toward `END`.

## 2026-07-10 - One-Deck Mixed Rule/Epsilon Curriculum Result

ERAWAN result:

- SLURM job: `73877`, script
  `scripts/slurm/phase5_one_deck_public_mixed_curriculum.sbatch`.
- Run name: `phase5_dragapult_vs_lucario_mixed`.
- Controlled learner: built-in `sample_dragapult`, synthetic deck index `101`.
- Opponent: built-in rule-based `sample_lucario`.
- The job completed all 10 generations with 1000 epsilon training games and 100
  zero-exploration eval games per generation. Stderr contained repeated PyTorch
  nested-tensor warnings only; eval reports had 0 errors and 0 timeouts.

Retained rule bootstrap:

- Games requested/started: 1000 / 1000.
- Steps: 84,036.
- Wins/losses/draws: 424 / 576 / 0.
- Errors/timeouts: 0 / 0.
- Attack taken rate: 0.2586.
- Attach taken rate: 0.4127.
- End selected rate: 0.0160.

Eval summary:

| Generation | Eval wins | Eval losses | Draws | Win rate | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 33 | 66 | 1 | 0.330 | fail |
| 2 | 49 | 50 | 1 | 0.490 | fail |
| 3 | 48 | 52 | 0 | 0.480 | fail |
| 4 | 44 | 55 | 1 | 0.440 | fail |
| 5 | 50 | 50 | 0 | 0.500 | pass at threshold |
| 6 | 43 | 57 | 0 | 0.430 | fail |
| 7 | 52 | 48 | 0 | 0.520 | pass |
| 8 | 46 | 53 | 1 | 0.460 | fail |
| 9 | 39 | 61 | 0 | 0.390 | fail |
| 10 | 45 | 55 | 0 | 0.450 | fail |

Epsilon-window behavior:

| Generation | Epsilon | Train wins | Train losses | Attack taken rate | Attach taken rate | End selected rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 68 | 932 | 0.1742 | 0.4151 | 0.1418 |
| 2 | 0.9 | 64 | 936 | 0.1807 | 0.4096 | 0.1321 |
| 3 | 0.8 | 85 | 915 | 0.1716 | 0.4239 | 0.1215 |
| 4 | 0.7 | 87 | 913 | 0.1676 | 0.4245 | 0.1112 |
| 5 | 0.6 | 97 | 903 | 0.1766 | 0.4311 | 0.0980 |
| 6 | 0.5 | 113 | 886 | 0.1753 | 0.4278 | 0.0867 |
| 7 | 0.4 | 147 | 853 | 0.1830 | 0.4219 | 0.0736 |
| 8 | 0.3 | 179 | 821 | 0.1984 | 0.4325 | 0.0613 |
| 9 | 0.2 | 253 | 745 | 0.2153 | 0.4325 | 0.0472 |
| 10 | 0.1 | 363 | 634 | 0.2388 | 0.4169 | 0.0335 |

Conclusion:

- The retained rule-vs-rule bootstrap anchor fixed the main failure mode from
  the sparse epsilon-only run. The model no longer collapses toward ending turns:
  generation-10 `END` selections were 0.0335 instead of 0.5777 in the
  epsilon-only run, while attach rates stayed near the rule bootstrap rate.
- The best checkpoint is generation 7:
  `models/rl/phase5_one_deck_public_mixed/phase5_dragapult_vs_lucario_mixed/gen-0007/specialists/deck-101.pt`.
- Generation 7 scored 52 / 100 against rule Lucario, and generation 5 scored
  exactly 50 / 100. This is the first focused one-deck run to reach the 50%
  rule-agent gate, but the 100-game eval is still noisy.
- Do not package based only on the 100-game report. Run a larger zero-exploration
  confirmation eval for generation 7, preferably 500-1000 games, before treating
  it as promoted.

## 2026-07-10 - One-Deck Mixed Gen7 1000-Game Confirmation

ERAWAN result:

- SLURM job: `73901`, script
  `scripts/slurm/phase5_public_agent_eval_conda.sbatch`.
- Evaluated checkpoint:
  `models/rl/phase5_one_deck_public_mixed/phase5_dragapult_vs_lucario_mixed/gen-0007/specialists`.
- Controlled learner: built-in `sample_dragapult`, synthetic deck index `101`.
- Opponent: built-in rule-based `sample_lucario`.
- Games per matchup: 1000.
- Stderr contained one PyTorch nested-tensor warning only.

Result:

- Wins/losses/draws: 462 / 537 / 1.
- Timeouts/errors: 0 / 0.
- Win rate: 0.462.
- Public-agent 50% gate: fail.

Baseline comparison:

- Rule-vs-rule Dragapult vs Lucario baseline from the mixed run's retained
  generation-0 bootstrap was 424 / 1000, or 0.424.
- The gen7 mixed checkpoint beat the fixed 0.424 baseline:
  462 / 1000, one-sided binomial tail `p ~= 0.0084`.
- Compared against the sampled 424 / 1000 rule baseline, the two-proportion
  one-sided z test gives `p ~= 0.0436`; this just clears a one-sided 95%
  baseline-improvement threshold.
- Wilson 95% interval for gen7 confirmation win rate is approximately
  0.431-0.493, so the current checkpoint still does not have credible evidence
  of being above 50%.

Conclusion:

- The mixed rule/epsilon curriculum produced a real improvement over the
  rule-only Dragapult baseline, but not enough to pass the 50% target against
  rule Lucario.
- Next training should continue from the gen7 checkpoint rather than gen10,
  and should keep the retained rule bootstrap anchor. Good next knobs are more
  mixed generations with lower epsilon, larger rule/eval windows, or a mild
  dense tactical reward while retaining the rule demonstrations.

## 2026-07-11 - Rule-Only Epoch Diagnostic Implementation

Implementation:

- Added resume support to `rl-train-phase5-generalist` via
  `--initial-checkpoint`, so supervised rule-demonstration training can continue
  one epoch at a time while writing an evaluation checkpoint after every epoch.
- Added `scripts/slurm/phase5_one_deck_rule_epoch_bc.sbatch`.
- The diagnostic trains two synthetic public-deck specialists from rule-vs-rule
  trajectories only:
  - `deck-101.pt`: sample Dragapult ex trained from rule Dragapult vs rule
    Lucario;
  - `deck-102.pt`: sample Mega Lucario ex trained from rule Lucario vs rule
    Dragapult.
- The script evaluates after each supervised epoch:
  - trained Dragapult vs rule Lucario;
  - trained Lucario vs rule Dragapult.
- Default schedule: at least 10 epochs, at most 50 epochs, early stop if the
  combined two-matchup eval win count does not improve for 10 epochs.
- By default the script reuses the retained Dragapult-vs-Lucario rule bootstrap
  from `phase5_dragapult_vs_lucario_mixed` if it exists, and generates the
  Lucario-vs-Dragapult rule bootstrap once.
- Updated `docs/phase-5-erawan-runbook.md` with the ERAWAN submit command and
  artifact paths.

Rationale:

- This isolates supervised imitation capacity from PPO/self-play. If the neural
  policy cannot match or surpass rule-agent behavior under repeated rule-only
  epochs, the next bottleneck is likely representation/action scoring rather
  than exploration.

## 2026-07-11 - Rule-Only Epoch Diagnostic Result

ERAWAN result:

- SLURM job: `73948`, script
  `scripts/slurm/phase5_one_deck_rule_epoch_bc.sbatch`.
- Run name: `phase5_dragapult_lucario_rule_epoch_bc`.
- The job completed epochs 1-18 and early-stopped at epoch 18 after 10
  non-improving combined-eval epochs.
- Evaluation reports: 36 matchup reports, 0 errors, 0 timeouts, 4 total draws.
- Stderr contained repeated PyTorch nested-tensor warnings only.

Rule-vs-rule baselines:

- Dragapult rule vs Lucario rule:
  424 / 1000 wins = 0.424. This reused the retained rule bootstrap from the
  mixed-rule experiment.
- Lucario rule vs Dragapult rule:
  191 / 1000 wins = 0.191. This was generated by job `73948`.
- Combined two-direction baseline:
  615 / 2000 wins = 0.3075.

Epoch eval summary:

| Epoch | Combined wins / 200 | Dragapult vs Lucario | Lucario vs Dragapult | Dragapult train acc | Lucario train acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 52 | 37 / 100 | 15 / 100 | 0.997 | 0.989 |
| 2 | 63 | 44 / 100 | 19 / 100 | 1.000 | 1.000 |
| 3 | 59 | 39 / 100 | 20 / 100 | 1.000 | 1.000 |
| 4 | 56 | 35 / 100 | 21 / 100 | 1.000 | 1.000 |
| 5 | 65 | 45 / 100 | 20 / 100 | 1.000 | 1.000 |
| 6 | 63 | 43 / 100 | 20 / 100 | 1.000 | 1.000 |
| 7 | 55 | 38 / 100 | 17 / 100 | 1.000 | 1.000 |
| 8 | 77 | 50 / 100 | 27 / 100 | 1.000 | 1.000 |
| 9 | 60 | 40 / 100 | 20 / 100 | 1.000 | 1.000 |
| 10 | 58 | 43 / 100 | 15 / 100 | 1.000 | 1.000 |
| 11 | 50 | 32 / 100 | 18 / 100 | 1.000 | 1.000 |
| 12 | 59 | 41 / 100 | 18 / 100 | 1.000 | 1.000 |
| 13 | 62 | 41 / 100 | 21 / 100 | 1.000 | 1.000 |
| 14 | 56 | 39 / 100 | 17 / 100 | 1.000 | 1.000 |
| 15 | 69 | 49 / 100 | 20 / 100 | 1.000 | 1.000 |
| 16 | 67 | 45 / 100 | 22 / 100 | 1.000 | 1.000 |
| 17 | 60 | 44 / 100 | 16 / 100 | 1.000 | 1.000 |
| 18 | 65 | 46 / 100 | 19 / 100 | 1.000 | 1.000 |

Best checkpoint:

- Best combined epoch: epoch 8, with 77 / 200 combined wins.
- Best Dragapult-vs-Lucario epoch: epoch 8, with 50 / 100 wins.
- Best Lucario-vs-Dragapult epoch: epoch 8, with 27 / 100 wins.
- Candidate checkpoint directory:
  `models/rl/phase5_one_deck_rule_epoch_bc/phase5_dragapult_lucario_rule_epoch_bc/epoch-0008/specialists`.

Baseline comparison:

- Raw combined baseline was surpassed first at epoch 2:
  63 / 200 = 0.315 versus baseline 0.3075.
- The first statistically meaningful combined pass was epoch 8:
  77 / 200 = 0.385. Against fixed baseline 0.3075, one-sided binomial
  `p ~= 0.0118`; against sampled rule baselines, two-proportion one-sided
  `p ~= 0.0122`.
- Dragapult alone never reached the one-sided 95% fixed-baseline threshold.
  Best was 50 / 100 at epoch 8 versus baseline 0.424; the 100-game threshold
  is 52 / 100.
- Lucario alone reached the one-sided 95% fixed-baseline threshold at epoch 8:
  27 / 100 versus baseline 0.191.

Conclusion:

- Repeated rule-only supervised epochs can memorize the rule-demonstration
  dataset quickly: both train accuracies reached 1.000 by epoch 2.
- More epochs did not yield monotonic eval improvement. The best epoch was 8,
  then the run failed to improve for 10 epochs and stopped at epoch 18.
- This suggests the remaining bottleneck is not simply "more supervised epochs."
  The model likely imitates the rule policy on rule-visited states, then drifts
  into model-visited states where it lacks corrective labels.
- Next diagnostic should be rule-teacher dataset aggregation: run the trained
  model against the rule opponent, record the model-visited states, label those
  states with rule-agent choices, add them to the rule bootstrap, and retrain.
  This is closer to DAgger and should directly address distribution shift.

## 2026-07-12 - One-Deck Rule-Teacher DAgger Implementation

Implementation:

- Extended `RecordingPolicyAgent` with optional teacher-label recording. The
  behavior agent still chooses the action executed in the game, but trajectory
  records can now store a different teacher-selected target action along with
  `behavior_selected_indices`, `teacher_selected_indices`, `teacher_agent`, and
  `teacher_forced_target` metadata.
- Added `--teacher-agent rule` to
  `rl-generate-phase5-public-agent-trajectories`. The trajectory summary now
  records `teacher_agent`; the command also accepts `--agent phase5-symbolic`
  for deterministic model behavior during correction collection.
- Added `scripts/slurm/phase5_one_deck_rule_teacher_dagger.sbatch`.
  The script starts from the rule-only epoch-8 two-specialist checkpoint,
  reuses or regenerates the retained Dragapult-vs-Lucario and
  Lucario-vs-Dragapult rule bootstrap data, collects model-visited correction
  windows labeled by the rule teacher, aggregates those correction JSONL files
  across iterations, retrains `deck-101.pt` and `deck-102.pt`, and evaluates
  both directions after each iteration.
- Updated `docs/phase-5-erawan-runbook.md` with the default ERAWAN submit
  command and artifact paths.

Experiment intent:

- Test the distribution-shift hypothesis from job `73948`: the neural policy
  fit rule-visited states quickly, but failed to improve monotonically when it
  acted in the environment. DAgger should expose the states the current model
  actually visits and give supervised rule labels there.
- The correction windows are intentionally retained for this bounded diagnostic
  because DAgger requires aggregation. They live under
  `$GAME_DATA_ROOT/phase5_one_deck_rule_teacher_dagger/...`, while reports and
  checkpoints stay in the repo paths.

Next ERAWAN step:

- Run `scripts/slurm/phase5_one_deck_rule_teacher_dagger.sbatch` with the
  runbook defaults, then inspect `iter-000*/iteration_summary.json`, the two
  teacher trajectory reports per iteration, and the two 100-game eval reports
  per iteration.

## 2026-07-12 - One-Deck Rule-Teacher DAgger Result And Context Fix

ERAWAN result:

- SLURM job: `73988`, script
  `scripts/slurm/phase5_one_deck_rule_teacher_dagger.sbatch`.
- Run name: `phase5_dragapult_lucario_rule_teacher_dagger`.
- The job completed base eval plus iterations 1-10, then early-stopped after
  five non-improving DAgger iterations.
- Eval reports: 22 matchup reports, 2200 total eval games, 0 errors, 0
  timeouts, 6 draws.
- Stderr contained repeated PyTorch nested-tensor warnings only.

Eval summary:

| Iteration | Combined wins / 200 | Dragapult vs Lucario | Lucario vs Dragapult |
| ---: | ---: | ---: | ---: |
| 0 | 80 | 52 / 100 | 28 / 100 |
| 1 | 57 | 40 / 100 | 17 / 100 |
| 2 | 61 | 45 / 100 | 16 / 100 |
| 3 | 66 | 49 / 100 | 17 / 100 |
| 4 | 56 | 38 / 100 | 18 / 100 |
| 5 | 71 | 46 / 100 | 25 / 100 |
| 6 | 63 | 38 / 100 | 25 / 100 |
| 7 | 69 | 48 / 100 | 21 / 100 |
| 8 | 66 | 44 / 100 | 22 / 100 |
| 9 | 66 | 45 / 100 | 21 / 100 |
| 10 | 47 | 33 / 100 | 14 / 100 |

Teacher-collection and train diagnostics:

- Correction windows were generated by the model behavior policy and labeled by
  the rule teacher.
- Dragapult correction windows across iterations 1-10: 5000 games, 412,168
  trajectory steps, 2186 wins, 2808 losses, 6 draws, 0 errors/timeouts.
- Lucario correction windows across iterations 1-10: 5000 games, 260,393
  trajectory steps, 994 wins, 4001 losses, 5 draws, 0 errors/timeouts.
- Train reports reached accuracy 1.000 and final loss 0.0 throughout, even as
  eval performance stayed below the starting checkpoint. By iteration 10, the
  Dragapult aggregated train set contained 495,729 examples across 11 datasets,
  and Lucario contained 312,304 examples across 11 datasets.

Conclusion:

- This DAgger run did not improve over the starting checkpoint. The base
  checkpoint at iteration 0 was best overall: 80 / 200 combined, 52 / 100
  Dragapult, 28 / 100 Lucario.
- The best post-update checkpoint was iteration 5 at 71 / 200 combined, 46 /
  100 Dragapult, and 25 / 100 Lucario. It did not beat the base checkpoint and
  was not statistically meaningful against the combined rule-vs-rule baseline
  of 0.3075 (`p ~= 0.085` one-sided fixed-baseline binomial).
- The final iteration degraded to 47 / 200 combined.
- Root-cause diagnosis: the first DAgger implementation stored the rule teacher
  action in `TrajectoryStep.chosen_indices`, and the trainer also used
  `chosen_indices` to build the previous-action context. When the model's
  behavior action differed from the teacher action, later frames described a
  state reached by the model action but a previous-action history containing
  the teacher action. That makes teacher-forced trajectories internally
  inconsistent.

Implementation fix:

- Added `Phase5TurnContext.observe_indices(...)`.
- For trajectories with `teacher_forced_target`, the generalist and PPO
  trajectory trainers now keep the teacher action as the supervised target but
  use `behavior_selected_indices` for previous-action history.
- Added a focused unit test covering teacher-forced trajectory context.

Next ERAWAN step:

- Treat job `73988` as a pre-fix diagnostic, not a promoted result.
- Pull the new code and rerun the same DAgger command. If the rerun still fails
  to beat iteration 0, stop DAgger and switch to a cleaner value/policy
  evaluation diagnostic rather than adding more correction windows.

## 2026-07-13 - Fixed One-Deck Rule-Teacher DAgger Result

ERAWAN result:

- SLURM job: `74006`, script
  `scripts/slurm/phase5_one_deck_rule_teacher_dagger.sbatch`.
- Run name: `phase5_dragapult_lucario_rule_teacher_dagger_fixed`.
- The job completed base eval plus iterations 1-9, then early-stopped after
  five non-improving iterations.
- Evaluation reports: 20 matchup reports, 2000 total eval games, 0 errors, 0
  timeouts, 4 draws.
- Stderr contained repeated PyTorch nested-tensor warnings only.

Eval summary:

| Iteration | Combined wins / 200 | Dragapult vs Lucario | Lucario vs Dragapult |
| ---: | ---: | ---: | ---: |
| 0 | 63 | 52 / 100 | 11 / 100 |
| 1 | 71 | 47 / 100 | 24 / 100 |
| 2 | 62 | 35 / 100 | 27 / 100 |
| 3 | 61 | 40 / 100 | 21 / 100 |
| 4 | 78 | 53 / 100 | 25 / 100 |
| 5 | 60 | 47 / 100 | 13 / 100 |
| 6 | 62 | 45 / 100 | 17 / 100 |
| 7 | 67 | 52 / 100 | 15 / 100 |
| 8 | 57 | 43 / 100 | 14 / 100 |
| 9 | 63 | 50 / 100 | 13 / 100 |

Best checkpoint:

- Best combined checkpoint: iteration 4, with 78 / 200 combined wins.
- Best Dragapult-vs-Lucario checkpoint: iteration 4, with 53 / 100 wins.
- Best Lucario-vs-Dragapult checkpoint: iteration 2, with 27 / 100 wins.
- Candidate model directory for confirmation:
  `models/rl/phase5_one_deck_rule_teacher_dagger/phase5_dragapult_lucario_rule_teacher_dagger_fixed/iter-0004/specialists`.

Teacher-collection and train diagnostics:

- Dragapult correction windows across iterations 1-9: 4500 games, 371,380
  trajectory steps, 1941 wins, 2547 losses, 12 draws, 0 errors/timeouts.
- Lucario correction windows across iterations 1-9: 4500 games, 234,230
  trajectory steps, 886 wins, 3611 losses, 3 draws, 0 errors/timeouts.
- Train reports again reached accuracy 1.000 and final loss 0.0 throughout.
  By iteration 9, the Dragapult aggregated train set contained 455,005 examples
  across 10 datasets, and Lucario contained 286,141 examples across 10
  datasets.

Baseline comparison:

- Iteration 4 combined win rate was 0.390. Against the combined rule-vs-rule
  baseline of 0.3075, fixed-baseline one-sided binomial `p ~= 0.0080`.
- Iteration 4 Dragapult win rate was 0.530. Against the retained Dragapult
  rule-vs-rule baseline of 0.424, fixed-baseline one-sided binomial
  `p ~= 0.0210`.
- Iteration 2 Lucario win rate was 0.270. Against the Lucario rule-vs-rule
  baseline of 0.191, fixed-baseline one-sided binomial `p ~= 0.0338`.
- The fixed run's base eval was noisy: 63 / 200 combined, driven by Lucario
  scoring only 11 / 100. Do not conclude iteration 4 is promoted based only on
  the 100-game eval.

Conclusion:

- The teacher-context fix changed the DAgger result from clearly negative to a
  small positive signal, especially for Dragapult.
- The signal is still not promotion-ready. Dragapult crossed 50% in one
  100-game eval, but Lucario did not; combined iteration 4 is similar to the
  previous noisy base eval from job `73988` and needs larger confirmation.
- Do not run more DAgger iterations yet. Confirm iteration 4 first with a
  larger zero-exploration eval in both directions, preferably 1000 games per
  matchup. If the larger eval confirms Dragapult but not Lucario, split the
  next experiments by deck instead of updating both together.

## 2026-07-17 - Fixed DAgger Iteration 4 Confirmation

ERAWAN result:

- SLURM jobs: `74012` and `74013`, script
  `scripts/slurm/phase5_public_agent_eval_conda.sbatch`.
- Evaluated checkpoint:
  `models/rl/phase5_one_deck_rule_teacher_dagger/phase5_dragapult_lucario_rule_teacher_dagger_fixed/iter-0004/specialists`.
- Agent: `phase5-symbolic`.
- Games per matchup: 1000.
- Stderr contained PyTorch nested-tensor warnings only.

Results:

| Controlled deck | Opponent | Wins / games | Losses | Draws | Errors | Timeouts | Win rate | Wilson 95% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dragapult ex | Rule Mega Lucario ex | 449 / 1000 | 548 | 3 | 0 | 0 | 0.449 | 0.418-0.480 |
| Mega Lucario ex | Rule Dragapult ex | 202 / 1000 | 798 | 0 | 0 | 0 | 0.202 | 0.178-0.228 |

Baseline comparison:

- Combined confirmation: 651 / 2000 = 0.3255. Against the combined
  rule-vs-rule baseline of 0.3075, fixed-baseline one-sided binomial
  `p ~= 0.0432`.
- Dragapult confirmation: 449 / 1000 = 0.449. Against the retained
  rule-vs-rule Dragapult baseline of 0.424, fixed-baseline one-sided binomial
  `p ~= 0.0587`, which is not a 5% pass. It is also below the earlier mixed
  gen7 1000-game confirmation of 462 / 1000.
- Lucario confirmation: 202 / 1000 = 0.202. Against the Lucario rule-vs-rule
  baseline of 0.191, fixed-baseline one-sided binomial `p ~= 0.198`, not
  meaningful.

Conclusion:

- The 100-game fixed-DAgger iteration-4 bump was not confirmed. Both decks fail
  the 50% public-agent gate, and neither individual matchup shows a strong
  improvement over rule-vs-rule baseline at 1000 games.
- The teacher-context fix made DAgger less harmful, but this DAgger recipe is
  not enough to produce a promotable one-deck policy.
- Stop scaling this two-deck DAgger loop for now. Next work should split the
  problem by deck and inspect action-quality failures in confirmed games,
  especially Dragapult, where the mixed rule/epsilon gen7 checkpoint remains
  the better 1000-game candidate.

## 2026-07-17 - Fractional Prize Reward Shaping

Implementation:

- Added opt-in public-agent tactical reward modes:
  - `fractional-prize`;
  - `basic-fractional-prize`.
- Fractional prize progress is computed as:
  `prizes_taken + sum(card_prize_value * damage_fraction)` over the opponent's
  active and bench Pokemon from the controlled player's perspective.
- `card_prize_value` is `3` for Mega ex, `2` for regular ex, and `1` otherwise.
  This matches the Dragapult example: after KOing Mega Lucario ex and placing
  60 damage on Riolu, progress is `3 + 60/80`; placing the same 60 damage on
  Solrock is `3 + 60/110`.
- The per-step fractional reward is:
  `fractional_prize_weight * (our_delta - fractional_opponent_weight * opponent_delta)`.
- The trajectory writer compares each controlled-agent decision board against
  the next recorded controlled-agent decision board. If the decision is the
  final recorded frame, it falls back to final prize counts so terminal KOs
  still receive prize-progress reward.
- The old attack/attach dense reward remains unchanged under `basic`. The new
  `basic-fractional-prize` mode adds the old basic reward to the fractional
  prize-progress reward, which is useful for early one-deck PPO because energy
  attachment itself often has delayed credit.
- Trajectory reports now summarize `basic_reward_sum`,
  `fractional_prize_reward_sum`, `fractional_prize_delta_sum`,
  `fractional_opponent_prize_delta_sum`, and average versions of those fields.
- `rl-generate-phase5-public-agent-trajectories` now accepts:
  - `--tactical-reward-mode {none,basic,fractional-prize,basic-fractional-prize}`;
  - `--tactical-fractional-prize-weight`;
  - `--tactical-fractional-opponent-weight`.
- SLURM wrappers now pass through fractional-prize weights:
  - `scripts/slurm/phase5_public_agent_trajectories.sbatch`;
  - `scripts/slurm/phase5_one_deck_public_epsilon_curriculum.sbatch`;
  - `scripts/slurm/phase5_one_deck_public_mixed_curriculum.sbatch`;
  - `scripts/slurm/phase5_one_deck_rule_teacher_dagger.sbatch`.

Validation:

- `py_compile` passed for `src/ptcg_abc/rl/public_opponents.py` and
  `src/ptcg_abc/cli.py` using a temporary pycache prefix.
- `PYTHONPATH=src python -m unittest tests.test_public_agents` passed.
- `bash -n` passed for the updated SLURM scripts.

Next ERAWAN experiment:

- Run a clean one-deck Dragapult-vs-Lucario mixed curriculum from scratch with
  retained rule-vs-rule bootstrap data plus per-generation epsilon-vs-rule data.
- Use `OUTCOME_REWARD_SCALE=0.0`,
  `TACTICAL_REWARD_MODE=basic-fractional-prize`,
  `TACTICAL_FRACTIONAL_PRIZE_WEIGHT=0.25`, and
  `TACTICAL_FRACTIONAL_OPPONENT_WEIGHT=1.0`.
- Compare generation evals against the retained rule-vs-rule baseline and the
  previous mixed gen7 1000-game confirmation result.

ERAWAN submission:

- Pushed implementation commit `ec05113`.
- Submitted SLURM job `74307` for run
  `phase5_dragapult_vs_lucario_fractional_prize` using
  `scripts/slurm/phase5_one_deck_public_mixed_curriculum.sbatch`.
- ERAWAN job id was recorded to
  `experiments/rl/phase5_one_deck_fractional_prize_job.txt`.

## 2026-07-17 - Fractional Prize Mixed Curriculum Result

ERAWAN result:

- SLURM job: `74307`.
- Run name: `phase5_dragapult_vs_lucario_fractional_prize`.
- Script: `scripts/slurm/phase5_one_deck_public_mixed_curriculum.sbatch`.
- Controlled deck: sample Dragapult ex, synthetic deck index `101`.
- Opponent: sample Mega Lucario ex rule-based public agent.
- Schedule: 10 generations, 1000 retained rule-vs-rule bootstrap games, 1000
  epsilon-vs-rule games per generation, 100 zero-exploration eval games per
  generation.
- Reward config: `OUTCOME_REWARD_SCALE=0.0`,
  `TACTICAL_REWARD_MODE=basic-fractional-prize`,
  `TACTICAL_FRACTIONAL_PRIZE_WEIGHT=0.25`,
  `TACTICAL_FRACTIONAL_OPPONENT_WEIGHT=1.0`.
- Stderr contained only PyTorch nested-tensor warnings. No eval errors or
  timeouts occurred.

Eval results:

| Generation | Epsilon | Eval wins / games | Win rate |
| ---: | ---: | ---: | ---: |
| 1 | 1.0 | 16 / 100 | 0.16 |
| 2 | 0.9 | 20 / 100 | 0.20 |
| 3 | 0.8 | 17 / 100 | 0.17 |
| 4 | 0.7 | 20 / 100 | 0.20 |
| 5 | 0.6 | 19 / 100 | 0.19 |
| 6 | 0.5 | 15 / 100 | 0.15 |
| 7 | 0.4 | 24 / 100 | 0.24 |
| 8 | 0.3 | 22 / 100 | 0.22 |
| 9 | 0.2 | 26 / 100 | 0.26 |
| 10 | 0.1 | 20 / 100 | 0.20 |

Training-data and reward diagnostics:

- Retained rule bootstrap: 432 wins, 565 losses, 3 draws over 1000 games;
  82,070 trajectory steps. This matches the retained Dragapult-vs-Lucario
  rule-vs-rule baseline range.
- Rule bootstrap reward summary:
  - average total tactical reward: `+0.00622`;
  - average fractional-prize component: `-0.00460`;
  - average basic attack/attach component: `+0.01082`.
- Epsilon-vs-rule training windows improved as epsilon decayed, but remained
  weak: generation 1 scored 68 / 1000, generation 8 scored 146 / 1000,
  generation 9 scored 144 / 1000, and generation 10 scored 171 / 1000.
- Epsilon-window fractional reward stayed negative in every generation, though
  it became less negative as epsilon decayed:
  - generation 1 average fractional component: `-0.03264`;
  - generation 10 average fractional component: `-0.01503`.

Conclusion:

- This fractional-prize mixed PPO run is not promotable. The best
  zero-exploration checkpoint was generation 9 at 26 / 100, far below the
  retained Dragapult rule-vs-rule baseline around 424 / 1000 and below the
  earlier mixed gen7 confirmation of 462 / 1000.
- The reward implementation is operationally valid, but the shaped signal is
  not aligned enough for this PPO recipe. The fractional component is negative
  even for the rule bootstrap window, so it can push the learner away from
  useful rule-agent behavior.
- Likely issue: the current reward compares one controlled-agent decision board
  to the next controlled-agent decision board. That interval can include
  multi-step attack subdecisions, Pokemon promotion, and the opponent's response,
  so the dense reward is not a clean immediate reward for the selected action.
- Do not continue this fractional-prize curriculum as-is. The next fix should
  either record immediate post-action board deltas from the simulator or move
  fractional-prize scoring into an attack/turn-end rollout evaluator, then
  retest on a smaller one-generation diagnostic before another 10-generation
  run.

## 2026-07-22 - Post-Action Fractional Prize Credit Assignment

Implementation:

- Added a simulator-to-recorder post-action observation hook. After
  `battle_select(...)`, `run_battle(...)` now calls
  `observe_after_action(...)` on the acting agent when that hook exists.
- `RecordingPolicyAgent` stores a `post_action_board` on the just-recorded
  `RecordedPolicyFrame`, using the acting player index as the forced board
  perspective. This prevents the board summary from flipping to the opponent's
  perspective when the engine advances `yourIndex`.
- `summarize_board(...)` now accepts `your_index_override` for this same-player
  post-action perspective.
- Public-agent fractional reward now prefers the immediate `post_action_board`,
  then falls back to the next recorded decision frame, then final prize counts.
- Tactical reward summaries now include `fractional_after_board_sources` so
  ERAWAN reports can confirm whether rewards came from `post-action`,
  `next-frame`, or `final-prizes`.

Validation:

- `py_compile` passed for `src/ptcg_abc/rl/featurizer.py`,
  `src/ptcg_abc/rl/workflow.py`, `src/ptcg_abc/simulator.py`, and
  `src/ptcg_abc/rl/public_opponents.py`.
- `PYTHONPATH=src python -m unittest tests.test_public_agents` passed.
- `git diff --check` passed aside from existing Windows line-ending warnings.
- A local one-game trajectory smoke could not run because this checkout does
  not have `data/kaggle/input/sample_submission`; use the ERAWAN diagnostics to
  confirm `fractional_after_board_sources.post-action` is populated.

Next ERAWAN diagnostics:

- Queue two short three-generation Dragapult-vs-Lucario mixed-curriculum jobs
  in parallel:
  - post-action `basic-fractional-prize` with fractional weight `0.25`;
  - `basic` attack/attach reward only.
- Use 1000 retained rule-vs-rule bootstrap games, 1000 epsilon-vs-rule games
  per generation, 200 zero-exploration eval games per generation, and
  `OUTCOME_REWARD_SCALE=0.0`.
- Compare the two runs before launching any new 10-generation curriculum.

ERAWAN submission:

- Pushed implementation commit `e672f8d`.
- Submitted post-action fractional diagnostic as SLURM job `74745`, run name
  `phase5_dragapult_vs_lucario_postaction_frac025_diag`.
- Submitted basic-only A/B diagnostic as SLURM job `74746`, run name
  `phase5_dragapult_vs_lucario_basic_only_diag`.
- Job IDs were written on ERAWAN to:
  - `experiments/rl/phase5_one_deck_postaction_frac025_diag_job.txt`;
  - `experiments/rl/phase5_one_deck_basic_only_diag_job.txt`.

## 2026-07-22 - Post-Action Fractional A/B Diagnostic Result

ERAWAN result:

- SLURM jobs:
  - `74745`: `phase5_dragapult_vs_lucario_postaction_frac025_diag`;
  - `74746`: `phase5_dragapult_vs_lucario_basic_only_diag`.
- Both jobs completed three generations with 1000 rule-bootstrap games, 1000
  epsilon-vs-rule games per generation, and 200 zero-exploration eval games per
  generation.
- Both used `OUTCOME_REWARD_SCALE=0.0`; the only learning signal was the dense
  tactical reward.
- Stderr contained PyTorch nested-tensor warnings only.

Eval results:

| Run | Gen 1 | Gen 2 | Gen 3 |
| --- | ---: | ---: | ---: |
| Post-action `basic-fractional-prize`, weight 0.25 | 13 / 200 | 12 / 200 | 8 / 200 |
| `basic` attack/attach only | 20 / 200 | 9 / 200 | 12 / 200 |

Reward-source and behavior diagnostics:

- The post-action hook worked: every fractional reward in job `74745` used
  `fractional_after_board_sources = {'post-action': steps}`.
- Post-action fractional reward is now sane on rule data: the rule bootstrap
  average fractional component was `+0.00542` instead of negative, and the
  total average tactical reward was `+0.01611`.
- However, dense-only learning is not a viable objective. With
  `OUTCOME_REWARD_SCALE=0.0`, both diagnostics collapsed far below the
  rule-vs-rule baseline and below the previous mixed gen7 checkpoint.
- By generation 3, both dense-only runs over-favored tactical actions:
  - post-action fractional: attack taken rate `0.8979`, attach taken rate
    `0.7165`, eval `8 / 200`;
  - basic-only: attack taken rate `0.8933`, attach taken rate `0.6686`,
    eval `12 / 200`.

Conclusion:

- The post-action reward implementation is valid, but dense tactical reward
  alone does not preserve the game objective. The original successful mixed run
  used terminal outcome reward by default (`OUTCOME_REWARD_SCALE=1.0`), while
  these diagnostics removed it.
- Do not run more dense-only PPO curricula. The next diagnostic should keep
  terminal outcome reward and compare:
  - `OUTCOME_REWARD_SCALE=1.0`, `TACTICAL_REWARD_MODE=none`;
  - `OUTCOME_REWARD_SCALE=1.0`, `TACTICAL_REWARD_MODE=basic-fractional-prize`,
    `TACTICAL_FRACTIONAL_PRIZE_WEIGHT=0.25`.

Next ERAWAN diagnostics:

- Submitted `74766`, run name
  `phase5_dragapult_vs_lucario_outcome1_postaction_frac025_diag`, for terminal
  outcome reward plus post-action fractional shaping.
- Submitted `74767`, run name `phase5_dragapult_vs_lucario_outcome1_none_diag`,
  for terminal outcome reward with no tactical shaping.
- Both jobs use three generations, 1000 rule-bootstrap games, 1000
  epsilon-vs-rule games per generation, and 200 zero-exploration eval games per
  generation.

## 2026-07-22 - Outcome Plus Post-Action Fractional Diagnostic Result

ERAWAN result:

- SLURM job `74766` completed successfully in `01:27:34` with exit code `0`.
- Run name:
  `phase5_dragapult_vs_lucario_outcome1_postaction_frac025_diag`.
- Configuration: terminal outcome reward scale `1.0`,
  `basic-fractional-prize` tactical reward, fractional-prize weight `0.25`,
  three generations, 1000 rule-bootstrap games, 1000 epsilon-vs-rule games per
  generation, and 200 zero-exploration eval games per generation.
- Stderr contained only the known PyTorch nested-tensor warning. There were no
  trajectory or evaluation errors or timeouts.
- The outcome-only A/B control, SLURM job `74767`, was still running when this
  arm was inspected.

Eval results:

| Generation | Epsilon used for training window | Eval wins / games | Win rate |
| ---: | ---: | ---: | ---: |
| 1 | 1.00 | 51 / 200 | 0.255 |
| 2 | 0.55 | 44 / 200 | 0.220 |
| 3 | 0.10 | 45 / 200 | 0.225 |

Training-data and reward diagnostics:

- The fresh rule-vs-rule bootstrap reached 446 / 1000 wins, 550 losses, and
  4 draws, for a `0.446` baseline win rate.
- Epsilon-vs-rule windows reached 53 / 1000, 78 / 1000, and 221 / 1000 wins at
  epsilon `1.00`, `0.55`, and `0.10`, respectively. The generation-3 training
  window at 221 / 1000 is essentially identical to the generation-2
  zero-exploration checkpoint at 44 / 200, so the apparent trajectory gain is
  explained by lower exploration rather than by a successful PPO update.
- Every fractional reward used the immediate `post-action` board: 39,457,
  43,869, and 59,655 shaped steps across generations 1-3.
- Average fractional-prize reward improved as exploration decayed:
  `-0.01341`, `-0.00794`, then `+0.00079`; the rule bootstrap average was
  `+0.00556`.
- Average total tactical reward similarly moved from `-0.01069` to `+0.00109`
  to `+0.01351`, close to the rule-bootstrap value of `+0.01626` by
  generation 3.
- Generation-3 epsilon behavior still over-selected tactical actions relative
  to the rule bootstrap: attack-taken rate `0.6330` versus `0.2607`, and
  attach-taken rate `0.7579` versus `0.4131`.
- PPO consumed 121,781, 126,193, and 141,979 examples. Mean advantage remained
  negative in all three updates: `-0.2575`, `-0.0727`, and `-0.0767`.

Training-pipeline diagnostic:

- The mixed curriculum sets `REQUIRE_ON_POLICY=0` and sends both retained rule
  demonstrations and epsilon trajectories through the PPO loss.
- Rule-agent records have `policy_on_policy=false` and default
  `policy_logprob=0.0`, but `_train_ppo_batch(...)` still computes the PPO ratio
  `exp(current_logprob - old_logprob)` for them. This is not a valid PPO update
  for the rule half of the data, and it is not behavior cloning either.
- This provides a strong explanation for why the checkpoint does not preserve
  the 0.446 rule policy despite seeing the rule bootstrap every generation.
  The rule demonstrations should use a supervised policy loss, while PPO
  should use only valid model-generated on-policy records.

Conclusion and next step:

- Terminal outcome reward materially improves over the dense-only post-action
  fractional run (`51 / 200` versus `13 / 200` at generation 1), but this arm
  remains far below the rule-vs-rule baseline and shows no improvement after
  generation 1. It is not promotable.
- Wait for outcome-only job `74767` before attributing any benefit to
  fractional shaping. After that A/B result, fix the mixed BC/PPO objective
  separation before scaling another curriculum.

## 2026-07-22 - Terminal-Outcome Reward A/B Completed

ERAWAN result:

- Outcome-only control job `74767`, run name
  `phase5_dragapult_vs_lucario_outcome1_none_diag`, completed successfully in
  `01:43:10` with exit code `0`.
- Its configuration matched job `74766` except that tactical reward mode was
  `none`; terminal outcome reward scale remained `1.0`.
- Stderr contained only the known PyTorch nested-tensor warning. No trajectory
  or evaluation errors or timeouts occurred.

Eval A/B:

| Generation | Outcome + post-action fractional | Outcome only | Fractional delta |
| ---: | ---: | ---: | ---: |
| 1 | 51 / 200 (`0.255`) | 12 / 200 (`0.060`) | +39 wins |
| 2 | 44 / 200 (`0.220`) | 47 / 200 (`0.235`) | -3 wins |
| 3 | 45 / 200 (`0.225`) | 19 / 200 (`0.095`) | +26 wins |
| Total | 140 / 600 (`0.233`) | 78 / 600 (`0.130`) | +62 wins |

- Fresh rule-vs-rule baselines were comparable: 446 / 1000 in the fractional
  arm and 431 / 1000 in the outcome-only arm. Neither learned policy approached
  rule strength; the best checkpoint in this A/B was fractional generation 1
  at only `0.255`.
- Fractional shaping was beneficial inside this flawed training recipe: it
  improved the three-generation aggregate by 10.3 percentage points and
  stabilized generations 2-3 around `0.22` instead of the control's
  generation-3 collapse.
- The result does not establish a promotable reward weight because each arm is
  a single stochastic training run and all six checkpoints remain far below
  baseline.

Behavior diagnostic:

- At epsilon `1.0`, both arms began with nearly identical action behavior:
  attack-taken rates around `0.178-0.179` and attach-taken rates around
  `0.403-0.410`.
- By generation 3, the outcome-only arm's attack-taken rate collapsed to
  `0.0757` while its attach-taken rate rose to `0.7158`. It generated 80,048
  controlled-decision steps over 1000 games and evaluated at 19 / 200.
- The fractional arm instead reached attack-taken rate `0.6330`, attach-taken
  rate `0.7579`, 59,655 controlled-decision steps, and 45 / 200 eval wins.
- Fractional prize progress therefore supplies a useful anti-stalling signal:
  terminal outcome alone does not give the current trainer enough credit to
  preserve attacks. However, the fractional arm can still over-favor attacks
  and remains much weaker than the rule policy.

Pipeline conclusion:

- Do not queue another long curriculum with the current trainer. Reward tuning
  is no longer the primary blocker.
- Separate the mixed update into two mathematically valid objectives:
  - supervised behavior-cloning cross-entropy for retained rule demonstrations;
  - PPO only for records with valid model-policy log probabilities and
    `policy_on_policy=true`.
- Mix or interleave the two losses by examples rather than streaming the full
  rule dataset followed by the full epsilon dataset. The present order makes
  the final optimizer updates disproportionately reflect the epsilon window.
- Keep terminal outcome reward and retain post-action fractional shaping as an
  auxiliary PPO reward for the first corrected experiment, with an outcome-only
  corrected control before selecting the reward weight.

## 2026-07-22 - Corrected Balanced BC + On-Policy PPO Pipeline

Implementation:

- Added `Phase5EpsilonMixturePolicyAgent`. Its behavior distribution is
  `(1 - epsilon) * softmax(neural_logits / temperature) + epsilon * Uniform`,
  sampled without replacement for multi-action selections. It records the
  exact mixture log probability, epsilon, and temperature needed to recompute
  PPO ratios.
- Kept the existing hard `phase5-epsilon` agent for reproducibility. Hard
  epsilon-greedy is intentionally not accepted by the corrected PPO trainer
  because its selected-action probability is not a differentiable function of
  the neural logits.
- Added `rl-train-phase5-bc-ppo`. It applies supervised behavior cloning to
  rule demonstrations and clipped PPO policy/value loss only to records with
  `policy_on_policy=true` and mode `sample` or `epsilon_mixture`.
- Every optimizer step contains equal rule and on-policy example counts. The
  shorter input is cycled, and the report records available/used counts, reuse
  factors, rejected off-policy/mode-invalid/nonfinite rows, component losses,
  PPO ratio, and clip fraction.
- Added deterministic scratch-checkpoint seeding plus per-game exploration
  seeds. A base policy seed is offset by absolute game index so A/B arms can
  share random streams without repeating one action sequence in every game.
- Added
  `scripts/slurm/phase5_one_deck_public_bc_ppo_curriculum.sbatch` as a new,
  non-destructive workflow. It retains the shared rule bootstrap, deletes only
  a generation's model-policy JSONL after a successful update, evaluates with
  zero exploration, and saves at most one win and one loss replay.

Validation:

- `py_compile` passed for the modified agent, workflow, public-opponent,
  trainer, CLI, and test modules.
- `tests.test_rl_phase5_symbolic_training` passed 9 tests with 3 Torch-only
  tests skipped locally; `tests.test_public_agents` and
  `tests.test_phase5_full_agent_scaffolds` each passed 9 tests.
- Local PyTorch is unavailable and Windows has no Bash runtime. The exact
  trainer and SBATCH path therefore require a small ERAWAN SLURM smoke before
  the two full corrected A/B jobs are submitted.

Next step:

- Run the documented one-generation, four-train-game/four-eval-game ERAWAN
  smoke against the retained job-74766 rule bootstrap.
- If it writes a checkpoint and reports valid PPO rows with an initial ratio
  near `1.0`, submit matched three-generation corrected arms: terminal outcome
  plus post-action fractional weight `0.25`, and terminal outcome only.

## 2026-07-22 - Corrected BC + PPO Smoke Passed And A/B Submitted

ERAWAN smoke:

- SLURM job `74786`, run name
  `phase5_dragapult_vs_lucario_corrected_bcppo_smoke`, completed in `00:13:44`
  with exit code `0`.
- The shared bootstrap scan found 82,419 steps and 82,324 valid rule-BC
  examples. The four-game mixture-policy window produced 224 / 224 valid PPO
  examples with zero no-target, off-policy, unsupported-mode, or nonfinite
  skips.
- The smoke intentionally balanced against the full retained bootstrap, so its
  224 PPO examples were reused `367.52x` across 1,287 optimizer steps. Its high
  average clip fraction (`0.8053`) is a tiny-window stress artifact, not a
  training result to promote.
- The generation-1 checkpoint was written successfully. The four-game
  zero-exploration eval was 3 wins and 1 loss with no errors/timeouts; this is
  execution validation only.
- Cleanup behaved as intended: the consumed model-policy JSONL was deleted,
  while the shared rule bootstrap and the 2,189,354-byte checkpoint remain.
  Stderr contained only the known PyTorch nested-tensor warning.

Corrected A/B submission:

- Submitted SLURM job `74791`, run name
  `phase5_dragapult_vs_lucario_corrected_bcppo_frac025_diag`, for terminal
  outcome plus post-action `basic-fractional-prize` shaping at weight `0.25`.
- Submitted SLURM job `74792`, run name
  `phase5_dragapult_vs_lucario_corrected_bcppo_outcome_only_diag`, for terminal
  outcome only.
- Both jobs use three generations, 1,000 training games and 200 deterministic
  eval games per generation, epsilon `1.0 -> 0.55 -> 0.10`, the same retained
  job-74766 rule bootstrap, `INIT_SEED=20260722`, and
  `POLICY_SEED=20260722`. Both were confirmed running after submission.
- The remote shell mangled only the first attempt to print the captured IDs;
  SLURM accounting and each log header confirmed that exactly two jobs were
  submitted and mapped as above. No resubmission occurred.

Next step:

- Wait for `74791` and `74792`. Inspect all three `bc_ppo_report.json`,
  epsilon-mixture trajectory reports, deterministic eval JSON/Markdown,
  statuses, and stdout/stderr files. Compare eval win rate and attack/attach/END
  behavior against the flawed 74766/74767 A/B and the 0.431-0.446 rule
  baseline before deciding whether to extend generations.

## 2026-07-22 - Corrected BC + PPO A/B Completed

ERAWAN completion:

- Fractional job `74791` completed in `02:27:42`; outcome-only job `74792`
  completed in `02:24:15`. Both exited `0` with no gameplay errors or timeouts.
- Stderr for both jobs contained only the known PyTorch nested-tensor warning.
- Downloaded report-only archive
  `phase5_corrected_bcppo_74791_74792_reports.tar.gz` contains all six update
  reports, six trajectory reports, six eval JSON/Markdown/status groups, saved
  win/loss replays, and both SLURM logs. Checkpoints remain on ERAWAN and each
  consumed model-policy JSONL was deleted after its successful update.

Deterministic eval against rule Mega Lucario ex:

| Generation | Corrected outcome + fractional | Corrected outcome only | Old fractional 74766 | Old outcome 74767 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83 / 200 (`0.415`) | 84 / 200 (`0.420`) | 51 / 200 (`0.255`) | 12 / 200 (`0.060`) |
| 2 | 75 / 200 (`0.375`) | 88 / 200 (`0.440`) | 44 / 200 (`0.220`) | 47 / 200 (`0.235`) |
| 3 | 90 / 200 (`0.450`) | 82 / 200 (`0.410`) | 45 / 200 (`0.225`) | 19 / 200 (`0.095`) |
| Total | 248 / 600 (`0.413`) | 254 / 600 (`0.423`) | 140 / 600 (`0.233`) | 78 / 600 (`0.130`) |

- Correct objective separation removed the old collapse. Relative to the old
  trainer, the aggregate improved by 18.0 percentage points for fractional
  shaping and 29.3 points for outcome-only.
- Neither corrected arm passed the 0.50 gate and neither improved monotonically
  across generations. The best checkpoint was fractional generation 3 at
  `0.450`, statistically indistinguishable from the historical 0.431-0.446
  rule baselines.
- Fractional versus outcome-only is also inconclusive: aggregate 0.413 versus
  0.423 has an approximate unpaired two-sided `p=0.73`; the generation-3
  0.450 versus 0.410 difference has `p=0.42`. Do not select a shaping arm from
  these differences.

Training diagnostics:

- Every generation used the same 82,324 valid rule-BC examples. PPO examples
  were 41,016 / 58,289 / 80,219 for fractional and
  39,189 / 59,543 / 80,275 for outcome-only. Every PPO row was accepted with
  zero no-target, off-policy, unsupported-mode, or nonfinite skips.
- Rule BC reached approximately 0.997 accuracy in generation 1 and 1.000 in
  generations 2-3. Average BC loss fell from about `0.149` to below `0.0005`
  in generation 2 and below `0.000013` in generation 3.
- Average PPO ratios stayed essentially `1.0`; clip fraction was zero except
  for roughly `0.0003-0.0004` in generation 2. PPO policy-loss scalars were
  near zero. This is consistent with small policy movement under a full 50/50
  rule anchor, not evidence of a broken likelihood ratio.
- The arms had nearly identical action behavior. By generation 3, attack-taken
  rates were `0.2340` fractional and `0.2357` outcome-only; attach-taken rates
  were `0.4193` and `0.4115`; END rates were `0.0301` and `0.0306`.
- Fractional training-window wins rose from 75 to 339 / 1000 as epsilon decayed;
  outcome-only rose from 61 to 304 / 1000. Because zero-exploration eval did
  not improve monotonically, most of this rise is exploration decay rather
  than learned strength.

Conclusion:

- The corrected pipeline successfully preserves/clones rule behavior, but the
  repeated 50/50 full rule bootstrap pins the policy near the teacher. This
  experiment does not show useful online PPO improvement beyond behavior
  cloning.
- At generation-1 epsilon `1.0`, the behavior policy is exactly uniform:
  `pi_mix = Uniform`. Its derivative with respect to neural logits is zero, so
  generation 1 cannot produce a PPO policy-gradient update; it trains only BC
  and value/shared features. Future differentiable exploration must start
  below 1.0 or use a high-temperature neural policy.
- Before the next long run, change the schedule to BC pretraining once at
  generation 0, then PPO-dominant online updates with either no retained BC or
  a small anti-forgetting anchor. Add per-objective gradient norms/cosine
  diagnostics so BC, PPO policy, value, and entropy influence are measurable.
- Submitted matched 200-game rule-vs-rule baseline SLURM job `74836` using
  sample Dragapult against sample Lucario. Use it to finalize whether the 0.450
  checkpoint exceeds the teacher on the exact eval budget; do not infer that
  from the historical 1,000-game baselines alone.
- Baseline job `74836` failed before gameplay in three seconds because
  `rl-evaluate-phase5-public-agents` supported the rule agent internally but
  omitted `rule` from its CLI `--agent` choices. Added that missing choice and
  a parser regression test; no completed training/eval artifacts were affected.
- Retry job `74837` also failed before gameplay in three seconds: the public
  evaluator still validated the SLURM wrapper's default specialist directory
  even when `--agent rule` did not need a checkpoint. Rule evaluation now
  ignores model/specialist arguments before checkpoint validation, with a
  regression test that supplies a nonexistent specialist directory and proves
  dispatch reaches the evaluator.
- Matched baseline retry `74838` completed successfully in `00:01:31` with
  exit code `0`: rule Dragapult won 87 / 200 (`0.435`) against rule Mega
  Lucario, with zero draws, timeouts, or errors and empty stderr.
- Fractional generation 3 was 90 / 200, only three wins (+1.5 points) above
  the matched teacher. Under an unpaired two-proportion approximation this is
  not significant (`p` approximately `0.76`). Outcome-only generation 2 was
  88 / 200, one win above the teacher (`p` approximately `0.92`). No learned
  checkpoint has demonstrated strength above the behavior-cloning teacher.
- Downloaded the baseline JSON, Markdown, status, and SLURM logs into the
  untracked corrected-A/B analysis bundle. Keep canonical reports on ERAWAN;
  do not add downloaded copies under tracked `reports/` paths.
