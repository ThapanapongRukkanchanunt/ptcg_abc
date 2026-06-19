# Phase 4 RL Plan: Rule-Guided Hybrid Agent

## Summary

Build a Kaggle-first Phase 4 reinforcement learning workflow across all 9 required
decks, targeting a 50% overall win rate on the existing 360-game benchmark. Use
PyTorch for training on CMU ERAWAN via SLURM, but package Kaggle submissions with
exported lightweight inference plus the current rule agent as fallback. ERAWAN
access details are documented at https://hpc.cmu.ac.th/th/home/access.

Adapt the idea from "Rule-Guided Reinforcement Learning Policy Evaluation and
Improvement" by starting with rule-guided evaluation/improvement rather than full
rule mining first:

https://arxiv.org/abs/2503.09270

## Key Changes

- Add a new `ptcg_abc.rl` package for featurization, datasets, rewards, PyTorch
  policy/value models, PPO training, guidance rules, and reporting.
- Add a `HybridRlAgent` that scores legal options with the RL model, blends or
  falls back to `RuleBasedAgent`, and never emits illegal selections.
- Add reusable decision records:
  - `DecisionFrame`: normalized observation, select type/context, legal options,
    rule scores, selected indices, and reward metadata.
  - `ActionFrame`: option type, card/attack/area/index metadata, target features,
    rule score, and legal mask.
  - `TrajectoryStep`: decision frame, chosen indices, logprob, value, reward, and
    terminal flags.
- Add CLI commands:
  - `rl-collect-bc`: collect rule-agent demonstrations from all 9 decks vs 4
    benchmark decks.
  - `rl-train-bc`: warm-start an option-ranking actor/value model from rule
    demonstrations.
  - `rl-rollout`: generate on-policy trajectories with rule, RL, or hybrid agents.
  - `rl-train-ppo`: train PPO from rollout JSONL chunks.
  - `rl-evaluate`: run the 9x4 required benchmark for rule, RL, or hybrid agents.
  - `rl-evaluate-guidance`: test individual rule-guidance interventions against an
    unguided checkpoint.
  - `rl-package`: export one Kaggle zip per selected deck using hybrid fallback.
- Add SLURM templates under `scripts/slurm/`:
  - BC collection/training job.
  - PPO rollout job array.
  - PPO update job.
  - Evaluation/package job.
- Add a board-image renderer that converts each current observation into a
  fixed-size image of the game board for model input.
- Update `.gitignore` so `data/datasets/rl/`, `models/rl/`, `experiments/rl/`,
  and generated `submissions/` stay out of git.

## RL Design

- Use a dynamic legal-option scorer, not a fixed action-space model.
- Represent the current game state with two synchronized inputs:
  - A deterministic board image tensor rendered from the observation, showing both
    players' Active, Bench, Stadium, hand/deck/discard/prize counts, visible card
    IDs/names, HP/damage, energies, tools, and special conditions in fixed board
    locations.
  - Structured per-option features for legal action scoring, including option
    type, card/attack/area/index metadata, target features, rule score, and legal
    mask.
- Score each legal option independently against the shared board image and board
  features, then select according to the select context's `minCount`/`maxCount`.
- Include the current rule score and rule rank as model inputs so RL learns when to
  follow or override the rule policy.
- Train in three stages:
  - BC warm start: 20,000 rule-agent games across all 9 decks vs 4 benchmarks,
    alternating player seat.
  - PPO smoke: 360 rollout games and one tiny PPO update to validate the loop.
  - HPC PPO: 10 iterations, each with 10,000 rollout games, resumable by checkpoint.
- Reward defaults:
  - Win/loss/draw: `+1.0 / -1.0 / 0.0`.
  - Prize delta: `+0.2` per prize gained, `-0.15` per opponent prize gained.
  - Step cost: `-0.001`.
  - Illegal/empty-action recovery penalty: `-0.05`, though the agent should avoid
    this through masks.
- Export trained checkpoints to a JSON/NPZ inference format; Kaggle submission
  should not require PyTorch at runtime.

## Rule-Guided LEGIBLE Adaptation

- Implement guidance rules as positive force rules and negative block rules over
  `DecisionFrame`/`ActionFrame`.
- V1 guidance rules are manually seeded from current rule knowledge instead of
  mined:
  - Block obviously bad optional bench/fill/discard choices.
  - Force clear prize-taking attacks when legal.
  - Prefer damage-counter distributions that finish multiple targets.
  - Prefer Munkidori-style counter movement when it creates a KO or removes
    meaningful own damage.
  - Prefer energy attachments that activate known scaling/ability conditions.
- Add metamorphic-style generalizations:
  - Bench-slot permutation: equivalent targets in different bench slots should be
    treated consistently.
  - Same-card-copy permutation: identical cards in hand are interchangeable.
  - Player-perspective normalization: features are always from the acting player's
    perspective.
  - Prize monotonicity: direct prize-taking actions should not become less
    preferred when fewer prizes remain.
  - Energy-deficit monotonicity: an attacker closer to required energy should not
    be scored worse all else equal.
- `rl-evaluate-guidance` evaluates one generalized rule set at a time. Accepted
  rules are those that improve win rate over the same checkpoint by at least +2
  wins over 360 games with no added errors/timeouts.
- Accepted rules become part of the hybrid fallback/guidance set and are logged in
  `reports/phase4_guidance_rules.md`.

## Evaluation Gates

- Smoke gate: tests pass, `rl-collect-bc --games 36`, `rl-train-bc --epochs 1`,
  `rl-rollout --games 36`, and `rl-evaluate --games-per-matchup 1` complete with
  0 errors.
- Promotion gate: hybrid RL reaches at least 180 wins out of 360 games on the
  required 9x4 benchmark.
- Stability gate: a second fresh 360-game run must stay at or above 175 wins with
  0 errors and 0 timeouts.
- Packaging gate: `rl-package` creates one zip per top-performing deck, and each
  zip passes a local 40-step simulator smoke test.

## Test Plan

- Unit tests for featurizer determinism, legal-option masks, multi-select
  `minCount`/`maxCount`, reward calculation, and rule-guidance conflict handling.
- Parity test: PyTorch checkpoint inference and exported JSON/NPZ inference choose
  the same option ordering on fixed decision frames.
- Simulator tests for rule, RL, and hybrid agents over short battles with no
  illegal selections.
- CLI smoke tests using tiny generated datasets and temporary output directories.
- Regression benchmark test that verifies report schema for Phase 4 matches the
  existing Phase 3 benchmark shape.

## Assumptions

- CMU ERAWAN jobs are submitted with SLURM `sbatch`.
- ERAWAN access is through SSH to `erawan.cmu.ac.th` or JupyterLab at
  `https://erawan.cmu.ac.th:8000`, and requires the CMU network or CMU VPN.
- Heavy Phase 4 jobs must not run directly on the login node or notebook kernel;
  use Slurm jobs from the ERAWAN shell/Jupyter terminal.
- Training may depend on PyTorch, but Kaggle inference should remain self-contained
  with rule fallback.
- The current 9-deck by 4-benchmark Phase 3 grid remains the canonical Phase 4
  evaluation target.
- Generated datasets, checkpoints, rollout chunks, and submissions are reproducible
  artifacts, not committed source files.
