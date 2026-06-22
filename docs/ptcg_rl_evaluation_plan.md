# Pokémon TCG AI Battle — Stage-by-Stage Evaluation Plan for Codex

**Purpose:** This file records how to evaluate the model at each training stage.  
**Context:** This evaluation plan is designed for the advanced training pipeline using entity Transformer policy/value model, neural belief model, auxiliary tactical heads, one-turn root search, search distillation, and PPO self-play.

---

## 1. Evaluation philosophy

Every stage should be evaluated with three layers:

```text
1. Offline metrics
   Does the model learn the intended labels?

2. Controlled battle metrics
   Does it beat fixed agents under fixed deck matchups?

3. Diagnostic metrics
   Why did it win or lose? Is it improving for the right reason?
```

Do **not** rely on training loss alone. For this project, the real metric is battle performance across the 13-deck matrix.

---

## 2. Global evaluation setup

Use the same evaluation framework for all stages.

### 2.1 Evaluation opponent pool

```text
- 4 specialized benchmark agents
- 1 general low-level rule-based agent
- random/legal baseline
- previous neural checkpoints
- current neural model without search
- current neural model with search
```

### 2.2 Deck pool

```text
4 benchmark decks
+ 9 high-level tournament decks
= 13 total decks
```

### 2.3 Deck matrix

For each checkpoint, evaluate:

```text
my_deck × opponent_deck × player_order
```

Full matrix:

```text
13 × 13 × 2 = 338 conditions
```

### 2.4 Game count recommendations

```text
Quick check:
  1–3 games per condition

Promotion check:
  10 games per condition

Serious final check:
  30+ games per condition
```

### 2.5 Core battle metrics

Track these for every evaluation:

```text
win rate
average prize difference
average game length
timeout rate
search time per decision
illegal/error rate
deck-specific win rate
opponent-specific win rate
player-0 vs player-1 win rate
```

---

## 3. Stage 0 — Data generation and logging evaluation

Before training, verify that the dataset is valid.

### 3.1 What to check

```text
Can games finish?
Are legal actions stored correctly?
Are selected option indices valid?
Are damage/prize changes parsed correctly?
Are hidden labels saved only for training?
Are deck IDs correct?
Are agent IDs correct?
Are all 13 decks represented?
Are both player orders represented?
```

### 3.2 Metrics

```text
game completion rate
invalid action rate
crash rate
timeout rate
average turns per game
average decision points per game
action-type distribution
deck-pair coverage
player-order coverage
combat-summary parse success rate
```

### 3.3 Pass condition

```text
game completion rate ≥ 99%
invalid action rate = 0
all 13 decks appear as both player 0 and player 1
all 338 deck/order conditions have samples
damage/prize summaries pass spot checks
hidden labels are not exposed to inference code
```

### 3.4 Debugging requirement

Use the 1024×1024 image renderer heavily here. Randomly inspect 20–50 saved turns.

For selected turns, save:

```text
state PNG
state JSON
legal actions JSON
chosen action
next state JSON
combat summary
```

---

## 4. Stage 1 — Supervised imitation policy evaluation

This stage checks whether the model can imitate the rule-based agents.

### 4.1 Offline metrics

```text
top-1 legal-action accuracy
top-3 legal-action accuracy
action-type accuracy
cross-entropy loss
accuracy by deck
accuracy by teacher agent
accuracy by game phase
accuracy by action type
```

Top-3 accuracy is important because multiple Pokémon TCG actions may be nearly equivalent.

### 4.2 Battle metrics

Evaluate:

```text
neural policy without search
vs specialized benchmark agents
vs general rule-based agent
vs random/legal baseline
```

Use all 13 decks where possible.

### 4.3 Expected result

At this stage, the neural policy should:

```text
beat random/legal baseline reliably
approach the general rule-based agent
imitate specialized agents better on the 4 benchmark decks than on the 9 tournament decks
avoid invalid actions completely
```

### 4.4 Pass condition

Suggested threshold:

```text
top-1 accuracy ≥ 40–50%
top-3 accuracy ≥ 70%
beats random/legal baseline ≥ 80%
competitive with general rule-based agent
invalid action rate = 0
```

Do not expect it to beat specialized benchmark agents yet.

---

## 5. Stage 2 — Value and Q model evaluation

This stage checks whether the model can distinguish winning and losing positions.

### 5.1 Offline value metrics

```text
value MSE / Huber loss
win/loss AUC
Spearman correlation between V(state) and final result
calibration by predicted win probability
value error by game phase
value error by deck
value error by prize count
```

Evaluate value separately for:

```text
early game
mid game
late game
near-win positions
near-loss positions
```

Late-game value should become accurate first. Early-game value will be noisy.

### 5.2 Q/action-value metrics

```text
Q loss on chosen actions
correlation between Q(state, action) and final return
Q ranking accuracy when multiple actions have rollout/search labels
action-value calibration by action type
```

### 5.3 Battle comparison

Compare:

```text
A. policy only
B. policy + value tie-break / reranking
```

### 5.4 Pass condition

```text
value AUC > 0.65 initially
late-game value AUC > 0.80
value is monotonic with prize advantage on average
policy + value does not reduce battle win rate
```

Also compare value model against simple baselines:

```text
prize-count baseline
remaining-HP baseline
board-size baseline
```

The learned value model should beat these simple baselines.

---

## 6. Stage 3 — Auxiliary tactical head evaluation

This stage checks whether the model understands immediate tactical consequences.

### 6.1 Actual damage head

Metrics:

```text
damage MAE
damage RMSE
damage error by attack type
damage error by target type
damage error when prevention/reduction occurs
damage error on multi-prize Pokémon
```

### 6.2 KO head

Metrics:

```text
KO accuracy
KO precision
KO recall
KO AUC
false-positive KO rate
false-negative KO rate
```

### 6.3 Prize head

Metrics:

```text
prize gain MAE
exact prize gain accuracy
prize gain calibration
multi-prize target accuracy
```

### 6.4 Damage prevention head

Metrics:

```text
prevention AUC
prevention precision
prevention recall
false-negative rate for prevented attacks
```

False negatives are especially dangerous. If the model thinks damage will go through when it is prevented, it will make bad attacks.

### 6.5 Turn-end head

Metrics:

```text
turn-end prediction accuracy
turn-end false positive rate
turn-end false negative rate
```

### 6.6 Resource-loss head

Metrics:

```text
resource-loss MAE
correlation with hand/deck/discard changes
high-resource-loss detection accuracy
```

### 6.7 Self-risk head

Metrics:

```text
opponent next-turn prize prediction MAE
counter-KO prediction AUC
self-risk calibration
```

### 6.8 Pass condition

Suggested thresholds:

```text
KO AUC ≥ 0.80
damage-prevention AUC ≥ 0.75
turn-end prediction accuracy ≥ 95%
prize gain MAE reasonably low
damage MAE improves over nominal-damage baseline
```

### 6.9 Tactical test cases

Codex should create a small controlled tactical test suite:

```text
attack into normal target
attack into damage prevention
attack with damage reduction
attack for exact KO
attack with overkill
attack into multi-prize Pokémon
attack when weakness/resistance changes result
attack after temporary prevention effect
```

Each case should verify that predicted damage, KO probability, prize gain, and prevention probability are sensible.

---

## 7. Stage 4 — Neural belief model evaluation

This stage evaluates hidden-card prediction.

### 7.1 Offline metrics

```text
top-k hidden-card recall
negative log likelihood
Brier score
expected calibration error
own-prize prediction accuracy before first search
opponent-hand prediction accuracy after revealed actions
opponent-deck prediction accuracy by archetype
opponent-prize prediction accuracy
```

### 7.2 Own prize evaluation

Before first deck search:

```text
my prize distribution is probabilistic
```

After first deck search:

```text
own prize deduction should be exact
```

### 7.3 Masking/card-accounting metrics

```text
impossible-card probability mass
deck-count consistency
hand-count consistency
prize-count consistency
visible-card exclusion correctness
```

### 7.4 Pass condition

```text
own-prize deduction after first deck search = 100% exact
belief model beats uniform baseline
belief calibration improves after revealed actions
impossible cards always have probability zero after masking
count constraints are always satisfied
```

The most important test is not neural accuracy. It is masking and card-accounting correctness.

---

## 8. Stage 5 — One-turn root search evaluation

This stage checks whether search improves the policy.

### 8.1 Compare four agents

```text
A. direct policy only
B. direct policy + value
C. policy + one-turn root search
D. policy + one-turn root search + belief sampling
```

### 8.2 Metrics

```text
win rate improvement over direct policy
average rollout score of chosen action
search time per decision
timeout rate
average top-K candidate count
percentage of turns where search changes policy action
win rate when search changes action
average damage/prize reward of search-selected action
average value of end state
```

### 8.3 Critical diagnostic

Measure:

```text
When search disagrees with direct policy, does search win more often?
```

This is one of the most important search diagnostics.

### 8.4 Pass condition

```text
search improves win rate by at least 2–5 percentage points
timeout rate remains acceptable
search-changed decisions have positive expected value
policy + search beats policy only in the 13-deck league
```

### 8.5 Failure analysis

For failed search decisions, save:

```text
state image
state JSON
legal actions
policy scores
rollout scores
chosen action
actual outcome
combat summary
```

Review whether the failure came from:

```text
bad value estimate
bad rollout policy
bad belief sample
bad damage reward
bad turn-boundary detection
timeout/cutoff
```

---

## 9. Stage 6 — Search distillation evaluation

This stage checks whether direct policy absorbs search improvements.

### 9.1 Offline metrics

```text
KL divergence to search distribution
top-1 agreement with search best action
top-3 agreement with search top actions
cross-entropy to search-selected action
rank correlation between policy logits and rollout scores
```

### 9.2 Battle comparison

Compare:

```text
old direct policy
new distilled direct policy
old policy + search
new policy + search
```

Desired ladder:

```text
new direct policy > old direct policy
new direct policy closer to old policy + search
new policy + search > old policy + search
```

### 9.3 Pass condition

```text
distilled direct policy improves win rate
search still adds improvement after distillation
no deck-specific collapse
no timeout increase
```

If search no longer improves the distilled policy, interpret carefully:

```text
Good:
  policy absorbed search behavior

Bad:
  search labels were weak, overfit, or too deterministic
```

Use league results and ablation results to distinguish.

---

## 10. Stage 7 — PPO self-play evaluation

This stage checks whether reinforcement learning improves beyond imitation/search.

### 10.1 PPO training metrics

```text
episode return
policy loss
value loss
entropy
KL divergence from old policy
clip fraction
explained variance
average advantage
average reward by component
```

### 10.2 PPO collapse indicators

Watch for:

```text
entropy drops too fast
KL spikes
value loss explodes
win rate improves against self but drops against benchmarks
search policy becomes slower or unstable
deck-specific collapse
```

### 10.3 Battle evaluation

Run a fixed league after each PPO checkpoint:

```text
current PPO checkpoint
vs benchmark agents
vs general agent
vs previous neural checkpoints
vs current best with search
```

### 10.4 Pass condition

```text
league win rate improves
no overfitting to current opponent pool
timeout rate remains acceptable
no benchmark deck collapses
illegal/error rate = 0
```

### 10.5 Promotion rule

Promote checkpoint if:

```text
overall win rate improves by ≥ 2 percentage points
no benchmark deck drops by > 5 percentage points
timeout rate remains below threshold
illegal/error rate = 0
value calibration does not collapse
```

---

## 11. Stage 8 — Final inference agent evaluation

Final inference agent:

```text
entity Transformer policy/value model
+ neural belief model
+ one-turn root search
+ debug renderer disabled by default
```

### 11.1 Final ablation ladder

Evaluate these agents:

```text
general rule-based agent
supervised neural policy
neural policy + value
neural policy + belief
neural policy + root search
search-distilled policy
PPO policy
PPO policy + root search
```

The ladder should be mostly monotonic. If a component does not improve performance under time limits, disable it or revise it.

### 11.2 Final battle matrix

Use:

```text
13 decks
× all evaluation opponents
× both player orders
× enough games for stable estimates
```

Recommended final count:

```text
30+ games per deck/order/opponent condition
```

### 11.3 Final metrics

```text
overall win rate
win rate by deck
win rate by opponent
win rate by player order
average prize difference
timeout rate
search time distribution
illegal/error rate
value calibration
belief calibration
damage/KO/prize prediction quality
search disagreement success rate
```

---

## 12. Evaluation report format

For every checkpoint, generate a markdown report:

```text
reports/
  eval_sup_v1.md
  eval_belief_v1.md
  eval_search_v1.md
  eval_distill_v1.md
  eval_ppo_v1.md
```

Each report should include:

```text
1. checkpoint name
2. training data used
3. model config
4. feature flags enabled
5. offline metrics
6. tactical metrics
7. belief metrics if applicable
8. battle league table
9. deck-by-deck win rate
10. opponent-by-opponent win rate
11. player-order win rate
12. timeout/error summary
13. top failure cases
14. promotion recommendation
```

Also output machine-readable files:

```text
reports/
  league_sup_v1.csv
  deck_matrix_sup_v1.csv
  action_metrics_sup_v1.json
  tactical_metrics_sup_v1.json
  belief_metrics_sup_v1.json
  search_ablation_sup_v1.json
```

---

## 13. Standard evaluation commands

Codex should implement these commands.

### 13.1 Offline evaluation

```bash
python scripts/eval_offline.py \
  --model checkpoints/sup_v1/best.pt \
  --data data/validation_v1 \
  --out reports/offline_sup_v1.json
```

### 13.2 Tactical auxiliary evaluation

```bash
python scripts/eval_tactical.py \
  --model checkpoints/sup_v1/best.pt \
  --data data/validation_v1 \
  --out reports/tactical_sup_v1.json
```

### 13.3 Belief evaluation

```bash
python scripts/eval_belief.py \
  --belief checkpoints/belief_v1/best.pt \
  --data data/validation_v1 \
  --out reports/belief_v1.json
```

### 13.4 League evaluation

```bash
python scripts/evaluate_league.py \
  --model checkpoints/search_distill_v1/best.pt \
  --decks configs/decks/all_13.yaml \
  --opponents configs/agents/eval_pool.yaml \
  --games-per-condition 10 \
  --out reports/league_search_distill_v1.md
```

### 13.5 Search ablation

```bash
python scripts/evaluate_search_ablation.py \
  --model checkpoints/search_distill_v1/best.pt \
  --belief checkpoints/belief_v1/best.pt \
  --decks configs/decks/all_13.yaml \
  --games-per-condition 5 \
  --out reports/search_ablation_v1.md
```

### 13.6 PPO checkpoint evaluation

```bash
python scripts/evaluate_league.py \
  --model checkpoints/ppo_v1/best.pt \
  --belief checkpoints/belief_v1/best.pt \
  --enable-search \
  --decks configs/decks/all_13.yaml \
  --opponents configs/agents/eval_pool.yaml \
  --games-per-condition 10 \
  --out reports/league_ppo_v1_search.md
```

---

## 14. Recommended stage gates

| Stage | Must pass before moving on |
|---|---|
| Data generation | Games finish, valid actions, all decks covered |
| Imitation | Beats random, reasonable top-3 accuracy |
| Value/Q | Value predicts win/loss better than prize-count baseline |
| Auxiliary heads | KO/prize/damage predictions usable |
| Belief | Own-prize deduction exact; belief beats uniform |
| Root search | Improves direct policy without timeout |
| Distillation | Direct policy improves after search labels |
| PPO | League win rate improves without benchmark collapse |
| Final agent | Best ablation under time limit |

---

## 15. Promotion checklist

Before promoting any checkpoint:

```text
1. Overall win rate improves over current best.
2. No catastrophic deck-specific regression.
3. No benchmark deck drops by more than allowed threshold.
4. Timeout rate remains acceptable.
5. Illegal/error rate is zero.
6. Search time remains within budget.
7. Value calibration does not collapse.
8. Belief model obeys masks/count constraints.
9. Tactical heads remain stable.
10. Debug traces do not show systematic obvious misplays.
```

Suggested threshold:

```text
overall win rate improvement ≥ 2 percentage points
and
no individual benchmark deck drops by more than 5 percentage points
```

---

## 16. Most important final metric

For the contest, the final metric is:

```text
win rate of final inference agent
= entity Transformer policy/value
+ neural belief model
+ one-turn root search
```

evaluated over:

```text
13 decks
× benchmark agents
× general agent
× previous neural checkpoints
× both player orders
```

But for engineering, the most important diagnostic is:

```text
Does each added component improve battle win rate over the previous component?
```

The ablation ladder is mandatory:

```text
general rule-based agent
→ supervised neural policy
→ neural policy + value
→ neural policy + belief
→ neural policy + root search
→ search-distilled policy
→ PPO policy
→ PPO policy + root search
```

Only keep a component enabled in the final agent if it improves the ladder under the time limit.

---

## 17. Codex deliverables

Codex should implement:

```text
scripts/eval_offline.py
scripts/eval_tactical.py
scripts/eval_belief.py
scripts/evaluate_league.py
scripts/evaluate_search_ablation.py
scripts/summarize_eval_reports.py
```

Codex should produce:

```text
offline metric JSON
tactical metric JSON
belief metric JSON
league CSV
deck matrix CSV
search ablation JSON
markdown summary report
debug traces for selected failures
```

The evaluation system should be runnable independently of training so that every checkpoint can be compared consistently.
