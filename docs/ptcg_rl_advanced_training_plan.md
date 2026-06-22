# Pokémon TCG AI Battle — Advanced Training Plan for Codex

**File purpose:** Detailed training implementation plan for Codex.  
**Decision policy:** This plan intentionally chooses the stronger “later” option for major design decisions: entity Transformer, neural belief model, auxiliary tactical heads, search-improved training, policy-pool self-play, and PPO-style reinforcement learning.  
**Assumption:** The project already has the strategy architecture file and the Kaggle sample notebook available as implementation references.

---

## 0. Current assets

The project has these current assets:

```text
Rule-based agents:
  1. Four specialized benchmark agents.
     - Each plays one specific benchmark deck.
     - These are treated as stronger but narrow teachers/evaluation anchors.

  2. One general low-level rule-based agent.
     - Can play any deck.
     - Lower quality but useful for broad coverage.

Decks:
  1. Four benchmark decks.
     - Paired with the four specialized rule-based agents.

  2. Nine additional high-level tournament decks.
     - No specialized rule-based agent yet.
     - Useful for generalization and self-play.

Total deck pool:
  13 decks
```

The training system should exploit these assets in stages:

```text
specialized rule-based agents → high-quality narrow demonstrations
general rule-based agent      → broad low-level demonstrations
13-deck pool                  → generalization and self-play diversity
search-improved policy        → stronger generated labels
policy-pool self-play         → iterative improvement
```

---

## 1. Training objectives

Codex should implement training for the following components.

### 1.1 Main legal-action policy

The policy ranks simulator-provided legal options.

```text
Input:
  GameState
  + BeliefState
  + legal action candidate

Output:
  policy logit / action score for that legal option
```

The model must not output from a fixed global action space. It must score only the legal options provided by the simulator.

```text
legal options = [a1, a2, ..., aN]

model computes:
  score(s, a1)
  score(s, a2)
  ...
  score(s, aN)

selected action:
  argmax_i score(s, ai)
```

### 1.2 State value model

The value model estimates long-term win probability from the current player’s perspective.

```text
Input:
  GameState + BeliefState

Output:
  V(state) in [-1, 1]

Meaning:
  +1 = likely win
   0 = even
  -1 = likely loss
```

This is used for:

```text
1. End-state evaluation after one-turn root search.
2. PPO advantage estimation.
3. Self-play training.
4. League evaluation diagnostics.
```

### 1.3 Q/action-value head

The Q head estimates action-specific value.

```text
Input:
  GameState + BeliefState + legal action candidate

Output:
  Q(state, action)
```

Use it for:

```text
1. Direct action ranking.
2. Search candidate filtering.
3. Offline action-value training.
4. Auxiliary stabilization for PPO.
```

### 1.4 Auxiliary tactical heads

Train tactical prediction heads from simulator outcomes.

Required heads:

```text
actual_damage_head(state, action)
damage_prevention_probability_head(state, action)
KO_probability_head(state, action)
expected_prize_gain_head(state, action)
turn_end_probability_head(state, action)
resource_loss_head(state, action)
self_KO_risk_next_turn_head(state)
```

These heads help the model learn Pokémon-specific tactical consequences without fully hand-coding card effects.

### 1.5 Neural belief model

Use a neural belief model rather than only count-based belief.

The belief model predicts hidden-card distributions.

```text
Input:
  public GameState
  + game log summary
  + opponent observed actions
  + opponent deck/archetype prior if known

Output:
  opponent hand card distribution
  opponent deck card distribution
  opponent prize card distribution
  own prize distribution before deck search
```

After the first own-deck search, own prize belief should become exact through deduction.

### 1.6 Search-imitation policy

Use one-turn root search as a teacher.

```text
current policy + value model + search
→ improved action distribution
→ train policy to imitate search
```

Use soft labels from rollout scores when possible, not only hard argmax labels.

### 1.7 PPO / policy-gradient training

Use PPO after supervised and search-imitation training stabilize the policy.

PPO should operate over legal-action distributions:

```text
π(a | s, legal_options)
```

The policy distribution is a softmax over legal options only.

---

## 2. Chosen architecture: entity Transformer + action scorer

Use the stronger architecture immediately.

### 2.1 Overview

```text
GameState + BeliefState
    → EntityEncoder
    → TransformerEncoder
    → pooled state embedding

LegalAction candidate
    → ActionEncoder
    → action embedding

state embedding + action embedding
    → policy logit
    → Q-value
    → tactical auxiliary predictions

state embedding
    → V(state)
    → self-risk prediction
```

### 2.2 Entity representation

Every visible or inferred object becomes an entity row.

Entity types:

```text
self active Pokémon
opponent active Pokémon
self benched Pokémon
opponent benched Pokémon
self hand card
self discard card
opponent discard card
stadium
attached energy
attached tool
known own prize card
belief summary entity
turn/global entity
```

### 2.3 Entity features

Each entity should include:

```text
card_id embedding
owner embedding
zone embedding
slot embedding
card type embedding
Pokémon type embedding
stage embedding
is_basic
is_evolved
is_ex
is_rule_box
hp_max
damage
remaining_hp
prize_value
attached_energy_by_type
status flags
can_attack_now
can_retreat_now
has_tool
turns_in_play
known_or_belief_flag
belief_probability
```

### 2.4 Global features

Add a global feature vector:

```text
turn number
current player
my prizes remaining
opponent prizes remaining
my deck count
opponent deck count
my hand count
opponent hand count
supporter used flag
energy attached flag
retreat used flag
stadium present flag
game phase estimate
time budget state if available
```

Represent global features as a special `[GLOBAL]` entity or concatenate after Transformer pooling.

### 2.5 Action features

Each legal option candidate should include:

```text
option_index
action_type
source_card_id
source_zone
source_slot
target_card_id
target_owner
target_zone
target_slot
cost_card_ids
selected_indices
attack_id
nominal_damage
energy_cost_satisfied
draw_effect_flag
search_effect_flag
switch_effect_flag
gust_effect_flag
discard_effect_flag
evolve_effect_flag
attach_energy_flag
ability_flag
ends_turn_flag
multi_select_count
```

### 2.6 Model dimensions

Start with:

```python
ModelConfig = {
    "card_embedding_dim": 96,
    "zone_embedding_dim": 16,
    "owner_embedding_dim": 8,
    "type_embedding_dim": 16,
    "entity_hidden_dim": 256,
    "transformer_layers": 4,
    "transformer_heads": 8,
    "transformer_ff_dim": 512,
    "action_hidden_dim": 256,
    "joint_hidden_dim": 384,
    "dropout": 0.10,
}
```

If inference is too slow, reduce to:

```python
ModelConfigFast = {
    "card_embedding_dim": 64,
    "entity_hidden_dim": 192,
    "transformer_layers": 3,
    "transformer_heads": 6,
    "transformer_ff_dim": 384,
    "action_hidden_dim": 192,
    "joint_hidden_dim": 256,
}
```

### 2.7 Model heads

```text
Policy head:
  joint(state_embedding, action_embedding) → policy_logit

Q head:
  joint(state_embedding, action_embedding) → Q(state, action)

Damage head:
  joint(state_embedding, action_embedding) → predicted_actual_damage

Damage prevention head:
  joint(state_embedding, action_embedding) → probability damage prevented

KO head:
  joint(state_embedding, action_embedding) → probability of KO

Prize head:
  joint(state_embedding, action_embedding) → expected prizes gained

End-turn head:
  joint(state_embedding, action_embedding) → probability action ends turn

Resource loss head:
  joint(state_embedding, action_embedding) → resource loss estimate

Value head:
  state_embedding → V(state)

Self-risk head:
  state_embedding → expected opponent prize gain next turn
```

---

## 3. Neural belief model

### 3.1 Purpose

The belief model estimates hidden information and improves search initialization.

Hidden zones:

```text
own prizes before first deck search
own deck before first deck search if order/content unknown
opponent hand
opponent deck
opponent prizes
```

### 3.2 Inputs

```text
public GameState
visible cards
known decklists if available
public logs
opponent actions
opponent revealed cards
turn number
deck/archetype ID if known
```

### 3.3 Outputs

For card vocabulary size `C`:

```text
opp_hand_logits[C]
opp_deck_logits[C]
opp_prize_logits[C]
my_prize_logits[C] before own deck search
```

Convert to distributions constrained by known counts.

### 3.4 Hard constraints

The model output must be corrected by deterministic card accounting.

For each player:

```text
total decklist counts
- visible cards
- known discard
- known attached cards
- known hand if self
= possible hidden cards
```

Apply masking so impossible cards receive probability zero.

### 3.5 Own prize deduction override

When an own deck search reveals remaining deck contents:

```text
known_my_prizes =
  original_decklist
  - visible_own_cards
  - revealed_remaining_deck
```

After this event:

```text
my_prize_distribution = exact known prize cards
```

This deterministic result overrides neural belief.

### 3.6 Training data for belief

Use self-play and simulator-generated full hidden states, but be careful:

```text
Training may use hidden labels.
Inference must not read hidden simulator internals.
```

Training labels:

```text
actual opponent hand contents
actual opponent deck contents
actual opponent prize contents
actual own prize contents before deck search
```

Losses:

```text
multi-label BCE / cross entropy for hidden-card membership
count-constrained distribution loss
KL loss between predicted distribution and actual hidden card counts
```

---

## 4. Data collection

### 4.1 Dataset format

Save every decision point.

```json
{
  "game_id": "...",
  "turn_id": 5,
  "step_id": 17,
  "player_id": 0,
  "agent_name": "specialized_benchmark_ogerpon",
  "deck_id": "ogerpon_box",
  "opponent_deck_id": "charizard_ex",
  "raw_observation": "... optional or compressed ...",
  "game_state_json": "...",
  "belief_state_json": "...",
  "legal_actions_json": "...",
  "chosen_local_index": 3,
  "chosen_option_index": 12,
  "action_type": "attack",
  "reward": 0.0,
  "next_game_state_json": "...",
  "done": false,
  "final_result_from_current_player": 1.0,
  "combat_summary": {
    "actual_damage": 180,
    "damage_prevented": 0,
    "ko_happened": true,
    "prize_gain": 2,
    "turn_ended": true
  },
  "hidden_labels_for_training_only": {
    "opponent_hand": [],
    "opponent_deck": [],
    "opponent_prizes": [],
    "own_prizes": []
  }
}
```

### 4.2 Debug traces

For selected games only:

```text
debug_runs/
  match_000001/
    turn_001.png
    turn_001_state.json
    turn_001_legal_actions.json
    turn_001_policy_scores.json
    turn_001_rollout_scores.json
    turn_001_chosen_action.txt
```

Do not store PNGs in the main replay buffer.

---

## 5. Dataset generation plan

### 5.1 Initial rule-based dataset

Generate demonstrations using:

```text
4 specialized benchmark agents
1 general low-level agent
13 decks
```

Recommended initial game counts:

```text
Specialized mirror/cross games on 4 benchmark decks:
  20,000 games

Specialized vs general on benchmark decks:
  20,000 games

General agent on all 13 decks:
  40,000 games

General vs random/noisy policy on all 13 decks:
  10,000 games

Total initial:
  90,000 games
```

If time is short computationally, start with:

```text
10,000 specialized games
10,000 specialized-vs-general games
20,000 general games
5,000 noisy games

Total minimum:
  45,000 games
```

### 5.2 Deck-pair sampling

Use balanced deck-pair sampling.

```python
for my_deck in deck_pool:
    for opp_deck in deck_pool:
        sample games for both player orders
```

Because 13 decks create:

```text
13 × 13 = 169 deck pairings
338 player-order pairings
```

At minimum, ensure every deck appears as both player 0 and player 1.

### 5.3 Agent assignment

For the four benchmark decks:

```text
Use specialized benchmark agent when available.
Also run general agent for comparison.
```

For the nine tournament decks:

```text
Use general low-level rule-based agent first.
Later replace with neural policy + search.
```

---

## 6. Training phases

## Phase 1: supervised imitation pretraining

### 6.1 Goal

Teach the entity Transformer basic game mechanics and deck-independent sequencing.

### 6.2 Training data

Use rule-based demonstration data.

Prioritize:

```text
specialized agent decisions > general agent decisions > random/noisy decisions
```

### 6.3 Loss

For each state with legal actions:

```text
policy_logits = model(state, legal_actions)
target = chosen action from rule-based agent
L_policy = cross_entropy(policy_logits, target)
```

### 6.4 Weighted imitation

Use source-dependent weights:

```python
source_weight = {
    "specialized_benchmark": 1.00,
    "general_rule_based": 0.60,
    "random_noisy": 0.10,
}
```

Loss:

```text
L_imitation = source_weight * CE(policy_logits, chosen_action)
```

### 6.5 Metrics

```text
top-1 legal-action accuracy
top-3 legal-action accuracy
action-type accuracy
deck-specific accuracy
benchmark-agent imitation accuracy
general-agent imitation accuracy
```

Top-3 accuracy matters because many TCG actions are equivalent or near-equivalent.

---

## Phase 2: value and Q training

### 6.6 Goal

Train the model to evaluate states and state-action pairs.

### 6.7 Value target

For each state:

```text
V_target = final_result_from_current_player
```

Where:

```text
win = +1
loss = -1
draw = 0
```

### 6.8 Q target

For chosen action:

```text
Q_target = discounted return from this state-action
```

Use:

```text
return_t = final_result + shaped_reward_sum_after_t
```

Start with final result only. Add shaped rewards later.

### 6.9 Loss

```text
L_value = Huber(V(state), V_target)
L_Q = Huber(Q(state, chosen_action), Q_target)
```

### 6.10 Metrics

```text
value MSE / Huber loss
value calibration by game phase
win/loss AUC
Spearman correlation between V(state) and final result
Q prediction loss
```

---

## Phase 3: auxiliary tactical training

### 6.11 Goal

Teach tactical consequences.

### 6.12 Labels

From simulator transition and combat logs:

```text
actual_damage
damage_prevented
KO_happened
prize_gain
turn_ended
resource_loss
self_KO_risk_next_turn
```

### 6.13 Losses

```text
L_damage = Huber(predicted_actual_damage, actual_damage)
L_prevent = BCE(predicted_prevention_probability, prevented_label)
L_KO = BCE(predicted_KO_probability, KO_label)
L_prize = Huber(predicted_prize_gain, prize_gain)
L_end = BCE(predicted_turn_end_probability, ended_label)
L_resource = Huber(predicted_resource_loss, resource_loss)
L_self_risk = Huber(predicted_self_risk, opponent_prizes_next_turn)
```

### 6.14 Combined supervised loss

```text
L_total =
    1.00 * L_policy
  + 0.50 * L_value
  + 0.25 * L_Q
  + 0.10 * L_damage
  + 0.10 * L_prevent
  + 0.10 * L_KO
  + 0.20 * L_prize
  + 0.05 * L_end
  + 0.05 * L_resource
  + 0.10 * L_self_risk
```

Tune later based on validation.

---

## Phase 4: neural belief model training

### 6.15 Goal

Train hidden-card prediction.

### 6.16 Labels

Use simulator-internal hidden state labels only during training data generation.

Labels:

```text
opponent hand cards
opponent deck cards
opponent prize cards
own prize cards before deck search
```

### 6.17 Loss

Use masked multi-label/count loss.

```text
L_belief =
    BCE(opp_hand_distribution, actual_opp_hand_counts)
  + BCE(opp_deck_distribution, actual_opp_deck_counts)
  + BCE(opp_prize_distribution, actual_opp_prize_counts)
  + BCE(my_prize_distribution, actual_my_prize_counts)
```

Apply masks so impossible cards are excluded.

### 6.18 Calibration metrics

```text
top-k hidden-card recall
negative log likelihood
Brier score
expected calibration error
own-prize prediction accuracy before first search
opponent-hand prediction accuracy after revealed actions
```

---

## Phase 5: search-improved data generation

### 6.19 Goal

Use one-turn root search to create stronger labels.

### 6.20 Search policy

For each decision point at start of turn:

```text
1. Policy pre-scores all legal options.
2. Select top-K candidates.
3. For each candidate:
     create belief-informed search state
     apply candidate first action
     let policy roll out until turn ends
     score final state.
4. Use rollout scores as improved policy target.
```

### 6.21 Initial search config

```python
RootSearchConfig = {
    "root_top_k": 8,
    "num_belief_samples": 2,
    "max_rollout_actions": 40,
    "rollout_temperature": 0.1,

    "lambda_value": 1.00,
    "lambda_damage": 0.30,
    "lambda_prize": 0.70,
    "lambda_board": 0.10,
    "lambda_hand": 0.05,
    "lambda_resource": 0.10,
    "lambda_self_risk": 0.10,
}
```

If too slow:

```python
RootSearchConfigFast = {
    "root_top_k": 4,
    "num_belief_samples": 1,
    "max_rollout_actions": 30,
}
```

### 6.22 Soft search labels

For top-K candidates:

```text
target_distribution = softmax(rollout_score / temperature)
```

Use KL training:

```text
L_search_distill = KL(search_distribution || policy_distribution)
```

Also store hard best action.

---

## Phase 6: search distillation

### 6.23 Goal

Make the direct policy approximate the stronger search policy.

### 6.24 Training data

Use search-improved games.

Each sample includes:

```text
state
legal actions
policy prescores
rollout scores for top-K
search-selected action
final result
combat summary
```

### 6.25 Loss

```text
L_search =
    KL(search_soft_distribution, policy_distribution)
  + CE(policy_logits, search_best_action)
```

Use a mixture:

```text
L_policy_total =
    0.50 * L_imitation
  + 0.50 * L_search
```

After enough search data:

```text
L_policy_total =
    0.20 * L_imitation
  + 0.80 * L_search
```

---

## Phase 7: PPO self-play

### 6.26 Goal

Improve beyond rule-based and search-distilled behavior.

### 6.27 Opponent pool

Use:

```text
4 specialized benchmark agents
general rule-based agent
previous neural checkpoints
current best neural checkpoint
current policy without search
current policy with search
```

### 6.28 Deck pool

Use all 13 decks.

Sampling:

```text
my_deck ~ uniform over 13 decks
opp_deck ~ uniform over 13 decks
opponent_agent ~ weighted opponent pool
```

Suggested opponent distribution:

```python
opponent_sampling = {
    "specialized_benchmark_agents": 0.25,
    "general_rule_based_agent": 0.15,
    "previous_neural_checkpoints": 0.30,
    "current_best_neural": 0.20,
    "random_or_noisy_agent": 0.10,
}
```

### 6.29 PPO action distribution

At each decision:

```text
legal action logits → softmax over legal actions only
```

Store:

```text
state
legal action features
chosen action index
log_prob
value estimate
reward
done
```

### 6.30 Reward

Use final outcome plus small shaping.

```text
reward =
    terminal_win_loss_reward
  + 0.10 * prize_delta
  + 0.03 * damage_progress
  + 0.02 * board_setup_delta
  - 0.02 * resource_loss
  - 0.03 * self_risk
```

Terminal:

```text
win = +1
loss = -1
draw = 0
```

Keep shaping small to avoid greedy damage-only play.

### 6.31 PPO losses

```text
L_PPO_policy = clipped surrogate objective
L_PPO_value = value loss
L_entropy = entropy bonus over legal actions
L_aux = tactical auxiliary losses
```

Combined:

```text
L =
    L_PPO_policy
  + 0.50 * L_PPO_value
  - 0.01 * entropy
  + 0.10 * L_aux
```

### 6.32 PPO config

```python
PPOConfig = {
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "learning_rate": 3e-5,
    "batch_size": 1024,
    "minibatch_size": 128,
    "epochs_per_update": 3,
    "max_grad_norm": 1.0,
}
```

---

## Phase 8: policy-pool iteration

### 6.33 Iterative loop

```text
repeat:
  1. Generate self-play games with current policy.
  2. Mix opponents from pool.
  3. Train with PPO + supervised/search replay.
  4. Evaluate in league.
  5. Promote checkpoint if better.
  6. Add promoted checkpoint to opponent pool.
```

### 6.34 Replay mixture per training update

```python
training_batch_mix = {
    "rule_based_imitation": 0.15,
    "search_distillation": 0.35,
    "ppo_self_play": 0.40,
    "belief_training": 0.10,
}
```

Adjust after stabilization:

```python
training_batch_mix_late = {
    "rule_based_imitation": 0.05,
    "search_distillation": 0.25,
    "ppo_self_play": 0.60,
    "belief_training": 0.10,
}
```

---

## 7. Evaluation protocol

### 7.1 League opponents

Evaluate every checkpoint against:

```text
4 specialized benchmark agents
general rule-based agent
previous best neural policy
previous best neural policy + search
current policy without search
current policy with search
```

### 7.2 Deck matrix

Use all 13 decks.

Minimum evaluation:

```text
checkpoint deck × opponent deck × both player orders
```

For 13 decks:

```text
13 × 13 × 2 = 338 deck/order conditions
```

Run at least 3 games per condition for quick evaluation:

```text
338 × 3 = 1,014 games
```

Run 10 games per condition for promotion-quality evaluation:

```text
338 × 10 = 3,380 games
```

### 7.3 Metrics

```text
overall win rate
win rate by deck
win rate by opponent
win rate by player order
average prize difference
average game length
timeout rate
search time per decision
policy entropy
damage progress per turn
prize gain per turn
counter-KO rate
resource waste rate
value calibration
belief calibration
```

### 7.4 Promotion rule

Promote a checkpoint only if:

```text
1. Overall win rate improves over current best.
2. No catastrophic deck-specific regression.
3. Timeout rate remains acceptable.
4. Direct policy improves or search policy improves.
5. Value calibration does not collapse.
```

Suggested promotion threshold:

```text
overall win rate improvement ≥ 2 percentage points
and
no individual benchmark deck drops by more than 5 percentage points
```

---

## 8. Time and compute management

Because this plan uses advanced components, Codex should implement switches.

```python
TrainingFeatureFlags = {
    "use_entity_transformer": True,
    "use_neural_belief": True,
    "use_auxiliary_heads": True,
    "use_search_distillation": True,
    "use_ppo": True,
    "use_policy_pool": True,
    "use_debug_renderer": False,
}
```

For emergency fallback:

```python
FastMode = {
    "transformer_layers": 2,
    "root_top_k": 4,
    "num_belief_samples": 1,
    "ppo_epochs_per_update": 1,
    "evaluation_games_per_condition": 1,
}
```

---

## 9. Codex implementation milestones

### Milestone 1: Data schema and log writer

Implement:

```text
GameState serialization
LegalAction serialization
BeliefState serialization
Transition schema
Dataset writer
Dataset loader
```

Deliverables:

```text
data/samples/*.jsonl or parquet
scripts/generate_rule_based_dataset.py
tests/test_dataset_schema.py
```

### Milestone 2: Entity/action encoder

Implement:

```text
EntityEncoder
ActionEncoder
batch collator for variable legal-action counts
masking for legal-action softmax
```

Deliverables:

```text
ptcg_agent/encoding/entity_encoder.py
ptcg_agent/encoding/action_encoder.py
ptcg_agent/training/collate.py
tests/test_batch_collator.py
```

### Milestone 3: Transformer policy/value model

Implement:

```text
EntityTransformer
ActionScorer
ValueHead
QHead
AuxiliaryHeads
```

Deliverables:

```text
ptcg_agent/models/entity_policy_value.py
tests/test_model_forward.py
```

### Milestone 4: Rule-based dataset generation

Implement match runners for:

```text
specialized benchmark agents
general low-level agent
all 13 decks
```

Deliverables:

```text
scripts/generate_rule_based_dataset.py
configs/decks/*.json
configs/agents/*.json
```

### Milestone 5: Supervised training

Implement:

```text
imitation loss
value loss
Q loss
auxiliary tactical losses
checkpointing
validation metrics
```

Deliverables:

```text
scripts/train_supervised.py
configs/train_supervised.yaml
```

### Milestone 6: Neural belief model

Implement:

```text
belief encoder
hidden-card prediction heads
deterministic mask/correction
own-prize deduction override
belief metrics
```

Deliverables:

```text
ptcg_agent/models/belief_model.py
scripts/train_belief.py
tests/test_belief_masks.py
tests/test_prize_deduction.py
```

### Milestone 7: Search-improved dataset generation

Implement:

```text
one-turn root search
top-K candidate filtering
belief-informed hidden-state sampler
rollout scoring
soft search target generation
```

Deliverables:

```text
scripts/generate_search_dataset.py
ptcg_agent/search/root_search.py
ptcg_agent/search/scoring.py
```

### Milestone 8: Search distillation training

Implement:

```text
KL loss from search distribution
hard best-action CE
mixed imitation/search replay
```

Deliverables:

```text
scripts/train_search_distill.py
configs/train_search_distill.yaml
```

### Milestone 9: PPO self-play

Implement:

```text
self-play rollout runner
PPO buffer
legal-action PPO loss
GAE
opponent pool
deck sampler
checkpoint promotion
```

Deliverables:

```text
scripts/train_ppo_selfplay.py
ptcg_agent/rl/ppo.py
ptcg_agent/rl/opponent_pool.py
ptcg_agent/rl/deck_sampler.py
```

### Milestone 10: League evaluation

Implement:

```text
round-robin evaluation
13-deck matrix
checkpoint comparison
promotion report
debug trace generation for losses
```

Deliverables:

```text
scripts/evaluate_league.py
reports/league_*.md
```

---

## 10. Recommended command sequence

Codex should eventually support commands like:

```bash
# 1. Generate rule-based dataset
python scripts/generate_rule_based_dataset.py \
  --decks configs/decks/all_13.yaml \
  --agents configs/agents/rule_based.yaml \
  --games 90000 \
  --out data/rule_based_v1

# 2. Train supervised model
python scripts/train_supervised.py \
  --config configs/train_supervised.yaml \
  --data data/rule_based_v1 \
  --out checkpoints/sup_v1

# 3. Train belief model
python scripts/train_belief.py \
  --config configs/train_belief.yaml \
  --data data/rule_based_v1 \
  --out checkpoints/belief_v1

# 4. Generate search-improved dataset
python scripts/generate_search_dataset.py \
  --model checkpoints/sup_v1/best.pt \
  --belief checkpoints/belief_v1/best.pt \
  --decks configs/decks/all_13.yaml \
  --games 20000 \
  --out data/search_v1

# 5. Train search distillation
python scripts/train_search_distill.py \
  --base checkpoints/sup_v1/best.pt \
  --data data/search_v1 \
  --out checkpoints/search_distill_v1

# 6. PPO self-play
python scripts/train_ppo_selfplay.py \
  --init checkpoints/search_distill_v1/best.pt \
  --belief checkpoints/belief_v1/best.pt \
  --decks configs/decks/all_13.yaml \
  --out checkpoints/ppo_v1

# 7. League evaluation
python scripts/evaluate_league.py \
  --model checkpoints/ppo_v1/best.pt \
  --decks configs/decks/all_13.yaml \
  --opponents configs/agents/eval_pool.yaml \
  --games-per-condition 10 \
  --out reports/league_ppo_v1.md
```

---

## 11. Final deliverable expected from Codex

Codex should produce a training system where the advanced training flow is:

```text
rule-based demonstrations
→ entity Transformer supervised pretraining
→ value/Q/auxiliary tactical training
→ neural belief model training
→ one-turn root-search data generation
→ search distillation
→ PPO self-play with opponent pool
→ 13-deck league evaluation
→ checkpoint promotion
```

The first complete advanced run should output:

```text
checkpoints/sup_v1/best.pt
checkpoints/belief_v1/best.pt
checkpoints/search_distill_v1/best.pt
checkpoints/ppo_v1/best.pt
reports/league_ppo_v1.md
```

The final inference agent should use:

```text
entity Transformer policy/value model
+ neural belief model
+ one-turn root search
+ debug renderer disabled by default
```

---

## 12. References

1. Uploaded notebook: `reinforcement-learning-and-mcts-sample-code.ipynb`.
   - Use as implementation reference for `cg` imports, `SearchState`, `search_begin`, `search_step`, `search_end`, MCTS node structure, sparse encoding, self-play sample collection, and value/policy training.

2. Previous strategy file: `ptcg_rl_strategy_recommendation.md`.
   - Use as implementation reference for the belief-aware one-turn root search and rollout scoring.

3. Kaggle Pokémon TCG AI Battle competition overview.
   - The competition provides observations, game logs, board state, and legal options; agents return selected option indices.
   - URL: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview

4. cabt / cg documentation.
   - Use for exact observation/search object details and Search API signatures.
   - URL: https://matsuoinstitute.github.io/cabt/

5. Official Pokémon TCG AI Battle Challenge site.
   - URL: https://ptcg-abc.pokemon.co.jp/
