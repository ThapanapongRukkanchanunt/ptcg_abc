# Pokémon TCG AI Battle RL Strategy

**Purpose:** Implementation strategy for a stronger contest agent built on top of the Kaggle sample notebook `reinforcement-learning-and-mcts-sample-code.ipynb`.

**Core recommendation:** Use the Kaggle sample as a technical reference for `cg` / `cabt` integration, Search API usage, sparse encoding, self-play scaffolding, and model structure. Replace the generic low-budget MCTS decision logic with a Pokémon-specific **belief-aware one-turn root search** that evaluates top first actions by simulating to the end of the current turn and scoring the resulting state using value prediction plus actual damage/prize reward.

---

## 1. Summary of the recommended agent

The final agent should have this high-level flow:

```text
raw cabt / Kaggle observation
    ├── StateAdapter → canonical GameState
    ├── LegalOptionAdapter → simulator-provided legal choices
    ├── GameMemory / BeliefState updater
    └── DebugBoardRenderer → 1024×1024 PNG only for debugging

GameState + BeliefState + LegalOptions
    → symbolic state encoder
    → legal-action / legal-option encoder
    → policy/value/action scorer

At start of turn:
    → policy pre-score all legal options
    → select top-K candidate first actions
    → for each candidate:
          clone simulator with belief-informed hidden state
          apply candidate first action
          let policy continue until end of current turn
          compute rollout score:
              V(end_state)
            + damage/prize tactical reward
            + board setup reward
            - resource waste penalty
            - self-risk penalty
    → return best simulator option index

During the rest of the turn:
    → normally use direct policy
    → optionally rerun small search for high-impact/uncertain decisions
```

The key principle:

```text
Simulator owns legality and effect resolution.
Agent owns ranking, memory, belief, and value estimation.
```

---

## 2. What to keep from the Kaggle sample notebook

The uploaded notebook is valuable as an implementation reference. It demonstrates:

1. Loading the `cg` library inside Kaggle.
2. Importing core API objects and functions:
   - `Observation`
   - `SearchState`
   - `SelectContext`
   - `OptionType`
   - `all_card_data`
   - `all_attack`
   - `search_begin`
   - `search_step`
   - `search_end`
   - `to_observation_class`
   - `battle_start`
   - `battle_select`
   - `battle_finish`
3. Sparse state/action encoding into encoder and decoder vectors.
4. A Transformer-like model with value and policy outputs.
5. Basic MCTS data structures:
   - `Node`
   - `Child`
   - `visit`
   - `total`
   - `value`
   - `children`
6. Self-play sample generation using MCTS decisions.
7. Training with value and policy targets.

The sample code sets:

```python
SEARCH_COUNT = 10
```

This makes the notebook a useful baseline, not a strong search policy. The recommended agent should preserve the working API integration but replace the generic low-budget MCTS logic with the domain-specific search described below.

---

## 3. What to replace from the sample notebook

The sample notebook creates search states using crude hidden-information filling. For example, it samples own deck/prizes randomly and fills opponent hidden zones with dummy card IDs:

```python
search_state = search_begin(
    obs,
    your_deck=random.sample(your_deck, state.players[your_index].deckCount),
    your_prize=random.sample(your_deck, len(state.players[your_index].prize)),
    opponent_deck=[1072] * state.players[1 - your_index].deckCount,
    opponent_prize=[1] * len(state.players[1 - your_index].prize),
    opponent_hand=[1] * state.players[1 - your_index].handCount,
    opponent_active=[1072] if len(active) > 0 and active[0] == None else []
)
```

This is acceptable for a sample notebook but should be replaced.

Replace it with:

```text
GameMemory + BeliefState
    → sampled or deduced hidden state
    → search_begin(...)
```

Specifically:

1. Use exact own hand, board, discard, and known deck information.
2. Deduce own prize cards after the first deck search.
3. Maintain probability distributions for unknown opponent hand/deck/prizes.
4. Sample hidden states from belief when running search.
5. Never read hidden simulator internals that are not available to the agent.

---

## 4. Proposed module layout

Use a modular project layout so Codex can implement and test each piece separately.

```text
ptcg_agent/
  agent.py
  config.py

  state/
    schema.py
    adapter.py
    memory.py
    belief.py
    option_adapter.py
    turn_boundary.py

  encoding/
    sparse_vector.py
    state_encoder.py
    action_encoder.py
    entity_encoder.py
    zone_encoder.py

  models/
    policy_value_model.py
    action_scorer.py
    checkpoint.py

  search/
    search_api_wrapper.py
    turn_rollout.py
    root_search.py
    scoring.py
    hidden_state_sampler.py

  rl/
    replay_buffer.py
    self_play.py
    train_policy_value.py
    train_offline.py
    heuristic_bots.py
    league_eval.py

  debug/
    board_renderer.py
    trace_writer.py
    replay_exporter.py

  tests/
    test_state_adapter.py
    test_option_adapter.py
    test_prize_deduction.py
    test_damage_reward.py
    test_turn_rollout.py
    test_search_scoring.py
```

---

## 5. StateAdapter

### Purpose

Convert raw Kaggle/cabt observation into a stable internal `GameState`.

The rest of the code should not depend directly on raw observation structure.

### Interface

```python
class StateAdapter:
    def parse(self, obs_dict: dict) -> GameState:
        ...
```

### Example schema

```python
from dataclasses import dataclass
from collections import Counter
from typing import Optional

@dataclass
class CardRef:
    card_id: int
    instance_id: int | None = None
    name: str | None = None

@dataclass
class PokemonSlot:
    card_id: int
    owner: int
    zone: str              # "active" or "bench"
    slot_index: int | None
    hp_max: int | None
    damage: int
    remaining_hp: int | None
    prize_value: int
    attached_energy: Counter[int]
    attached_tools: list[int]
    status: list[str]
    is_ex: bool
    is_basic: bool
    is_evolved: bool

@dataclass
class TurnFlags:
    supporter_used: bool
    energy_attached: bool
    retreat_used: bool
    turn_player: int
    is_my_turn: bool

@dataclass
class GameState:
    turn: int
    player_id: int
    current_player: int

    my_prizes_remaining: int
    opp_prizes_remaining: int

    my_deck_count: int
    opp_deck_count: int
    my_hand_count: int
    opp_hand_count: int

    my_active: Optional[PokemonSlot]
    opp_active: Optional[PokemonSlot]
    my_bench: list[Optional[PokemonSlot]]
    opp_bench: list[Optional[PokemonSlot]]

    my_hand: list[CardRef]
    my_discard: list[CardRef]
    opp_discard: list[CardRef]

    stadium: Optional[CardRef]
    turn_flags: TurnFlags
    public_log: list[dict]
```

### Notes

- Keep `GameState` deterministic and serializable.
- Add raw IDs needed to convert back to simulator options.
- Do not include hidden information unless it is actually known.

---

## 6. LegalOptionAdapter

### Purpose

The Kaggle environment provides legal options each turn. Therefore, the agent should not implement full Pokémon TCG legality. It should parse and normalize simulator-provided choices.

The Kaggle competition overview states that each turn the agent receives observation data including current board state, game logs, and a list of legal options, then returns selected option indices.

### Interface

```python
class LegalOptionAdapter:
    def parse(self, obs_dict: dict) -> list[LegalAction]:
        ...
```

### Schema

```python
@dataclass
class LegalAction:
    option_index: int             # original simulator option index
    local_index: int              # 0..N-1 after parsing
    action_type: str              # "attack", "play_card", "attach_energy", etc.

    source_card_id: int | None
    source_zone: str | None
    source_slot: int | None

    target_card_id: int | None
    target_owner: int | None
    target_zone: str | None
    target_slot: int | None

    cost_card_ids: list[int]
    selected_indices: list[int]   # for multi-select options
    ends_turn: bool
    raw_option: dict | object
```

### Rule

```text
Do not decide legality.
Only parse simulator legal options and preserve original indices.
```

The policy returns one of the legal option indices, not a global fixed action ID.

---

## 7. GameMemory and BeliefState

### Purpose

Maintain information that persists across observations:

1. Original decklist.
2. Seen cards.
3. Known own prizes after deck search.
4. Own deck belief.
5. Opponent hand/deck/prize belief.
6. Temporary effects inferred from logs.
7. Debug trace.

### Schema

```python
@dataclass
class GameMemory:
    original_decklist: Counter[int]

    known_my_prizes: Counter[int] | None
    my_prize_belief: dict[int, float]

    opp_hand_belief: dict[int, float]
    opp_deck_belief: dict[int, float]
    opp_prize_belief: dict[int, float]

    seen_cards: Counter[int]
    known_my_deck: Counter[int] | None

    last_obs_hash: str | None
    turn_start_state: GameState | None
    current_turn_id: int | None
```

### Prize deduction after first deck search

When a search effect reveals the remaining own deck, infer own prizes as:

```text
known_my_prizes =
    original_decklist
  - known_visible_own_cards
  - revealed_remaining_deck
```

Where `known_visible_own_cards` includes:

```text
hand + active + bench + discard + attached cards + stadium/tool if owned
```

Implementation:

```python
def deduce_own_prizes(
    original_decklist: Counter[int],
    visible_own_cards: Counter[int],
    revealed_deck: Counter[int],
) -> Counter[int]:
    prizes = original_decklist - visible_own_cards - revealed_deck
    return +prizes
```

### Important restriction

Use this only when the deck is legitimately revealed by the simulator due to a search effect. Do not access hidden simulator internals.

---

## 8. Hidden-state sampler

### Purpose

Create plausible hidden states for `search_begin`.

### Input

```text
GameState
GameMemory / BeliefState
number of samples K
```

### Output

```python
@dataclass
class HiddenStateSample:
    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]
```

### Sampling policy

For own side:

```text
if known_my_deck exists:
    sample/order from known_my_deck
else:
    sample from remaining original decklist after visible cards

if known_my_prizes exists:
    use known_my_prizes exactly
else:
    sample own prizes from remaining deck belief
```

For opponent side:

```text
sample opponent hand/deck/prizes from belief distribution
fallback to placeholder only if no belief model exists yet
```

Start with `num_belief_samples = 1`. Later use `4` or `8` for high-impact turns.

---

## 9. Search API wrapper

### Purpose

Wrap `search_begin`, `search_step`, and `search_end` so the rest of the code does not call the raw API directly.

### Interface

```python
class SearchAPIWrapper:
    def begin(self, obs_dict: dict, hidden: HiddenStateSample) -> SearchState:
        ...

    def step(self, state: SearchState, selected_indices: list[int]) -> tuple[SearchState, dict]:
        ...

    def end(self, state: SearchState) -> None:
        ...
```

### Notes

- Always call `search_end` in a `finally` block.
- Keep logs of selected option indices.
- Preserve simulator output after each `search_step` for combat/result parsing.
- Support multi-select actions because simulator actions may be a list of selected indices, not only one index.

---

## 10. Symbolic state encoding

Do not use the 1024×1024 rendered image as the model input. Use it only for debugging. The training and inference model should consume symbolic tensors.

### Minimum recommended encoder

```text
Zone-count features
+ global scalar features
+ legal-action features
```

### Zone-count features

```text
my_hand[card_id]
my_discard[card_id]
my_active[card_id]
my_bench[slot][card_id]

opp_discard[card_id]
opp_active[card_id]
opp_bench[slot][card_id]

my_deck_count
opp_deck_count
my_prizes_remaining
opp_prizes_remaining
```

### Entity features

Later, add entity rows:

```text
owner
zone
slot
card_id
card_type
pokemon_type
stage
hp_remaining
damage
energy_attached_by_type
status_condition
can_attack
can_retreat
has_tool
is_ex
is_basic
is_evolved
```

### Belief features

```text
known_my_prizes vector
opp_hand_belief vector
opp_deck_belief vector
opp_prize_belief vector
```

---

## 11. Legal-action encoding

Each legal action should become a feature vector.

Features:

```text
action_type
source_card_id
source_zone
target_card_id
target_owner
target_zone
target_slot
cost_card_ids
attack_id
attack_nominal_damage
energy_cost_satisfied
draw_effect_flag
search_effect_flag
switch_effect_flag
gust_effect_flag
discard_effect_flag
ends_turn_flag
```

The model should score:

```text
Q(state, legal_action_i)
```

rather than predicting a global fixed action ID.

---

## 12. Policy/value model

### Recommended initial model

```text
state_features → MLP/Transformer encoder → state_embedding
action_features_i → MLP action encoder → action_embedding_i

concat(state_embedding, action_embedding_i)
    → action score / policy logit
    → optional Q-value
```

Also produce:

```text
V(state) → scalar value estimate
```

### Model outputs

```python
@dataclass
class ModelOutput:
    value: float
    action_scores: list[float]
    action_probs: list[float]
```

### Auxiliary heads

Add later:

```text
actual_damage_prediction
damage_prevention_probability
KO_probability
expected_prizes_taken
resource_value
self_KO_risk
```

These auxiliary labels can be generated from simulator rollouts.

---

## 13. One-turn root search

### Purpose

At the start of the turn, compare the most promising first actions by simulating each to the end of the current turn.

This is a domain-specific alternative to generic MCTS.

### Why this works for Pokémon TCG

A first action often determines the entire turn line:

```text
play search card
→ choose target
→ attach energy
→ evolve
→ retreat
→ attack
```

The quality of the first action is often only clear after the attack or end command has resolved.

### Algorithm

```python
def select_action(raw_obs, memory, policy_model, value_model, config):
    state = state_adapter.parse(raw_obs)
    memory.update_from_observation(raw_obs, state)

    if raw_obs_contains_revealed_own_deck(raw_obs):
        memory.update_known_prizes_from_deck_search(raw_obs, state)

    legal_actions = legal_option_adapter.parse(raw_obs)

    if not should_run_root_search(state, memory, legal_actions, config):
        return direct_policy_select(state, legal_actions, policy_model)

    prescores = policy_model.score_actions(state, legal_actions)
    candidate_indices = top_k(prescores, config.root_top_k)

    best_score = -float("inf")
    best_local_idx = None

    for local_idx in candidate_indices:
        action = legal_actions[local_idx]
        scores = []

        for _ in range(config.num_belief_samples):
            hidden = hidden_state_sampler.sample(state, memory)
            score = evaluate_first_action_by_turn_rollout(
                raw_obs=raw_obs,
                hidden=hidden,
                first_action=action,
                root_state=state,
                policy_model=policy_model,
                value_model=value_model,
                config=config,
            )
            scores.append(score)

        expected_score = mean(scores)

        if expected_score > best_score:
            best_score = expected_score
            best_local_idx = local_idx

    return legal_actions[best_local_idx].option_index
```

---

## 14. Turn rollout

### Purpose

Given a candidate first action, use the Search API to simulate until the current turn ends.

### Algorithm

```python
def evaluate_first_action_by_turn_rollout(
    raw_obs,
    hidden,
    first_action,
    root_state,
    policy_model,
    value_model,
    config,
):
    search_state = search_api.begin(raw_obs, hidden)

    try:
        trajectory = []
        combat_events = []

        # Apply candidate first action.
        search_state, info = search_api.step(search_state, first_action.selected_indices)
        trajectory.append(first_action)
        combat_events.extend(parse_combat_events(info))

        step_count = 0

        while not turn_has_ended(search_state, root_state) and step_count < config.max_rollout_actions:
            rollout_obs = extract_observation(search_state)
            rollout_state = state_adapter.parse(rollout_obs)
            rollout_legal = legal_option_adapter.parse(rollout_obs)

            next_action = policy_select_action(
                rollout_state,
                rollout_legal,
                policy_model,
                temperature=config.rollout_temperature,
            )

            search_state, info = search_api.step(search_state, next_action.selected_indices)
            trajectory.append(next_action)
            combat_events.extend(parse_combat_events(info))
            step_count += 1

        end_obs = extract_observation(search_state)
        end_state = state_adapter.parse(end_obs)

        combat_summary = summarize_combat(root_state, end_state, combat_events)

        score = score_turn_rollout(
            root_state=root_state,
            end_state=end_state,
            trajectory=trajectory,
            combat_summary=combat_summary,
            value_model=value_model,
            config=config,
        )

        return score

    finally:
        search_api.end(search_state)
```

---

## 15. Turn-boundary detection

The rollout should stop when any of these happens:

```text
1. The current player changes.
2. The agent chooses an explicit END / PASS action.
3. An attack resolves and the turn ends.
4. Game is over.
5. max_rollout_actions is reached.
```

Implement:

```python
def turn_has_ended(search_state, root_state) -> bool:
    obs = extract_observation(search_state)
    state = state_adapter.parse(obs)
    return (
        state.current_player != root_state.current_player
        or is_game_over(state)
    )
```

Also use action metadata if available:

```python
if action.ends_turn:
    stop_after_resolution = True
```

---

## 16. Rollout scoring

### Core formula

```text
score =
    V(end_state)
  + λ_damage * damage_progress
  + λ_prize * prize_delta
  + λ_board * board_setup_delta
  + λ_hand * hand_quality_delta
  - λ_resource * resource_loss
  - λ_self_risk * self_risk
```

### Initial configuration

```python
RootSearchConfig = {
    "enabled": True,
    "root_top_k": 8,
    "num_belief_samples": 1,
    "max_rollout_actions": 30,
    "rollout_temperature": 0.0,

    "lambda_damage": 0.30,
    "lambda_prize": 0.70,
    "lambda_board": 0.10,
    "lambda_hand": 0.05,
    "lambda_resource": 0.10,
    "lambda_self_risk": 0.10,
}
```

Later:

```python
RootSearchConfig = {
    "root_top_k": 6,
    "num_belief_samples": 4,
    "max_rollout_actions": 40,
}
```

---

## 17. Damage reward

### Use actual damage

Use actual damage after prevention, reduction, weakness, resistance, and other effects. Do not use printed damage.

The Search API/simulator should resolve the actual result.

### Formula

```text
damage_progress =
  sum over opponent Pokémon damaged this turn:
      capped_actual_damage / target_max_hp * target_prize_value
```

Where:

```text
capped_actual_damage = min(actual_damage, target_remaining_hp_before_damage)
```

Add KO bonus:

```text
if target was knocked out:
    damage_progress += target_prize_value
```

### Example

```text
Target max HP = 300
Target remaining HP before damage = 180
Prize value = 2
Actual damage = 220

capped_actual_damage = min(220, 180) = 180

damage_progress = 180 / 300 * 2 = 1.20
KO bonus = 2.00

total = 3.20
```

### Why cap damage?

Overkill should not be over-rewarded.

```text
500 damage to a 30 HP Pokémon is still only a KO, not a 500-damage strategic gain.
```

---

## 18. Prize reward

Use real prize progress:

```text
prize_delta =
    prizes_taken_by_us_this_turn
  - prizes_taken_by_opponent_this_turn
```

Usually opponent does not take prizes during our turn, but card effects may create edge cases. Keep the formula general.

```python
def compute_prize_delta(root_state, end_state):
    return (
        root_state.opp_prizes_remaining - end_state.opp_prizes_remaining
    ) - (
        root_state.my_prizes_remaining - end_state.my_prizes_remaining
    )
```

---

## 19. Board setup reward

Reward having future attackers and energy setup.

Possible features:

```text
active_can_attack_now
backup_attacker_ready
total_energy_in_play
energy_on_relevant_attackers
evolved_support_pokemon_online
bench_development
draw/search engine online
```

Initial approximate formula:

```python
board_setup_delta =
    attacker_ready_score(end_state)
  - attacker_ready_score(root_state)
```

Example attacker score:

```python
def attacker_ready_score(state):
    score = 0.0

    for slot in [state.my_active] + state.my_bench:
        if slot is None:
            continue
        if can_attack_next_turn(slot):
            score += 1.0 * slot.prize_value
        elif has_partial_energy(slot):
            score += 0.3 * slot.prize_value

    return score
```

---

## 20. Resource loss penalty

Penalize using resources in ways that do not improve the board.

Examples:

```text
discarding key cards
discarding rare energy
using Supporter with little value
using once-per-turn ability with no benefit
attaching energy to low-value target
benching a vulnerable 2-prize Pokémon unnecessarily
```

Start simple:

```python
resource_loss =
    important_discard_penalty
  + wasted_supporter_penalty
  + wasted_energy_attachment_penalty
```

Add a card importance table later.

```python
CARD_IMPORTANCE = {
    # card_id: importance_weight
}
```

---

## 21. Self-risk penalty

Estimate risk of opponent taking a KO next turn.

Approximate early version:

```text
self_risk =
  if our active has low remaining HP and gives 2 prizes:
      high
  else:
      low
```

Better later:

```text
self_risk = opponent_policy_value_model.estimated_prizes_next_turn
```

The purpose is to prevent greedy plays that deal damage but leave an easy counter-KO.

---

## 22. Handling damage prevention

Do not hard-code all damage-prevention card effects first.

Recommended approach:

```text
1. Let simulator resolve the attack/effect.
2. Parse actual damage from logs or state difference.
3. Compute damage reward from actual damage.
4. Train auxiliary heads to predict actual damage, prevention probability, KO probability, and prize gain.
5. Manually tag only high-impact defensive cards later.
```

This directly handles cases like:

```text
printed damage = 200
actual damage = 0
damage_progress = 0
```

---

## 23. Direct policy mode

Use direct policy when not running search:

```python
def direct_policy_select(state, legal_actions, policy_model):
    scores = policy_model.score_actions(state, legal_actions)
    local_idx = argmax(scores)
    return legal_actions[local_idx].option_index
```

Run direct policy:

```text
1. After the first action of the turn, by default.
2. When not our turn.
3. When legal option count is small and low-impact.
4. When time budget is low.
5. During training rollouts.
```

Optionally rerun search for high-impact decisions:

```text
attack action
end-turn action
discard choice
search target choice
gust target choice
retreat target choice
top-two policy scores close
```

---

## 24. Candidate filtering

Full rollout for every legal action can be expensive. Use policy pre-scoring:

```text
legal options = N
policy scores all N cheaply
take top K
roll out only top K
```

Recommended initial values:

```python
root_top_k = 8
num_belief_samples = 1
```

When legal options are very numerous:

```python
root_top_k = min(8, max(3, int(sqrt(N))))
```

---

## 25. Time management

The submission must avoid running out of time.

Maintain:

```python
@dataclass
class TimeBudget:
    match_time_remaining: float
    turn_time_soft_limit: float
    action_time_soft_limit: float
```

Policy:

```text
if time is abundant:
    top_k = 8, belief_samples = 2-4
if time is normal:
    top_k = 6-8, belief_samples = 1
if time is low:
    direct policy only
```

Use hard caps:

```python
max_rollout_actions = 30
max_search_seconds_per_root = 1.0
```

Tune based on Kaggle runtime.

---

## 26. Debug image renderer

The 1024×1024 board image renderer should be kept for debugging only.

Do not put images in replay buffer for normal RL.

Use debug output:

```text
debug_runs/
  match_00012/
    turn_001.png
    turn_001_state.json
    turn_001_legal_actions.json
    turn_001_prescores.json
    turn_001_rollout_scores.json
    turn_001_chosen_action.txt
```

This will make it easy to inspect:

```text
What did the board look like?
Which legal actions existed?
What did the model think?
Which candidate won rollout search?
Did actual damage differ from printed damage?
```

---

## 27. Training plan

### Stage 1: heuristic bots

Implement simple bots:

```text
RandomLegalBot
AttackIfPossibleBot
EnergyGreedyBot
PrizeGreedyBot
SetupGreedyBot
DrawSupporterGreedyBot
```

Use them to generate early games and validate the environment wrapper.

### Stage 2: imitation pretraining

Train policy head:

```text
state + legal options → heuristic action
```

Loss:

```text
cross-entropy over legal options
```

### Stage 3: offline value learning

From completed games:

```text
(state, chosen_action) → final outcome
```

Train:

```text
V(state)
Q(state, action)
```

Labels:

```text
win = +1
loss = -1
draw = 0
```

### Stage 4: self-play RL

Train against opponent pool:

```text
random bot
heuristic bots
previous policy snapshots
current best policy
```

Avoid only training current policy against itself.

### Stage 5: search-enhanced self-play

Generate self-play data with the one-turn root search. Train policy to imitate the search-improved decision.

This is similar in spirit to AlphaZero-style policy improvement, but the search is domain-specific and turn-bounded.

---

## 28. Replay buffer

Store symbolic transitions:

```python
Transition = {
    "state_features": encoded_state,
    "legal_action_features": encoded_legal_actions,
    "chosen_local_index": chosen_local_idx,
    "chosen_option_index": chosen_option_index,

    "reward": reward,
    "next_state_features": next_encoded_state,
    "next_legal_action_features": next_encoded_legal_actions,
    "done": done,

    "combat_summary": {
        "actual_damage": ...,
        "damage_prevented": ...,
        "ko_count": ...,
        "prize_delta": ...,
    },

    "debug": {
        "raw_option": ...,
        "trajectory": ...,
    },
}
```

Do not store PNGs in normal replay buffer.

---

## 29. Evaluation league

Evaluate by round-robin:

```text
candidate_policy
vs RandomLegalBot
vs AttackIfPossibleBot
vs PrizeGreedyBot
vs SetupGreedyBot
vs previous_best_policy
vs current_policy_without_search
vs current_policy_with_search
```

Metrics:

```text
win rate
average prize difference
average game length
timeout rate
illegal action rate
first-turn setup failure rate
average damage progress
average prizes per turn
resource waste rate
counter-KO rate
```

Promote a checkpoint only if it improves league results, not just training loss.

---

## 30. Submission-time agent

At submission time, the agent should only run inference and search. It should not train.

```text
agent.py
    load card metadata
    load config
    load model weights
    initialize GameMemory per game
    receive observation
    parse GameState
    update GameMemory / BeliefState
    parse legal options
    decide direct policy vs root search
    return simulator option index / indices
```

Recommended submission tree:

```text
submission/
  agent.py
  config.json
  card_vocab.json
  model_weights.pt

  ptcg_agent/
    state/
    encoding/
    models/
    search/
    debug/
```

Debug rendering should be disabled by default:

```python
DEBUG_RENDER = False
```

---

## 31. Implementation milestones for Codex

### Milestone 1: Refactor sample notebook into modules

- Extract Search API usage.
- Extract sparse encoding.
- Extract model.
- Extract MCTS/self-play code.
- Make scripts runnable outside notebook.

### Milestone 2: Implement state and option adapters

- `GameState`
- `LegalAction`
- `StateAdapter`
- `LegalOptionAdapter`
- Unit tests using saved observations.

### Milestone 3: Implement memory and prize deduction

- `GameMemory`
- `deduce_own_prizes`
- update from deck search
- tests using artificial deck/search examples.

### Milestone 4: Implement direct policy baseline

- state encoder
- action encoder
- action scorer
- direct action selection.

### Milestone 5: Implement one-turn root search

- hidden-state sampler
- Search API wrapper
- rollout until turn end
- rollout scorer
- top-K candidate filtering.

### Milestone 6: Implement debug traces

- save PNG board images
- save action scores
- save rollout trajectories
- save combat summary.

### Milestone 7: Training pipeline

- heuristic bots
- imitation training
- offline value learning
- self-play
- policy pool
- league evaluation.

---

## 32. Key risks and mitigations

### Risk 1: Myopic damage greed

Problem:

```text
Agent chooses high damage now but loses prize race next turn.
```

Mitigation:

```text
Keep V(end_state) as the main score.
Add self-risk penalty.
Tune damage reward conservatively.
```

### Risk 2: Search too slow

Mitigation:

```text
Use policy top-K filtering.
Use one belief sample initially.
Use direct policy when time is low.
Run search mainly at start of turn.
```

### Risk 3: Belief sampling wrong

Mitigation:

```text
Start with own-prize deduction first.
Use simple opponent belief.
Fallback to placeholders if needed.
Improve belief gradually.
```

### Risk 4: Turn boundary detection incorrect

Mitigation:

```text
Use current_player change as primary signal.
Use action.ends_turn as secondary signal.
Add max_rollout_actions guard.
Write tests from real traces.
```

### Risk 5: Damage reward corrupted by healing/moving damage

Mitigation:

```text
Prefer combat logs if available.
Fallback to state difference only when logs are unavailable.
Cap damage by prior remaining HP.
```

---

## 33. Difference from the Kaggle sample notebook

| Aspect | Kaggle sample notebook | Recommended implementation |
|---|---|---|
| Search style | Generic MCTS | One-turn root search |
| Search budget | `SEARCH_COUNT = 10` | Top-K candidate rollouts |
| Hidden information | Random/dummy fill | Belief state + own-prize deduction |
| Evaluation | Neural value + MCTS visit count | `V(end_state) + tactical reward` |
| Damage handling | Resolved by simulator during search | Resolved by simulator, then explicitly rewarded by actual damage/prize progress |
| Legal actions | Uses simulator options | Same, but normalized through `LegalOptionAdapter` |
| Training | Self-play sample | Heuristic pretraining + offline value + self-play + search improvement |
| Debugging | Notebook-level | Dedicated renderer and trace files |

---

## 34. References

1. **Uploaded notebook:** `reinforcement-learning-and-mcts-sample-code.ipynb`. Used as the primary implementation reference for `cg` imports, sparse encoder/decoder style, model value/policy heads, `SearchState`, `search_begin`, `search_step`, `search_end`, MCTS node structure, and self-play sample generation.

2. **Kaggle competition overview:** Pokémon TCG AI Battle Challenge. The competition overview states that the agent receives an observation each turn, including game logs, current board state, and legal options, and returns selected option indices.  
   URL: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview

3. **Kaggle strategy competition page:** PTCG AI Battle Challenge Strategy. Useful as broader context for strategy-agent development.  
   URL: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy

4. **cabt / cg documentation:** Use for exact function signatures and observation/search object details. Search API functions of interest: `search_begin`, `search_step`, `search_end`, `to_observation_class`.  
   URL: https://matsuoinstitute.github.io/cabt/

5. **Official Pokémon TCG AI Battle Challenge site:** Provides public competition context and automated battle / leaderboard framing.  
   URL: https://ptcg-abc.pokemon.co.jp/

---

## 35. Final implementation directive

Codex should implement the Kaggle sample first as a runnable baseline, then progressively replace the decision engine with:

```text
belief-aware top-K one-turn root search
+ policy rollout to end of turn
+ value/end-state scoring
+ actual damage/prize reward
+ debug trace rendering
```

The first working version should prioritize correctness and observability over model strength:

```text
1. Direct policy works.
2. Search API wrapper works.
3. Turn rollout terminates safely.
4. Damage/prize scoring is logged.
5. Debug PNG + JSON trace can explain every chosen action.
```

Only after those are stable should Codex tune model architecture, search depth, belief sampling count, and reward weights.
