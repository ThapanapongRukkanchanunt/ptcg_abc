# Rule Inventory

Date: 2026-06-18

## Scope

This document separates rules that are currently implemented from rules learned from
Kaggle example agents and worth considering next.

## Sources

- Local baseline: `src/ptcg_abc/agent/rule_based.py`
- Kaggle example: https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck
- Kaggle example: https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck
- Kaggle example: https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-dragapult-ex-deck
- Kaggle example: https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck

## Current Implemented Rules

These rules are implemented in `RuleBasedAgent` and `select_option_indices`. The
Phase 3 closeout agent uses card metadata when it is available from Kaggle's
`all_card_data()` and `all_attack()` APIs.

1. Initial deck selection
   - If `obs.select` is `None`, return the configured 60-card deck ID list.

2. Empty selection
   - If no selectable options exist, return an empty selection.

3. Selection count
   - For `MAIN`, `COUNT`, and `YES_NO` selections, choose one option when possible.
   - For discard-like or forced-minimum contexts, choose the minimum legal count.
   - For other multi-select contexts, choose the maximum legal count.

4. Yes/no prompts
   - For the opening first-player prompt, choose `NO`, meaning prefer going second.
   - For other yes/no prompts, choose `YES` when available.

5. Main-action ordering
   - Current main-action priority is:
     attach, evolve, ability, play, attack, retreat, discard, end.

6. General option ordering
   - Current non-main priority is:
     attack, evolve, ability, play, attach, card, energy, energy card, tool card,
     skill, special condition, yes, no, retreat, discard, end, number.

7. Count choices
   - For count selections, prefer the largest offered number.

8. Damage targeting
   - For damage and damage-counter contexts, prefer opponent-controlled targets when the
     active player is known.

9. Healing and recovery targeting
   - For heal, remove-damage-counter, and recover-special-condition contexts, prefer
     self-controlled targets when the active player is known.

10. Active-area tie break
    - When otherwise tied, options associated with the Active Spot are preferred over
      non-active options.

11. Determinism
    - Ties fall back to the original option order, so the policy is deterministic.

12. Metadata-backed scoring
    - When metadata is available, score every legal option and select the highest
      scoring legal choices rather than relying only on fixed priority order.

13. Board feature extraction
    - Count cards across field, hand, discard, and field-plus-hand.
    - Track active attacker readiness, bench attacker readiness, hand energy count,
      stadium state, supporter state, deck count, and current prize counts.

14. Attack planning
    - Estimate the best current attack plan from active and switchable benched Pokemon.
    - Use attack damage, energy cost, weakness, resistance, knockout potential, and
      prize value.
    - Reuse the attack plan to guide attachment, retreat, Boss-like plays, target
      selection, and attack choice.

15. Prize-aware target value
    - Value knockouts by prize count, ex/Mega ex status, attached energy, tools,
      evolution stage, and remaining HP.
    - Account for known prize modifiers such as Legacy Energy and Lillie's Pearl.

16. Setup and search
    - Prefer missing core Pokemon over duplicates.
    - Prefer evolution targets when the matching previous stage is already in play.
    - Penalize extra copies when enough are already in field or hand.

17. Energy attachment
    - Prefer attachments that complete the planned attack.
    - Prefer powering a next attacker when the active Pokemon can already attack.
    - Penalize over-attaching beyond useful attack requirements.

18. Retreat and switch
    - Retreat only when a better benched attacker is ready or the stored attack plan
      requires a bench attacker.

19. Draw, recovery, and discard
    - Restrict deck-search/draw effects when deck count is low.
    - Use recovery effects only when discard contains useful Pokemon or Energy.
    - Prefer discarding duplicates and low-value cards while protecting current plans.

20. Prize-map planning
    - Build a turn-local prize route from available attackers and opponent board targets.
    - Prefer routes that take the remaining prizes in the fewest attack commands.
    - Use route setup cost to account for missing energy, retreat/switch needs, and
      Boss-like bench targeting.
    - Model spread and damage-counter attacks that can damage multiple opponent
      Pokemon in one attack command.
    - Count multiple knockouts from one multi-target attack when scoring a route.
    - Ignore routes that do not contain an actionable prize-taking step from a primary
      attacker.
    - Boost mapped spread or damage-counter targets when a planned route assigns
      damage to multiple opponent Pokemon.
    - Reuse the first prize-map step as the immediate attack plan.

21. Automatic key-attacker identification
    - Mark attackers used in the current prize map as key attackers.
    - Score deck Pokemon with attacks against the current opponent board to identify
      high-value attackers even before they are in play.
    - Reject low-damage utility basics as primary key attackers unless they have a
      high-value route.
    - Boost search, benching, evolution, attachment, and discard-protection rules for
      key attackers and their recursive pre-evolution chains.

22. Evolution setup and repeat acceleration
    - Prefer key evolution-chain basics during opening Active selection when they are
      available.
    - Support repeat attachment abilities such as Iono's Bellibolt ex's Electric
      Streamer when enough matching Energy is in hand.
    - Keep broad chain scoring narrow by default because the first aggressive version
      regressed the 27-deck Dragapult benchmark.

23. Phase 3 benchmark coverage
    - The copied Dragapult benchmark now appends required sample decks for Crustle,
      Mega Lucario ex, Mega Abomasnow ex, and Iono's Bellibolt ex.
    - Coverage status is written into benchmark reports so missing required decks are
      visible immediately.

## Rule Families Learned From Kaggle Examples

These are reusable patterns learned from the four example agents. The Phase 3 closeout
agent implements the generic versions; deck-specific card IDs and exact combo scripts
remain intentionally out of scope.

### Board Feature Extraction

- Build a `get_card(area, index, player)` helper so all option scoring can inspect the
  actual card or Pokemon behind an option.
- Load official card metadata with `all_card_data()` so rules can use card type, HP,
  stage, weakness, resistance, retreat cost, ex/Mega ex flags, skills, and attacks.
- Count cards by ID across field, hand, discard, deck estimate, prize estimate, and
  field-plus-hand.
- Track stadium, supporter-used, manual-energy-used, turn number, prize counts, and
  known previous-turn logs.
- Track whether active and benched attackers are already attack-ready.

### Scoring Model

- Score every legal option, then select the highest scoring options.
- Use negative scores for optional actions that should be skipped.
- Preserve min/max legality after scoring.
- Keep attack scores lower than setup scores during main actions, because attacks end
  the turn and should usually happen after sequencing.

### Attack Planning

- Evaluate possible attackers from Active and Bench, not only the current Active Pokemon.
- Evaluate possible opponent targets from Active and Bench when Boss-like effects or
  switching effects are available.
- Estimate attack damage, including weakness and resistance when metadata is available.
- Check whether a manual energy attachment can make an attacker ready this turn.
- Prefer game-winning knockouts over ordinary value.
- Store the best attack plan and reuse it to guide Boss, switch, retreat, energy attach,
  attack choice, and target choice.
- Build a prize route before immediate action scoring, so setup decisions can serve a
  route to the remaining prizes instead of only the highest current damage option.

### Prize Mapping

- Prize count should use card metadata:
  one prize by default, two for ex, three for Mega ex.
- Prize count can be reduced by card modifiers such as Legacy Energy and Lillie's Pearl.
- Knockout target value should include prize value, attached energies, attached tools,
  evolution stage, HP, and special high- or low-value support Pokemon.
- Multi-target attacks or damage-counter effects should plan combinations of knockouts,
  not only the active target.

### Setup And Bench

- Prefer benching missing core basics before duplicate basics.
- Prefer evolutions when the matching pre-evolution is already in play and eligible.
- Stop benching or searching for extra copies once the board already has enough copies.
- Avoid filling bench slots with low-value liabilities when better setup targets remain.
- Choose opening Active based on role: setup Pokemon when going first, attacker or pivot
  when going second, and avoid fragile support Pokemon when possible.

### Energy Attachment

- Attach energy to the Pokemon closest to reaching an attack requirement.
- If the active Pokemon can already attack, consider powering a benched next attacker.
- Avoid over-attaching beyond a Pokemon's useful requirement.
- Avoid attaching to support Pokemon unless they are stuck active or need to retreat.
- If a planned attack needs one more energy this turn, prioritize that attachment.

### Retreat And Switch

- Retreat only when there is a better benched attacker or pivot ready.
- Use switching cards or movement abilities when they advance the stored attack plan.
- Do not retreat just because retreat is legal.

### Search, Draw, And Recovery

- Search targets should be scored by current board need, hand duplication, and evolution
  chain completion.
- Ultra Ball-like effects should be used when there are low-value discard cards or missing
  key Pokemon.
- Recovery cards should be used only when discard contains useful Pokemon or energy.
- Energy Retrieval-like effects should be used when enough energy is in discard.
- Draw or deck-thinning effects should be restricted when deck count is low.

### Discard Choices

- Prefer discarding duplicate cards and currently low-value cards.
- Prefer discarding basic energy only for decks that benefit from energy in discard.
- Avoid discarding needed evolution pieces, active plans, and the only useful supporter.

### Supporter And Stadium Sequencing

- Before playing a supporter, score all playable supporters and choose the best one for
  the current plan.
- Boss-like supporters should be played only when the stored attack plan has a valuable
  bench target.
- Stadiums should be played when they replace an opposing stadium or enable a relevant
  ability/plan.

### Ability Use

- Use abilities when they produce immediate value or enable the current plan.
- Avoid draw abilities when deck count is low.
- Track once-per-turn ability effects when needed so the agent does not plan as if the
  same ability remains available.

### Log-Aware Rules

- Use previous-turn logs to detect whether one of our Pokemon was knocked out.
- Use previous-turn attack logs to detect lock effects such as item lock.
- Use knockout detection to raise the value of comeback cards such as draw or stamp
  effects.

## Phase 4 Follow-Up Ideas

1. Add focused deck-family profiles derived from the public sample agents while keeping
   the generic scorer as the default.
2. Add anti-Dragapult defensive rules for Phantom Dive, especially avoiding optional
   low-HP bench liabilities and prioritizing damage-counter-safe board states.
3. Add variable-damage estimates for attacks such as Mega Abomasnow ex's Hammer-lanche
   and Iono's Voltaic Chain.
4. Use the Phase 3 scorer as a fixed baseline opponent for reinforcement learning.
5. Train or tune weights for setup, attachment, attack, and target scoring.
6. Re-run the 31-deck Phase 3 benchmark after each focused profile change until the
   average win rate approaches the 50% target.
