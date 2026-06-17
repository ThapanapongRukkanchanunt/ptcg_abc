# Phase 3 Baseline Checkpoint

Date: 2026-06-17

## Status

The first generic rule-based agent baseline is runnable against the local Kaggle sample
simulator.

This is a checkpoint inside Phase 3, not the final rule strategy. The agent is now useful
as a stable baseline because it can:

- Load a saved TEF-POR Limitless corpus deck.
- Convert card names to official Kaggle numeric card IDs through `EN_Card_Data.csv`.
- Resolve duplicate card-name matches by choosing the lowest Kaggle Card ID.
- Return the 60-card deck for the initial Kaggle agent call.
- Choose legal option indices with deterministic deck-agnostic rules.
- Run a short native simulator smoke match without random choices.

## Added Code

- `src/ptcg_abc/card_db.py`: official Kaggle card ID lookup.
- `src/ptcg_abc/corpus.py`: corpus JSONL loader and deck-to-ID conversion.
- `src/ptcg_abc/agent/rule_based.py`: first generic action selector.
- `src/ptcg_abc/simulator.py`: local Kaggle sample-simulator smoke runner.
- `python -m ptcg_abc agent-smoke`: command-line entry point for the smoke check.

## Current Rule Policy

The selector is intentionally simple and deck-agnostic:

- During main actions, prefer sequencing actions before attacking:
  attach, evolve, ability, play, attack, retreat, discard, then end.
- For attack selections, choose a legal attack option deterministically.
- For count selections, prefer the largest offered number.
- For discard-like contexts, choose the minimum allowed count.
- For damage contexts, prefer opponent targets when the active player is known.
- For healing and recovery contexts, prefer self targets when the active player is known.
- For yes/no prompts, generally say yes, but choose to go second at the opening prompt.

## Verification

Unit tests:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\thaip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests
```

Result:

```text
Ran 19 tests
OK
```

Simulator smoke:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\thaip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ptcg_abc agent-smoke --max-steps 50
```

Result summary:

- Deck 0: Dragapult ex / None / Hiromu Sasaki 1st
- Deck 1: Dragapult ex / None / Andrew Hedrick 1st
- Ambiguous names resolved by lowest Kaggle Card ID: 1
- Ambiguous card: Dunsparce, IDs 65 and 305
- Simulator started: yes
- Steps completed: 50
- Engine error: none

## Resume Command

From a checkout with Kaggle data and the generated corpus present:

```powershell
$env:PYTHONPATH='src'
& 'C:\Users\thaip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ptcg_abc agent-smoke --max-steps 50
```

## Next Work

Phase 3 should continue by improving rule quality:

- Load attack/card metadata so attack choices can use damage and energy requirements.
- Improve setup choices using HP, retreat cost, evolution role, and deck counts.
- Add card sequencing rules for common Trainers and Energy attachment targets.
- Add prize mapping and knockout planning once state inspection helpers are in place.
- Package the baseline as a Kaggle `main.py` submission after local behavior is stable.
