#!/usr/bin/env bash
#SBATCH --job-name=ptcg-p5-eq-audit
#SBATCH --output=experiments/rl/slurm-%j-phase5-equivalent-audit.out
#SBATCH --error=experiments/rl/slurm-%j-phase5-equivalent-audit.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/rl reports

PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"
INPUT="${INPUT:?Set INPUT to a trajectory JSONL}"
REPORT_JSON="${REPORT_JSON:?Set REPORT_JSON}"

"$PY" - "$INPUT" "$REPORT_JSON" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

source = Path(sys.argv[1])
report_path = Path(sys.argv[2])
rows = malformed = multi_target = missing_equivalence = 0
games: set[int] = set()
rows_by_game: Counter[int] = Counter()
target_sizes: Counter[int] = Counter()

with source.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        rows += 1
        decision = dict(payload.get("decision", {}) or {})
        metadata = dict(decision.get("reward_metadata", {}) or {})
        game_index = int(metadata.get("game_index", -1) or -1)
        if game_index >= 0:
            games.add(game_index)
            rows_by_game[game_index] += 1
        equivalent = metadata.get("phase5_search_equivalent_indices")
        if not isinstance(equivalent, list) or not equivalent:
            missing_equivalence += 1
            equivalent = metadata.get("phase5_search_indices", [])
        size = len({int(value) for value in equivalent or []})
        target_sizes[size] += 1
        multi_target += int(size > 1)

report = {
    "input": source.as_posix(),
    "bytes": source.stat().st_size,
    "rows": rows,
    "malformed_rows": malformed,
    "games": len(games),
    "minimum_rows_per_game": min(rows_by_game.values(), default=0),
    "maximum_rows_per_game": max(rows_by_game.values(), default=0),
    "missing_equivalence_rows": missing_equivalence,
    "multi_target_rows": multi_target,
    "multi_target_rate": multi_target / rows if rows else 0.0,
    "target_size_distribution": dict(sorted(target_sizes.items())),
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if malformed or rows == 0 or len(games) != 5000 or multi_target == 0:
    raise SystemExit(2)
PY
