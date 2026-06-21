# ERAWAN Phase 4 Image-Size Progression Runbook

This runbook only covers the current Phase 4 path in this repo: the
image-size progression experiment driven by `rl-image-progression` and
`scripts/slurm/phase4_image_progression_conda.sbatch`.

The current experiment repeats this loop for each image-size label:

1. Run Hybrid vs Hybrid self-play for 1,000 games rotating across all 9 prepared decks.
2. Update the model from the rollout records.
3. Run Hybrid vs Benchmark evaluation for 100 games per matchup.
4. Save one compact replay per matchup, for 36 replays per iteration.

The planned sweep runs 10 iterations for image sizes `1024`, `512`, and `256`.
Rollout records are compact symbolic records; the image size is tracked as the
experiment dimension and model metadata for comparing these three runs.
By default, self-play rotates through the full 9x9 ordered matchup grid,
including mirror matchups. Use `DECK_A_INDEX` and `DECK_B_INDEX` only when you
intentionally want to reproduce a fixed two-deck run.

## 0. Push Latest Changes From This PC

Run this from Windows PowerShell on this PC before pulling on ERAWAN:

```powershell
cd C:\Users\thaip\Documents\ptcg_ai_battle_challenge
git status --short --branch
git add docs/erawan-runbook.md `
  scripts/slurm/phase4_image_progression_conda.sbatch `
  src/ptcg_abc/cli.py `
  src/ptcg_abc/rl/workflow.py `
  tests/test_rl_phase4.py
git commit -m "Rotate Phase 4 self-play decks"
git push origin main
```

Use explicit file paths here. Do not use `git add .` if local copied Kaggle or
ERAWAN data folders are present.

Verify GitHub has the latest commit:

```powershell
git status --short --branch
git log -1 --oneline
```

## 1. Pull The Latest Repo On ERAWAN

Run on ERAWAN from your home directory:

```bash
cd ~/ptcg_abc
git fetch origin
git status --short --branch
git pull --ff-only origin main
git rev-parse --short HEAD
```

Expected: the branch is up to date with `origin/main`, and the latest commit
contains `scripts/slurm/phase4_image_progression_conda.sbatch`.

Verify the progression command exists:

```bash
cd ~/ptcg_abc
~/ptcg_abc/.conda_ptcg/bin/python -m ptcg_abc --help | grep rl-image-progression
```

If `.conda_ptcg` is not created yet, do Step 2 first, then rerun this command.

## 2. Prepare The Conda Environment

Use the ERAWAN miniconda module and call the environment Python directly.
Do not rely on `conda activate` inside SLURM jobs.

```bash
cd ~/ptcg_abc
module purge
module load miniconda3/3.12
test -x ~/ptcg_abc/.conda_ptcg/bin/python || conda create -y -p ~/ptcg_abc/.conda_ptcg python=3.12 pip
```

Install the repo and runtime packages:

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
"$PY" --version
"$PY" -m pip install -U pip setuptools wheel
"$PY" -m pip install -e .
"$PY" -m pip install "numpy>=1.26" "Pillow>=10.0" "pypdf>=4.0"
"$PY" -m pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Check imports on the login node:

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
"$PY" - <<'PY'
import sys
import torch
import ptcg_abc

print(sys.version)
print(sys.executable)
print("torch", torch.__version__)
print("torch cuda runtime", torch.version.cuda)
print("cuda available on login node", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

It is acceptable if CUDA is `False` on the login node. CUDA must be checked
inside a SLURM GPU allocation in Step 5.

## 3. Prepare Kaggle Input Files

If your files are already in `data/input/`, point the repo default path at that
directory:

```bash
cd ~/ptcg_abc
mkdir -p data/kaggle
test -e data/kaggle/input || ln -s ../input data/kaggle/input
ls data/kaggle/input
```

Build the legal-card list from the copied Kaggle CSV:

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
"$PY" -m ptcg_abc discover-legal-cards \
  --input-dir data/kaggle/input \
  --legal-source data/kaggle/input/EN_Card_Data.csv \
  --legal-cards data/kaggle/legal_cards.txt
```

Verify the required files:

```bash
cd ~/ptcg_abc
test -f data/kaggle/input/EN_Card_Data.csv && echo "EN card data OK"
test -f data/kaggle/input/Card_ID\ List_EN.pdf && echo "card art PDF OK"
test -d data/kaggle/input/sample_submission && echo "sample submission OK"
test -f data/kaggle/legal_cards.txt && echo "legal cards OK"
```

## 4. Login-Node Smoke Test

Keep this tiny. It verifies the latest progression command, data paths, model
update path, benchmark path, and replay writing without starting the real run.

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
"$PY" -m ptcg_abc rl-image-progression \
  --image-size 256 \
  --iterations 1 \
  --selfplay-games 1 \
  --eval-games-per-matchup 1 \
  --max-steps 20 \
  --saved-replays-per-matchup 1 \
  --replay-trace-limit 10 \
  --update-epochs 1 \
  --dataset-root data/datasets/rl/image_progression_smoke \
  --model-root models/rl/image_progression_smoke \
  --report-root reports/image_progression_smoke \
  --output-root experiments/rl/image_progression_smoke
```

Inspect the smoke result:

```bash
cd ~/ptcg_abc
cat experiments/rl/image_progression_smoke/image-256/progression_summary.json
find experiments/rl/image_progression_smoke/image-256/iter-01/replays -type f | wc -l
```

Expected: `errors` is `0`, and the replay count is `36`.

## 5. GPU Smoke Test Through SLURM

Run the same current progression script as a tiny GPU job. This checks the
conda interpreter and CUDA from inside the allocation.

```bash
cd ~/ptcg_abc
JOB=$(
  IMAGE_SIZES="256" \
  ITERATIONS=1 \
  SELFPLAY_GAMES=1 \
  EVAL_GAMES_PER_MATCHUP=1 \
  MAX_STEPS=20 \
  SAVED_REPLAYS_PER_MATCHUP=1 \
  REPLAY_TRACE_LIMIT=10 \
  UPDATE_EPOCHS=1 \
  sbatch --parsable --gres=gpu:1 --time=00:30:00 scripts/slurm/phase4_image_progression_conda.sbatch
)
mkdir -p experiments/rl/image_progression_smoke
echo "$JOB" | tee experiments/rl/image_progression_smoke/latest_slurm_job.txt
```

Monitor the GPU smoke job:

```bash
cd ~/ptcg_abc
JOB=$(cat experiments/rl/image_progression_smoke/latest_slurm_job.txt)
squeue -j "$JOB"
tail -n 80 experiments/rl/slurm-${JOB}-image-progression.out
tail -n 80 experiments/rl/slurm-${JOB}-image-progression.err
```

Expected in the output:

```text
torch 2.5.1+cu121
cuda True
device NVIDIA A100-SXM4-80GB
```

The exact GPU name can differ if the job lands on H100 instead of A100.

## 6. Submit The Full Image-Size Progression Experiment

This is the current full experiment:

```bash
cd ~/ptcg_abc
PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
mkdir -p "$PROJECT_RUN"
JOB=$(
  IMAGE_SIZES="1024 512 256" \
  ITERATIONS=10 \
  SELFPLAY_GAMES=1000 \
  EVAL_GAMES_PER_MATCHUP=100 \
  MAX_STEPS=600 \
  SAVED_REPLAYS_PER_MATCHUP=1 \
  REPLAY_TRACE_LIMIT=60 \
  UPDATE_EPOCHS=1 \
  DATASET_ROOT="$PROJECT_RUN/data/datasets/rl/image_progression" \
  MODEL_ROOT="$PROJECT_RUN/models/rl/image_progression" \
  REPORT_ROOT="$PROJECT_RUN/reports/image_progression" \
  OUTPUT_ROOT="$PROJECT_RUN/experiments/rl/image_progression" \
  sbatch --parsable --gres=gpu:1 scripts/slurm/phase4_image_progression_conda.sbatch
)
mkdir -p experiments/rl/image_progression
echo "$JOB" | tee experiments/rl/image_progression/latest_slurm_job.txt
```

The script writes logs here:

```bash
cd ~/ptcg_abc
JOB=$(cat experiments/rl/image_progression/latest_slurm_job.txt)
echo experiments/rl/slurm-${JOB}-image-progression.out
echo experiments/rl/slurm-${JOB}-image-progression.err
```

## 7. Monitor The Running Job

Check whether the job is still queued or running:

```bash
cd ~/ptcg_abc
JOB=$(cat experiments/rl/image_progression/latest_slurm_job.txt)
squeue -j "$JOB"
```

Watch the latest normal output:

```bash
cd ~/ptcg_abc
JOB=$(cat experiments/rl/image_progression/latest_slurm_job.txt)
tail -f experiments/rl/slurm-${JOB}-image-progression.out
```

Watch errors or tracebacks:

```bash
cd ~/ptcg_abc
JOB=$(cat experiments/rl/image_progression/latest_slurm_job.txt)
tail -f experiments/rl/slurm-${JOB}-image-progression.err
```

If the job leaves the queue, inspect the final logs:

```bash
cd ~/ptcg_abc
JOB=$(cat experiments/rl/image_progression/latest_slurm_job.txt)
cat experiments/rl/slurm-${JOB}-image-progression.out
cat experiments/rl/slurm-${JOB}-image-progression.err
```

## 8. Inspect The Progression Trend

Each image size gets its own summary:

```bash
cd ~/ptcg_abc
PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
ls "$PROJECT_RUN"/experiments/rl/image_progression/image-1024/progression_summary.json
ls "$PROJECT_RUN"/experiments/rl/image_progression/image-512/progression_summary.json
ls "$PROJECT_RUN"/experiments/rl/image_progression/image-256/progression_summary.json
```

Print a compact trend table:

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
export PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
"$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PROJECT_RUN"])
for size in (1024, 512, 256):
    path = root / "experiments" / "rl" / "image_progression" / f"image-{size}" / "progression_summary.json"
    print(f"\nimage-{size}")
    if not path.exists():
        print("  not started")
        continue
    data = json.loads(path.read_text())
    for item in data.get("summaries", []):
        ev = item.get("evaluation", {})
        print(
            f"  iter {item['iteration']:02d}: "
            f"win_rate={ev.get('win_rate', 0):.3f} "
            f"wins={ev.get('wins', 0)}/{ev.get('games', 0)} "
            f"timeouts={ev.get('timeouts', 0)} "
            f"errors={ev.get('errors', 0)} "
            f"model={item.get('model_path')}"
        )
PY
```

Check replay counts:

```bash
cd ~/ptcg_abc
PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
find "$PROJECT_RUN"/experiments/rl/image_progression/image-1024 -path '*/replays/*.json' | wc -l
find "$PROJECT_RUN"/experiments/rl/image_progression/image-512 -path '*/replays/*.json' | wc -l
find "$PROJECT_RUN"/experiments/rl/image_progression/image-256 -path '*/replays/*.json' | wc -l
```

Expected after a complete run: `360` replay files per image size.

## 9. Inspect A Specific Iteration

Open the markdown benchmark for any image size and iteration:

```bash
cd ~/ptcg_abc
PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
cat "$PROJECT_RUN"/reports/image_progression/image-1024/iter-01-benchmark.md
cat "$PROJECT_RUN"/reports/image_progression/image-512/iter-01-benchmark.md
cat "$PROJECT_RUN"/reports/image_progression/image-256/iter-01-benchmark.md
```

List the 36 compact replay traces for an iteration:

```bash
cd ~/ptcg_abc
PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
find "$PROJECT_RUN"/experiments/rl/image_progression/image-1024/iter-01/replays -maxdepth 1 -type f | sort
```

View one replay trace:

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
"$PY" -m json.tool "$PROJECT_RUN"/experiments/rl/image_progression/image-1024/iter-01/replays/deck-01-vs-Crustle-game-001.json | head -n 120
```

If the exact replay filename differs, list the directory first and replace the
path in the command.

## 10. Select The Best Model

After the job finishes, pick the model with the highest benchmark win rate:

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
export PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
BEST_MODEL=$(
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PROJECT_RUN"])
best = None
for size in (1024, 512, 256):
    path = root / "experiments" / "rl" / "image_progression" / f"image-{size}" / "progression_summary.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    for item in data.get("summaries", []):
        ev = item.get("evaluation", {})
        row = (
            float(ev.get("win_rate", 0.0)),
            int(ev.get("wins", 0)),
            int(ev.get("games", 0)),
            size,
            int(item.get("iteration", 0)),
            item.get("model_path", ""),
        )
        if best is None or row > best:
            best = row
if best is None:
    raise SystemExit("no progression summaries found")
print(best[5])
PY
)
mkdir -p "$PROJECT_RUN"/experiments/rl/image_progression
echo "$BEST_MODEL" | tee "$PROJECT_RUN"/experiments/rl/image_progression/best_model.txt
```

Print the selected model metadata:

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
BEST_MODEL=$(cat "$PROJECT_RUN"/experiments/rl/image_progression/best_model.txt)
"$PY" -m json.tool "$BEST_MODEL" | head -n 80
```

## 11. Package A Kaggle Submission

Use the selected model. The current progression self-play defaults to deck `9`
vs deck `9`, so package deck `9` unless you decide to submit a different
prepared deck.

```bash
cd ~/ptcg_abc
PY=~/ptcg_abc/.conda_ptcg/bin/python
PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
BEST_MODEL=$(cat "$PROJECT_RUN"/experiments/rl/image_progression/best_model.txt)
PACKAGE_DECK_INDEX=9
"$PY" -m ptcg_abc rl-package \
  --model "$BEST_MODEL" \
  --deck-index "$PACKAGE_DECK_INDEX" \
  --output-dir submissions/phase4_image_progression
```

Verify the package path:

```bash
cd ~/ptcg_abc
PACKAGE_DECK_INDEX=9
ls -lh submissions/phase4_image_progression/deck-${PACKAGE_DECK_INDEX}/submission.tar.gz
```

## 12. Optional: Copy Results Back

Create a compact archive of summaries, benchmark markdown, and replay traces:

```bash
cd ~/ptcg_abc
PROJECT_RUN=/project/SIGGI/$USER/ptcg_abc_phase4
tar -czf experiments/rl/phase4_image_progression_results.tar.gz \
  -C "$PROJECT_RUN" \
  experiments/rl/image_progression \
  reports/image_progression \
  models/rl/image_progression
ls -lh experiments/rl/phase4_image_progression_results.tar.gz
```

From your local machine, copy the archive back:

```bash
scp thapanapong.r@erawan.cmu.ac.th:~/ptcg_abc/experiments/rl/phase4_image_progression_results.tar.gz .
```
