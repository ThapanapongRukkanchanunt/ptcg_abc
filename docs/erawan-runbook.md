# ERAWAN Phase 4 Runbook

This is the tested ERAWAN path for Phase 4 RL jobs. It records the working
setup after the first successful login smoke and GPU check on June 20, 2026.

## Known ERAWAN Notes

- The login node may report `torch.cuda.is_available() == False`; that is normal.
- GPU availability must be checked inside a SLURM GPU allocation.
- Do not use the repo `.venv` for SLURM if it points at `/usr/bin/python`.
  In one failed setup, `.venv/bin/python` printed Python 3.6 in a batch job even
  though it printed Python 3.12 interactively.
- Do not use ERAWAN `python/3.11.1` for PyTorch if it lacks `_ctypes`.
- The working environment used `miniconda3/3.12` with a local conda env at
  `.conda_ptcg`.
- The working torch build was `torch 2.5.1+cu121`; a newer `cu130` wheel failed
  on one login check because the visible driver was too old there.

## One-Time Environment Setup

Run from the repo root on ERAWAN:

```bash
cd ~/ptcg_abc

module purge
module load miniconda3/3.12

conda create -y -p ~/ptcg_abc/.conda_ptcg python=3.12 pip

~/ptcg_abc/.conda_ptcg/bin/python --version
~/ptcg_abc/.conda_ptcg/bin/python -m pip install -U pip setuptools wheel
~/ptcg_abc/.conda_ptcg/bin/python -m pip install --ignore-requires-python -e .
~/ptcg_abc/.conda_ptcg/bin/python -m pip install numpy
~/ptcg_abc/.conda_ptcg/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Do not rely on `conda activate`; use the direct Python path:

```bash
~/ptcg_abc/.conda_ptcg/bin/python
```

Verify imports on the login node:

```bash
~/ptcg_abc/.conda_ptcg/bin/python - <<'PY'
import sys, torch, ptcg_abc
print(sys.version)
print(sys.executable)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

It is acceptable if CUDA is `False` here.

## Kaggle Data

If Kaggle files are under `data/input/`, create the default project path:

```bash
mkdir -p data/kaggle
ln -sfn ../input data/kaggle/input
```

Create the legal-card list without Kaggle credentials:

```bash
~/ptcg_abc/.conda_ptcg/bin/python -m ptcg_abc discover-legal-cards \
  --input-dir data/kaggle/input \
  --legal-source data/kaggle/input/EN_Card_Data.csv \
  --legal-cards data/kaggle/legal_cards.txt
```

Phase 4 needs these files:

```bash
test -f data/kaggle/input/EN_Card_Data.csv && echo "card data OK"
test -d data/kaggle/input/sample_submission && echo "sample submission OK"
test -f data/kaggle/legal_cards.txt && echo "legal cards OK"
```

## Login-Node Smoke

Keep this tiny. The goal is wiring, not performance.

```bash
~/ptcg_abc/.conda_ptcg/bin/python -m ptcg_abc rl-collect-bc \
  --games 4 \
  --max-steps 60 \
  --output data/datasets/rl/bc_smoke.jsonl

~/ptcg_abc/.conda_ptcg/bin/python -m ptcg_abc rl-train-bc \
  --backend torch \
  --dataset data/datasets/rl/bc_smoke.jsonl \
  --checkpoint models/rl/bc_smoke.pt \
  --model models/rl/bc_smoke_export.json \
  --epochs 1

~/ptcg_abc/.conda_ptcg/bin/python -m ptcg_abc rl-evaluate \
  --agent hybrid \
  --model models/rl/bc_smoke_export.json \
  --games-per-matchup 1 \
  --max-steps 120
```

Pass condition: `errors: 0`. Timeouts are acceptable at these short step caps.

## GPU Check Job

Create the GPU check script:

```bash
cat > scripts/slurm/gpu_check_conda.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=gpu-check
#SBATCH --output=experiments/rl/slurm-%j-gpu-check.out
#SBATCH --error=experiments/rl/slurm-%j-gpu-check.err
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load miniconda3/3.12

cd "$SLURM_SUBMIT_DIR"
PY="$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python"

nvidia-smi || true
"$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
EOF
```

Submit:

```bash
sbatch --gres=gpu:1 scripts/slurm/gpu_check_conda.sbatch
```

Check:

```bash
cat experiments/rl/slurm-<jobid>-gpu-check.out
cat experiments/rl/slurm-<jobid>-gpu-check.err
```

Known good result:

```text
NVIDIA A100-SXM4-80GB
torch 2.5.1+cu121
torch cuda runtime 12.1
cuda available True
```

## Medium BC Trend Job

Use the checked-in conda script:

```text
scripts/slurm/phase4_bc_conda.sbatch
```

The script loads `miniconda3/3.12`, uses the direct
`$SLURM_SUBMIT_DIR/.conda_ptcg/bin/python` interpreter, exports `PYTHONPATH`,
checks CUDA inside the allocation, and prints the torch checkpoint format. It
saves the training-side CNN actor/value checkpoint to `models/rl/bc_model.pt`
and the Kaggle-safe JSON fallback to `models/rl/bc_model.json`.

Useful environment overrides:

```text
BC_GAMES=1000
BC_EPOCHS=2
MAX_STEPS=600
BC_LEARNING_RATE=0.02
BC_DATASET=data/datasets/rl/bc_decisions.jsonl
BC_CHECKPOINT=models/rl/bc_model.pt
BC_MODEL=models/rl/bc_model.json
BC_REPORT=experiments/rl/bc_train_report.json
```

Submit the medium trend job:

```bash
BC_GAMES=1000 BC_EPOCHS=2 MAX_STEPS=600 \
sbatch --gres=gpu:1 scripts/slurm/phase4_bc_conda.sbatch
```

Monitor:

```bash
squeue -u "$USER"
cat experiments/rl/slurm-<jobid>-bc.out
cat experiments/rl/slurm-<jobid>-bc.err
```

Evaluate after the job finishes:

```bash
~/ptcg_abc/.conda_ptcg/bin/python -m ptcg_abc rl-evaluate \
  --agent hybrid \
  --model models/rl/bc_model.json \
  --games-per-matchup 10 \
  --max-steps 600 \
  --report-json reports/phase4_medium_benchmark.json \
  --report-md reports/phase4_medium_benchmark.md
```

Compare the medium result to the Phase 3 baseline:

```text
Phase 3 baseline: 138 wins / 360 games, win rate 0.383
```

## Large BC And Package

If the medium trend has `errors: 0`, launch the large BC run:

```bash
BC_GAMES=20000 BC_EPOCHS=3 MAX_STEPS=600 \
sbatch --gres=gpu:1 scripts/slurm/phase4_bc_conda.sbatch
```

Evaluate:

```bash
~/ptcg_abc/.conda_ptcg/bin/python -m ptcg_abc rl-evaluate \
  --agent hybrid \
  --model models/rl/bc_model.json \
  --games-per-matchup 10 \
  --max-steps 600 \
  --report-json reports/phase4_large_bc_benchmark.json \
  --report-md reports/phase4_large_bc_benchmark.md
```

Package a Kaggle submission for the selected deck:

```bash
~/ptcg_abc/.conda_ptcg/bin/python -m ptcg_abc rl-package \
  --model models/rl/bc_model.json \
  --deck-index 1 \
  --output-dir submissions/phase4
```

Expected package path:

```text
submissions/phase4/deck-1/submission.tar.gz
```

## Image-Size Progression Experiment

Use this after the compact board snapshots are accepted and you want the
10-iteration progression sweep:

```bash
IMAGE_SIZES="256 512 1024" \
ITERATIONS=10 \
SELFPLAY_GAMES=1000 \
EVAL_GAMES_PER_MATCHUP=100 \
MAX_STEPS=600 \
DECK_A_INDEX=9 \
DECK_B_INDEX=9 \
sbatch --gres=gpu:1 scripts/slurm/phase4_image_progression_conda.sbatch
```

This runs three separate experiments. For each image size, each iteration does:

```text
1,000 Hybrid-vs-Hybrid self-play games
reward-weighted model update
3,600 Hybrid-vs-Benchmark evaluation games
36 compact replay traces, one per 9x4 matchup
```

Outputs:

```text
data/datasets/rl/image_progression/image-<size>/
models/rl/image_progression/image-<size>/
reports/image_progression/image-<size>/
experiments/rl/image_progression/image-<size>/progression_summary.json
experiments/rl/image_progression/image-<size>/iter-<NN>/replays/
```

The runner keeps rollout records compact for storage. The current update step
uses the existing reward-weighted JSON model path; the `image_size` value is
recorded in model metadata and replay/progression artifacts so the 256, 512, and
1024 sweeps can be compared cleanly.
