# Kaggle-First Workflow

Use the Kaggle competition files as the source of legal cards before collecting Limitless decks.

## 1. Add Kaggle credentials

Create an API token in Kaggle account settings, then use either option:

```powershell
$env:KAGGLE_USERNAME="your_kaggle_username"
$env:KAGGLE_KEY="your_kaggle_api_key"
```

Or place `kaggle.json` at:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

Do not commit Kaggle credentials.

## 2. Install the local package

```powershell
python -m pip install -e .
```

If `python` is not on PATH in this Codex workspace, use the bundled Python runtime shown by the app's workspace dependencies.

## 3. Download Kaggle data and discover legal cards

```powershell
python -m ptcg_abc kaggle-setup
```

If you already have the competition archive, use it directly:

```powershell
python -m ptcg_abc kaggle-setup --archive pokemon-tcg-ai-battle.zip
```

This writes:

- `data/kaggle/raw/files.json`
- `data/kaggle/raw/pokemon-tcg-ai-battle.zip`
- `data/kaggle/input/`
- `data/kaggle/legal_cards.txt`
- `reports/legal_card_candidates.md`

If multiple plausible legal-card files are found, choose the right one from the candidate report and rerun:

```powershell
python -m ptcg_abc discover-legal-cards --legal-source path\to\legal_cards_file.csv
```

## 4. Compare Limitless cards against Kaggle legality

```powershell
python -m ptcg_abc missing-limitless
```

This writes `reports/missing_limitless_cards.md`, with an empty column for the alternative card you want to provide.
