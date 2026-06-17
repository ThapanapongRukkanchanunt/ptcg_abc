from pathlib import Path


COMPETITION_SLUG = "pokemon-tcg-ai-battle"
LIMITLESS_BASE_URL = "https://limitlesstcg.com"
LIMITLESS_FORMAT = "TEF-POR"

DATA_DIR = Path("data")
KAGGLE_RAW_DIR = DATA_DIR / "kaggle" / "raw"
KAGGLE_INPUT_DIR = DATA_DIR / "kaggle" / "input"
LEGAL_CARDS_PATH = DATA_DIR / "kaggle" / "legal_cards.txt"
LIMITLESS_RAW_DIR = DATA_DIR / "raw" / "limitless"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
