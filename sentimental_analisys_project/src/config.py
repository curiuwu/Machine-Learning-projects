from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

DATASET_URL = "hf://datasets/k1tub/sentiment_dataset/data/train-00000-of-00001.parquet"
RAW_DATA_FILENAME = "sentiment_reviews_raw.parquet"
PROCESSED_DATA_FILENAME = "sentiment_reviews_processed.parquet"