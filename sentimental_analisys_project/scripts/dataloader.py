from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_URL = "hf://datasets/k1tub/sentiment_dataset/data/train-00000-of-00001.parquet"


def load_and_save_data(output_dir: Path = RAW_DATA_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATASET_URL)

    raw_data_path = output_dir / "sentiment_reviews_raw.parquet"
    df.to_parquet(raw_data_path, index=False)

    return raw_data_path

