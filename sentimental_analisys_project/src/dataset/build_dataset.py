from pathlib import Path

import pandas as pd

from src.config import DATASET_URL, RAW_DATA_DIR, RAW_DATA_FILENAME


def load_raw_dataset(dataset_url: str = DATASET_URL) -> pd.DataFrame:
    return pd.read_parquet(dataset_url)


def save_raw_dataset(
    df: pd.DataFrame,
    output_dir: Path = RAW_DATA_DIR,
    filename: str = RAW_DATA_FILENAME,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_parquet(output_path, index=False)

    return output_path


def load_and_save_data(
    output_dir: Path = RAW_DATA_DIR,
    dataset_url: str = DATASET_URL,
    filename: str = RAW_DATA_FILENAME,
) -> Path:
    df = load_raw_dataset(dataset_url)
    return save_raw_dataset(df=df, output_dir=output_dir, filename=filename)


def main() -> None:
    output_path = load_and_save_data()
    print(f"Saved raw data to {output_path}")


if __name__ == "__main__":
    main()
