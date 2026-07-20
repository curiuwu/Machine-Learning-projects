import re
from pathlib import Path

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILENAME,
    RAW_DATA_DIR,
    RAW_DATA_FILENAME,
)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = text.replace("ё", "е")
    return re.findall(r"[а-яa-z0-9]+", text)


def build_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df[["text", "label"]].copy()
    clean_df = clean_df.dropna()
    clean_df["text"] = clean_df["text"].astype(str)
    clean_df["label"] = clean_df["label"].astype(int)
    clean_df["clean_text"] = clean_df["text"].apply(clean_text)
    clean_df["tokens"] = clean_df["text"].apply(tokenize)
    clean_df["tokens_len"] = clean_df["tokens"].apply(len)
    clean_df = clean_df[clean_df["tokens_len"] > 0].copy()
    return clean_df


def clean_and_save_dataset(
    input_path: Path = RAW_DATA_DIR / RAW_DATA_FILENAME,
    output_dir: Path = PROCESSED_DATA_DIR,
    filename: str = PROCESSED_DATA_FILENAME,
) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    clean_df = build_clean_dataset(df)

    output_path = output_dir / filename
    clean_df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    output_path = clean_and_save_dataset()
    print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    main()
