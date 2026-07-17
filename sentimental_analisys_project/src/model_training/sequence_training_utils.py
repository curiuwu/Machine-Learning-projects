from pathlib import Path

import mlflow
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.dataset.build_dataset import load_and_save_data
from src.dataset.clean_dataset import build_clean_dataset
from src.dataset.torch_dataset import ReviewsDataset
from src.config import RAW_DATA_DIR, RAW_DATA_FILENAME, MODELS_DIR

def load_sequence_dataframe() -> pd.DataFrame:
    raw_data_path = RAW_DATA_DIR / RAW_DATA_FILENAME

    if not raw_data_path.is_file():
        print("Raw data not found. Downloading from Hugging Face...")
        raw_data_path = load_and_save_data()
    else:
        print(f"Raw data found: {raw_data_path}")

    df = pd.read_parquet(raw_data_path)
    df = build_clean_dataset(df)
    df = df[["tokens", "label"]].copy()

    return df

def split_sequence_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        list(X),
        list(y),
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_sequence_dataset(tokenized_text, labels, word2idx: dict[str, int], max_len: int = 100) -> ReviewsDataset:
    dataset = ReviewsDataset(
        tokenized_text=list(tokenized_text),
        labels=list(labels),
        word2idx=word2idx,
        max_len=max_len
    )

    return dataset

def build_dataloader(dataset: Dataset, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    return dataloader

def save_and_log_word2vec(
    w2v_model: Word2Vec,
    model_dir_name: str,
    artifact_path: str = "word2vec",
) -> Path:
    word2vec_dir = MODELS_DIR / model_dir_name
    word2vec_dir.mkdir(parents=True, exist_ok=True)

    word2vec_path = word2vec_dir / "word2vec.model"

    w2v_model.save(str(word2vec_path))

    mlflow.log_artifacts(
        local_dir=str(word2vec_dir),
        artifact_path=artifact_path
    )

    return word2vec_path
