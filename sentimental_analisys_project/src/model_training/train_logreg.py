import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, LearningCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.linear_model  import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

from src.dataset.build_dataset import load_and_save_data
from src.dataset.clean_dataset import build_clean_dataset

from src.config import RAW_DATA_DIR, RAW_DATA_FILENAME, MODELS_DIR

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature



mlflow.set_tracking_uri("http://localhost:5000")

EXPERIMENT_NAME = "sentiment_reviews"
REGISTERED_MODEL_NAME = "sentiment-review-classifier"


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def load_data() -> pd.DataFrame:
    raw_data_path = RAW_DATA_DIR / RAW_DATA_FILENAME

    if not raw_data_path.is_file():
        print("Raw data not found. Downloading from Hugging Face...")
        raw_data_path = load_and_save_data()
    else:
        print(f"Raw data found: {raw_data_path}")

    df =  pd.read_parquet(raw_data_path)
    df = build_clean_dataset(df)
    df = df[["clean_text", "label"]].copy()

    return df

def split_data(df: pd.DataFrame):
    X = df["clean_text"].values
    y = df["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, 
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_pipeline() -> Pipeline:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=10**5,
            sublinear_tf=True
        )),
        ("logreg", LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1
        ))
    ])

    return pipeline

def evaluate_model(model:Pipeline, X, y) -> dict[str, float]:
    preds = model.predict(X)
    
    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, average="macro", zero_division=0)
    recall = recall_score(y, preds, average="macro", zero_division=0)
    f1 = f1_score(y, preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

def train_logreg():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="tfidf_logreg"):
        df = load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df=df)

        model = build_pipeline()

        mlflow.log_params({
            "model_type": "tfidf_logreg",
            "random_state": 42,
            "test_size": 0.15,
            "val_size": 0.15,
            "tfidf_ngram_range": "(1, 2)",
            "tfidf_min_df": 2,
            "tfidf_max_df": 0.95,
            "tfidf_max_features": 100000,
            "tfidf_sublinear_tf": True,
            "logreg_C": 2.0,
            "logreg_max_iter": 2000,
            "logreg_class_weight": "balanced",
            "logreg_solver": "lbfgs",
        })

        model.fit(X_train, y_train)

        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_metrics(prefix_metrics(val_metrics, "val"))
        mlflow.log_metrics(prefix_metrics(test_metrics, "test"))

        local_model_dir = MODELS_DIR / "logreg"
        local_model_dir.mkdir(parents=True, exist_ok=True)

        local_model_path = local_model_dir / "tfidf_logreg.pkl"
        joblib.dump(model, local_model_path)

        mlflow.log_artifact(str(local_model_path), artifact_path="local_exports")

        input_example = X_train[:5]
        signature = infer_signature(input_example, model.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_logreg()