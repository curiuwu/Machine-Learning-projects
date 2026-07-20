import copy
import json
import random
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn

from src.config import MODELS_DIR, PLOTS_DIR, REPORTS_DIR
from src.embedding.build_embedding_matrix import build_embedding_matrix
from src.embedding.build_vocab import build_word2idx
from src.embedding.word2vec import train_word2vec
from src.model_training.sequence_training_utils import (
    build_dataloader,
    build_sequence_dataset,
    load_sequence_dataframe,
    save_and_log_word2vec,
    split_sequence_data,
)
from src.model_training.torch_loops import evaluate, train_one_epoch
from src.models.lstm import LSTMModel
from src.plots.classification import build_and_save_confusion_matrix, make_classification_report
from src.plots.train_curves import build_and_save_torch_training_curves


mlflow.set_tracking_uri("http://localhost:5000")

EXPERIMENT_NAME = "sentiment_reviews"
RUN_NAME = "word2vec_lstm"
REGISTERED_MODEL_NAME = "sentiment-review-lstm"

MODEL_DIR_NAME = "lstm"
RANDOM_STATE = 42

CLASS_NAMES = ["neutral", "positive", "negative"]
LABELS = list(range(len(CLASS_NAMES)))
ID2LABEL = {idx: label for idx, label in enumerate(CLASS_NAMES)}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}

W2V_CONFIG = {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,
    "workers": 4,
    "sg": 1,
    "epochs": 10,
}

MODEL_CONFIG = {
    "model_type": "word2vec_lstm",
    "max_len": 100,
    "hidden_size": 96,
    "num_layers": 1,
    "num_classes": len(CLASS_NAMES),
    "dropout": 0.3,
    "freeze_embeddings": True,
}

TRAINING_CONFIG = {
    "random_state": RANDOM_STATE,
    "batch_size": 64,
    "num_epochs": 10,
    "patience": 3,
    "min_delta": 1e-4,
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "label_smoothing": 0.05,
    "grad_clip_max_norm": 1.0,
    "train_size": 0.70,
    "val_size": 0.15,
    "test_size": 0.15,
}

PREPROCESSING_CONFIG = {
    "tokenizer": "src.dataset.clean_dataset.tokenize",
    "normalization": "lowercase, replace ё with е, keep russian/latin letters and digits",
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
    "max_len": MODEL_CONFIG["max_len"],
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def make_metrics_dict(
    loss: float,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
) -> dict[str, float]:
    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def save_json(data: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return output_path


def save_lstm_inference_bundle(
    model: LSTMModel,
    embedding_matrix: torch.Tensor,
    word2idx: dict[str, int],
    model_config: dict,
    training_config: dict,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_state_path = output_dir / "model_state_dict.pt"
    embedding_matrix_path = output_dir / "embedding_matrix.pt"
    checkpoint_path = output_dir / "checkpoint.pt"
    word2idx_path = output_dir / "word2idx.pkl"
    id2label_path = output_dir / "id2label.json"
    label2id_path = output_dir / "label2id.json"
    model_config_path = output_dir / "model_config.json"
    training_config_path = output_dir / "training_config.json"
    preprocessing_config_path = output_dir / "preprocessing_config.json"

    model = model.to("cpu")
    embedding_matrix = embedding_matrix.detach().cpu()

    torch.save(model.state_dict(), model_state_path)
    torch.save(embedding_matrix, embedding_matrix_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "preprocessing_config": PREPROCESSING_CONFIG,
            "id2label": ID2LABEL,
            "label2id": LABEL2ID,
        },
        checkpoint_path,
    )
    joblib.dump(word2idx, word2idx_path)

    save_json(ID2LABEL, id2label_path)
    save_json(LABEL2ID, label2id_path)
    save_json(model_config, model_config_path)
    save_json(training_config, training_config_path)
    save_json(PREPROCESSING_CONFIG, preprocessing_config_path)

    return {
        "model_state": model_state_path,
        "embedding_matrix": embedding_matrix_path,
        "checkpoint": checkpoint_path,
        "word2idx": word2idx_path,
        "id2label": id2label_path,
        "label2id": label2id_path,
        "model_config": model_config_path,
        "training_config": training_config_path,
        "preprocessing_config": preprocessing_config_path,
    }


def train_lstm() -> tuple[LSTMModel, dict[str, float], dict[str, float]]:
    set_seed()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_model_dir = MODELS_DIR / MODEL_DIR_NAME

        mlflow.set_tags(
            {
                "model_family": "torch_sequence",
                "model_type": MODEL_CONFIG["model_type"],
                "embedding_type": "word2vec",
                "deployment_target": "fastapi",
            }
        )
        mlflow.log_params(
            {
                **{f"w2v_{key}": value for key, value in W2V_CONFIG.items()},
                **{f"model_{key}": value for key, value in MODEL_CONFIG.items()},
                **{f"train_{key}": value for key, value in TRAINING_CONFIG.items()},
                "device": str(device),
            }
        )

        df = load_sequence_dataframe()
        X = df["tokens"].tolist()
        y = df["label"].astype(int).tolist()

        X_train, X_val, X_test, y_train, y_val, y_test = split_sequence_data(X=X, y=y)

        mlflow.log_dict(
            {
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "labels": LABELS,
                "class_names": CLASS_NAMES,
            },
            artifact_file="data/split_summary.json",
        )

        w2v_model = train_word2vec(tokenized_texts=X_train, **W2V_CONFIG)
        word2vec_path = save_and_log_word2vec(
            w2v_model=w2v_model,
            model_dir_name=f"{MODEL_DIR_NAME}/word2vec",
            artifact_path="word2vec",
        )

        words = w2v_model.wv.index_to_key
        word2idx = build_word2idx(words=words)
        embedding_matrix = build_embedding_matrix(w2v_model=w2v_model, word2idx=word2idx)

        mlflow.log_metrics(
            {
                "vocab_size": len(word2idx),
                "embedding_dim": W2V_CONFIG["vector_size"],
                "word2vec_vocab_size": len(words),
            }
        )

        train_dataset = build_sequence_dataset(
            tokenized_text=X_train,
            labels=y_train,
            word2idx=word2idx,
            max_len=MODEL_CONFIG["max_len"],
        )
        val_dataset = build_sequence_dataset(
            tokenized_text=X_val,
            labels=y_val,
            word2idx=word2idx,
            max_len=MODEL_CONFIG["max_len"],
        )
        test_dataset = build_sequence_dataset(
            tokenized_text=X_test,
            labels=y_test,
            word2idx=word2idx,
            max_len=MODEL_CONFIG["max_len"],
        )

        train_loader = build_dataloader(
            dataset=train_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True,
        )
        val_loader = build_dataloader(
            dataset=val_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
        )
        test_loader = build_dataloader(
            dataset=test_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
        )

        model = LSTMModel(
            embedding_matrix=embedding_matrix,
            hidden_size=MODEL_CONFIG["hidden_size"],
            num_classes=MODEL_CONFIG["num_classes"],
            num_layers=MODEL_CONFIG["num_layers"],
            dropout=MODEL_CONFIG["dropout"],
            freeze_embeddings=MODEL_CONFIG["freeze_embeddings"],
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=TRAINING_CONFIG["label_smoothing"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
        )

        best_macro_f1 = -float("inf")
        best_epoch = 0
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        for epoch in range(TRAINING_CONFIG["num_epochs"]):
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_precision"].append(train_precision)
            history["train_recall"].append(train_recall)
            history["train_f1"].append(train_f1)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_precision"].append(val_precision)
            history["val_recall"].append(val_recall)
            history["val_f1"].append(val_f1)

            current_lr = optimizer.param_groups[0]["lr"]
            train_metrics = make_metrics_dict(
                train_loss,
                train_acc,
                train_precision,
                train_recall,
                train_f1,
            )
            val_metrics = make_metrics_dict(
                val_loss,
                val_acc,
                val_precision,
                val_recall,
                val_f1,
            )

            mlflow.log_metrics(prefix_metrics(train_metrics, "train"), step=epoch + 1)
            mlflow.log_metrics(prefix_metrics(val_metrics, "val"), step=epoch + 1)
            mlflow.log_metric("learning_rate", current_lr, step=epoch + 1)

            print(
                f"Epoch {epoch + 1}/{TRAINING_CONFIG['num_epochs']} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_acc:.4f} | "
                f"Train macro-F1: {train_f1:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f} | "
                f"Val macro-F1: {val_f1:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if val_f1 > best_macro_f1 + TRAINING_CONFIG["min_delta"]:
                best_macro_f1 = val_f1
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler.step(val_f1)

            if epochs_without_improvement >= TRAINING_CONFIG["patience"]:
                print(
                    f"Early stopping: validation macro-F1 не улучшается "
                    f"{TRAINING_CONFIG['patience']} эпохи подряд. "
                    f"Лучшая эпоха: {best_epoch}."
                )
                break

        model.load_state_dict(best_state_dict)
        print(f"Лучший validation macro-F1: {best_macro_f1:.4f}, эпоха: {best_epoch}")

        test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, test_preds = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        test_metrics = make_metrics_dict(
            test_loss,
            test_acc,
            test_precision,
            test_recall,
            test_f1,
        )

        mlflow.log_metric("best_val_f1_macro", best_macro_f1)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metrics(prefix_metrics(test_metrics, "test"))

        plots_dir = PLOTS_DIR / MODEL_DIR_NAME
        reports_dir = REPORTS_DIR / MODEL_DIR_NAME

        training_curves_path = build_and_save_torch_training_curves(
            history=history,
            output_dir=plots_dir,
            filename="training_curves.png",
        )
        confusion_matrix_path = build_and_save_confusion_matrix(
            y_true=test_labels,
            y_pred=test_preds,
            labels=LABELS,
            class_names=CLASS_NAMES,
            output_dir=plots_dir,
            filename="confusion_matrix.png",
        )
        report_path = make_classification_report(
            y_true=test_labels,
            y_pred=test_preds,
            labels=LABELS,
            class_names=CLASS_NAMES,
            output_dir=reports_dir,
            filename="classification_report.json",
        )
        history_path = save_json(history, reports_dir / "training_history.json")

        model = model.to("cpu")
        saved_paths = save_lstm_inference_bundle(
            model=model,
            embedding_matrix=embedding_matrix,
            word2idx=word2idx,
            model_config=MODEL_CONFIG,
            training_config={
                **TRAINING_CONFIG,
                "best_epoch": best_epoch,
                "best_val_f1_macro": best_macro_f1,
                "word2vec_path": str(word2vec_path),
            },
            output_dir=local_model_dir,
        )

        mlflow.log_artifact(str(training_curves_path), artifact_path="plots")
        mlflow.log_artifact(str(confusion_matrix_path), artifact_path="plots")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(history_path), artifact_path="reports")
        mlflow.log_artifacts(str(local_model_dir), artifact_path="local_exports/lstm")

        requirements_path = Path("requirements.txt")
        if requirements_path.is_file():
            mlflow.log_artifact(str(requirements_path), artifact_path="env")

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
        )

        print(f"Saved FastAPI bundle to: {local_model_dir}")
        print(f"Saved model artifacts: {saved_paths}")

        return model, {"best_val_f1_macro": best_macro_f1}, test_metrics


if __name__ == "__main__":
    train_lstm()
