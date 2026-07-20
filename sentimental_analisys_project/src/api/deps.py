import json
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from src.config import ARTIFACTS_DIR
from src.inference.BasePredictor import BasePredictor
from src.inference.factory import create_predictor


_predictor: BasePredictor | None = None
_load_error: str | None = None


def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "lstm").lower()


def get_model_artifact_dir() -> Path | None:
    value = os.getenv("MODEL_ARTIFACT_DIR")
    return Path(value) if value else None


def load_predictor() -> None:
    global _predictor, _load_error

    try:
        _predictor = create_predictor(
            model_name=get_model_name(),
            artifacts_dir=get_model_artifact_dir(),
            auto_load=True,
        )
        _load_error = None
    except Exception as exc:
        _predictor = None
        _load_error = str(exc)


def get_predictor() -> BasePredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {_load_error}",
        )

    return _predictor


def get_model_info() -> dict[str, Any]:
    if _predictor is None:
        return {
            "model_name": get_model_name(),
            "model_loaded": False,
            "artifact_dir": str(get_model_artifact_dir()) if get_model_artifact_dir() else None,
            "error": _load_error,
        }

    info = _predictor.info()
    info["model_name"] = get_model_name()
    return info


def get_model_metrics() -> dict[str, Any]:
    model_name = get_model_name()
    report_path = ARTIFACTS_DIR / "reports" / model_name / "classification_report.json"
    training_config_path = ARTIFACTS_DIR / "models" / model_name / "training_config.json"

    report = _read_json(report_path)
    training_config = _read_json(training_config_path)

    summary = {}
    if report:
        macro_avg = report.get("macro avg", {})
        weighted_avg = report.get("weighted avg", {})
        summary = {
            "accuracy": report.get("accuracy"),
            "macro_precision": macro_avg.get("precision"),
            "macro_recall": macro_avg.get("recall"),
            "macro_f1": macro_avg.get("f1-score"),
            "weighted_f1": weighted_avg.get("f1-score"),
        }

    if training_config:
        summary["best_epoch"] = training_config.get("best_epoch")
        summary["best_val_f1_macro"] = training_config.get("best_val_f1_macro")

    return {
        "model_name": model_name,
        "summary": summary,
        "classification_report": report,
        "training_config": training_config,
        "paths": {
            "classification_report": str(report_path),
            "training_config": str(training_config_path),
        },
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
