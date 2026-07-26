from pathlib import Path

from src.config import ARTIFACTS_DIR, MODELS_DIR
from src.inference.BasePredictor import BasePredictor
from src.inference.LogReg_predictor import LogRegPredictor
from src.inference.Sequence_predictor import SequencePredictor


def create_predictor(
    model_name: str = "lstm",
    artifacts_dir: str | Path | None = None,
    auto_load: bool = True,
) -> BasePredictor:
    model_name = model_name.lower()

    if model_name == "logreg":
        if artifacts_dir is not None and (Path(artifacts_dir) / "tfidf_logreg.pkl").is_file():
            predictor = LogRegPredictor(model_dir=artifacts_dir)
        else:
            predictor = LogRegPredictor(artifacts_dir=artifacts_dir or ARTIFACTS_DIR)

    elif model_name in {"rnn", "lstm", "bilstm"}:
        model_artifact_dir = Path(artifacts_dir) if artifacts_dir is not None else MODELS_DIR / model_name
        predictor = SequencePredictor(
            artifacts_dir=model_artifact_dir
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if auto_load:
        predictor.load()

    return predictor
