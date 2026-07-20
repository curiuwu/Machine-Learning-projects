from pathlib import Path

from src.config import ARTIFACTS_DIR, MODELS_DIR
from src.inference.BasePredictor import BasePredictor
from src.inference.LogReg_predictor import LogRegPredictor
from src.inference.Sequence_predictor import SequencePredictor

def create_predictor(
        model_name: str = "lstm",
        artifacts_dir: str | Path = ARTIFACTS_DIR,
        auto_load: bool = True
) -> BasePredictor:
    artifacts_dir = Path(artifacts_dir)

    if model_name == "logreg":
        predictor = LogRegPredictor(artifacts_dir=artifacts_dir)

    elif model_name in {"rnn", "lstm"}:
        predictor = SequencePredictor(
            artifacts_dir==MODELS_DIR / model_name
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    if auto_load:
        predictor.load()

    return predictor