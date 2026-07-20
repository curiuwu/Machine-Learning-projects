import json
from pathlib import Path
from typing import Any

import joblib

from src.config import ARTIFACTS_DIR
from src.dataset.clean_dataset import clean_text
from src.inference.BasePredictor import BasePredictor


class LogRegPredictor(BasePredictor):
    def __init__(self, artifacts_dir: str | Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.model_path = self.artifacts_dir / "models" / "logreg" / "tfidf_logreg.pkl"
        self.id2label = self._load_id2label()
        self.model = None

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"LogReg model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)

    def predict(self, text: str) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call .load() before predict().")

        clean = clean_text(text)
        pred_id = self.model.predict([clean])[0]
        pred_id = int(pred_id)
        pred_label = self.id2label[pred_id]

        probs = self.model.predict_proba([clean])[0]
        probabilities = {
            self.id2label[int(class_id)]: float(prob)
            for class_id, prob in zip(self.model.classes_, probs)
        }
        confidence = max(probabilities.values())

        result = {
            "text": text,
            "clean_text": clean,
            "label_id": pred_id,
            "label": pred_label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

        return result

    def info(self) -> dict[str, Any]:
        return {
            "model_type": "tfidf_logreg",
            "model_path": str(self.model_path),
            "model_loaded": self.model is not None,
            "classes": [
                self.id2label[class_id]
                for class_id in sorted(self.id2label)
            ],
        }

    def _load_id2label(self) -> dict[int, str]:
        candidate_paths = [
            self.artifacts_dir / "models" / "logreg" / "id2label.json",
            self.artifacts_dir / "models" / "rnn" / "id2label.json",
            self.artifacts_dir / "models" / "lstm" / "id2label.json",
        ]

        for path in candidate_paths:
            if path.is_file():
                with path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
                return {int(class_id): label for class_id, label in data.items()}

        return {
            0: "neutral",
            1: "positive",
            2: "negative",
        }
