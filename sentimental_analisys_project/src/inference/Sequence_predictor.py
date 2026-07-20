import json
from pathlib import Path
from typing import Any

import joblib
import torch

from src.dataset.clean_dataset import tokenize
from src.embedding.build_vocab import PAD_TOKEN, UNK_TOKEN
from src.inference.BasePredictor import BasePredictor
from src.config import ARTIFACTS_DIR
from src.models.rnn import RNNModel
from src.models.lstm import LSTMModel



class SequencePredictor(BasePredictor):
    def __init__(self, artifacts_dir: str | Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.word2idx = None
        self.id2label = None
        self.model_config = None
        self.max_len = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        self.model_config = self._load_json("model_config.json")
        self.id2label = self._load_id2label("id2label.json")
        self.word2idx = joblib.load(self.artifacts_dir / "word2idx.pkl")

        self.max_len = self.model_config["max_len"]

        embedding_matrix = torch.load(
            self.artifacts_dir / "embedding_matrix.pt",
            map_location=self.device
        )

        self.model = self._build_model(embedding_matrix)
        state_dict = torch.load(
            self.artifacts_dir / "model_state_dict.pt",
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        tokens, input_ids, length = self._encode_text(text)

        with torch.no_grad():
            logits = self.model(input_ids, length)
            probs = torch.softmax(logits, dim=1)
            pred_id = int(torch.argmax(probs, dim=1).item())

        pred_label = self.id2label[pred_id]
        probabilities = {
            self.id2label[i]: float(probs[0][i].cpu())
            for i in range(len(self.id2label))
        }

        confidence = max(probabilities.values())
        result = {
            "text": text,
            "tokens": tokens,
            "label_id": pred_id,
            "label": pred_label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

        return result

    def info(self) -> dict[str, Any]:
        model_type = None
        classes = []

        if self.model_config is not None:
            model_type = self.model_config.get("model_type")

        if self.id2label is not None:
            classes = [
                self.id2label[class_id]
                for class_id in sorted(self.id2label)
            ]

        return {
            "model_type": model_type,
            "artifact_dir": str(self.artifacts_dir),
            "model_loaded": self.model is not None,
            "classes": classes,
            "device": str(self.device),
        }

    def _build_model(self, embedding_matrix: torch.Tensor):
        model_type = self.model_config["model_type"]

        if model_type == "word2vec_rnn":
            model_cls = RNNModel
        elif model_type == "word2vec_lstm":
            model_cls = LSTMModel
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        return model_cls(
            embedding_matrix=embedding_matrix,
            hidden_size=self.model_config["hidden_size"],
            num_layers=self.model_config["num_layers"],
            num_classes=self.model_config["num_classes"],
            dropout=self.model_config["dropout"],
            freeze_embeddings=self.model_config["freeze_embeddings"],
        )

    def _encode_text(self, text: str):
        tokens = tokenize(text)

        ids = [
            self.word2idx.get(token, self.word2idx[UNK_TOKEN])
            for token in tokens
        ]

        ids = ids[:self.max_len]

        if len(ids) == 0:
            ids = [self.word2idx[UNK_TOKEN]]

        length = len(ids)
        padding_length = self.max_len - length
        ids = ids + [self.word2idx[PAD_TOKEN]] * padding_length

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        length_tensor = torch.tensor([length], dtype=torch.long, device=self.device)

        return tokens, input_ids, length_tensor

    def _load_json(self, filename: str) -> dict:
        with (self.artifacts_dir / filename).open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_id2label(self, filename: str) -> dict[int, str]:
        data = self._load_json(filename)
        return {int(key): value for key, value in data.items()}
