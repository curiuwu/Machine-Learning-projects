from abc import ABC, abstractmethod
from typing import Any


class BasePredictor(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def predict(self, text: str) -> dict[str, Any]:
        pass

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        return [self.predict(text) for text in texts]

    @abstractmethod
    def info(self) -> dict[str, Any]:
        pass
