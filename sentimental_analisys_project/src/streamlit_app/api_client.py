import os
from typing import Any

import requests


DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")


class SentimentApiClient:
    def __init__(self, api_url: str = DEFAULT_API_URL, timeout: float = 15.0) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def predict(self, text: str) -> dict[str, Any]:
        return self._post("/predict", {"text": text})

    def predict_batch(self, texts: list[str]) -> dict[str, Any]:
        return self._post("/predict_batch", {"texts": texts})

    def info(self) -> dict[str, Any]:
        return self._get("/info")

    def metrics(self) -> dict[str, Any]:
        return self._get("/metrics")

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(f"{self.api_url}{path}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.api_url}{path}", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
