from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class ErrorResponse(BaseModel):
    detail: str
