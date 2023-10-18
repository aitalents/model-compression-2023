from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from transformers import pipeline


@dataclass
class TextClassificationModelData:
    model_name: str
    label: str
    score: float


class BaseTextClassificationModel(ABC):

    def __init__(self, name: str, model_path: str, tokenizer: str):
        self.name = name
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Callable:
        ...

    @abstractmethod
    def __call__(self, input_text: str) -> TextClassificationModelData:
        ...


class TransformerTextClassificationModel(BaseTextClassificationModel):

    def _load_model(self):
        sentiment_task = pipeline(
                "sentiment-analysis",
                model=self.model_path,
                tokenizer=self.model_path,
                device=self.device
                )
        return sentiment_task

    def __call__(self, input_text: str) -> TextClassificationModelData:
        if isinstance(input_text, str):
            prediction = self.model(input_text)[0]
            prediction = TextClassificationModelData(self.name, **prediction)
            return prediction
        else:
            raise TypeError("Model input text must be str type")

