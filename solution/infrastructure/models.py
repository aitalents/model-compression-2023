from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from optimum.onnxruntime import ORTOptimizer, ORTModelForSequenceClassification


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
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._load_model()

    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def __call__(self, inputs) -> List[TextClassificationModelData]:
        ...


class TransformerTextClassificationModel(BaseTextClassificationModel):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)

    def tokenize_texts(self, texts: List[str]):
        inputs = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to GPU
        return inputs

    def _results_from_logits(self, logits: torch.Tensor):
        id2label = self.model.config.id2label

        label_ids = logits.argmax(dim=1)
        scores = logits.softmax(dim=-1)
        results = [
            {"label": id2label[label_id.item()], "score": score[label_id.item()].item()}
            for label_id, score in zip(label_ids, scores)
        ]
        return results

    def __call__(self, inputs) -> List[TextClassificationModelData]:
        logits = self.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).logits
        predictions = self._results_from_logits(logits)
        predictions = [
            TextClassificationModelData(self.name, **prediction)
            for prediction in predictions
        ]
        return predictions


class OptimizedTransformerTextClassificationModel(TransformerTextClassificationModel):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path, export=True
        )
        self.model = self.model.to(self.device)


# models = [
#     {
#         "name": "cardiffnlp",
#         "model_path": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
#         "tokenizer": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
#     },
#     {
#         "name": "ivanlau",
#         "model_path": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
#         "tokenizer": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
#     },
#     {
#         "name": "svalabs",
#         "model_path": "svalabs/twitter-xlm-roberta-crypto-spam",
#         "tokenizer": "svalabs/twitter-xlm-roberta-crypto-spam",
#     },
#     {
#         "name": "EIStakovskii",
#         "model_path": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
#         "tokenizer": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
#     },
#     {
#         "name": "jy46604790",
#         "model_path": "jy46604790/Fake-News-Bert-Detect",
#         "tokenizer": "jy46604790/Fake-News-Bert-Detect",
#     },
# ]


# def test():
#     for model in models:
#         print(f"Testing {model['name']}")
#         model = OptimizedTransformerTextClassificationModel(**model)
#         # model = TransformerTextClassificationModel(**model)
#         inputs = model.tokenize_texts(["Hello world!"])
#         predictions = model(inputs)
#         print(predictions)


# test()
