import gc
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import torch
from optimum.onnxruntime import ORTOptimizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoOptimizationConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time


# from optimum.bettertransformer import BetterTransformer


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute")
        return result

    return wrapper


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
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to GPU
        return inputs

    def _results_from_logits(self, logits: torch.Tensor):
        id2label = self.model.config.id2label

        label_ids = logits.argmax(dim=1)
        scores = logits.softmax(dim=-1)
        results = [
            {
                "label": id2label[label_id.item()],
                "score": score[label_id.item()].item()
            }
            for label_id, score in zip(label_ids, scores)
        ]
        return results

    @torch.inference_mode()
    @measure_execution_time
    def __call__(self, inputs) -> List[TextClassificationModelData]:
        logits = self.model(**inputs).logits
        predictions = self._results_from_logits(logits)
        predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        return predictions


class OptimizedTransformerTextClassificationModel(BaseTextClassificationModel):

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.model = None
        self.model = self._model_optimizer()
        self.model = self.model.to(self.device)

    def _model_optimizer(self) -> ORTModelForSequenceClassification:
        if self.model:
            return self.model
        if not os.path.isdir(f'optimized_models/{self.name}'):
            model = ORTModelForSequenceClassification.from_pretrained(self.model_path, export=True)
            print(f'Optimizing model: {self.name}')
            optimizer = ORTOptimizer.from_pretrained(model)
            optimizer.optimize(save_dir=os.path.join('./optimized_models', self.name),
                               optimization_config=AutoOptimizationConfig.O2())

            del optimizer
            del model
            gc.collect()

        optimized_model = ORTModelForSequenceClassification.from_pretrained(
            os.path.join('./optimized_models', self.name)
        )
        print(f'Loaded optimized model: {self.name}')
        return optimized_model

    def tokenize_texts(self, texts: List[str]):
        inputs = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to GPU
        return inputs

    def _results_from_logits(self, logits: torch.Tensor):
        id2label = self.model.config.id2label

        label_ids = logits.argmax(dim=1)
        scores = logits.softmax(dim=-1)
        results = [
            {
                "label": id2label[label_id.item()],
                "score": score[label_id.item()].item()
            }
            for label_id, score in zip(label_ids, scores)
        ]
        return results

    @torch.inference_mode()
    @measure_execution_time
    def __call__(self, inputs) -> List[TextClassificationModelData]:
        logits = self.model(**inputs).logits
        predictions = self._results_from_logits(logits)
        predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        return predictions
