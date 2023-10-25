
import os
from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

from optimum.pipelines import pipeline
from optimum.onnxruntime import (
    AutoOptimizationConfig,
    AutoQuantizationConfig,
    ORTModelForSequenceClassification,
    ORTOptimizer
)


MODELS_PATH = '/src/.cache/optimum_models'


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
        self.device = 0 if torch.cuda.is_available() else -1
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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )
        self.model = self.model.to(self.device)

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

    def __call__(self, inputs) -> List[TextClassificationModelData]:
        logits = self.model(**inputs).logits
        return [
            TextClassificationModelData(self.name, **prediction)
            for prediction in self._results_from_logits(logits)
        ]


class OptimumOptimTransformer(TransformerTextClassificationModel):

    def __init__(self, name: str, model_path: str, tokenizer: str):
        super().__init__(name, model_path, tokenizer)

    def _optimize_model(self) -> str:
        save_dir = os.path.join(MODELS_PATH, f'{self.name}_opt')
        if os.path.isdir(save_dir):
            return save_dir
        model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path, export=True,
        )
        optimization_config = AutoOptimizationConfig.O3()
        optimizer = ORTOptimizer.from_pretrained(model)
        os.makedirs(save_dir)
        optimizer.optimize(
            save_dir=save_dir,
            optimization_config=optimization_config,
        )
        return save_dir

    def _warmup_model(self) -> None:
        dummy_input = 'complete a transaction from savings to checking of $2000'
        for _ in range(10):
            _ = self.pipe(dummy_input)

    def _load_model(self):
        print(f'Start {self.name} model loading.')
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        opt_model_dir = self._optimize_model()
        self.model = ORTModelForSequenceClassification.from_pretrained(
            opt_model_dir,
            provider="CUDAExecutionProvider",
        )
        self.pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            accelerator="ort",
            device=self.device,
        )
        self._warmup_model()
        print(f'Model {self.name} has been loaded successfully.')

    def __call__(self, texts: List[str]) -> List[TextClassificationModelData]:
        return [
            TextClassificationModelData(self.name, **prediction)
            for prediction in self.pipe(texts)
        ]


class TransformersFactory:
    def __init__(self):
        self.models_zoo = {
            'cardiffnlp': OptimumOptimTransformer,
            'ivanlau': OptimumOptimTransformer,
            'svalabs': OptimumOptimTransformer,
            'EIStakovskii': OptimumOptimTransformer,
            'jy46604790': OptimumOptimTransformer,
        }

    def create(
            self,
            model_name: str,
            model_path: str,
            tokenizer: str,
    ) -> TransformerTextClassificationModel:
        return self.models_zoo[model_name](model_name, model_path, tokenizer)
