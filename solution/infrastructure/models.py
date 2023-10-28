from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import List

import torch

import os
import onnxruntime
from optimum.pipelines import pipeline
from optimum.onnxruntime import ORTOptimizer, ORTModelForSequenceClassification, OptimizationConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig


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

    def __call__(self, inputs) -> List[TextClassificationModelData]:
        logits = self.model(**inputs).logits
        predictions = self._results_from_logits(logits)
        predictions = [TextClassificationModelData(self.name, **prediction) for prediction in predictions]
        return predictions


class OptTransformerTextClassificationModel(BaseTextClassificationModel):
    def _optimize_model(self) -> str:
        if self.device == 'cpu':
            saved_model_path = self._optimize_for_cpu()
        else:
            saved_model_path = self._optimize_for_gpu()
        return saved_model_path

    def _optimize_for_gpu(self):
        model_save_dir = f'/src/infrastructure/{self.name}_opt'

        # optional caching
        # if os.path.exists(model_save_dir) and len(os.listdir(model_save_dir)):
        #     return model_save_dir

        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 0
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path,
            export=True,
            provider="CUDAExecutionProvider",
            session_options=session_options
        )

        optimization_config = OptimizationConfig(
            optimize_for_gpu=True,
            optimization_level=99,
            fp16=True,
            enable_transformers_specific_optimizations=True,
        )
        optimizer = ORTOptimizer.from_pretrained(onnx_model)
        optimizer.optimize(save_dir=model_save_dir, optimization_config=optimization_config)
        return model_save_dir

    def _optimize_for_cpu(self):
        model_save_dir = f'/src/infrastructure/{self.name}_opt'

        # optional caching
        # if os.path.exists(model_save_dir) and len(os.listdir(model_save_dir)):
        #     return model_save_dir

        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 0
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path,
            export=True,
            provider='CPUExecutionProvider',
            session_options=session_options
        )

        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quantizer = ORTQuantizer.from_pretrained(onnx_model)
        model_save_dir = f'{model_save_dir}_dq'
        quantizer.quantize(save_dir=model_save_dir, quantization_config=qconfig)
        return model_save_dir

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        opt_model_path = self._optimize_model()

        self.pipeline = pipeline(
            "text-classification",
            model=opt_model_path,
            tokenizer=self.tokenizer,
            device=self.device,
            accelerator="ort"
        )
        print(f'Pipeline is set, device: {self.device}')

    def tokenize_texts(self, texts: List[str]):
        return texts

    def __call__(self, inputs: List[str]) -> List[TextClassificationModelData]:
        predictions = []
        for txt in inputs:
            sample = {'text': txt}
            res = self.pipeline(sample)

            predictions.append(TextClassificationModelData(
                model_name=self.name,
                label=res['label'],
                score=res['score'],
            ))
        return predictions
