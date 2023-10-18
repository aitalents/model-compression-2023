from typing import List
import asyncio

from pydantic import ValidationError

from infrastructure.models import TextClassificationModelData
from service.recognition import TextClassificationService
from handlers.data_models import ResponseSchema, RecognitionSchema


class PredictionHandler:

    def __init__(self, recognition_service: TextClassificationService, timeout: float):
        self.recognition_service = recognition_service
        self.timeout = timeout

    async def handle(self, model_name, model_queue, max_batch_size: int):
        while True:
            inputs = None
            texts = []
            queues = []

            try:
                while True:
                    (text, response_queue) = await asyncio.wait_for(model_queue.get(), timeout=self.timeout)
                    texts.append(text)
                    queues.append(response_queue)
            except asyncio.exceptions.TimeoutError:
                pass

            if texts:
                model = next(
                        (model for model in self.recognition_service.service_models if model.name == model_name),
                        None
                        )
                if model:
                    for text_batch in self._perform_batches(texts, max_batch_size):
                        inputs = model.tokenize_texts(texts)
                        outs = model(inputs)
                        for rq, out in zip(queues, outs):
                            await rq.put(out)

    def serialize_answer(self, results: List[TextClassificationModelData]) -> ResponseSchema:
        res_model = {rec.model_name: self._recognitions_to_schema(rec) for rec in results}
        return ResponseSchema(**res_model)

    def _recognitions_to_schema(self, recognition: TextClassificationModelData) -> RecognitionSchema:
        if recognition.model_name != "ivanlau":
            recognition.label = recognition.label.upper()
        return RecognitionSchema(score=recognition.score, label=recognition.label)

    def _perform_batches(self, texts: List[str], max_batch_size):
        for i in range(0, len(texts), max_batch_size):
            yield texts[i:i + max_batch_size]

