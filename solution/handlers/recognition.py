from typing import List
import asyncio

from infrastructure.models import TextClassificationModelData
from service.recognition import TextClassificationService
from handlers.data_models import ResponseSchema, RecognitionSchema


class PredictionHandler:

    def __init__(
            self,
            recognition_service: TextClassificationService,
            timeout: float
    ):
        self.recognition_service = recognition_service
        self.timeout = timeout

    async def handle(self, model_name, model_queue):
        while True:
            texts = []
            queues = []

            try:
                while True:
                    (text, response_queue) = await asyncio.wait_for(
                        model_queue.get(), timeout=self.timeout
                    )
                    texts.append(text)
                    queues.append(response_queue)
            except asyncio.exceptions.TimeoutError:
                pass

            if texts:
                outs = self.recognition_service.models[model_name](texts)
                for rq, out in zip(queues, outs):
                    await rq.put(out)

    def serialize_answer(
            self,
            results: List[TextClassificationModelData]
    ) -> ResponseSchema:
        res_model = {
            rec.model_name: self._recognitions_to_schema(rec) for rec in results
        }
        return ResponseSchema(**res_model)

    @staticmethod
    def _recognitions_to_schema(
            recognition: TextClassificationModelData
    ) -> RecognitionSchema:
        if recognition.model_name != "ivanlau":
            recognition.label = recognition.label.upper()
        return RecognitionSchema(
            score=recognition.score, label=recognition.label
        )
