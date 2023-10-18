from typing import List

from pydantic import ValidationError

from infrastructure.models import TextClassificationModelData
from service.recognition import TextClassificationService
from handlers.data_models import ResponseSchema, RecognitionSchema


class PredictionHandler:

    def __init__(self, recognition_service: TextClassificationService):
        self.recognition_service = recognition_service

    def handle(self, body: str) -> ResponseSchema:
        query_results = self.recognition_service.get_results(body)
        result = self.serialize_answer(query_results)
        return result

    def serialize_answer(self, results: List[TextClassificationModelData]) -> ResponseSchema:
        results = {rec.model_name: self._recognitions_to_schema(rec) for rec in results}
        return ResponseSchema(**results)

    def _recognitions_to_schema(self, recognition: TextClassificationModelData) -> RecognitionSchema:
        if recognition.model_name != "ivanlau":
            recognition.label = recognition.label.upper()
        return RecognitionSchema(score=recognition.score, label=recognition.label)

