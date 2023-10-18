from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

from infrastructure.models import BaseTextClassificationModel, TextClassificationModelData


class TextClassificationService:

    def __init__(self, models: List[BaseTextClassificationModel]):
        self.service_models = models

    def get_results(self, input_texts: List[str]) -> List[List[TextClassificationModelData]]:
        results = [model(input_texts) for model in self.service_models]
        return results

