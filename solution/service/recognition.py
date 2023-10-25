
from typing import List, Dict

from infrastructure.models import (
    BaseTextClassificationModel,
    TextClassificationModelData
)


class TextClassificationService:

    def __init__(self, models: Dict[str, BaseTextClassificationModel]):
        self.models = models

    def get_results(
            self,
            input_texts: List[str],
    ) -> List[List[TextClassificationModelData]]:
        return [model(input_texts) for name, model in self.models.keys()]
