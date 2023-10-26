from typing import List

from pydantic import BaseModel, validator


class RecognitionSchema(BaseModel):
    score: float
    label: str


class ResponseSchema(BaseModel):
    cardiffnlp: RecognitionSchema
    ivanlau: RecognitionSchema
    svalabs: RecognitionSchema
    EIStakovskii: RecognitionSchema
    jy46604790: RecognitionSchema

