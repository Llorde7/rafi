from pydantic import BaseModel
from typing import Optional


class EmotionScore(BaseModel):
    emotion: str
    confidence: float


class ClassifierInput(BaseModel):
    text: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ClassifierOutput(BaseModel):
    text: str
    translation: Optional[str]
    top_3: list[EmotionScore]
    reasoning: str
