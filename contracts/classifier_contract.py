from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class GoEmotionLabel(str, Enum):
    ADMIRATION = "admiration"
    AMUSEMENT = "amusement"
    ANGER = "anger"
    ANNOYANCE = "annoyance"
    APPROVAL = "approval"
    CARING = "caring"
    CONFUSION = "confusion"
    CURIOSITY = "curiosity"
    DESIRE = "desire"
    DISAPPOINTMENT = "disappointment"
    DISAPPROVAL = "disapproval"
    DISGUST = "disgust"
    EMBARRASSMENT = "embarrassment"
    EXCITEMENT = "excitement"
    FEAR = "fear"
    GRATITUDE = "gratitude"
    GRIEF = "grief"
    JOY = "joy"
    LOVE = "love"
    NERVOUSNESS = "nervousness"
    NEUTRAL = "neutral"
    OPTIMISM = "optimism"
    FRUSTRATION = "frustration"
    PRIDE = "pride"
    REALIZATION = "realization"
    RELIEF = "relief"
    REMORSE = "remorse"
    SADNESS = "sadness"
    SURPRISE = "surprise"


class EmotionScore(BaseModel):
    emotion: GoEmotionLabel
    confidence: float = Field(ge=0.0, le=1.0)


class ClassifierInput(BaseModel):
    text: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ClassifierOutput(BaseModel):
    text: str
    translation: Optional[str]
    top_3: list[EmotionScore] = Field(min_length=3, max_length=3)
    reasoning: str

    @model_validator(mode="after")
    def validate_top_3(self) -> "ClassifierOutput":
        total_confidence = sum(item.confidence for item in self.top_3)
        if abs(total_confidence - 1.0) > 0.02:
            raise ValueError(
                f"top_3 confidence scores must sum to 1.0, got {total_confidence:.4f}"
            )

        labels = [item.emotion for item in self.top_3]
        if len(set(labels)) != len(labels):
            raise ValueError("top_3 emotions must be unique")

        confidences = [item.confidence for item in self.top_3]
        if confidences != sorted(confidences, reverse=True):
            raise ValueError("top_3 emotions must be sorted by confidence descending")

        return self
