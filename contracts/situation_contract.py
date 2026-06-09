from enum import Enum
from typing import Optional
from pydantic import BaseModel


class SituationType(str, Enum):
    acute_task_crisis    = "acute_task_crisis"
    convergent_overload  = "convergent_overload"
    interpersonal_event  = "interpersonal_event"
    loss_or_absence      = "loss_or_absence"
    identity_pressure    = "identity_pressure"
    chronic_low_mood     = "chronic_low_mood"
    ambiguous            = "ambiguous"


class SituationOutput(BaseModel):
    situation_type:           SituationType
    situation_summary:        str    # one sentence — concrete facts only, not emotion restatement
    has_concrete_deadline:    bool   # True if a specific time pressure is named
    has_external_referents:   bool   # True if tasks, people, places, or events are named
    situation_confidence:     float  # 0.0–1.0