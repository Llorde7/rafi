from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime

from contracts.classifier_contract import ClassifierOutput
from contracts.causal_contract import CausalOutput
from contracts.planner_contract import PlannerOutput


class PipelineEnvelope(BaseModel):
    turn_id: Optional[UUID]       = None
    session_id: Optional[str]     = None
    user_id: Optional[str]        = None
    timestamp: datetime           = datetime.utcnow()
    raw_text: str

    classifier_output: Optional[ClassifierOutput] = None
    causal_output:     Optional[CausalOutput]     = None
    planner_output:    Optional[PlannerOutput]    = None