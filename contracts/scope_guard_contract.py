from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ScopeCategory(str, Enum):
    # In-scope: pipeline proceeds
    EMOTIONAL_DISTRESS      = "emotional_distress"       # grief, anxiety, low mood, overwhelm
    ACADEMIC_DISTRESS       = "academic_distress"        # pressure, failure fear, burnout — emotional referent
    RELATIONAL_DISTRESS     = "relational_distress"      # family, friendship, romantic tension
    IDENTITY_DISTRESS       = "identity_distress"        # belonging, self-worth, purpose
    WELLBEING               = "wellbeing"                # sleep, loneliness, motivation, general coping
    AMBIGUOUS_IN_SCOPE      = "ambiguous_in_scope"       # uncertain but emotionally weighted — proceed

    # Out-of-scope: pipeline blocked
    ACADEMIC_TASK           = "academic_task"            # write my essay, solve this problem
    FACTUAL_QUERY           = "factual_query"            # what is X, when did Y, how does Z work
    GENERAL_CONVERSATION    = "general_conversation"     # small talk, greetings with no distress signal
    ENTERTAINMENT           = "entertainment"            # jokes, games, stories
    TECHNICAL_REQUEST       = "technical_request"        # code help, device troubleshooting


IN_SCOPE_CATEGORIES = {
    ScopeCategory.EMOTIONAL_DISTRESS,
    ScopeCategory.ACADEMIC_DISTRESS,
    ScopeCategory.RELATIONAL_DISTRESS,
    ScopeCategory.IDENTITY_DISTRESS,
    ScopeCategory.WELLBEING,
    ScopeCategory.AMBIGUOUS_IN_SCOPE,
}


class ScopeGuardInput(BaseModel):
    text: str
    session_id: Optional[str] = None
    # Prior turns help detect deferred distress — "btw can you write my essay"
    # after 3 turns of expressed anxiety is still in scope.
    prior_turn_count: int = 0
    prior_was_in_scope: bool = False


class ScopeGuardOutput(BaseModel):
    is_in_scope: bool
    category: ScopeCategory
    confidence: float                          # 0.0–1.0
    reasoning: str                             # one sentence — for logging, not user-facing
