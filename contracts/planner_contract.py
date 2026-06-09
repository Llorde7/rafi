from enum import Enum
from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Legacy enums — still used by session_framework_contract.py and contracts/__init__.py
# ---------------------------------------------------------------------------

class TherapeuticFramework(str, Enum):
    PERSON_CENTRED    = "person_centred"
    CBT               = "cbt"
    MI                = "mi"
    SOLUTION_FOCUSED  = "solution_focused"
    NONE              = "none"


class ResponseStrategy(str, Enum):
    VALIDATE        = "validate"
    REFRAME         = "reframe"
    PSYCHOEDUCATE   = "psychoeducate"
    PROBE           = "probe"
    MOTIVATE        = "motivate"
    REFLECT         = "reflect"
    ACTIVE_LISTEN   = "active_listen"
    GROUND          = "ground"
    SOLUTION_ELICIT = "solution_elicit"


class PlannerConfidence(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# ---------------------------------------------------------------------------
# Technique cluster — replaces single Strategy
# ---------------------------------------------------------------------------

class Technique(BaseModel):
    name:          str   # e.g. "redefine_success_criteria"
    modality:      str   # cbt | abt | person_centred | mi | solution_focused | psychoeducation
    purpose:       str   # what this technique must accomplish — one sentence
    sequence_note: str   # why it comes at this position in the cluster


class TechniqueCluster(BaseModel):
    techniques:           list[Technique]  # ordered 1–3, each sets up the next
    cluster_rationale:    str              # why this cluster fits situation + cognitive_pattern + intent
    executor_instruction: str             # unified instruction to TRACE
    rag_context:          Optional[str]
    kb_sources:           list[str] = []


# ---------------------------------------------------------------------------
# Planner input — extended with situation and intent fields
# ---------------------------------------------------------------------------

class PlannerInput(BaseModel):
    text:                    str

    # Situation fields — from Stage1Output (fused classifier)
    situation_type:          str
    situation_summary:       str
    has_concrete_deadline:   bool = False
    has_external_referents:  bool = False

    # Cognitive and risk fields — from CausalOutput (fused causal engine)
    cognitive_pattern:       str
    behavioral_risk:         str

    # Intent state — from detect_intent() deterministic function
    intent_state:            str   # processing | action_seeking | transitioning | crisis

    # Emotion fields — from Stage1Output
    top_emotion:             str
    emotion_confidence:      float

    # Causal fields — from CausalOutput
    cause_type:              str
    confidence_category:     str
    causal_chain:            list[str]
    temporal_pattern:        Optional[str]
    planner_instruction:     str           # from CAE — proceed | ask_first | hold
    clarifying_question:     Optional[str]

    # Session context
    session_framework:       str
    session_history_summary: str


# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------

class PlannerOutput(BaseModel):
    framework:                      str
    intent_state_received:          str              # confirms intent was honoured
    technique_cluster:              TechniqueCluster
    clarifying_question:            Optional[str]
    clarifying_question_overridden: bool = False
    response_directive:             str
    escalate_to_safety:             bool = False
    escalation_reason:              Optional[str]
    kb_retrieval_attempted:         bool = False
    error:                          Optional[str]


# ---------------------------------------------------------------------------
# Technique selection reference map
# Provided to planner prompt as reference — not enforced in code
# ---------------------------------------------------------------------------

TECHNIQUE_SELECTION_MAP = {
    # (situation_type, cognitive_pattern): [primary, secondary, tertiary]
    ("acute_task_crisis",   "all_or_nothing"):            ["redefine_success_criteria", "reconstruct_from_partial_progress", "reality_testing"],
    ("acute_task_crisis",   "catastrophising"):           ["present_focus_reframe", "behavioural_activation"],
    ("convergent_overload", "catastrophising"):           ["task_triage", "cognitive_narrowing", "behavioural_activation"],
    ("convergent_overload", "all_or_nothing"):            ["problem_decomposition", "redefine_success_criteria"],
    ("interpersonal_event", "personalisation"):           ["externalise_attribution", "perspective_taking"],
    ("interpersonal_event", "mind_reading"):              ["reality_testing", "perspective_taking"],
    ("loss_or_absence",     "any"):                       ["validation", "meaning_making"],
    ("identity_pressure",   "should_statements"):         ["defusion", "self_compassion_reframe"],
    ("identity_pressure",   "personalisation"):           ["externalise_attribution", "defusion"],
    ("chronic_low_mood",    "emotional_reasoning"):       ["psychoeducation", "behavioural_activation"],
    ("ambiguous",           "any"):                       ["validation", "open_exploration"],
}

# ---------------------------------------------------------------------------
# Intent constraints — enforced in planner_engine.py before LLM call
# ---------------------------------------------------------------------------

# These techniques are disallowed as primary when intent = action_seeking
ACTION_SEEKING_DISALLOWED_PRIMARY = {"reflect", "active_listen", "probe"}

# These techniques are disallowed when intent = processing
PROCESSING_DISALLOWED = {"solution_elicit", "task_triage", "behavioural_activation", "problem_decomposition"}