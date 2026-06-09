from enum import Enum
from pydantic import BaseModel


class IntentState(str, Enum):
    processing     = "processing"      # student needs to feel understood
    action_seeking = "action_seeking"  # student is ready for practical help
    transitioning  = "transitioning"   # student is shifting between the two
    crisis         = "crisis"          # safety event — separate path


# ---------------------------------------------------------------------------
# Phrase lists — extend Swahili/Sheng variants in a later session
# ---------------------------------------------------------------------------

ACTION_SEEKING_PHRASES = [
    "what do i do",
    "what should i do",
    "how do i",
    "how do i handle",
    "help me with",
    "i need help with",
    "tell me what to do",
    "what can i do",
    "give me advice",
    "what's the next step",
    "how can i fix",
    "i need a plan",
    "what do i need to do",
    "help me figure out",
    "nisaidie",        # Swahili: help me
    "nifanye nini",    # Swahili: what should I do
    "nataka msaada",   # Swahili: I want help
]

PROCESSING_PHRASES = [
    "i just needed to say",
    "i don't know what to do",
    "i feel like",
    "it's been hard",
    "i've been struggling",
    "i can't stop thinking",
    "i just feel",
    "i keep feeling",
    "it's hard to",
    "i don't understand why",
]

TRANSITIONING_PHRASES = [
    "thanks for listening",
    "i feel heard",
    "that helps",
    "okay so",
    "so now what",
    "what happens next",
]


# ---------------------------------------------------------------------------
# Deterministic intent detection — no LLM call
# ---------------------------------------------------------------------------

def detect_intent(
    text: str,
    trajectory_flag: str,
    prior_intent: IntentState,
    turn_count: int,
) -> IntentState:
    """
    Evaluate in order — first match wins.

    1. Crisis trajectory            → crisis
    2. Explicit action phrase       → action_seeking
    3. Explicit transition phrase   → transitioning (may combine with action)
    4. Explicit processing phrase
       AND turn <= 2                → processing
    5. Prior intent was action_seeking
       AND no processing phrase     → action_seeking  (sticky)
    6. Prior intent was processing
       AND no action phrase         → processing      (sticky)
    7. Turn 1–2 default             → processing
    8. Turn 3+ default              → transitioning
    """
    text_lower = text.lower()

    if trajectory_flag == "crisis":
        return IntentState.crisis

    has_action    = any(p in text_lower for p in ACTION_SEEKING_PHRASES)
    has_processing = any(p in text_lower for p in PROCESSING_PHRASES)
    has_transition = any(p in text_lower for p in TRANSITIONING_PHRASES)

    if has_action:
        return IntentState.action_seeking

    if has_transition and not has_processing:
        # "thanks for listening but I need help" — transition phrase alone
        # means the student is moving toward action
        return IntentState.transitioning

    if has_processing and turn_count <= 2:
        return IntentState.processing

    if prior_intent == IntentState.action_seeking and not has_processing:
        return IntentState.action_seeking

    if prior_intent == IntentState.processing and not has_action:
        return IntentState.processing

    if turn_count <= 2:
        return IntentState.processing

    return IntentState.transitioning