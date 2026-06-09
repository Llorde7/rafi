import json
import os
from typing import Optional
from groq import AsyncGroq
from contracts.planner_contract import (
    PlannerInput, PlannerOutput, TechniqueCluster, Technique,
    ACTION_SEEKING_DISALLOWED_PRIMARY, PROCESSING_DISALLOWED,
    TECHNIQUE_SELECTION_MAP,
)
from contracts.intent_contract import IntentState

MODEL  = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Lazy client — reads GROQ_API_KEY from env at first call, after load_dotenv()
# ---------------------------------------------------------------------------

_client: AsyncGroq | None = None

def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


# ---------------------------------------------------------------------------
# Acute panic guard — deterministic, runs before LLM call
# ---------------------------------------------------------------------------

PANIC_SIGNALS = [
    "can't breathe", "can't stop shaking", "hands are shaking",
    "heart is racing", "i'm going to fail everything",
    "everything is falling apart right now",
    "i don't know what's happening to me",
]

def is_acute_panic(text: str, arousal: str, valence: float) -> bool:
    """
    Returns True only when ALL of:
    - arousal = high
    - valence < -0.5
    - explicit somatic or disintegration signal in text
    A coherent, articulate request is never acute panic.
    """
    if arousal != "high" or valence >= -0.5:
        return False
    text_lower = text.lower()
    return any(signal in text_lower for signal in PANIC_SIGNALS)


# ---------------------------------------------------------------------------
# Intent constraint injection
# ---------------------------------------------------------------------------

def _intent_constraint_block(intent_state: str) -> str:
    if intent_state == IntentState.action_seeking:
        return """
INTENT STATE: ACTION_SEEKING — HARD CONSTRAINT

The student has explicitly signalled they are ready for practical help.
They may have said "what do I do", "help me with", "how do I handle this",
or "thanks for listening but I need help on what to actually do."

REQUIRED:
- The technique cluster MUST include at least one action-oriented technique:
  redefine_success_criteria, task_triage, problem_decomposition,
  behavioural_activation, reconstruct_from_partial_progress, solution_elicit
- Validation is permitted only as a brief secondary technique, not primary
- probe, reflect, active_listen are DISALLOWED as the primary technique
- Do NOT generate a clarifying question

CRITICAL: A student who says "thanks for listening but I need help on what to do"
is ACTION_SEEKING. Responding with grounding, reflection, or further exploration
at this moment is a clinical error.
"""
    if intent_state == IntentState.processing:
        return """
INTENT STATE: PROCESSING

The student needs to feel understood before anything else.
They are not yet ready for practical advice or steps.

REQUIRED:
- Cluster primary must be validate, reflect, or active_listen
- Do NOT offer advice, steps, plans, or solutions
- solution_elicit, task_triage, behavioural_activation, problem_decomposition
  are DISALLOWED
- A single technique is sufficient if it creates genuine holding space
"""
    if intent_state == IntentState.transitioning:
        return """
INTENT STATE: TRANSITIONING

The student is moving from emotional processing toward action.
They may have acknowledged feeling heard and are beginning to ask what to do.

REQUIRED:
- Cluster should bridge: one validation technique, then one action technique
- Do not force either mode exclusively
- The transition should feel natural — validation first, then open the door to action
"""
    return ""  # crisis handled upstream


# ---------------------------------------------------------------------------
# Technique selection map as formatted string for prompt injection
# ---------------------------------------------------------------------------

def _technique_map_block() -> str:
    lines = ["TECHNIQUE SELECTION REFERENCE MAP (situation_type × cognitive_pattern → cluster):"]
    for (sit, cog), techniques in TECHNIQUE_SELECTION_MAP.items():
        lines.append(f"  {sit} + {cog}: {' → '.join(techniques)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-LLM enforcement — validate cluster against intent constraints
# ---------------------------------------------------------------------------

def _enforce_intent_constraints(
    cluster: TechniqueCluster,
    intent_state: str,
    clarifying_question: Optional[str],
) -> tuple[TechniqueCluster, Optional[str], bool]:
    """
    Returns (cluster, clarifying_question, was_overridden).
    Modifies cluster in-place if violations detected.
    """
    overridden = False

    if intent_state == IntentState.action_seeking:
        if clarifying_question:
            clarifying_question = None
            overridden = True
        if cluster.techniques and cluster.techniques[0].name in ACTION_SEEKING_DISALLOWED_PRIMARY:
            print(f"[PLANNER OVERRIDE] Primary technique '{cluster.techniques[0].name}' "
                  f"disallowed for ACTION_SEEKING intent. Override required.")
            overridden = True
            cluster.techniques = [
                t for t in cluster.techniques
                if t.name not in ACTION_SEEKING_DISALLOWED_PRIMARY
            ]

    if intent_state == IntentState.processing:
        disallowed = [
            t for t in cluster.techniques
            if t.name in PROCESSING_DISALLOWED
        ]
        if disallowed:
            print(f"[PLANNER OVERRIDE] Techniques {[t.name for t in disallowed]} "
                  f"disallowed for PROCESSING intent. Removed.")
            cluster.techniques = [
                t for t in cluster.techniques
                if t.name not in PROCESSING_DISALLOWED
            ]
            overridden = True

    return cluster, clarifying_question, overridden


# ---------------------------------------------------------------------------
# Main system prompt
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """You are a senior clinical supervisor for an AI counselling system serving Kenyan university students.

You receive the output of the emotion classifier, situation assessor, and causal analysis engine.
Your job is to select a technique cluster — an ordered set of 1–3 therapeutic techniques — that
addresses this student's specific situation, cognitive pattern, and behavioral risk in sequence.

═══════════════════════════════════════════
TECHNIQUE CLUSTER DESIGN
═══════════════════════════════════════════

A technique cluster is ordered — each technique sets up the next.
The cluster should address situation_type, cognitive_pattern, and behavioral_risk in that sequence.

Rules:
- Maximum 3 techniques. Minimum 1.
- Each technique must have a clear purpose — what it accomplishes in this specific turn
- The primary technique (index 0) must be appropriate for the intent state (see constraint block)
- The cluster must read as ONE natural counselling move when executed by TRACE
  (seams between techniques must not show in the response)

For each technique provide:
- name: a specific technique name (e.g. redefine_success_criteria, not just "cbt")
- modality: cbt | abt | person_centred | mi | solution_focused | psychoeducation
- purpose: what THIS technique must accomplish for THIS student in THIS turn
- sequence_note: why it comes at this position

cluster_rationale: one paragraph — why this cluster fits the student's situation,
cognitive pattern, intent state, and behavioral risk together.

executor_instruction: a single unified instruction to the response generator.
  Specific enough that TRACE knows exactly what to do without reading the cluster details.
  NOT "use CBT" — rather "Briefly acknowledge the student's disappointment,
  then reframe the success criterion for tonight from 'complete project' to
  'presentable story of what you tried', then ask what they already have to work with."

═══════════════════════════════════════════
ESCALATION
═══════════════════════════════════════════

escalate_to_safety: true ONLY if the student's message contains explicit self-harm language,
suicidal ideation, or immediate safety risk.

The following are NOT escalation triggers:
- Wanting to quit school or take time off
- Academic failure or withdrawal
- General expressions of overwhelm or distress

escalation_reason: must reference specific language from the student's message.
False positive escalation on mild signals is worse than a missed one — the
deterministic crisis gate catches genuine trajectory crises.

═══════════════════════════════════════════
OUTPUT FORMAT — STRICT
═══════════════════════════════════════════

Single JSON object. No preamble. No markdown fences.

{
  "framework": "<cbt|mi|person_centred|solution_focused|none>",
  "intent_state_received": "<echo the intent state>",
  "technique_cluster": {
    "techniques": [
      {
        "name": "<specific technique name>",
        "modality": "<modality>",
        "purpose": "<what this accomplishes>",
        "sequence_note": "<why this position>"
      }
    ],
    "cluster_rationale": "<paragraph>",
    "executor_instruction": "<specific unified instruction>",
    "rag_context": null,
    "kb_sources": []
  },
  "clarifying_question": "<string or null>",
  "clarifying_question_overridden": false,
  "response_directive": "<one sentence>",
  "escalate_to_safety": false,
  "escalation_reason": null,
  "kb_retrieval_attempted": false,
  "error": null
}
"""


async def plan(inp: PlannerInput, arousal: str = "medium", valence: float = -0.3) -> PlannerOutput:

    # Deterministic panic guard — before LLM call
    if is_acute_panic(inp.text, arousal, valence):
        return _ground_response(inp)

    # Build dynamic system prompt with intent constraint block
    system_prompt = (
        BASE_SYSTEM_PROMPT
        + "\n\n"
        + _intent_constraint_block(inp.intent_state)
        + "\n\n"
        + _technique_map_block()
    )

    user_content = f"""STUDENT MESSAGE: {inp.text}

SITUATION TYPE: {inp.situation_type}
SITUATION SUMMARY: {inp.situation_summary}
HAS CONCRETE DEADLINE: {inp.has_concrete_deadline}
HAS EXTERNAL REFERENTS: {inp.has_external_referents}

TOP EMOTION: {inp.top_emotion} ({inp.emotion_confidence:.2f})
CAUSE TYPE: {inp.cause_type}
COGNITIVE PATTERN: {inp.cognitive_pattern}
BEHAVIORAL RISK: {inp.behavioral_risk}
CONFIDENCE CATEGORY: {inp.confidence_category}
PLANNER INSTRUCTION FROM CAE: {inp.planner_instruction}
CAUSAL CHAIN: {json.dumps(inp.causal_chain)}
TEMPORAL PATTERN: {inp.temporal_pattern or 'none'}

INTENT STATE: {inp.intent_state}
SESSION FRAMEWORK: {inp.session_framework}

CLARIFYING QUESTION FROM CAE: {inp.clarifying_question or 'none'}

SESSION HISTORY SUMMARY: {inp.session_history_summary or 'none — early session'}
"""

    client = _get_client()
    response = await client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        max_tokens=800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())

    cluster = TechniqueCluster(
        techniques=[Technique(**t) for t in data["technique_cluster"]["techniques"]],
        cluster_rationale=data["technique_cluster"]["cluster_rationale"],
        executor_instruction=data["technique_cluster"]["executor_instruction"],
        rag_context=data["technique_cluster"].get("rag_context"),
        kb_sources=data["technique_cluster"].get("kb_sources", []),
    )

    clarifying_question = data.get("clarifying_question")
    cluster, clarifying_question, overridden = _enforce_intent_constraints(
        cluster, inp.intent_state, clarifying_question
    )

    return PlannerOutput(
        framework=data["framework"],
        intent_state_received=data.get("intent_state_received", inp.intent_state),
        technique_cluster=cluster,
        clarifying_question=clarifying_question,
        clarifying_question_overridden=overridden or data.get("clarifying_question_overridden", False),
        response_directive=data.get("response_directive", ""),
        escalate_to_safety=data.get("escalate_to_safety", False),
        escalation_reason=data.get("escalation_reason"),
        kb_retrieval_attempted=data.get("kb_retrieval_attempted", False),
        error=data.get("error"),
    )


def _ground_response(inp: PlannerInput) -> PlannerOutput:
    """Fallback for deterministic acute panic detection."""
    return PlannerOutput(
        framework="person_centred",
        intent_state_received=inp.intent_state,
        technique_cluster=TechniqueCluster(
            techniques=[Technique(
                name="ground",
                modality="person_centred",
                purpose="Regulate acute distress before any other intervention",
                sequence_note="Only technique — acute panic state",
            )],
            cluster_rationale="Acute panic detected — grounding before any other strategy.",
            executor_instruction=(
                "The student is in acute distress right now. "
                "Give one slow-exhale grounding directive. "
                "Then one sentence of calm presence. No question. No reflection. No analysis."
            ),
            rag_context=None,
            kb_sources=[],
        ),
        clarifying_question=None,
        clarifying_question_overridden=False,
        response_directive="Ground the student before any other move.",
        escalate_to_safety=False,
        escalation_reason=None,
        kb_retrieval_attempted=False,
        error=None,
    )