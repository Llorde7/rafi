import asyncio
import json
import logging
import os
from time import perf_counter
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
from upstash_redis.asyncio import Redis

from database import engine, get_db, Base
from models import Session as DBSession, Turn, UserEmotionalProfile
from contracts.causal_contract import HistoryTurn
from contracts.pipeline_envelope import PipelineEnvelope
from contracts.trajectory_contract import SessionTrajectory, TrajectoryFlag
from contracts.trace_contract import TraceTurn, TonePreference
from contracts.session_framework_contract import SessionFramework
from contracts.intent_contract import IntentState
from schemas import (
    ClassifyRequest,
    CreateSessionRequest,
    SessionHistoryResponse,
    SessionResponse,
    TurnResponse,
)
from orchestrator import run_pipeline

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)
redis: Redis = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(
                text("ALTER TABLE turns ADD COLUMN IF NOT EXISTS planner_output JSON")
            )
            await conn.execute(
                text("ALTER TABLE turns ADD COLUMN IF NOT EXISTS trace_output JSON")
            )
            await conn.execute(
                text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tone_preference VARCHAR DEFAULT 'friendly'")
            )
        logger.info("Database connection + migrations OK")
    except Exception as e:
        logger.exception(
            "Database startup failed (DB_URL=%s, host=%s). "
            "App will start anyway, but DB-backed endpoints will fail. "
            "Check: 1) Supabase project is not paused, 2) Internal vs External URL, "
            "3) Render outbound network allows port 5432 to Supabase.",
            os.getenv("DB_URL", "<not set>")[:80],
            engine.url.host,
        )
    try:
        redis = Redis.from_env()
        logger.info("Redis client initialized")
    except Exception:
        logger.exception("Redis.from_env() failed; redis-backed endpoints will fail")
        redis = None
    yield


@app.get("/")
async def root():
    return {"status": "ok", "service": "EmpathAI"}


app = FastAPI(title="EmpathAI", lifespan=lifespan)

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Redis helpers ─────────────────────────────────────────────────────────────
SESSION_META_TTL = 3600

async def get_classifier_history(session_id: str) -> list[dict]:
    raw = await redis.get(f"session:{session_id}:classifier_history")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse classifier_history for session {session_id}: {e}. Raw content: {raw[:200]}")
        return []


async def get_causal_history(session_id: str) -> list[HistoryTurn]:
    raw = await redis.get(f"session:{session_id}:causal_history")
    if not raw:
        return []
    try:
        return [HistoryTurn(**h) for h in json.loads(raw)]
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse causal_history for session {session_id}: {e}. Raw content: {raw[:200]}")
        return []


async def get_cached_trajectory(session_id: str) -> SessionTrajectory | None:
    raw = await redis.get(f"session:{session_id}:trajectory")
    if not raw:
        return None
    try:
        return SessionTrajectory(**json.loads(raw))
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse trajectory for session {session_id}: {e}. Raw content: {raw[:200]}")
        return None


async def get_trace_history(session_id: str) -> list[dict]:
    raw = await redis.get(f"session:{session_id}:trace_history")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse trace_history for session {session_id}: {e}. Raw content: {raw[:200]}")
        return []


async def get_session_tone(session_id: str) -> str:
    raw = await redis.get(f"session:{session_id}:tone")
    return raw if raw else "friendly"


async def get_cached_session_framework(session_id: str) -> dict | None:
    raw = await redis.get(f"session:{session_id}:session_framework")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse session_framework for session {session_id}: {e}")
        return None


async def get_trajectory(session_id: str, user_id: str | None, db: AsyncSession) -> SessionTrajectory:
    """Load trajectory from Redis. On first turn, seed cross-session baseline from Postgres."""
    cached = await get_cached_trajectory(session_id)
    if cached:
        return cached

    traj = SessionTrajectory(session_id=session_id)
    if user_id:
        result = await db.execute(
            select(UserEmotionalProfile).where(UserEmotionalProfile.user_id == user_id)
        )
        profile = result.scalar_one_or_none()
        if profile:
            traj = traj.model_copy(update={
                "cross_session_baseline": profile.mean_valence,
                "sessions_seen": profile.sessions_seen,
            })
    return traj


async def save_all_histories(
    session_id: str,
    classifier_history: list[dict],
    causal_history: list[HistoryTurn],
    trajectory: SessionTrajectory,
    trace_history: list[dict],
    session_framework: SessionFramework,
):
    await asyncio.gather(
        redis.setex(
            f"session:{session_id}:classifier_history",
            SESSION_META_TTL,
            json.dumps(classifier_history[-6:])
        ),
        redis.setex(
            f"session:{session_id}:causal_history",
            SESSION_META_TTL,
            json.dumps([h.model_dump(mode="json") for h in causal_history[-6:]])
        ),
        redis.setex(
            f"session:{session_id}:trajectory",
            SESSION_META_TTL,
            trajectory.model_dump_json()
        ),
        redis.setex(
            f"session:{session_id}:trace_history",
            SESSION_META_TTL,
            json.dumps(trace_history[-6:])
        ),
        redis.setex(
            f"session:{session_id}:session_framework",
            SESSION_META_TTL,
            session_framework.model_dump_json()
        ),
    )


async def close_session_trajectory(
    session_id: str,
    user_id: str | None,
    trajectory: SessionTrajectory,
    causal_history: list[HistoryTurn],
    db: AsyncSession,
):
    """Persist cross-session profile to Postgres when a session ends."""
    if not user_id or not trajectory.valence_series:
        return

    session_mean_valence = (
        sum(trajectory.valence_series) / len(trajectory.valence_series)
    )

    cause_types = [h.cause_type for h in causal_history if h.cause_type]
    cause_type_counts: dict[str, int] = {}
    for ct in cause_types:
        cause_type_counts[ct] = cause_type_counts.get(ct, 0) + 1
    ranked_causes = sorted(cause_type_counts, key=lambda k: -cause_type_counts[k])

    result = await db.execute(
        select(UserEmotionalProfile).where(UserEmotionalProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()

    if profile is None:
        profile = UserEmotionalProfile(
            user_id=user_id,
            sessions_seen=1,
            mean_valence=session_mean_valence,
            dominant_cause_types=ranked_causes[:3],
            last_session_flag=trajectory.current_flag.value,
            last_session_end_emotion=(
                trajectory.dominant_emotions[-1]
                if trajectory.dominant_emotions else None
            ),
        )
        db.add(profile)
    else:
        n = profile.sessions_seen
        new_mean = (profile.mean_valence * n + session_mean_valence) / (n + 1)
        prior_counts = {ct: (3 - i) for i, ct in enumerate(profile.dominant_cause_types)}
        for i, ct in enumerate(ranked_causes):
            prior_counts[ct] = prior_counts.get(ct, 0) + (len(ranked_causes) - i)
        merged_causes = sorted(prior_counts, key=lambda k: -prior_counts[k])

        profile.sessions_seen            = n + 1
        profile.mean_valence             = round(new_mean, 4)
        profile.dominant_cause_types     = merged_causes[:3]
        profile.last_session_flag        = trajectory.current_flag.value
        profile.last_session_end_emotion = (
            trajectory.dominant_emotions[-1]
            if trajectory.dominant_emotions else None
        )
        profile.updated_at = datetime.utcnow()

    await db.commit()


# ─── /session POST ─────────────────────────────────────────────────────────────

@app.post("/session", response_model=SessionResponse)
async def create_session(
    req: CreateSessionRequest,
    db: AsyncSession = Depends(get_db)
):
    db_session = DBSession(
        user_id=req.user_id,
        language=req.language,
        tone_preference=req.tone_preference,
    )
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)

    await redis.setex(
        f"session:{db_session.id}:tone",
        SESSION_META_TTL,
        req.tone_preference,
    )

    return SessionResponse(
        session_id=db_session.id,
        user_id=db_session.user_id,
        language=db_session.language,
        tone_preference=db_session.tone_preference,
        created_at=db_session.created_at
    )


# ─── /classify ────────────────────────────────────────────────────────────────

@app.post("/classify", response_model=TurnResponse)
async def classify_emotion(
    req: ClassifyRequest,
    db: AsyncSession = Depends(get_db)
):
    request_started = perf_counter()

    session_started = perf_counter()
    if req.session_id:
        result = await db.execute(
            select(DBSession).where(DBSession.id == req.session_id)
        )
        db_session = result.scalar_one_or_none()
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        db_session = DBSession(user_id=req.user_id)
        db.add(db_session)
        await db.flush()
    logger.info(
        "API timing | session_id=%s stage=session_lookup duration_ms=%.1f",
        str(db_session.id),
        (perf_counter() - session_started) * 1000,
    )

    session_id_str = str(db_session.id)
    user_id        = req.user_id or db_session.user_id

    # ── Load histories ────────────────────────────────────────────────────────
    history_started = perf_counter()
    classifier_history, causal_history, cached_trajectory, trace_history, tone, sf_raw = await asyncio.gather(
        get_classifier_history(session_id_str),
        get_causal_history(session_id_str),
        get_cached_trajectory(session_id_str),
        get_trace_history(session_id_str),
        get_session_tone(session_id_str),
        get_cached_session_framework(session_id_str),
    )
    trajectory = (
        cached_trajectory
        if cached_trajectory is not None
        else await get_trajectory(session_id_str, user_id, db)
    )

    # Reconstruct or default session framework
    if sf_raw:
        session_framework = SessionFramework(**sf_raw)
    else:
        session_framework = SessionFramework()

    # Determine prior intent: last entry in trace_history or default to processing
    prior_intent_str = "processing"
    if trace_history and len(trace_history) > 0:
        last_trace = trace_history[-1]
        prior_intent_str = last_trace.get("intent_state", "processing")

    logger.info(
        "API timing | session_id=%s stage=load_histories duration_ms=%.1f",
        session_id_str,
        (perf_counter() - history_started) * 1000,
    )

    # ── Build session context dict ─────────────────────────────────────────────
    session_context = {
        "classifier_history":      classifier_history,
        "causal_history":          [h.model_dump(mode="json") for h in causal_history],
        "trajectory":              trajectory,
        "trace_history":           trace_history,  # list of TraceTurn dicts
        "session_framework":       session_framework.model_dump(),
        "tone":                    tone,
        "prior_intent":            prior_intent_str,
        "session_history_summary": _build_session_summary(classifier_history),
    }

    # ── Build cross-session summary ─────────────────────────────────────────────
    cross_session_summary: str | None = None
    if user_id:
        profile_result = await db.execute(
            select(UserEmotionalProfile).where(UserEmotionalProfile.user_id == user_id)
        )
        profile = profile_result.scalar_one_or_none()
        if profile and profile.sessions_seen > 0:
            cross_session_summary = (
                f"This user has had {profile.sessions_seen} prior session(s). "
                f"Their average emotional valence across sessions is {profile.mean_valence:.2f} "
                f"(range -1.0 negative to 1.0 positive). "
                f"Their most common stressor patterns have been: {', '.join(profile.dominant_cause_types[:3])}. "
                f"Their last session ended with a trajectory flag of '{profile.last_session_flag}' "
                f"and dominant emotion of '{profile.last_session_end_emotion}'."
            )
    if cross_session_summary:
        session_context["session_history_summary"] = cross_session_summary

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline_started = perf_counter()
    envelope, updated_trajectory, updated_session_framework, intent_state = await run_pipeline(
        text=req.text,
        session_id=session_id_str,
        session_context=session_context,
    )
    logger.info(
        "API timing | session_id=%s stage=pipeline duration_ms=%.1f",
        session_id_str,
        (perf_counter() - pipeline_started) * 1000,
    )

    # ── Persist turn ──────────────────────────────────────────────────────────
    db_started = perf_counter()
    turn = Turn(
        session_id=db_session.id,
        text=req.text,
        translation=getattr(envelope.classifier_output, "translation", None),
        top_3=[e.model_dump(mode="json") for e in envelope.classifier_output.top_3],
        reasoning=envelope.classifier_output.reasoning,
        causal_analysis=envelope.causal_output.model_dump(mode="json"),
        planner_output=(
            envelope.planner_output.model_dump(mode="json")
            if envelope.planner_output else None
        ),
        trace_output=(
            envelope.trace_output.model_dump(mode="json")
            if envelope.trace_output else None
        ),
    )
    db.add(turn)
    await db.flush()
    await db.commit()
    logger.info(
        "API timing | session_id=%s stage=db_commit_refresh duration_ms=%.1f",
        session_id_str,
        (perf_counter() - db_started) * 1000,
    )

    # ── Update histories ──────────────────────────────────────────────────────
    # Build history turn from envelope
    top_emotion = (
        envelope.classifier_output.top_3[0].emotion.value
        if envelope.classifier_output.top_3 else "neutral"
    )
    top_confidence = (
        envelope.classifier_output.top_3[0].confidence
        if envelope.classifier_output.top_3 else 0.0
    )

    classifier_history.append({
        "text": req.text,
        "translation": getattr(envelope.classifier_output, "translation", None),
        "top_3": [e.model_dump(mode="json") for e in envelope.classifier_output.top_3],
        "reasoning": envelope.classifier_output.reasoning,
        "situation_type": envelope.situation_type,
    })

    causal_history.append(HistoryTurn(
        turn_index=len(causal_history) + 1,
        text=req.text,
        top_emotion=top_emotion,
        cause_type=envelope.causal_output.cause_type.value,
        valence=0.0,  # will be properly computed from trajectory
    ))

    # Append to trace history if TRACE ran
    if envelope.trace_output and not envelope.trace_output.error:
        trace_history.append({
            "student_message": req.text,
            "trace_response": envelope.trace_output.response_text,
            "strategy_used": envelope.trace_output.strategy_used,
            "turn_index": len(trace_history) + 1,
            "intent_state": intent_state.value,
        })

    redis_started = perf_counter()
    await save_all_histories(
        session_id_str,
        classifier_history,
        causal_history,
        updated_trajectory,
        trace_history,
        updated_session_framework,
    )
    logger.info(
        "API timing | session_id=%s stage=redis_save duration_ms=%.1f",
        session_id_str,
        (perf_counter() - redis_started) * 1000,
    )
    logger.info(
        "API timing | session_id=%s stage=total duration_ms=%.1f",
        session_id_str,
        (perf_counter() - request_started) * 1000,
    )

    return TurnResponse(
        turn_id=turn.id,
        session_id=db_session.id,
        text=turn.text,
        translation=turn.translation,
        top_3=[e.model_dump(mode="json") for e in envelope.classifier_output.top_3],
        reasoning=envelope.classifier_output.reasoning,
        causal_analysis=envelope.causal_output.model_dump(mode="json"),
        planner_output=(
            envelope.planner_output.model_dump(mode="json")
            if envelope.planner_output else None
        ),
        trace_output=(
            envelope.trace_output.model_dump(mode="json")
            if envelope.trace_output else None
        ),
        created_at=turn.created_at
    )


# ─── /session/{id} GET ────────────────────────────────────────────────────────

@app.get("/session/{session_id}", response_model=SessionHistoryResponse)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DBSession)
        .where(DBSession.id == session_id)
        .options(selectinload(DBSession.turns))
    )
    db_session = result.scalar_one_or_none()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionHistoryResponse(
        session_id=db_session.id,
        language=db_session.language,
        created_at=db_session.created_at,
        turns=[
            TurnResponse(
                turn_id=t.id,
                session_id=db_session.id,
                text=t.text,
                translation=t.translation,
                top_3=t.top_3,
                reasoning=t.reasoning,
                causal_analysis=t.causal_analysis,
                planner_output=t.planner_output,
                trace_output=t.trace_output,
                created_at=t.created_at
            )
            for t in db_session.turns
        ]
    )


# ─── /session/{id} DELETE ─────────────────────────────────────────────────────

@app.delete("/session/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DBSession).where(DBSession.id == session_id)
    )
    db_session = result.scalar_one_or_none()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_id = db_session.user_id
    trajectory_raw = await redis.get(f"session:{session_id}:trajectory")
    if trajectory_raw and user_id:
        try:
            trajectory = SessionTrajectory(**json.loads(trajectory_raw))
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse trajectory in delete_session: {e}")
            trajectory = None

        causal_raw = await redis.get(f"session:{session_id}:causal_history")
        causal_history = []
        if causal_raw:
            try:
                causal_history = [HistoryTurn(**h) for h in json.loads(causal_raw)]
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse causal_history in delete_session: {e}")

        if trajectory:
            await close_session_trajectory(session_id, user_id, trajectory, causal_history, db)

    await db.delete(db_session)
    await db.commit()

    await redis.delete(f"session:{session_id}:classifier_history")
    await redis.delete(f"session:{session_id}:causal_history")
    await redis.delete(f"session:{session_id}:trajectory")
    await redis.delete(f"session:{session_id}:trace_history")
    await redis.delete(f"session:{session_id}:tone")

    return {"deleted": session_id}


# ─── /session/{id}/close POST ─────────────────────────────────────────────────

@app.post("/session/{session_id}/close")
async def close_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DBSession).where(DBSession.id == session_id)
    )
    db_session = result.scalar_one_or_none()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_id = db_session.user_id
    trajectory_raw = await redis.get(f"session:{session_id}:trajectory")
    if not trajectory_raw:
        return {"status": "no trajectory to persist"}

    try:
        trajectory = SessionTrajectory(**json.loads(trajectory_raw))
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse trajectory in persist_trajectory: {e}")
        return {"status": "error", "error": "Failed to parse trajectory"}

    causal_raw = await redis.get(f"session:{session_id}:causal_history")
    causal_history = []
    if causal_raw:
        try:
            causal_history = [HistoryTurn(**h) for h in json.loads(causal_raw)]
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse causal_history in persist_trajectory: {e}")

    await close_session_trajectory(session_id, user_id, trajectory, causal_history, db)

    await redis.delete(f"session:{session_id}:classifier_history")
    await redis.delete(f"session:{session_id}:causal_history")
    await redis.delete(f"session:{session_id}:trajectory")
    await redis.delete(f"session:{session_id}:trace_history")
    await redis.delete(f"session:{session_id}:tone")

    return {
        "status": "closed",
        "session_id": session_id,
        "trajectory_flag": trajectory.current_flag.value,
        "turns": trajectory.turn_count,
    }


# ─── /user/{user_id}/profile GET ──────────────────────────────────────────────

@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(UserEmotionalProfile).where(UserEmotionalProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found for user")

    return {
        "user_id":                  profile.user_id,
        "sessions_seen":            profile.sessions_seen,
        "mean_valence":             profile.mean_valence,
        "dominant_cause_types":     profile.dominant_cause_types,
        "last_session_flag":        profile.last_session_flag,
        "last_session_end_emotion": profile.last_session_end_emotion,
        "updated_at":               profile.updated_at,
    }


# ─── Helper ───────────────────────────────────────────────────────────────────

def _build_session_summary(classifier_history: list[dict]) -> str:
    """Compress classifier history into a one-line summary for the planner."""
    if not classifier_history:
        return ""
    parts = []
    for h in classifier_history[-4:]:
        emotion = "unknown"
        if h.get("top_3"):
            emotion = h["top_3"][0].get("emotion", "unknown") if isinstance(h["top_3"], list) else "unknown"
        sit = h.get("situation_type", "unknown")
        parts.append(f"[{emotion}/{sit}]")
    return " ".join(parts)