import json
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
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
from schemas import (
    ClassifyRequest,
    CreateSessionRequest,
    SessionHistoryResponse,
    SessionResponse,
    TurnResponse,
)
from orchestrator import (
    run_pipeline,
    build_history_turn,
    advance_trajectory,
    get_escalation_flag,
)

load_dotenv()
redis: Redis = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    redis = Redis.from_env()
    yield


app = FastAPI(title="EmpathAI", lifespan=lifespan)


# ─── Redis helpers ─────────────────────────────────────────────────────────────

async def get_classifier_history(session_id: str) -> list[dict]:
    raw = await redis.get(f"session:{session_id}:classifier_history")
    return json.loads(raw) if raw else []


async def get_causal_history(session_id: str) -> list[HistoryTurn]:
    raw = await redis.get(f"session:{session_id}:causal_history")
    if not raw:
        return []
    return [HistoryTurn(**h) for h in json.loads(raw)]


async def get_trajectory(session_id: str, user_id: str | None, db: AsyncSession) -> SessionTrajectory:
    """
    Load trajectory from Redis. On first turn of a session, seed cross-session
    baseline from Postgres if the user has a prior profile.
    """
    raw = await redis.get(f"session:{session_id}:trajectory")
    if raw:
        return SessionTrajectory(**json.loads(raw))

    # New session — build fresh trajectory, seed from user profile if available
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
):
    TTL = 3600
    await redis.setex(
        f"session:{session_id}:classifier_history",
        TTL,
        json.dumps(classifier_history[-6:])
    )
    await redis.setex(
        f"session:{session_id}:causal_history",
        TTL,
        json.dumps([h.model_dump() for h in causal_history[-6:]])
    )
    await redis.setex(
        f"session:{session_id}:trajectory",
        TTL,
        trajectory.model_dump_json()
    )


async def close_session_trajectory(
    session_id: str,
    user_id: str | None,
    trajectory: SessionTrajectory,
    causal_history: list[HistoryTurn],
    db: AsyncSession,
):
    """
    Persist cross-session profile to Postgres when a session ends.
    Only for authenticated users. Uses a rolling mean for valence baseline.
    """
    if not user_id or not trajectory.valence_series:
        return

    session_mean_valence = (
        sum(trajectory.valence_series) / len(trajectory.valence_series)
    )

    # Collect cause_types from this session
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
            last_session_flag=trajectory.current_flag,
            last_session_end_emotion=(
                trajectory.dominant_emotions[-1]
                if trajectory.dominant_emotions else None
            ),
        )
        db.add(profile)
    else:
        # Rolling mean: weight prior sessions equally
        n = profile.sessions_seen
        new_mean = (profile.mean_valence * n + session_mean_valence) / (n + 1)

        # Merge cause type rankings
        prior_counts = {ct: (3 - i) for i, ct in enumerate(profile.dominant_cause_types)}
        for i, ct in enumerate(ranked_causes):
            prior_counts[ct] = prior_counts.get(ct, 0) + (len(ranked_causes) - i)
        merged_causes = sorted(prior_counts, key=lambda k: -prior_counts[k])

        profile.sessions_seen            = n + 1
        profile.mean_valence             = round(new_mean, 4)
        profile.dominant_cause_types     = merged_causes[:3]
        profile.last_session_flag        = trajectory.current_flag
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
    db_session = DBSession(user_id=req.user_id, language=req.language)
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)

    return SessionResponse(
        session_id=db_session.id,
        user_id=db_session.user_id,
        language=db_session.language,
        created_at=db_session.created_at
    )


# ─── /classify ────────────────────────────────────────────────────────────────

@app.post("/classify", response_model=TurnResponse)
async def classify_emotion(
    req: ClassifyRequest,
    db: AsyncSession = Depends(get_db)
):
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

    session_id_str = str(db_session.id)
    user_id        = req.user_id or db_session.user_id

    # ── Load histories ────────────────────────────────────────────────────────
    classifier_history = await get_classifier_history(session_id_str)
    causal_history     = await get_causal_history(session_id_str)
    trajectory         = await get_trajectory(session_id_str, user_id, db)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    envelope: PipelineEnvelope = run_pipeline(
        text=req.text,
        session_id=session_id_str,
        user_id=user_id,
        classifier_history=classifier_history,
        causal_history=causal_history,
        trajectory=trajectory,
    )

    # ── Advance trajectory ────────────────────────────────────────────────────
    updated_trajectory = advance_trajectory(trajectory, envelope)

    # ── Check for actionable flags ────────────────────────────────────────────
    escalation_flag = get_escalation_flag(updated_trajectory)
    # TODO: when MIND-SAFE is integrated, pass escalation_flag into it here.
    # For now it is available on the envelope for downstream consumers.

    # ── Persist turn ──────────────────────────────────────────────────────────
    turn = Turn(
        session_id=db_session.id,
        text=req.text,
        translation=envelope.classifier_output.translation,
        top_3=[e.model_dump() for e in envelope.classifier_output.top_3],
        reasoning=envelope.classifier_output.reasoning,
        causal_analysis=envelope.causal_output.model_dump()
    )
    db.add(turn)
    await db.commit()
    await db.refresh(turn)

    # ── Update histories ──────────────────────────────────────────────────────
    new_history_turn = build_history_turn(req.text, envelope)

    classifier_history.append({
        "text": req.text,
        "translation": envelope.classifier_output.translation,
        "top_3": [e.model_dump() for e in envelope.classifier_output.top_3],
        "reasoning": envelope.classifier_output.reasoning,
    })
    causal_history.append(new_history_turn)

    await save_all_histories(
        session_id_str,
        classifier_history,
        causal_history,
        updated_trajectory,
    )

    return TurnResponse(
        turn_id=turn.id,
        session_id=db_session.id,
        text=turn.text,
        translation=turn.translation,
        top_3=[e.model_dump() for e in envelope.classifier_output.top_3],
        reasoning=envelope.classifier_output.reasoning,
        causal_analysis=envelope.causal_output.model_dump(),
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

    # Persist trajectory to user profile before deleting
    user_id = db_session.user_id
    trajectory_raw = await redis.get(f"session:{session_id}:trajectory")
    if trajectory_raw and user_id:
        trajectory = SessionTrajectory(**json.loads(trajectory_raw))
        causal_raw = await redis.get(f"session:{session_id}:causal_history")
        causal_history = (
            [HistoryTurn(**h) for h in json.loads(causal_raw)]
            if causal_raw else []
        )
        await close_session_trajectory(session_id, user_id, trajectory, causal_history, db)

    await db.delete(db_session)
    await db.commit()

    await redis.delete(f"session:{session_id}:classifier_history")
    await redis.delete(f"session:{session_id}:causal_history")
    await redis.delete(f"session:{session_id}:trajectory")

    return {"deleted": session_id}


# ─── /session/{id}/close POST ─────────────────────────────────────────────────
# Explicit session-close endpoint. Call this when the user ends the conversation
# gracefully (e.g. they click "end session"). Persists cross-session profile
# without deleting the session record or turn history.

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

    trajectory = SessionTrajectory(**json.loads(trajectory_raw))
    causal_raw = await redis.get(f"session:{session_id}:causal_history")
    causal_history = (
        [HistoryTurn(**h) for h in json.loads(causal_raw)]
        if causal_raw else []
    )

    await close_session_trajectory(session_id, user_id, trajectory, causal_history, db)

    # Clear Redis caches — session is closed
    await redis.delete(f"session:{session_id}:classifier_history")
    await redis.delete(f"session:{session_id}:causal_history")
    await redis.delete(f"session:{session_id}:trajectory")

    return {
        "status": "closed",
        "session_id": session_id,
        "trajectory_flag": trajectory.current_flag,
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