import json
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from upstash_redis.asyncio import Redis

from database import engine, get_db, Base
from models import Session as DBSession, Turn
from contracts.causal_contract import HistoryTurn
from contracts.pipeline_envelope import PipelineEnvelope
from schemas import (
    ClassifyRequest,
    CreateSessionRequest,
    SessionHistoryResponse,
    SessionResponse,
    TurnResponse,
)
from orchestrator import run_pipeline, build_history_turn

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


# ─── Redis helpers ────────────────────────────────────────────────────────────

async def get_classifier_history(session_id: str) -> list[dict]:
    raw = await redis.get(f"session:{session_id}:classifier_history")
    return json.loads(raw) if raw else []


async def get_causal_history(session_id: str) -> list[HistoryTurn]:
    raw = await redis.get(f"session:{session_id}:causal_history")
    if not raw:
        return []
    return [HistoryTurn(**h) for h in json.loads(raw)]


async def save_histories(
    session_id: str,
    classifier_history: list[dict],
    causal_history: list[HistoryTurn]
):
    await redis.setex(
        f"session:{session_id}:classifier_history",
        3600,
        json.dumps(classifier_history[-6:])
    )
    await redis.setex(
        f"session:{session_id}:causal_history",
        3600,
        json.dumps([h.model_dump() for h in causal_history[-6:]])
    )


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

    classifier_history = await get_classifier_history(session_id_str)
    causal_history = await get_causal_history(session_id_str)

    envelope: PipelineEnvelope = run_pipeline(
        text=req.text,
        session_id=session_id_str,
        user_id=req.user_id,
        classifier_history=classifier_history,
        causal_history=causal_history,
    )

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

    new_history_turn = build_history_turn(req.text, envelope)

    classifier_history.append({
        "text": req.text,
        "translation": envelope.classifier_output.translation,
        "top_3": [e.model_dump() for e in envelope.classifier_output.top_3],
        "reasoning": envelope.classifier_output.reasoning,
    })
    causal_history.append(new_history_turn)

    await save_histories(session_id_str, classifier_history, causal_history)

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


# ─── /session/{id} ────────────────────────────────────────────────────────────

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
    await db.delete(db_session)
    await db.commit()
    await redis.delete(f"session:{session_id}:classifier_history")
    await redis.delete(f"session:{session_id}:causal_history")
    return {"deleted": session_id}
