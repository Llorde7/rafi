import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from upstash_redis.asyncio import Redis

from app.api.schemas import (
    CausalRequest,
    CausalResponse,
    ClassifyRequest,
    CreateSessionRequest,
    SessionHistoryResponse,
    SessionResponse,
    TurnResponse,
)
from app.core.database import Base, engine, get_db
from app.models import Session as DBSession, Turn
from app.services.causal_engine import analyse
from app.services.classifier import classify

load_dotenv()

redis: Redis = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text("ALTER TABLE turns ADD COLUMN IF NOT EXISTS causal_analysis JSON")
        )
    redis = Redis.from_env()
    yield


app = FastAPI(title="EmpathAI", lifespan=lifespan)


async def get_history(session_id: str) -> list[dict]:
    raw = await redis.get(f"session:{session_id}:history")
    return json.loads(raw) if raw else []


async def save_history(session_id: str, history: list[dict]):
    await redis.setex(
        f"session:{session_id}:history",
        3600,
        json.dumps(history),
    )


@app.post("/session", response_model=SessionResponse)
async def create_session(
    req: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
):
    db_session = DBSession(user_id=req.user_id, language=req.language)
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)

    return SessionResponse(
        session_id=db_session.id,
        user_id=db_session.user_id,
        language=db_session.language,
        created_at=db_session.created_at,
    )


@app.post("/classify", response_model=TurnResponse)
async def classify_emotion(
    req: ClassifyRequest,
    db: AsyncSession = Depends(get_db),
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
    history = await get_history(session_id_str)

    classification = classify(req.text, history)
    causal = analyse(req.text, classification, history)

    turn = Turn(
        session_id=db_session.id,
        text=req.text,
        translation=classification.get("translation"),
        top_3=classification["top_3"],
        reasoning=classification.get("reasoning", ""),
        causal_analysis=causal,
    )
    db.add(turn)
    await db.commit()
    await db.refresh(turn)

    history.append(
        {
            "text": req.text,
            "translation": classification.get("translation"),
            "top_3": classification["top_3"],
            "reasoning": classification.get("reasoning", ""),
            "causal_analysis": causal,
        }
    )
    await save_history(session_id_str, history[-6:])

    return TurnResponse(
        turn_id=turn.id,
        session_id=db_session.id,
        text=turn.text,
        translation=turn.translation,
        top_3=classification["top_3"],
        reasoning=turn.reasoning,
        causal_analysis=causal,
        created_at=turn.created_at,
    )


@app.post("/analyse", response_model=CausalResponse)
async def causal_analysis(
    req: CausalRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DBSession).where(DBSession.id == req.session_id)
    )
    db_session = result.scalar_one_or_none()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    session_id_str = str(db_session.id)
    history = await get_history(session_id_str)

    classification = {
        "top_3": [e.model_dump() for e in req.top_3],
        "reasoning": req.reasoning,
    }
    causal = analyse(req.text, classification, history)

    return CausalResponse(
        session_id=req.session_id,
        text=req.text,
        **causal,
    )


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
                created_at=t.created_at,
            )
            for t in db_session.turns
        ],
    )


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
    await redis.delete(f"session:{session_id}:history")
    return {"deleted": session_id}
