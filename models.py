from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Text, Float, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import uuid


class Session(Base):
    __tablename__ = "sessions"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id         = Column(String, nullable=True)
    language        = Column(String, default="en")
    tone_preference = Column(String, default="friendly")   # "friendly" | "clinical"
    created_at      = Column(DateTime, default=datetime.utcnow)
    turns           = relationship("Turn", back_populates="session", order_by="Turn.created_at")


class Turn(Base):
    __tablename__ = "turns"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id      = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    text            = Column(Text, nullable=False)
    translation     = Column(Text, nullable=True)
    top_3           = Column(JSON, nullable=False)
    reasoning       = Column(Text, nullable=True)
    causal_analysis = Column(JSON, nullable=True)
    planner_output  = Column(JSON, nullable=True)
    trace_output    = Column(JSON, nullable=True)    # TraceOutput — student-facing response
    created_at      = Column(DateTime, default=datetime.utcnow)
    session         = relationship("Session", back_populates="turns")


class UserEmotionalProfile(Base):
    """
    Persisted cross-session emotional baseline per user.
    Written/updated at session close. Read at session open.
    Only populated for authenticated users (user_id not null).
    """
    __tablename__ = "user_emotional_profiles"

    user_id                  = Column(String, primary_key=True)
    sessions_seen            = Column(Integer, default=0)
    mean_valence             = Column(Float, default=0.0)
    dominant_cause_types     = Column(JSON, default=list)   # list[str], ranked
    last_session_flag        = Column(String, nullable=True)
    last_session_end_emotion = Column(String, nullable=True)
    updated_at               = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)