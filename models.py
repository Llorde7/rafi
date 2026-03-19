from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import uuid


class Session(Base):
    __tablename__ = "sessions"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id    = Column(String, nullable=True)
    language   = Column(String, default="en")
    created_at = Column(DateTime, default=datetime.utcnow)
    turns      = relationship("Turn", back_populates="session", order_by="Turn.created_at")


class Turn(Base):
    __tablename__ = "turns"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id      = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    text            = Column(Text, nullable=False)
    translation     = Column(Text, nullable=True)
    top_3           = Column(JSON, nullable=False)
    reasoning       = Column(Text, nullable=True)
    causal_analysis = Column(JSON, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    session         = relationship("Session", back_populates="turns")
