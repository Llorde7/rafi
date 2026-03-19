from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from uuid import uuid4
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DB_URL")

# detect whether we're on the direct connection (5432) or pooler (6543)
# and configure the engine accordingly
if ":5432" in DATABASE_URL:
    # Ensure we're using asyncpg dialect
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={
            # unique prepared statement names prevent conflicts
            # when the connection is reused across requests
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
            "server_settings": {
                "jit": "off",
                "statement_timeout": "60000",
            },
        },
    )
else:
    # transaction pooler (port 6543) — disable prepared statements entirely
    # Ensure we're using asyncpg dialect
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    
    engine = create_async_engine(
        DATABASE_URL + "?prepared_statement_cache_size=0",
        echo=False,
        poolclass=NullPool,
    )

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session