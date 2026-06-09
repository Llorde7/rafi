from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from uuid import uuid4
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL or DB_URL must be set")

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# Ensure SSL for external Postgres hosts (Supabase, Neon, Render PG, etc.)
# Supabase and most managed Postgres providers require SSL.
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

if "sslmode" not in DATABASE_URL:
    parsed = urlparse(DATABASE_URL)
    query = parse_qs(parsed.query)
    query["sslmode"] = ["require"]
    DATABASE_URL = urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

if ":5432" in DATABASE_URL:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
            "timeout": 10,
            "server_settings": {
                "jit": "off",
                "statement_timeout": "60000",
            },
        },
    )
else:
    engine = create_async_engine(
        DATABASE_URL + "&prepared_statement_cache_size=0"
        if "?" in DATABASE_URL
        else DATABASE_URL + "?prepared_statement_cache_size=0",
        echo=False,
        poolclass=NullPool,
        connect_args={"timeout": 10},
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
