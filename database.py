from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from uuid import uuid4
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL or DB_URL must be set")

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# Strip sslmode from the URL — for asyncpg, SSL is configured via
# connect_args={"ssl": True}, not via the URL query string. SQLAlchemy
# would otherwise forward sslmode=require to asyncpg's connect() as a
# kwarg, which raises TypeError: connect() got an unexpected keyword
# argument 'sslmode'.
# Supabase and most managed Postgres providers require SSL by default,
# so we always enable it here for any non-localhost host.
parsed = urlparse(DATABASE_URL)
if parsed.query:
    query = parse_qs(parsed.query)
    query.pop("sslmode", None)
    query.pop("ssl", None)
    DATABASE_URL = urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
else:
    DATABASE_URL = urlunparse(parsed._replace(query=""))

# Determine if this is a local DB (no SSL needed) or remote (SSL required)
host = (parsed.hostname or "").lower()
needs_ssl = host not in ("localhost", "127.0.0.1", "::1") and host != ""

if ":5432" in DATABASE_URL:
    connect_args = {
        "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
        "timeout": 10,
        "server_settings": {
            "jit": "off",
            "statement_timeout": "60000",
        },
    }
    if needs_ssl:
        connect_args["ssl"] = True
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args=connect_args,
    )
else:
    sep = "&" if "?" in DATABASE_URL else "?"
    url_with_cache_off = (
        DATABASE_URL + sep + "prepared_statement_cache_size=0"
    )
    connect_args = {"timeout": 10}
    if needs_ssl:
        connect_args["ssl"] = True
    engine = create_async_engine(
        url_with_cache_off,
        echo=False,
        poolclass=NullPool,
        connect_args=connect_args,
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
