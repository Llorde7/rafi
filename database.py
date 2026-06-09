from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from uuid import uuid4
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import os
import ssl


def _build_ssl_context(verify: bool = True) -> ssl.SSLContext:
    """
    Build an SSL context for asyncpg.

    By default uses the system CA bundle (certifi if installed, then
    fall back to ssl.create_default_context()). If verify=False, disables
    certificate verification (insecure — use only for diagnostics or trusted
    internal networks).

    The context is explicitly used as `ssl=ctx` in connect_args so that
    asyncpg does not silently fall back to its own default (which is
    `ssl=True` with no CA file, causing "self-signed certificate in
    certificate chain" errors on hosts that present a CA-signed cert
    chain that Python cannot locate).
    """
    if not verify:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    # Prefer certifi's CA bundle if installed (most reliable on minimal
    # container images that lack /etc/ssl/certs).
    try:
        import certifi  # type: ignore
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    return ctx

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

# Build the SSL context. Set DB_SSL_DISABLE=1 to skip cert verification
# (only use for local development or trusted internal networks).
ssl_verify = os.getenv("DB_SSL_DISABLE", "0") not in ("1", "true", "True")
ssl_ctx = _build_ssl_context(verify=ssl_verify) if needs_ssl else None

if ":5432" in DATABASE_URL:
    connect_args = {
        "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
        "timeout": 10,
        "server_settings": {
            "jit": "off",
            "statement_timeout": "60000",
        },
    }
    if ssl_ctx is not None:
        connect_args["ssl"] = ssl_ctx
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
    if ssl_ctx is not None:
        connect_args["ssl"] = ssl_ctx
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
