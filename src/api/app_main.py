from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import os
import platform
import asyncio
from urllib.parse import urlparse

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover
    asyncpg = None

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None

app = FastAPI(
    title="Kimera SWM - Minimal API",
    description="Stable, minimal entrypoint to keep development reliable while the larger system is cleaned up.",
    version="0.1.1",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


class StatusResponse(BaseModel):
    app: str
    version: str
    environment: str
    pythonpath: str
    database_url_set: bool
    system: Dict[str, str]


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Kimera SWM minimal API is running", "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/system/status", response_model=StatusResponse)
def system_status() -> StatusResponse:
    return StatusResponse(
        app="kimera-swm-minimal",
        version="0.1.1",
        environment=os.getenv("KIMERA_ENV", "development"),
        pythonpath=os.getenv("PYTHONPATH", ""),
        database_url_set=bool(os.getenv("DATABASE_URL")),
        system={
            "os": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
    )


@app.get("/system/db-check")
async def db_check() -> Dict[str, str]:
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        return {"status": "unknown", "detail": "DATABASE_URL not set"}

    parsed = urlparse(db_url)
    if parsed.scheme.startswith("postgres"):
        if asyncpg is None:
            return {"status": "missing", "detail": "asyncpg not installed"}
        try:
            conn = await asyncpg.connect(dsn=db_url)
            try:
                await conn.execute("SELECT 1;")
            finally:
                await conn.close()
            return {"status": "ok", "driver": "asyncpg", "db": parsed.path.lstrip("/")}
        except Exception as e:  # pragma: no cover
            return {"status": "fail", "error": str(e)}
    elif parsed.scheme.startswith("sqlite"):
        return {"status": "ok", "driver": "sqlite", "db": parsed.path}
    else:
        return {"status": "unknown", "detail": f"scheme {parsed.scheme} not supported"}


@app.get("/system/redis-check")
def redis_check() -> Dict[str, str]:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    if redis is None:
        return {"status": "missing", "detail": "redis-py not installed"}
    try:
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=1.5)
        pong = client.ping()
        return {"status": "ok" if pong else "fail", "url": redis_url}
    except Exception as e:  # pragma: no cover
        return {"status": "fail", "url": redis_url, "error": str(e)}
