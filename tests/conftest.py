"""Shared fixtures.

Tests run against a real PostgreSQL + pgvector database (``OVP_TEST_DATABASE_URL``, default
``postgresql://postgres@localhost:5432/ovp_test``) with fake models. The schema is rebuilt once
per session and every table is truncated before each test. Database tests are skipped when the
server is unreachable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlsplit, urlunsplit

TEST_DB_URL = os.environ.get("OVP_TEST_DATABASE_URL", "postgresql://postgres@localhost:5432/ovp_test")
_STORAGE = Path(tempfile.mkdtemp(prefix="ovp_test_storage_"))

# Settings are read lazily from the environment: pin them before any ``ovp`` import.
os.environ.update(
    DATABASE_URL=TEST_DB_URL,
    OVP_STORAGE_DIR=str(_STORAGE),
    OVP_PRELOAD="0",
    OVP_DEVICE="cpu",
    OVP_MEDIA_CONCURRENCY="3",
    OVP_JOB_CONCURRENCY="2",
    OVP_FRONTEND_DIST=str(_STORAGE / "no-frontend"),
)
for _key in ("OVP_REID_MATCH", "OVP_REID_REVIEW", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OVP_COMPAT_BASE_URL"):
    os.environ.pop(_key, None)

import asyncpg  # noqa: E402
import pytest  # noqa: E402

import ovp.context  # noqa: E402
import ovp.runtime  # noqa: E402
from ovp.context import AppContext  # noqa: E402
from ovp.storage.db import Database  # noqa: E402
from ovp.storage.files import FileStore  # noqa: E402

from .fakes import FakeDetector, FakeRegistry, FakeReID, FakeTextEmbedder, FakeVLM  # noqa: E402

# A CPU runtime without probing torch for accelerators (keeps start-up fast).
ovp.runtime._runtime = ovp.runtime.Runtime("cpu")

TABLES = ("qa", "embeddings", "entities", "persons", "media")


def db_url(name: str) -> str:
    parts = urlsplit(TEST_DB_URL)
    return urlunsplit(parts._replace(path=f"/{name}"))


async def admin_connect() -> asyncpg.Connection:
    return await asyncpg.connect(db_url("postgres"))


async def recreate_database(name: str) -> str:
    """Drop and create database ``name`` with pgvector; returns its URL."""
    conn = await admin_connect()
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()
    url = db_url(name)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await conn.close()
    return url


async def drop_database(name: str) -> None:
    conn = await admin_connect()
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
async def database() -> AsyncIterator[Database]:
    try:
        url = await recreate_database(urlsplit(TEST_DB_URL).path.lstrip("/"))
    except (OSError, asyncpg.PostgresError) as e:
        pytest.skip(f"PostgreSQL unavailable at {TEST_DB_URL}: {e}")
    database = Database(url)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def db(database: Database) -> Database:
    assert database.pool
    await database.pool.execute(f"TRUNCATE {', '.join(TABLES)} CASCADE")
    return database


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    return FileStore(tmp_path / "storage")


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry(detector=FakeDetector(), vlm=FakeVLM(), reid=FakeReID(), text_embedder=FakeTextEmbedder())


@pytest.fixture
def app_ctx(db: Database, store: FileStore, registry: FakeRegistry) -> Any:
    ctx = AppContext(db, store, registry)  # type: ignore[arg-type]
    ctx.ready = True
    previous = ovp.context._ctx
    ovp.context._ctx = ctx
    yield ctx
    ovp.context._ctx = previous
