"""SQLAlchemy engine and session factory."""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from bbbot.config import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _enable_sqlite_fk(dbapi_conn, connection_record):
    """Enable foreign keys for SQLite connections."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.db.url

        # Ensure SQLite directory exists
        if url.startswith("sqlite:///"):
            db_path = Path(url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(url, echo=settings.db.echo)

        if url.startswith("sqlite"):
            event.listen(_engine, "connect", _enable_sqlite_fk)

    return _engine


def get_session() -> Session:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory()


def init_db():
    """Create all tables."""
    from bbbot.db.models import Base
    Base.metadata.create_all(get_engine())
