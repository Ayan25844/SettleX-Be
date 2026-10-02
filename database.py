import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

RAW_DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if RAW_DATABASE_URL:
    if RAW_DATABASE_URL.startswith(("http://", "https://")):
        raise ValueError(
            "DATABASE_URL must be a database connection URL, not an HTTP(S) URL. "
            "For Supabase, use its PostgreSQL connection string, such as "
            "postgresql+psycopg://user:password@host:5432/postgres."
        )

    # Supabase PostgreSQL connection string normalization for psycopg (v3) driver
    if RAW_DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = RAW_DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
    elif RAW_DATABASE_URL.startswith("postgresql://") and not RAW_DATABASE_URL.startswith("postgresql+"):
        DATABASE_URL = RAW_DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    else:
        DATABASE_URL = RAW_DATABASE_URL
else:
    # Fallback to local SQLite if DATABASE_URL is not configured
    DATABASE_URL = "sqlite:///./settlex.db"

is_sqlite = DATABASE_URL.startswith("sqlite")

engine_kwargs = {}
if is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_recycle"] = 300

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db() -> None:
    """
    Initialize database tables safely without destroying existing data.
    SQLAlchemy's create_all will only create tables that do not already exist.
    """
    import models.user  # noqa: F401
    import models.profile  # noqa: F401
    import models.match  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to provide a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
