"""Shared Postgres connection. ingest.py, train_model.py, and app.py all read
DATABASE_URL from .env so the connection string lives in exactly one place."""

import os

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

load_dotenv()


def get_engine() -> Engine:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill in "
            "your Supabase connection string."
        )
    return create_engine(database_url)
