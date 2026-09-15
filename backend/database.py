import os

import psycopg2
from psycopg2.extras import RealDictCursor


DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not configured")

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor,
    )