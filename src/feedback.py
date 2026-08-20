"""Session-oriented SQLite request and feedback logging."""

import hashlib
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_DIR

FEEDBACK_DB_PATH = os.getenv(
    "FEEDBACK_DB_PATH", os.path.join(DATA_DIR, "feedback.sqlite3")
)
_IP_HASH_SALT = os.getenv("FEEDBACK_IP_SALT") or secrets.token_hex(16)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connection() -> sqlite3.Connection:
    path = Path(FEEDBACK_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize_feedback_db() -> None:
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                ip_hash TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                band TEXT NOT NULL,
                signals_json TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                helpful INTEGER NOT NULL,
                reason TEXT,
                FOREIGN KEY (request_id) REFERENCES requests(request_id)
            );
            """
        )


def hash_ip(ip: str) -> str:
    return hashlib.sha256(
        (_IP_HASH_SALT + ":" + (ip or "unknown")).encode("utf-8")
    ).hexdigest()


def log_request(
    *,
    request_id: str,
    ip: str,
    question: str,
    answer: str,
    band: str,
    signals: dict,
    latency_ms: float,
    error: str | None = None,
) -> None:
    with _connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO requests
            (request_id, timestamp, ip_hash, question, answer, band,
             signals_json, latency_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                _now(),
                hash_ip(ip),
                question,
                answer,
                band,
                json.dumps(signals, sort_keys=True),
                latency_ms,
                error,
            ),
        )


def log_feedback(
    *, request_id: str, helpful: bool, reason: str | None = None
) -> None:
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO feedback (request_id, timestamp, helpful, reason)
            VALUES (?, ?, ?, ?)
            """,
            (request_id, _now(), int(helpful), reason),
        )
