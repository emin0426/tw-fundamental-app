"""Tiered local SQLite cache to prevent API throttling and speed up repeat queries."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Optional


class TieredCache:
    """Simple thread-safe SQLite key-value cache with per-entry TTL."""

    TTL_QUOTE = 5 * 60             # 5 minutes for latest price
    TTL_MARKET_VAL = 24 * 3600     # 1 day for market-wide PE/PB snapshot
    TTL_MONTHLY_REV = 24 * 3600    # 1 day for monthly revenue
    TTL_FINANCIALS = 7 * 24 * 3600 # 7 days for quarterly/annual financials
    TTL_STOCK_LIST = 7 * 24 * 3600 # 7 days for stock symbol/name map
    TTL_NEWS = 6 * 3600            # 6 hours for news & announcements
    TTL_AI_REPORT = 24 * 3600      # 1 day for AI summaries (saves free quota)

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path:
            self.db_path = Path(db_path)
        else:
            default_dir = Path.cwd() / ".cache"
            try:
                default_dir.mkdir(parents=True, exist_ok=True)
                test_file = default_dir / ".write_test"
                test_file.write_text("ok", encoding="utf-8")
                test_file.unlink(missing_ok=True)
                self.db_path = default_dir / "tw_fundamental_cache.db"
            except OSError:
                # Fallback to temp directory on read-only cloud containers
                tmp_dir = Path(tempfile.gettempdir()) / "tw_fundamental_cache"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                self.db_path = tmp_dir / "tw_fundamental_cache.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_store (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM cache_store WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            payload, expires_at = row
            if now > expires_at:
                conn.execute("DELETE FROM cache_store WHERE cache_key = ?", (key,))
                return None
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        now = time.time()
        payload = json.dumps(value, ensure_ascii=False, default=str)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_store (cache_key, payload, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (key, payload, now + ttl_seconds, now),
            )

    def clear_prefix(self, prefix: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cache_store WHERE cache_key LIKE ?",
                (f"{prefix}%",),
            )
            return cur.rowcount

    def clear_all(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM cache_store")
