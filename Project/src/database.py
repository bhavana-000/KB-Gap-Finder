"""
database.py
SQLite schema setup for KB Gap Finder.
Tables: tickets, clusters, kb_articles, gaps, generated_articles, logs
"""

import sqlite3
import os
from pathlib import Path

DB_PATH = os.environ.get("KB_DB_PATH", "kb_gap_finder.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS tickets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id   TEXT UNIQUE,
            date        TEXT,
            category    TEXT,
            subject     TEXT,
            description TEXT,
            status      TEXT,
            resolution  TEXT,
            cluster_id  INTEGER,
            embedding_json TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS clusters (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            label           TEXT,
            topic_summary   TEXT,
            ticket_count    INTEGER DEFAULT 0,
            has_kb_match    INTEGER DEFAULT 0,
            kb_article_id   INTEGER,
            similarity_score REAL DEFAULT 0.0,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS kb_articles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT UNIQUE,
            title       TEXT,
            content     TEXT,
            embedding_json TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS gaps (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id      INTEGER REFERENCES clusters(id),
            topic_summary   TEXT,
            ticket_count    INTEGER,
            priority        TEXT,
            status          TEXT DEFAULT 'pending',
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS generated_articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            gap_id          INTEGER REFERENCES gaps(id),
            cluster_id      INTEGER,
            title           TEXT,
            content         TEXT,
            llm_provider    TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            step        TEXT,
            message     TEXT,
            level       TEXT DEFAULT 'INFO',
            created_at  TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def log(step: str, message: str, level: str = "INFO"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO logs (step, message, level) VALUES (?, ?, ?)",
        (step, message, level)
    )
    conn.commit()
    conn.close()


def get_logs(limit: int = 100):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM logs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_run_data():
    """Clear all processed data to allow a fresh run."""
    conn = get_connection()
    conn.executescript("""
        DELETE FROM generated_articles;
        DELETE FROM gaps;
        DELETE FROM clusters;
        UPDATE tickets SET cluster_id = NULL, embedding_json = NULL;
        DELETE FROM kb_articles;
        DELETE FROM logs;
    """)
    conn.commit()
    conn.close()
