"""
ticket_analyzer.py
Step 1 of the agent loop: Read and preprocess support tickets from CSV.
Stores tickets in SQLite. Returns list of ticket dicts.
"""

import csv
import json
import re
import math
import sqlite3
from pathlib import Path
from typing import List, Dict
from src.database import get_connection, log


# ---------------------------------------------------------------------------
# TF-IDF based text embeddings (pure Python, no external ML needed at this step)
# We produce a simple vector here; the clustering step uses sklearn TF-IDF
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_ticket_text(ticket: Dict) -> str:
    """Merge subject + description + category into one string for embedding."""
    parts = [
        ticket.get("category", ""),
        ticket.get("subject", ""),
        ticket.get("description", ""),
    ]
    return clean_text(" ".join(p for p in parts if p))


def load_tickets_from_csv(csv_path: str) -> List[Dict]:
    """
    Read tickets from CSV file.
    Returns list of ticket dicts.
    """
    tickets = []
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ticket CSV not found: {csv_path}")

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(dict(row))

    log("ticket_analyzer", f"Read {len(tickets)} tickets from {csv_path}")
    return tickets


def store_tickets(tickets: List[Dict]) -> int:
    """
    Upsert tickets into the DB.
    Returns count inserted/updated.
    """
    conn = get_connection()
    count = 0
    for t in tickets:
        try:
            conn.execute("""
                INSERT INTO tickets (ticket_id, date, category, subject, description, status, resolution)
                VALUES (:ticket_id, :date, :category, :subject, :description, :status, :resolution)
                ON CONFLICT(ticket_id) DO UPDATE SET
                    date=excluded.date, category=excluded.category,
                    subject=excluded.subject, description=excluded.description,
                    status=excluded.status, resolution=excluded.resolution
            """, t)
            count += 1
        except Exception as e:
            log("ticket_analyzer", f"Skipped ticket {t.get('ticket_id')}: {e}", "WARN")
    conn.commit()
    conn.close()
    log("ticket_analyzer", f"Stored {count} tickets in DB")
    return count


def get_all_tickets() -> List[Dict]:
    """Fetch all tickets from DB as list of dicts."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM tickets").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_ticket_analyzer(csv_path: str) -> List[Dict]:
    """
    Full Step 1: load CSV -> store to DB -> return tickets with combined text.
    """
    tickets = load_tickets_from_csv(csv_path)
    store_tickets(tickets)
    # Enrich each with combined text for downstream use
    for t in tickets:
        t["combined_text"] = combine_ticket_text(t)
    log("ticket_analyzer", "Ticket analyzer complete")
    return tickets
