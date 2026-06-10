"""
tests/test_kb_gap_finder.py
Test suite for KB Gap Finder
Run: pytest tests/ -v
"""

import json
import os
import sys
import tempfile
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Use a file-based temp DB so all get_connection() calls share the same database
_TEST_DB = str(Path(tempfile.gettempdir()) / "kb_gap_test.db")
os.environ["KB_DB_PATH"] = _TEST_DB

from src.database import init_db, get_connection, log, get_logs, clear_run_data
from src.ticket_analyzer import (
    clean_text, combine_ticket_text, load_tickets_from_csv,
    store_tickets, run_ticket_analyzer,
)
from src.categorizer import (
    cluster_tickets, infer_cluster_label, get_top_terms,
)
from src.kb_loader import parse_markdown, load_kb_articles, embed_kb_articles
from src.gap_detector import (
    detect_gaps, compute_max_similarity, prioritize_gap,
)
from src.article_generator import (
    build_prompt, generate_template_article, generate_article,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_db():
    """Fresh DB tables for every test."""
    # Remove old test DB if present, reinitialise
    if Path(_TEST_DB).exists():
        Path(_TEST_DB).unlink()
    init_db()
    yield
    # Cleanup after test
    if Path(_TEST_DB).exists():
        Path(_TEST_DB).unlink()


@pytest.fixture
def sample_tickets():
    return [
        {"ticket_id": "T001", "date": "2024-01-01", "category": "ERP Login",
         "subject": "Cannot login to ERP", "description": "Invalid credentials error on login page",
         "status": "Closed", "resolution": "Password reset"},
        {"ticket_id": "T002", "date": "2024-01-02", "category": "ERP Login",
         "subject": "ERP password forgot", "description": "Student forgot password and cannot access ERP portal",
         "status": "Closed", "resolution": "Sent reset link"},
        {"ticket_id": "T003", "date": "2024-01-03", "category": "Attendance",
         "subject": "Attendance not updated", "description": "Attendance marked present but ERP shows absent",
         "status": "Closed", "resolution": "Faculty corrected manually"},
        {"ticket_id": "T004", "date": "2024-01-04", "category": "Attendance",
         "subject": "Wrong attendance percentage", "description": "Attendance percentage is wrong in ERP",
         "status": "Closed", "resolution": "Database sync triggered"},
        {"ticket_id": "T005", "date": "2024-01-05", "category": "Hostel",
         "subject": "Room allotment not showing", "description": "Hostel room allotment page shows no data",
         "status": "Closed", "resolution": "Allotment processed manually"},
        {"ticket_id": "T006", "date": "2024-01-06", "category": "Transport",
         "subject": "Bus route not visible", "description": "Cannot see bus route and stop details in ERP",
         "status": "Closed", "resolution": "Route assigned in backend"},
    ]


@pytest.fixture
def sample_kb_articles():
    return [
        {
            "filename": "erp_login_guide.md",
            "title": "ERP Login Guide",
            "content": "# ERP Login Guide\n\nHow to login and reset password.",
            "clean_text": "ERP Login Guide How to login and reset password forgot password credentials",
        },
        {
            "filename": "attendance_guide.md",
            "title": "Attendance Guide",
            "content": "# Attendance Guide\n\nView and track your attendance.",
            "clean_text": "Attendance Guide view track attendance percentage absent present biometric",
        },
    ]


@pytest.fixture
def sample_gap(sample_tickets):
    return {
        "cluster_id": 0,
        "label": "Hostel: room allotment hostel",
        "topic_summary": "Hostel: room allotment hostel. Example tickets: Room allotment not showing",
        "ticket_count": 3,
        "similarity_score": 0.05,
        "priority": "HIGH",
        "top_terms": ["hostel", "room", "allotment", "payment", "fee"],
        "sample_tickets": sample_tickets[:2],
        "db_id": 1,
    }


@pytest.fixture
def csv_file(sample_tickets, tmp_path):
    f = tmp_path / "tickets.csv"
    fieldnames = ["ticket_id", "date", "category", "subject", "description", "status", "resolution"]
    with open(f, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_tickets)
    return str(f)


@pytest.fixture
def kb_folder(tmp_path):
    folder = tmp_path / "kb"
    folder.mkdir()
    (folder / "erp_login_guide.md").write_text(
        "# ERP Login Guide\n\nHow to login to ERP portal and reset password.", encoding="utf-8"
    )
    (folder / "attendance_guide.md").write_text(
        "# Attendance Guide\n\nView and track attendance percentage.", encoding="utf-8"
    )
    return str(folder)


# ── Database Tests ──────────────────────────────────────────────────────────

class TestDatabase:
    def test_init_db_creates_tables(self):
        conn = get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        conn.close()
        assert "tickets" in table_names
        assert "clusters" in table_names
        assert "gaps" in table_names
        assert "generated_articles" in table_names

    def test_log_writes_entry(self):
        log("test_step", "Test message", "INFO")
        logs = get_logs(limit=5)
        assert any(l["message"] == "Test message" for l in logs)

    def test_get_logs_returns_list(self):
        log("test", "msg1")
        log("test", "msg2")
        logs = get_logs()
        assert isinstance(logs, list)
        assert len(logs) >= 2


# ── Ticket Analyzer Tests ───────────────────────────────────────────────────

class TestTicketAnalyzer:
    def test_clean_text_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_clean_text_removes_punctuation(self):
        result = clean_text("Hello, World!")
        assert "," not in result
        assert "!" not in result

    def test_combine_ticket_text_merges_fields(self, sample_tickets):
        t = sample_tickets[0]
        text = combine_ticket_text(t)
        assert "erp" in text.lower() or "login" in text.lower()

    def test_load_tickets_from_csv(self, csv_file):
        tickets = load_tickets_from_csv(csv_file)
        assert len(tickets) == 6
        assert tickets[0]["ticket_id"] == "T001"

    def test_load_tickets_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_tickets_from_csv("nonexistent.csv")

    def test_store_tickets_inserts_records(self, sample_tickets):
        count = store_tickets(sample_tickets)
        assert count == len(sample_tickets)

    def test_store_tickets_handles_duplicate(self, sample_tickets):
        store_tickets(sample_tickets)
        count = store_tickets(sample_tickets)
        assert count == len(sample_tickets)

    def test_run_ticket_analyzer_returns_enriched(self, csv_file):
        tickets = run_ticket_analyzer(csv_file)
        assert len(tickets) > 0
        assert "combined_text" in tickets[0]


# ── Categorizer Tests ────────────────────────────────────────────────────────

class TestCategorizer:
    def test_cluster_tickets_returns_correct_types(self, sample_tickets):
        tickets, clusters = cluster_tickets(sample_tickets, n_clusters=3)
        assert isinstance(tickets, list)
        assert isinstance(clusters, list)

    def test_cluster_tickets_assigns_cluster_id(self, sample_tickets):
        tickets, _ = cluster_tickets(sample_tickets, n_clusters=2)
        for t in tickets:
            assert "cluster_id" in t
            assert isinstance(t["cluster_id"], int)

    def test_cluster_tickets_produces_n_clusters(self, sample_tickets):
        _, clusters = cluster_tickets(sample_tickets, n_clusters=3)
        assert len(clusters) == 3

    def test_cluster_has_required_fields(self, sample_tickets):
        _, clusters = cluster_tickets(sample_tickets, n_clusters=2)
        for c in clusters:
            assert "label" in c
            assert "ticket_count" in c
            assert "topic_summary" in c

    def test_infer_cluster_label_uses_top_category(self, sample_tickets):
        erp_tickets = [t for t in sample_tickets if t["category"] == "ERP Login"]
        label = infer_cluster_label(erp_tickets, ["login", "password", "erp"])
        assert "ERP Login" in label


# ── KB Loader Tests ──────────────────────────────────────────────────────────

class TestKBLoader:
    def test_parse_markdown_extracts_title(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# My KB Article\n\nContent here.", encoding="utf-8")
        result = parse_markdown(md)
        assert result["title"] == "My KB Article"

    def test_parse_markdown_uses_filename_if_no_h1(self, tmp_path):
        md = tmp_path / "erp_login_guide.md"
        md.write_text("No heading here\n\nJust content.", encoding="utf-8")
        result = parse_markdown(md)
        assert "erp" in result["title"].lower() or "Erp" in result["title"]

    def test_load_kb_articles_reads_all_md(self, kb_folder):
        articles = load_kb_articles(kb_folder)
        assert len(articles) == 2

    def test_load_kb_raises_on_missing_folder(self):
        with pytest.raises(FileNotFoundError):
            load_kb_articles("nonexistent_folder")

    def test_embed_kb_articles_adds_embedding(self, kb_folder):
        articles = load_kb_articles(kb_folder)
        articles = embed_kb_articles(articles)
        for a in articles:
            assert "embedding_json" in a
            vec = json.loads(a["embedding_json"])
            assert isinstance(vec, list)


# ── Gap Detector Tests ───────────────────────────────────────────────────────

class TestGapDetector:
    def test_prioritize_gap_high_for_many_tickets(self):
        p = prioritize_gap(ticket_count=10, max_similarity=0.05)
        assert p == "HIGH"

    def test_prioritize_gap_low_for_few_tickets(self):
        p = prioritize_gap(ticket_count=1, max_similarity=0.05)
        assert p == "LOW"

    def test_detect_gaps_finds_uncovered_clusters(self, sample_tickets, sample_kb_articles):
        _, clusters = cluster_tickets(sample_tickets, n_clusters=3)
        clusters, gaps = detect_gaps(clusters, sample_kb_articles, threshold=0.25)
        assert isinstance(gaps, list)

    def test_detect_gaps_empty_kb_makes_all_gaps(self, sample_tickets):
        _, clusters = cluster_tickets(sample_tickets, n_clusters=2)
        clusters, gaps = detect_gaps(clusters, [], threshold=0.25)
        assert len(gaps) == len(clusters)

    def test_detect_gaps_enriches_clusters(self, sample_tickets, sample_kb_articles):
        _, clusters = cluster_tickets(sample_tickets, n_clusters=2)
        clusters, _ = detect_gaps(clusters, sample_kb_articles, threshold=0.25)
        for c in clusters:
            assert "has_kb_match" in c
            assert "similarity_score" in c

    def test_gaps_sorted_by_priority(self, sample_tickets):
        _, clusters = cluster_tickets(sample_tickets, n_clusters=3)
        _, gaps = detect_gaps(clusters, [], threshold=0.25)
        priorities = [g["priority"] for g in gaps]
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_p = sorted(priorities, key=lambda p: order.get(p, 3))
        assert priorities == sorted_p


# ── Article Generator Tests ──────────────────────────────────────────────────

class TestArticleGenerator:
    def test_build_prompt_returns_string(self, sample_gap):
        prompt = build_prompt(sample_gap)
        assert isinstance(prompt, str)
        assert "Knowledge Base" in prompt

    def test_build_prompt_includes_topic(self, sample_gap):
        prompt = build_prompt(sample_gap)
        assert "Hostel" in prompt or "hostel" in prompt.lower()

    def test_template_article_returns_markdown(self, sample_gap):
        content = generate_template_article(sample_gap)
        assert isinstance(content, str)
        assert content.startswith("#")

    def test_template_article_has_required_sections(self, sample_gap):
        content = generate_template_article(sample_gap)
        assert "Problem Statement" in content
        assert "Resolution" in content
        assert "Contact" in content

    def test_template_article_includes_ticket_count(self, sample_gap):
        content = generate_template_article(sample_gap)
        assert str(sample_gap["ticket_count"]) in content

    def test_generate_article_falls_back_to_template(self, sample_gap):
        with patch("src.article_generator.call_ollama", return_value=None), \
             patch("src.article_generator.call_anthropic", return_value=None):
            article = generate_article(sample_gap, llm_provider="auto")
            assert article["llm_provider"] == "template"
            assert len(article["content"]) > 100

    def test_generate_article_uses_ollama_if_available(self, sample_gap):
        mock_content = "# Hostel Guide\n\n## Problem Statement\n\nTest content."
        with patch("src.article_generator.call_ollama", return_value=mock_content):
            article = generate_article(sample_gap, llm_provider="ollama")
            assert article["llm_provider"] == "ollama"
            assert article["content"] == mock_content


# ── Happy Path Integration Tests ─────────────────────────────────────────────

class TestHappyPath:
    def test_full_pipeline_end_to_end(self, csv_file, kb_folder):
        """Full pipeline: load CSV -> cluster -> load KB -> detect gaps -> generate articles."""
        tickets = run_ticket_analyzer(csv_file)
        assert len(tickets) > 0

        from src.categorizer import run_categorizer
        tickets, clusters = run_categorizer(tickets, n_clusters=3)
        assert len(clusters) == 3

        from src.kb_loader import run_kb_loader
        kb_articles = run_kb_loader(kb_folder)
        assert len(kb_articles) == 2

        from src.gap_detector import run_gap_detector
        clusters, gaps = run_gap_detector(clusters, kb_articles, threshold=0.25)
        assert isinstance(gaps, list)

        from src.article_generator import run_article_generator
        with patch("src.article_generator.call_ollama", return_value=None), \
             patch("src.article_generator.call_anthropic", return_value=None):
            articles = run_article_generator(gaps, llm_provider="template")
            assert len(articles) == len(gaps)
            for art in articles:
                assert "title" in art
                assert "content" in art
                assert len(art["content"]) > 50

    def test_gaps_have_no_erp_login_or_attendance_gap(self, csv_file, kb_folder):
        """With ERP Login + Attendance KB articles, those clusters should be covered."""
        tickets = run_ticket_analyzer(csv_file)
        from src.categorizer import run_categorizer
        from src.kb_loader import run_kb_loader
        from src.gap_detector import run_gap_detector

        _, clusters = run_categorizer(tickets, n_clusters=3)
        kb_articles = run_kb_loader(kb_folder)
        clusters, gaps = run_gap_detector(clusters, kb_articles, threshold=0.25)

        covered = [c for c in clusters if c.get("has_kb_match")]
        assert len(covered) > 0, "Expected at least one cluster to be covered by KB"
