"""
AI Agent Workflow Loop - orchestrates all 5 pipeline steps:
  1. Ticket Analyzer
  2. Categorizer (clustering)
  3. KB Loader
  4. Gap Detector
  5. Article Generator

Yields status updates (generator) so the Streamlit UI can show live progress.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import init_db, clear_run_data, log, get_connection
from src.ticket_analyzer import run_ticket_analyzer
from src.categorizer import run_categorizer
from src.kb_loader import run_kb_loader
from src.gap_detector import run_gap_detector
from src.article_generator import generate_article, store_generated_article


def emit(step: str, message: str, data: Any = None) -> Dict:
    """Helper to build a status event dict."""
    log(step, message)
    return {"step": step, "message": message, "data": data}


def run_agent(
    ticket_csv: str,
    kb_folder: str,
    n_clusters: int = None,
    gap_threshold: float = 0.25,
    llm_provider: str = "auto",
    max_articles: int = None,
    fresh_run: bool = True,
) -> Generator[Dict, None, None]:
    """
    Full agent loop as a generator.
    Each yield = one status update for the UI.
    Final yield contains 'result' key with summary stats.
    """
    # Init DB
    init_db()
    if fresh_run:
        clear_run_data()
        yield emit("init", "Database initialised. Starting fresh run.")

    # -------------------------------------------------------------------------
    # Step 1: Ticket Analyzer
    # -------------------------------------------------------------------------
    yield emit("ticket_analyzer", "Step 1/5 - Reading support tickets from CSV...")
    try:
        tickets = run_ticket_analyzer(ticket_csv)
        yield emit("ticket_analyzer", f"Loaded {len(tickets)} tickets.", {"ticket_count": len(tickets)})
    except FileNotFoundError as e:
        yield emit("ticket_analyzer", f"ERROR: {e}", {"error": str(e)})
        return

    # -------------------------------------------------------------------------
    # Step 2: Categorization / Clustering
    # -------------------------------------------------------------------------
    yield emit("categorizer", "Step 2/5 - Clustering tickets by topic (TF-IDF + KMeans)...")
    tickets, clusters = run_categorizer(tickets, n_clusters=n_clusters)
    yield emit("categorizer", f"Found {len(clusters)} topic clusters.", {"cluster_count": len(clusters)})

    # -------------------------------------------------------------------------
    # Step 3: KB Loader
    # -------------------------------------------------------------------------
    yield emit("kb_loader", "Step 3/5 - Loading existing KB articles...")
    try:
        kb_articles = run_kb_loader(kb_folder)
        yield emit("kb_loader", f"Loaded {len(kb_articles)} KB articles.", {"kb_count": len(kb_articles)})
    except FileNotFoundError as e:
        yield emit("kb_loader", f"WARNING: {e} - proceeding with empty KB.", {"kb_count": 0})
        kb_articles = []

    # -------------------------------------------------------------------------
    # Step 4: Gap Detector
    # -------------------------------------------------------------------------
    yield emit("gap_detector", f"Step 4/5 - Detecting KB gaps (threshold={gap_threshold})...")
    clusters, gaps = run_gap_detector(clusters, kb_articles, threshold=gap_threshold)
    covered = len(clusters) - len(gaps)
    yield emit(
        "gap_detector",
        f"Found {len(gaps)} gaps. {covered}/{len(clusters)} clusters have KB coverage.",
        {"gap_count": len(gaps), "covered": covered},
    )

    # -------------------------------------------------------------------------
    # Step 5: Article Generator
    # -------------------------------------------------------------------------
    if gaps:
        gaps_to_generate = gaps[:max_articles] if max_articles else gaps
        skipped_count = len(gaps) - len(gaps_to_generate)
        yield emit(
            "article_generator",
            f"Step 5/5 - Generating {len(gaps_to_generate)} KB draft articles ({llm_provider})...",
        )
        if skipped_count:
            yield emit(
                "article_generator",
                f"Fast mode: skipped {skipped_count} lower-priority gaps. Increase article limit to generate all.",
            )
        articles = []

        max_workers = min(3, len(gaps_to_generate))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(generate_article, gap, llm_provider): gap
                for gap in gaps_to_generate
            }
            for future in as_completed(future_map):
                gap = future_map[future]
                gap_label = gap.get("label", gap.get("topic_summary", "KB gap"))[:60]
                try:
                    article = future.result()
                except Exception as error:
                    yield emit("article_generator", f"Article failed for {gap_label}: {error}")
                    continue

                article["id"] = store_generated_article(article)
                articles.append(article)
                yield emit(
                    "article_generator",
                    f"Generated article {len(articles)}/{len(gaps_to_generate)} using {article['llm_provider']}: {gap_label}",
                    {"article_count": len(articles), "article_total": len(gaps_to_generate)},
                )
        yield emit("article_generator", f"Generated {len(articles)} draft KB articles.", {"article_count": len(articles)})
    else:
        articles = []
        yield emit("article_generator", "Step 5/5 - No gaps found, no articles to generate.")

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    yield emit("completed", "Agent loop complete.", {
        "result": {
            "ticket_count": len(tickets),
            "cluster_count": len(clusters),
            "kb_article_count": len(kb_articles),
            "gap_count": len(gaps),
            "article_count": len(articles),
            "gaps": gaps,
            "clusters": clusters,
            "articles": articles,
        }
    })
