"""
gap_detector.py
Step 4: Compare ticket clusters against KB articles.
Uses TF-IDF cosine similarity to find clusters with no adequate KB coverage.
Outputs a prioritised Gap List.
"""

import json
import math
import re
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.database import get_connection, log


# Similarity threshold: clusters below this score are considered "gaps"
GAP_THRESHOLD = 0.25


def build_joint_vectorizer(cluster_texts: List[str], kb_texts: List[str]):
    """
    Fit a single TF-IDF vectorizer on ALL texts (clusters + KB articles)
    so that vectors are in the same embedding space for fair comparison.
    """
    all_texts = cluster_texts + kb_texts
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        ngram_range=(1, 2),
        min_df=1,
    )
    vectorizer.fit(all_texts)
    return vectorizer


def compute_max_similarity(cluster_vec, kb_matrix) -> Tuple[float, int]:
    """
    Return (max_similarity, best_kb_idx) between cluster vector and all KB vectors.
    """
    if kb_matrix.shape[0] == 0:
        return 0.0, -1
    sims = cosine_similarity(cluster_vec, kb_matrix)[0]
    best_idx = int(np.argmax(sims))
    return float(sims[best_idx]), best_idx


def prioritize_gap(ticket_count: int, max_similarity: float) -> str:
    """
    Priority = HIGH if many tickets and low KB coverage.
    """
    coverage_gap = 1.0 - max_similarity
    score = ticket_count * coverage_gap
    if score >= 8:
        return "HIGH"
    elif score >= 4:
        return "MEDIUM"
    return "LOW"


def detect_gaps(
    clusters: List[Dict],
    kb_articles: List[Dict],
    threshold: float = GAP_THRESHOLD,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Compare each cluster to KB articles.

    Returns:
      - clusters: enriched with has_kb_match, similarity_score, best_kb_title
      - gaps: list of gap dicts for clusters below threshold
    """
    if not clusters:
        return [], []

    # Build text for each cluster (topic_summary + sample ticket descriptions)
    cluster_texts = []
    for c in clusters:
        samples = " ".join([t.get("description", "") for t in c.get("sample_tickets", [])[:3]])
        cluster_texts.append(c.get("topic_summary", "") + " " + samples)

    kb_texts = [a.get("clean_text", a.get("content", "")) for a in kb_articles]

    if not kb_texts:
        # No KB articles at all -> every cluster is a gap
        for c in clusters:
            c["has_kb_match"] = False
            c["similarity_score"] = 0.0
            c["best_kb_title"] = None
        gaps = _build_gaps(clusters, threshold=0.0)
        return clusters, gaps

    # Fit joint vectorizer
    vectorizer = build_joint_vectorizer(cluster_texts, kb_texts)
    C = vectorizer.transform(cluster_texts)
    K = vectorizer.transform(kb_texts)

    gaps = []
    for i, cluster in enumerate(clusters):
        cluster_vec = C[i]
        max_sim, best_idx = compute_max_similarity(cluster_vec, K)
        cluster["has_kb_match"] = max_sim >= threshold
        cluster["similarity_score"] = round(max_sim, 4)
        cluster["best_kb_title"] = kb_articles[best_idx]["title"] if best_idx >= 0 else None
        cluster["best_kb_filename"] = kb_articles[best_idx].get("filename") if best_idx >= 0 else None

        if not cluster["has_kb_match"]:
            priority = prioritize_gap(cluster["ticket_count"], max_sim)
            gaps.append({
                "cluster_id": cluster["id"],
                "topic_summary": cluster["topic_summary"],
                "ticket_count": cluster["ticket_count"],
                "similarity_score": cluster["similarity_score"],
                "priority": priority,
                "sample_tickets": cluster.get("sample_tickets", []),
                "top_terms": cluster.get("top_terms", []),
                "label": cluster.get("label", ""),
            })

    # Sort gaps by priority then ticket count
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    gaps.sort(key=lambda g: (priority_order.get(g["priority"], 3), -g["ticket_count"]))

    log("gap_detector", f"Found {len(gaps)} gaps out of {len(clusters)} clusters (threshold={threshold})")
    return clusters, gaps


def _build_gaps(clusters: List[Dict], threshold: float) -> List[Dict]:
    gaps = []
    for c in clusters:
        if not c.get("has_kb_match", False):
            priority = prioritize_gap(c["ticket_count"], c.get("similarity_score", 0.0))
            gaps.append({
                "cluster_id": c["id"],
                "topic_summary": c["topic_summary"],
                "ticket_count": c["ticket_count"],
                "similarity_score": c.get("similarity_score", 0.0),
                "priority": priority,
                "sample_tickets": c.get("sample_tickets", []),
                "top_terms": c.get("top_terms", []),
                "label": c.get("label", ""),
            })
    return gaps


def store_gaps(gaps: List[Dict]) -> List[int]:
    """Persist gaps to DB, return list of gap IDs."""
    conn = get_connection()
    conn.execute("DELETE FROM gaps")
    gap_ids = []
    for g in gaps:
        cur = conn.execute("""
            INSERT INTO gaps (cluster_id, topic_summary, ticket_count, priority, status)
            VALUES (?, ?, ?, ?, 'pending')
        """, (g["cluster_id"], g["topic_summary"], g["ticket_count"], g["priority"]))
        gap_ids.append(cur.lastrowid)
        g["db_id"] = cur.lastrowid
    conn.commit()
    # Update cluster has_kb_match flags
    all_clusters = conn.execute("SELECT id FROM clusters").fetchall()
    for row in all_clusters:
        cid = row["id"]
        has_match = all(g["cluster_id"] != cid for g in gaps)
        conn.execute("UPDATE clusters SET has_kb_match=? WHERE id=?", (int(has_match), cid))
    conn.commit()
    conn.close()
    log("gap_detector", f"Stored {len(gaps)} gaps in DB")
    return gap_ids


def run_gap_detector(clusters: List[Dict], kb_articles: List[Dict], threshold: float = GAP_THRESHOLD):
    """Full Step 4: detect -> store -> return."""
    clusters, gaps = detect_gaps(clusters, kb_articles, threshold)
    store_gaps(gaps)
    log("gap_detector", "Gap Detector complete")
    return clusters, gaps
