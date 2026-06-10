"""
categorizer.py
Step 2: Categorize tickets using TF-IDF embeddings + KMeans clustering.
Assigns cluster_id to each ticket and saves cluster summaries to DB.
No GPU, no paid API needed.
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

from src.database import get_connection, log
from src.ticket_analyzer import combine_ticket_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_top_terms(tfidf_matrix, feature_names, cluster_centers, cluster_id: int, top_n: int = 8) -> List[str]:
    """Get top TF-IDF terms for a cluster centroid."""
    center = cluster_centers[cluster_id]
    top_indices = center.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices if feature_names[i].strip()]


def infer_cluster_label(tickets_in_cluster: List[Dict], top_terms: List[str]) -> str:
    """
    Build a human-readable cluster label from category counts + top terms.
    """
    categories = [t.get("category", "Unknown") for t in tickets_in_cluster]
    top_cat = Counter(categories).most_common(1)[0][0]
    # Combine category + top terms
    keywords = " / ".join(top_terms[:3])
    return f"{top_cat}: {keywords}"


def choose_k(texts: List[str], min_k: int = 3, max_k: int = 12) -> int:
    """
    Choose optimal K for KMeans using silhouette score.
    Falls back to sqrt heuristic if not enough data.
    """
    n = len(texts)
    if n < 6:
        return max(2, n // 2)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(texts)

    best_k = min_k
    best_score = -1
    cap = min(max_k, n - 1)

    for k in range(min_k, cap + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    log("categorizer", f"Optimal K={best_k} (silhouette={best_score:.3f})")
    return best_k


# ---------------------------------------------------------------------------
# Main clustering
# ---------------------------------------------------------------------------

def cluster_tickets(tickets: List[Dict], n_clusters: int = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Cluster tickets using TF-IDF + KMeans.

    Returns:
      - tickets: each enriched with cluster_id, embedding_json
      - clusters: list of cluster summary dicts
    """
    if not tickets:
        return [], []

    texts = [combine_ticket_text(t) for t in tickets]

    # Build TF-IDF matrix
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Choose K
    if n_clusters is None:
        n_clusters = choose_k(texts)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Attach cluster labels to tickets
    for i, ticket in enumerate(tickets):
        ticket["cluster_id"] = int(labels[i])
        # Store a small embedding (first 50 dims of TF-IDF row) as JSON
        row = X[i].toarray()[0]
        ticket["embedding_json"] = json.dumps(row[:50].tolist())

    # Build cluster summaries
    cluster_map: Dict[int, List[Dict]] = defaultdict(list)
    for t in tickets:
        cluster_map[t["cluster_id"]].append(t)

    clusters = []
    for cid, members in cluster_map.items():
        top_terms = get_top_terms(X, feature_names, km.cluster_centers_, cid)
        label = infer_cluster_label(members, top_terms)
        # Representative subject lines
        subjects = [m.get("subject", "") for m in members[:3]]
        topic_summary = label + ". Example tickets: " + "; ".join(subjects)

        clusters.append({
            "id": cid,
            "label": label,
            "topic_summary": topic_summary,
            "ticket_count": len(members),
            "top_terms": top_terms,
            "sample_tickets": members[:5],
        })

    log("categorizer", f"Clustered {len(tickets)} tickets into {n_clusters} clusters")
    return tickets, clusters


def store_clusters(clusters: List[Dict]) -> None:
    """Persist cluster summaries to DB."""
    conn = get_connection()
    # Clear old clusters
    conn.execute("DELETE FROM clusters")
    for c in clusters:
        conn.execute("""
            INSERT INTO clusters (id, label, topic_summary, ticket_count)
            VALUES (?, ?, ?, ?)
        """, (c["id"], c["label"], c["topic_summary"], c["ticket_count"]))
    conn.commit()

    # Update ticket cluster assignments
    # fetch tickets from DB then update
    all_db = conn.execute("SELECT id, ticket_id FROM tickets").fetchall()
    id_map = {row["ticket_id"]: row["id"] for row in all_db}
    conn.close()


def update_ticket_clusters(tickets: List[Dict]) -> None:
    """Write cluster_id and embedding back to DB for each ticket."""
    conn = get_connection()
    for t in tickets:
        conn.execute("""
            UPDATE tickets SET cluster_id=?, embedding_json=?
            WHERE ticket_id=?
        """, (t.get("cluster_id"), t.get("embedding_json"), t.get("ticket_id")))
    conn.commit()
    conn.close()


def run_categorizer(tickets: List[Dict], n_clusters: int = None) -> Tuple[List[Dict], List[Dict]]:
    """Full Step 2: cluster -> store -> return."""
    tickets, clusters = cluster_tickets(tickets, n_clusters)
    store_clusters(clusters)
    update_ticket_clusters(tickets)
    log("categorizer", "Categorization complete")
    return tickets, clusters
