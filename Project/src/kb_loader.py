"""
kb_loader.py
Step 3: Load existing KB articles from a folder of Markdown files.
Extracts title + content, stores in DB with TF-IDF embeddings for RAG matching.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from src.database import get_connection, log


def parse_markdown(filepath: Path) -> Dict:
    """Extract title and clean text from a Markdown file."""
    content = filepath.read_text(encoding="utf-8")
    # Title = first H1 heading, or filename
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else filepath.stem.replace("_", " ").title()
    # Clean text: strip markdown syntax for embedding
    clean = re.sub(r"#+\s*", "", content)       # remove headings
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)  # bold
    clean = re.sub(r"\*(.+?)\*", r"\1", clean)  # italic
    clean = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", clean)  # links
    clean = re.sub(r"`+", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return {
        "filename": filepath.name,
        "title": title,
        "content": content,
        "clean_text": clean,
    }


def load_kb_articles(kb_folder: str) -> List[Dict]:
    """Load all .md files from a folder. Returns list of article dicts."""
    folder = Path(kb_folder)
    if not folder.exists():
        raise FileNotFoundError(f"KB folder not found: {kb_folder}")

    articles = []
    for md_file in sorted(folder.glob("*.md")):
        try:
            article = parse_markdown(md_file)
            articles.append(article)
        except Exception as e:
            log("kb_loader", f"Failed to parse {md_file.name}: {e}", "WARN")

    log("kb_loader", f"Loaded {len(articles)} KB articles from {kb_folder}")
    return articles


def embed_kb_articles(articles: List[Dict]) -> List[Dict]:
    """
    Build TF-IDF embeddings for KB articles.
    Stores compressed (50-dim) vector as JSON for similarity comparison.
    """
    if not articles:
        return articles

    texts = [a["clean_text"] for a in articles]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    for i, article in enumerate(articles):
        row = X[i].toarray()[0]
        article["embedding_json"] = json.dumps(row[:50].tolist())
        article["full_vector"] = X[i]  # keep sparse for similarity (not stored to DB)

    # Store vectorizer vocabulary for later cluster-KB matching
    article_metadata = {
        "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        "n_features": X.shape[1],
    }
    # Save vocab so cluster embeddings can be aligned
    for a in articles:
        a["_vocab"] = article_metadata["vocabulary"]

    return articles


def store_kb_articles(articles: List[Dict]) -> None:
    """Upsert KB articles into DB."""
    conn = get_connection()
    conn.execute("DELETE FROM kb_articles")
    for a in articles:
        conn.execute("""
            INSERT INTO kb_articles (filename, title, content, embedding_json)
            VALUES (?, ?, ?, ?)
        """, (a["filename"], a["title"], a["content"], a.get("embedding_json", "")))
    conn.commit()
    conn.close()
    log("kb_loader", f"Stored {len(articles)} KB articles in DB")


def get_kb_articles() -> List[Dict]:
    """Fetch all KB articles from DB."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM kb_articles").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_kb_loader(kb_folder: str) -> List[Dict]:
    """Full Step 3: load -> embed -> store."""
    articles = load_kb_articles(kb_folder)
    articles = embed_kb_articles(articles)
    store_kb_articles(articles)
    log("kb_loader", "KB Loader complete")
    return articles
