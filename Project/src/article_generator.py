"""
article_generator.py
Step 5: Generate draft KB articles for each detected gap.
Primary: Ollama (Llama3, free, local)
Fallback: structured template (no LLM required)
"""

import json
import re
import requests
from typing import List, Dict, Optional
from src.database import get_connection, log


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_OPTIONS = {
    "temperature": 0.2,
    "num_predict": 280,
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(gap: Dict) -> str:
    """Build an LLM prompt to generate a KB article for a gap."""
    sample_tickets = gap.get("sample_tickets", [])
    ticket_lines = []
    for t in sample_tickets[:3]:
        subj = t.get("subject", "")
        desc = t.get("description", "")
        res = t.get("resolution", "")
        ticket_lines.append(f"- Issue: {subj}\n  Description: {desc}\n  Resolution: {res}")

    tickets_block = "\n".join(ticket_lines) if ticket_lines else "No sample tickets."
    top_terms = ", ".join(gap.get("top_terms", [])[:6])

    return f"""You are a technical writer for a university ERP support knowledge base.

A recurring support issue has been identified that has NO existing KB article.

Topic Summary: {gap.get('topic_summary', '')}
Key Terms: {top_terms}
Number of recurring tickets: {gap.get('ticket_count', 0)}

Sample support tickets for this issue:
{tickets_block}

Write a concise, student-friendly KB (Knowledge Base) article in Markdown format.
The article MUST include these sections:
1. Title (H1, descriptive)
2. Problem Statement - what issue students face
3. Possible Causes - 2-3 bullets
4. Step-by-Step Resolution - 3-5 numbered steps
5. FAQs - 1-2 common follow-up questions with answers
6. Contact / References - who to contact if unresolved

Use simple language. Be specific. Keep it under 250 words.
Output ONLY the Markdown article, no preamble.
"""


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    """Call local Ollama instance."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": OLLAMA_OPTIONS,
                "keep_alive": "10m",
            },
            timeout=120,
        )
        resp.raise_for_status()
        article_text = resp.json().get("response", "").strip()
        log("article_generator", f"Ollama generated article successfully with {model}")
        return article_text
    except requests.exceptions.ConnectionError:
        log("article_generator", "Ollama not running - using template fallback", "WARN")
        return None
    except Exception as e:
        log("article_generator", f"Ollama call failed: {e}", "WARN")
        return None


def call_anthropic(prompt: str) -> Optional[str]:
    """Call Anthropic API (optional, needs ANTHROPIC_API_KEY env var)."""
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    except Exception as e:
        log("article_generator", f"Anthropic call failed: {e}", "WARN")
        return None


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------

def generate_template_article(gap: Dict) -> str:
    """
    Pure-Python fallback KB article when no LLM is available.
    Structures real data from the gap/tickets into Markdown.
    """
    topic = gap.get("label", gap.get("topic_summary", "Unknown Issue"))
    # Extract category from label (format: "Category: terms")
    category = topic.split(":")[0].strip() if ":" in topic else topic
    terms = gap.get("top_terms", [])
    count = gap.get("ticket_count", 0)

    sample_tickets = gap.get("sample_tickets", [])
    # Collect unique resolutions
    resolutions = list({
        t.get("resolution", "").strip()
        for t in sample_tickets
        if t.get("resolution", "").strip()
    })
    # Collect unique subjects as problem examples
    subjects = [t.get("subject", "") for t in sample_tickets[:4] if t.get("subject")]

    # Build resolution steps from actual ticket resolutions
    res_steps = []
    for i, res in enumerate(resolutions[:4], 1):
        if res:
            res_steps.append(f"{i}. {res.capitalize()}")
    if not res_steps:
        res_steps = ["1. Contact IT support with your Student ID", "2. Describe the issue in detail"]

    causes = []
    term_cause_map = {
        "login": "Incorrect credentials or account lock after failed attempts",
        "password": "Expired or forgotten password",
        "session": "Session timeout due to inactivity",
        "attendance": "Sync delay between biometric system and ERP",
        "upload": "File size exceeds portal limit or unsupported format",
        "payment": "Payment gateway timeout or bank processing delay",
        "allot": "Allotment not yet processed in backend system",
        "generat": "Prerequisite steps (fee clearance, registration) not completed",
        "schedul": "Schedule data not yet updated in the portal",
        "form": "Form validation error or incomplete required fields",
    }
    for term in terms:
        for key, cause in term_cause_map.items():
            if key in term and cause not in causes:
                causes.append(cause)
    if not causes:
        causes = ["Data not yet synced in the ERP system", "Required prerequisite not completed"]
    causes = causes[:4]

    lines = [
        f"# {category} - Common Issues and Resolutions",
        "",
        f"> **This article covers {count} recurring support tickets related to {category.lower()}.**",
        "",
        "## Problem Statement",
        "",
        f"Students frequently report issues with **{category.lower()}** in the ERP portal.",
        "Common reported problems include:",
        "",
    ]
    for subj in subjects:
        lines.append(f"- {subj}")
    lines += [
        "",
        "## Possible Causes",
        "",
    ]
    for cause in causes:
        lines.append(f"- {cause}")
    lines += [
        "",
        "## Step-by-Step Resolution",
        "",
    ]
    lines += res_steps
    lines += [
        "",
        "## FAQs",
        "",
        f"**Q: How long does it take for {category.lower()} issues to be resolved?**",
        "A: Most issues are resolved within 1-2 working days after raising a support ticket.",
        "",
        f"**Q: What information should I include in my support ticket for {category.lower()} issues?**",
        "A: Include your Student ID, roll number, the exact error message or screenshot, and the date of the issue.",
        "",
        "**Q: Can I track the status of my request?**",
        "A: Yes. Login to the ERP portal and go to Student Services > My Tickets to track status.",
        "",
        "## Contact / References",
        "",
        "- **IT Helpdesk:** helpdesk@college.edu",
        "- **Phone:** Ext. 1234 (Mon-Fri, 9 AM - 5 PM)",
        "- **Student Services Office:** Block A, Room 101",
        "- **Self-service portal:** erp.college.edu/support",
        "",
        "---",
        f"*Article auto-generated from {count} support tickets | Review before publishing*",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_article(gap: Dict, llm_provider: str = "auto") -> Dict:
    """
    Generate a KB article for a single gap.
    Returns dict with title, content, llm_provider used.
    """
    prompt = build_prompt(gap)
    content = None
    provider_used = "template"

    if llm_provider in ("ollama", "auto"):
        content = call_ollama(prompt)
        if content:
            provider_used = "ollama"

    if content is None and llm_provider in ("anthropic", "auto"):
        content = call_anthropic(prompt)
        if content:
            provider_used = "anthropic"

    if content is None:
        content = generate_template_article(gap)
        provider_used = "template"

    # Extract title from generated content
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else gap.get("label", "KB Article")

    return {
        "gap_id": gap.get("db_id"),
        "cluster_id": gap.get("cluster_id"),
        "title": title,
        "content": content,
        "llm_provider": provider_used,
    }


def store_generated_article(article: Dict) -> int:
    """Persist a generated article to DB. Returns article ID."""
    conn = get_connection()
    cur = conn.execute("""
        INSERT INTO generated_articles (gap_id, cluster_id, title, content, llm_provider)
        VALUES (?, ?, ?, ?, ?)
    """, (article["gap_id"], article["cluster_id"], article["title"],
          article["content"], article["llm_provider"]))
    art_id = cur.lastrowid
    # Mark gap as 'generated'
    if article.get("gap_id"):
        conn.execute("UPDATE gaps SET status='generated' WHERE id=?", (article["gap_id"],))
    conn.commit()
    conn.close()
    return art_id


def run_article_generator(gaps: List[Dict], llm_provider: str = "auto") -> List[Dict]:
    """Full Step 5: generate articles for all gaps, store, return."""
    articles = []
    for gap in gaps:
        log("article_generator", f"Generating article for gap: {gap.get('label', gap.get('topic_summary', ''))[:60]}")
        article = generate_article(gap, llm_provider)
        art_id = store_generated_article(article)
        article["id"] = art_id
        articles.append(article)

    log("article_generator", f"Generated {len(articles)} KB articles")
    return articles
