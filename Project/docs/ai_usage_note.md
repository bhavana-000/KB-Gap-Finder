# AI Usage Note

**Project:** KB Gap Finder for ERP
**Date:** June 2025

---

## What AI Helped With

**1. Clustering architecture decision**
Prompted Claude to recommend an embedding + clustering approach that works without GPU or paid APIs. It suggested TF-IDF (sparse) + KMeans as the right tradeoff for a 1-2 day hackathon - fast, explainable, runs on any laptop.

**2. Joint vectorizer for RAG gap detection**
AI suggested fitting a single TF-IDF vectorizer on BOTH cluster texts and KB article texts together, so both live in the same embedding space for fair cosine similarity comparison. This was a non-obvious design choice.

**3. Silhouette score for auto-K selection**
Prompted: "How do I pick the number of clusters automatically?" AI suggested silhouette score sweep, which we implemented in `choose_k()`.

**4. Ollama prompt engineering**
AI helped write the KB article generation prompt that forces structured Markdown output with all 6 required sections (Title, Problem Statement, Causes, Resolution, FAQs, Contact).

**5. Streamlit agent loop pattern**
AI suggested using a Python generator (`yield`) for the agent loop so Streamlit can display live step-by-step progress without blocking.

**6. SQLite schema design**
AI generated the full schema with proper foreign keys between tickets, clusters, gaps, and generated_articles tables.

---

## What AI Got Wrong

| Issue | What Happened | Fix |
|---|---|---|
| `:memory:` SQLite in tests | AI suggested `os.environ["KB_DB_PATH"] = ":memory:"` for test isolation - but every `get_connection()` call opens a NEW in-memory DB, so tables are empty on second call | Switched to a temp file DB (`tempfile.gettempdir() / "kb_gap_test.db"`) |
| `sentence-transformers` suggestion | AI initially suggested using sentence-transformers for embeddings (requires PyTorch, slow install, GPU preferred) | Replaced with pure scikit-learn TF-IDF - faster, lighter, Windows-compatible |
| Parallel feed fetching | AI generated `asyncio.gather()` for parallel step execution - incompatible with Streamlit's threading model | Reverted to sequential execution in the generator |
| Markdown emoji in output | AI generated article templates with emoji (problem on Windows cp1252 terminals) | Replaced all emoji with ASCII equivalents |

---

## Best Prompts Used

### 1. Architecture prompt
```
I need to build a KB Gap Finder that reads closed support tickets (CSV),
groups them by topic without GPU or paid APIs, checks if each group has
a matching KB article (folder of .md files), and drafts a new article for
gaps. What is the simplest free Python stack for a 1-2 day hackathon?
```

### 2. RAG alignment prompt
```
I have TF-IDF vectors for ticket clusters and separate TF-IDF vectors for
KB articles, but they're fit on different corpora so cosine similarity is
meaningless. How do I fix this?
```
-> AI: "Fit a single TF-IDF vectorizer on the union of all texts before transforming."

### 3. Ollama prompt iteration
```
The Llama3 output keeps adding preamble like "Sure! Here is your article:"
before the Markdown. How do I suppress this?
```
-> AI: Add "Output ONLY the Markdown article, no preamble." to system prompt.

### 4. Streamlit live progress prompt
```
How do I show live step-by-step progress in Streamlit while a Python
pipeline is running, without threading or async?
```
-> AI: Use a Python generator that yields status dicts; iterate in Streamlit with st.markdown() inside the for loop.

### 5. Test isolation prompt
```
My pytest tests all fail because SQLite :memory: creates a new DB on each
connection. How do I share one in-memory DB across multiple connections
in the same process?
```
-> AI: "You can't with :memory: - use a temp file path instead."

---

*AI tools used: Claude codex  (Anthropic) for architecture, prompts, code review. GitHub Copilot for inline completions.*
