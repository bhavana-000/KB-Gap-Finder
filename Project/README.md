# KB Gap Finder for ERP

> AI-Powered Knowledge Base Gap Detection and Article Generation System

Reads closed support tickets (CSV), clusters them by topic using TF-IDF + KMeans embeddings, detects which topics have no matching KB article, and drafts new KB entries using Ollama (Llama3) or a structured template fallback.

---

## Architecture Overview

```
INPUT LAYER          PROCESSING LAYER (AI Agent Workflow)           OUTPUT LAYER
-----------    -----------------------------------------------    ---------------
                1. Ticket Analyzer  -> Read + store tickets
Tickets CSV ->  2. Categorizer      -> TF-IDF + KMeans cluster  -> Dashboard
                3. KB Loader        -> Parse .md articles             (Streamlit)
KB Articles ->  4. Gap Detector     -> Cosine similarity RAG check
                5. Article Generator-> Ollama/Llama3 or template

                DATA STORAGE: SQLite (kb_gap_finder.db)
```

### Components

| File | Role |
|---|---|
| `app.py` | Streamlit dashboard (UI) |
| `src/agent.py` | Agent loop orchestrator |
| `src/ticket_analyzer.py` | Step 1 - CSV reader, DB storage |
| `src/categorizer.py` | Step 2 - TF-IDF + KMeans clustering |
| `src/kb_loader.py` | Step 3 - Markdown KB article loader |
| `src/gap_detector.py` | Step 4 - Cosine similarity gap detection |
| `src/article_generator.py` | Step 5 - Ollama/template KB draft writer |
| `src/database.py` | SQLite schema + helpers |

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.ai) with Llama3 for LLM-powered article drafts

### Install
```bash
git clone https://github.com/<your-team>/kb-gap-finder
cd kb-gap-finder
pip install -r requirements.txt
```

### (Optional) Setup Ollama
```bash
# Install from https://ollama.ai
ollama pull llama3
ollama serve
```

---

## Run Instructions

### Streamlit Dashboard (recommended)
```bash
streamlit run app.py
```
Open http://localhost:8501

### CLI / headless run
```python
from src.agent import run_agent

for event in run_agent(
    ticket_csv="sample_data/tickets/support_tickets.csv",
    kb_folder="sample_data/kb_articles",
    llm_provider="auto",   # auto -> ollama -> anthropic -> template
):
    print(event["step"], event["message"])
```

### Run tests
```bash
pytest tests/ -v
```

---

## Ticket CSV Format

```
ticket_id,date,category,subject,description,status,resolution
T001,2024-01-10,ERP Login,Cannot login,Student unable to login...,Closed,Password reset
```

Required columns: `ticket_id`, `date`, `category`, `subject`, `description`, `status`, `resolution`

---

## KB Articles Format

Plain Markdown files in any folder. Each `.md` file = one KB article. The first `# Heading` becomes the article title.

---

## AI / Agent Capability

- **Clustering (RAG retrieval):** TF-IDF vectorization + KMeans finds natural topic groups in tickets
- **Gap detection (RAG check):** Joint TF-IDF space, cosine similarity compares cluster centroids to KB article vectors
- **Article generation:** Ollama (Llama3) generates structured Markdown KB articles from ticket context; template fallback requires no LLM

---

## Assumptions & Limitations

| Area | Note |
|---|---|
| Clustering K | Auto-selected via silhouette score; can be overridden in UI |
| Gap threshold | Default 0.25 cosine similarity; tune via slider |
| Ollama | Must be running locally (`ollama serve`) for LLM generation |
| CSV format | Expects specific column names (see above) |
| KB format | Only `.md` files supported in KB folder |
| Language | English text assumed for stop-word removal |

---

## Project Structure

```
kb-gap-finder/
- app.py                              Streamlit dashboard
- requirements.txt
- stack.yaml                          (not used - this project uses SQLite)
- src/
  - agent.py                          Agent loop orchestrator
  - database.py                       SQLite schema
  - ticket_analyzer.py                Step 1
  - categorizer.py                    Step 2
  - kb_loader.py                      Step 3
  - gap_detector.py                   Step 4
  - article_generator.py              Step 5
- tests/
  - test_kb_gap_finder.py             36 pytest cases
- sample_data/
  - tickets/support_tickets.csv       40 sample ERP support tickets
  - kb_articles/                      3 existing KB articles
- docs/
  - ai_usage_note.md
```
