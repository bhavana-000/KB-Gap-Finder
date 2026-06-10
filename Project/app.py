"""
app.py
KB Gap Finder - Streamlit Dashboard
Run: streamlit run app.py
"""

import sys
import os
import json
import csv
import io
import time
from pathlib import Path
import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.database import init_db, get_connection, get_logs, clear_run_data
from src.agent import run_agent

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="KB Gap Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    border-left: 4px solid #4CAF50;
    margin-bottom: 0.5rem;
}
.gap-card {
    background: #fff8f0;
    border-radius: 8px;
    padding: 1rem;
    border-left: 4px solid #FF9800;
    margin-bottom: 0.75rem;
}
.gap-card.HIGH { border-left-color: #f44336; }
.gap-card.MEDIUM { border-left-color: #FF9800; }
.gap-card.LOW { border-left-color: #2196F3; }
.step-done { color: #4CAF50; font-weight: 600; }
.step-active { color: #2196F3; font-weight: 600; }
.badge-high { background:#fdecea; color:#c62828; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; }
.badge-medium { background:#fff3e0; color:#e65100; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; }
.badge-low { background:#e3f2fd; color:#1565c0; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────

init_db()

if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "result" not in st.session_state:
    st.session_state.result = None
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar - Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("KB Gap Finder")
    st.caption("AI-Powered Knowledge Base Gap Detection")
    st.divider()

    st.subheader("1. Upload Tickets")
    uploaded_csv = st.file_uploader(
        "Support Tickets CSV", type=["csv"],
        help="CSV with columns: ticket_id, date, category, subject, description, status, resolution"
    )
    use_sample = st.checkbox("Use sample tickets", value=True)

    st.subheader("2. KB Articles Folder")
    kb_folder_input = st.text_input(
        "KB Articles folder path",
        value="sample_data/kb_articles",
        help="Folder containing .md KB articles"
    )

    st.subheader("3. Settings")
    n_clusters = st.slider("Number of clusters (0 = auto)", 0, 15, 0)
    gap_threshold = st.slider("Gap threshold (similarity)", 0.05, 0.60, 0.25, step=0.05,
                               help="Clusters below this similarity score to KB articles are flagged as gaps")
    llm_provider = st.selectbox(
        "LLM for article generation",
        ["auto", "ollama", "anthropic", "template"],
        index=0,
        help="auto = tries Ollama -> Anthropic -> template fallback"
    )

    st.divider()
    run_btn = st.button("Run Agent Pipeline", type="primary", use_container_width=True)
    if st.button("Clear / Reset", use_container_width=True):
        clear_run_data()
        st.session_state.run_complete = False
        st.session_state.result = None
        st.session_state.agent_logs = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

st.title("KB Gap Finder for ERP")
st.caption("Identifies recurring support issues with no Knowledge Base coverage, then drafts articles.")

# ─────────────────────────────────────────────────────────────────────────────
# Agent workflow diagram (static)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("Agent Workflow Steps", expanded=not st.session_state.run_complete):
    cols = st.columns(5)
    steps = [
        ("1", "Ticket Analyzer", "Read CSV tickets"),
        ("2", "Categorize", "Cluster by topic"),
        ("3", "Load KB", "Read MD articles"),
        ("4", "Detect Gaps", "Compare to KB"),
        ("5", "Draft Articles", "Generate KB drafts"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center;padding:12px;background:#f0f7ff;border-radius:8px;border:1px solid #c5d8f0">
                <div style="font-size:22px;font-weight:700;color:#1976d2">{num}</div>
                <div style="font-weight:600;font-size:13px">{title}</div>
                <div style="font-size:11px;color:#666">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    st.session_state.run_complete = False
    st.session_state.result = None
    st.session_state.agent_logs = []

    # Resolve ticket CSV path
    if uploaded_csv:
        tmp_path = Path("output/_uploaded_tickets.csv")
        tmp_path.parent.mkdir(exist_ok=True)
        tmp_path.write_bytes(uploaded_csv.getvalue())
        ticket_csv = str(tmp_path)
    elif use_sample:
        ticket_csv = "sample_data/tickets/support_tickets.csv"
    else:
        st.error("Please upload a CSV file or check 'Use sample tickets'")
        st.stop()

    # Validate KB folder
    if not Path(kb_folder_input).exists():
        st.warning(f"KB folder '{kb_folder_input}' not found - will treat as empty KB (all clusters = gaps)")

    st.subheader("Agent Running...")
    step_icons = {
        "init": "🔧", "ticket_analyzer": "📋", "categorizer": "🔵",
        "kb_loader": "📚", "gap_detector": "🔍", "article_generator": "✍️", "completed": "✅"
    }

    log_container = st.container()
    progress_bar = st.progress(0)
    step_weights = {"init": 5, "ticket_analyzer": 15, "categorizer": 25,
                    "kb_loader": 35, "gap_detector": 55, "article_generator": 85, "completed": 100}

    with log_container:
        for event in run_agent(
            ticket_csv=ticket_csv,
            kb_folder=kb_folder_input,
            n_clusters=n_clusters if n_clusters > 0 else None,
            gap_threshold=gap_threshold,
            llm_provider=llm_provider,
            fresh_run=True,
        ):
            step = event["step"]
            msg = event["message"]
            icon = step_icons.get(step, "•")
            st.markdown(f"{icon} **{msg}**")
            st.session_state.agent_logs.append(f"{icon} {msg}")
            progress_value = step_weights.get(step, 0)
            event_data = event.get("data") or {}
            if step == "article_generator" and event_data.get("article_count"):
                progress_value = min(95, 85 + event_data["article_count"])
            progress_bar.progress(progress_value)

            if step == "completed" and event.get("data", {}).get("result"):
                st.session_state.result = event["data"]["result"]
                st.session_state.run_complete = True

    if st.session_state.run_complete:
        progress_bar.progress(100)
        st.success("Pipeline complete! See results below.")
        time.sleep(0.3)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.run_complete and st.session_state.result:
    result = st.session_state.result
    st.divider()

    # ── KPIs ────────────────────────────────────────────────────────────────
    st.subheader("Dashboard Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tickets Analyzed", result["ticket_count"])
    k2.metric("Topic Clusters", result["cluster_count"])
    k3.metric("KB Gaps Found", result["gap_count"],
              delta=f"{result['gap_count']} need articles", delta_color="inverse")
    k4.metric("Articles Generated", result["article_count"])

    st.divider()

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "KB Gaps Detected",
        "Ticket Clusters",
        "Generated Articles",
        "Download",
        "Execution Logs",
    ])

    # ── Tab 1: Gaps ──────────────────────────────────────────────────────────
    with tab1:
        st.subheader(f"KB Gaps Detected ({result['gap_count']})")
        if not result["gaps"]:
            st.success("No gaps found! All clusters have KB coverage.")
        else:
            priority_filter = st.multiselect(
                "Filter by priority", ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"]
            )
            for gap in result["gaps"]:
                if gap["priority"] not in priority_filter:
                    continue
                badge = f'<span class="badge-{gap["priority"].lower()}">{gap["priority"]}</span>'
                with st.expander(
                    f"[{gap['priority']}] {gap.get('label', gap['topic_summary'])[:70]} — {gap['ticket_count']} tickets",
                    expanded=gap["priority"] == "HIGH"
                ):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Topic:** {gap['topic_summary'][:200]}")
                        st.markdown(f"**Key terms:** `{'`, `'.join(gap.get('top_terms', [])[:6])}`")
                        st.markdown(f"**Similarity to nearest KB article:** {gap['similarity_score']:.2%}")
                    with c2:
                        st.metric("Tickets", gap["ticket_count"])
                        st.markdown(f"Priority: **{gap['priority']}**")

                    if gap.get("sample_tickets"):
                        st.markdown("**Sample tickets:**")
                        rows = []
                        for t in gap["sample_tickets"][:5]:
                            rows.append({
                                "Subject": t.get("subject", ""),
                                "Description": t.get("description", "")[:80] + "...",
                                "Resolution": t.get("resolution", ""),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Tab 2: Clusters ──────────────────────────────────────────────────────
    with tab2:
        st.subheader(f"Ticket Clusters ({result['cluster_count']})")
        cluster_rows = []
        for c in result["clusters"]:
            cluster_rows.append({
                "Cluster": c.get("label", f"Cluster {c['id']}")[:50],
                "Tickets": c["ticket_count"],
                "Has KB?": "Yes" if c.get("has_kb_match") else "No",
                "KB Similarity": f"{c.get('similarity_score', 0):.2%}",
                "Nearest KB": c.get("best_kb_title", "—") or "—",
            })
        df = pd.DataFrame(cluster_rows)
        st.dataframe(
            df.style.apply(
                lambda row: ["background-color: #fdecea" if row["Has KB?"] == "No" else "" for _ in row],
                axis=1
            ),
            use_container_width=True, hide_index=True
        )

    # ── Tab 3: Generated Articles ─────────────────────────────────────────────
    with tab3:
        st.subheader(f"Generated KB Articles ({result['article_count']})")
        if not result["articles"]:
            st.info("No articles generated (no gaps found).")
        else:
            for art in result["articles"]:
                with st.expander(f"📄 {art['title']}", expanded=False):
                    st.caption(f"LLM: {art['llm_provider']} | Gap ID: {art.get('gap_id')}")
                    st.markdown(art["content"])

    # ── Tab 4: Downloads ──────────────────────────────────────────────────────
    with tab4:
        st.subheader("Download Outputs")

        # Gap report CSV
        if result["gaps"]:
            gap_rows = [{
                "Cluster": g.get("label", ""),
                "Topic": g.get("topic_summary", "")[:100],
                "Ticket Count": g["ticket_count"],
                "Priority": g["priority"],
                "Similarity Score": g["similarity_score"],
                "Top Terms": ", ".join(g.get("top_terms", [])[:5]),
            } for g in result["gaps"]]
            gap_csv_buf = io.StringIO()
            pd.DataFrame(gap_rows).to_csv(gap_csv_buf, index=False)
            st.download_button(
                "Download Gap Report (.csv)",
                gap_csv_buf.getvalue(),
                file_name="kb_gap_report.csv",
                mime="text/csv",
            )

        # Generated articles as individual .txt downloads
        if result["articles"]:
            for art in result["articles"]:
                safe_title = art["title"].replace(" ", "_").replace("/", "-")[:40]
                st.download_button(
                    f"Download: {art['title'][:50]} (.md)",
                    art["content"],
                    file_name=f"{safe_title}.md",
                    mime="text/markdown",
                    key=f"dl_{art.get('id', safe_title)}",
                )

    # ── Tab 5: Logs ───────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Execution Logs")
        logs = get_logs(limit=200)
        if logs:
            log_df = pd.DataFrame(logs)[["created_at", "step", "level", "message"]]
            log_df.columns = ["Time", "Step", "Level", "Message"]
            st.dataframe(log_df, use_container_width=True, hide_index=True, height=400)
        else:
            for line in st.session_state.agent_logs:
                st.text(line)

# ─────────────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.run_complete:
    st.info("Configure settings in the sidebar and click **Run Agent Pipeline** to start.")
    st.markdown("""
    **What this tool does:**
    1. Reads your closed support ticket CSV
    2. Clusters tickets by topic using TF-IDF + KMeans embeddings
    3. Loads your existing KB articles (Markdown files)
    4. Detects which topics have NO matching KB article
    5. Drafts a new KB article for each gap using Ollama (Llama3) or a structured template

    **Tech Stack:** Python + Streamlit + SQLite + scikit-learn + Ollama (free)
    """)
