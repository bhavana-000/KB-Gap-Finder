KB Gap Finder
AI-Powered Knowledge Base Gap Detection System
KB Gap Finder is an AI-driven solution that analyzes historical support tickets, identifies recurring issues that are not covered by existing Knowledge Base (KB) articles, and automatically generates draft KB articles to improve organizational knowledge management.
The system helps support teams reduce repetitive ticket handling by proactively identifying documentation gaps and recommending new KB content
Problem Statement
Organizations often receive recurring support requests for issues that are not documented in their Knowledge Base. This leads to:
•	Increased support workload
•	Repeated issue resolution efforts
•	Inconsistent responses from support teams
•	Poor self-service experience for users
KB Gap Finder addresses this challenge by automatically discovering undocumented recurring issues and generating draft KB articles.
Key Features
Ticket Analysis
•	Reads historical support tickets from CSV files.
•	Extracts issue descriptions and recurring patterns.
Topic Clustering
•	Groups semantically similar tickets into clusters.
•	Identifies frequently occurring support issues.
Knowledge Gap Detection
•	Compares ticket clusters against existing KB articles.
•	Detects topics with insufficient or missing documentation.
AI-Powered KB Generation
•	Generates draft KB articles for identified gaps.
•	Creates structured content including:
o	Title
o	Problem Statement
o	Root Cause
o	Resolution Steps
o	FAQs
Explainable Recommendations
•	Provides reasoning for every suggested KB article.
•	Displays similarity scores and supporting ticket evidence.
System Architecture
+--------------------+
| Support Tickets    |
| (CSV Dataset)      |
+---------+----------+
          |
          v
+--------------------+
| Ticket Analyzer    |
+---------+----------+
          |
          v
+--------------------+
| Topic Clustering   |
| (TF-IDF + KMeans)  |
+---------+----------+
          |
          v
+--------------------+
| KB Repository      |
| (Markdown Files)   |
+---------+----------+
          |
          v
+--------------------+
| Gap Detection      |
| Similarity Search  |
+---------+----------+
          |
          v
+--------------------+
| KB Article         |
| Generator          |
+---------+----------+
          |
          v
+--------------------+
| Streamlit Dashboard|
+--------------------+
AI Agent Workflow
The project demonstrates an Agent-Based AI workflow:
Step 1 – Observe
Read support tickets and existing KB articles.
Step 2 – Analyze
Extract important issue information from tickets.
Step 3 – Cluster
Group similar issues using machine learning.
Step 4 – Compare
Match ticket clusters with KB articles.
Step 5 – Detect
Identify knowledge gaps.
Step 6 – Generate
Create draft KB articles for uncovered topics.
Step 7 – Report
Present recommendations through the dashboard.
Technology Stack
Category	Technology
Frontend	Streamlit
Backend	Python
Database	SQLite
Data Processing	Pandas
Machine Learning	Scikit-Learn
Clustering	TF-IDF + KMeans
Similarity Search	Cosine Similarity
AI Assistance	OpenAI Codex, VS Code AI Completion
Version Control	GitHub

Project Structure
kb-gap-finder/
│
├── app.py
├── requirements.txt
│
├── src/
│   ├── agent.py
│   ├── ticket_analyzer.py
│   ├── categorizer.py
│   ├── kb_loader.py
│   ├── gap_detector.py
│   ├── article_generator.py
│   └── database.py
│
├── sample_data/
│   ├── tickets/
│   └── kb_articles/
│
├── tests/
│
└── docs/
    ├── README.md
    └── AI_Usage_Note.md
Installation
Prerequisites
•	Python 3.10+
•	Git
Clone Repository
git clone <repository-url>
cd kb-gap-finder
Install Dependencies
pip install -r requirements.txt
Running the Application
Launch the Streamlit dashboard:
streamlit run app.py
Open:
http://localhost:8501
Input Requirements
Support Tickets
CSV file containing:
ticket_id
subject
description
category
status
resolution
date
Knowledge Base Articles
Markdown (.md) files stored inside the KB repository folder.
Sample Output
The system generates:
•	Ticket Clusters
•	Similarity Scores
•	Knowledge Gap Report
•	Recommended KB Articles
•	Draft KB Content
AI Capability Demonstration
This project satisfies the challenge requirement through:
✅ AI-Assisted Development
✅ Agent Loop Implementation
✅ Knowledge Gap Detection
✅ Automated KB Draft Generation
✅ End-to-End Working Prototype
Assumptions
•	Historical ticket data is available in CSV format.
•	Existing KB articles are stored as Markdown files.
•	English-language support tickets are used.
•	Human review is required before publishing generated KB articles.
Limitations
•	Clustering quality depends on ticket data quality.
•	Generated KB articles may require refinement.
•	Similarity thresholds may require tuning.
•	Small datasets may reduce recommendation accuracy.
Future Enhancements
•	Real-Time Ticket Monitoring
•	Jira Integration
•	ServiceNow Integration
•	Multi-Language Support
•	Knowledge Graph Generation
•	Feedback-Based Learning
•	Automatic KB Publishing
•	MCP Tool Integration
