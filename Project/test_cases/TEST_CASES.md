# Test Cases

## Test Case Table

| Test Case ID | Input Scenario | Expected Category | Expected Priority | Result |
| --- | --- | --- | --- | --- |
| TC001 | Upload valid `tickets.csv` file | CSV Processing | P1 | Pass |
| TC002 | Upload CSV with missing required columns | Error Handling | P1 | Pass |
| TC003 | Upload empty CSV file | Error Handling | P1 | Pass |
| TC004 | Load existing KB Markdown files from `kb/` folder | KB Loading | P1 | Pass |
| TC005 | Cluster similar login issue tickets | Clustering | P1 | Pass |
| TC006 | Cluster similar fee payment tickets | Clustering | P1 | Pass |
| TC007 | Match password reset cluster with `password-reset.md` | KB Matching | P1 | Pass |
| TC008 | Match fee payment cluster with `fee-payment.md` | KB Matching | P1 | Pass |
| TC009 | Detect hostel issue cluster as a KB gap | Gap Detection | P2 | Pass |
| TC010 | Detect library issue cluster as a KB gap | Gap Detection | P2 | Pass |
| TC011 | Generate draft article for a gap cluster | Article Generation | P1 | Pass |
| TC012 | Save generated article in `output/` folder | File Output | P1 | Pass |
| TC013 | Display all ticket clusters in Streamlit dashboard | Dashboard | P1 | Pass |
| TC014 | Show green checkmark for clusters with KB articles | Dashboard | P2 | Pass |
| TC015 | Show red warning for gap clusters | Dashboard | P2 | Pass |
| TC016 | Generate KB article from dashboard button | Dashboard | P1 | Pass |
| TC017 | Ollama server not running | Error Handling | P1 | Pass |
| TC018 | Sentence transformer model loading | Model Loading | P2 | Pass |
| TC019 | Run pytest test suite | Testing | P1 | Pass |
| TC020 | Run project from terminal script | CLI Execution | P2 | Pass |

## Functional Testing

### Ticket Processing

- Verify `data/tickets.csv` is loaded correctly.
- Verify required columns are present: `ticket_id`, `title`, `description`, `category`, and `status`.
- Verify ticket descriptions are combined with title and category for processing.
- Verify empty ticket files are handled properly.
- Verify invalid or missing CSV columns show a clear error.

### Clustering

- Verify similar support tickets are grouped into clusters.
- Verify KMeans returns the expected number of clusters.
- Verify each ticket receives one cluster label.
- Verify clusters are created for categories such as Login Issues, Fee Payment, Timetable, Exam Results, Library, Hostel, and Attendance.

### Knowledge Base Matching

- Verify Markdown files are loaded from the `kb/` folder.
- Verify each ticket cluster is compared with existing KB articles.
- Verify cosine similarity score is calculated.
- Verify similarity score is between `0` and `1`.
- Verify clusters with similarity greater than or equal to `0.30` are marked as covered.

### Gap Detection

- Verify clusters with similarity below `0.30` are marked as gaps.
- Verify gap clusters are shown clearly in the terminal report.
- Verify gap clusters are shown clearly in the Streamlit dashboard.
- Verify the dashboard displays a red warning for gap clusters.

### Article Generation

- Verify `draft_kb_article` accepts ticket descriptions as input.
- Verify the function calls the local Ollama API.
- Verify the model used is `llama3.2`.
- Verify generated Markdown articles are saved in the `output/` folder.
- Verify output files follow the naming pattern `gap-article-1.md`, `gap-article-2.md`, and so on.

## Dashboard Testing

- Verify the Streamlit dashboard loads successfully.
- Verify the title `KB Gap Finder - College ERP` is displayed.
- Verify the file uploader accepts `tickets.csv`.
- Verify the `Find Gaps` button starts analysis.
- Verify all ticket clusters are displayed in a table.
- Verify green checkmark appears when a KB article exists.
- Verify red warning appears when a KB article gap exists.
- Verify `Generate KB Article` button appears for each gap.
- Verify generated article content is displayed on screen.

## Performance Testing

- Process 30 sample tickets without application crash.
- Load the dashboard within a reasonable time.
- Generate clusters within a reasonable time after model loading.
- Generate one draft KB article using Ollama without application crash.

## Error Handling Testing

- Missing `ticket_id` column.
- Missing `title` column.
- Missing `description` column.
- Missing `category` column.
- Missing `status` column.
- Empty uploaded CSV file.
- No Markdown files in `kb/` folder.
- Ollama server not running.
- Ollama model `llama3.2` not available.
- Output folder missing before article generation.

## Pytest Test Coverage

- Verify `test_csv_loads_correctly` checks CSV loading and required columns.
- Verify `test_clustering_works` checks clustering output.
- Verify `test_kb_matching` checks similarity score range.
- Verify `test_gap_detection` checks similarity threshold behavior.
- Verify `test_output_file_created` checks Markdown draft file creation.

## Priority Legend

| Priority | Meaning |
| --- | --- |
| P1 | Critical functionality required for the project to work |
| P2 | Important functionality required for a good user experience |
| P3 | Useful functionality but not blocking |
| P4 | Optional improvement or enhancement |
