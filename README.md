# Data Analyst Agent

An AI data-analysis agent that turns a CSV + a question into a full analysis
report: profiling → data quality → statistical tests (with correct test
selection) → baseline ML → charts → structured Markdown report.

Not "ChatGPT with a CSV": the agent calls deterministic tools (pandas / scipy /
scikit-learn) and explains **real computed results** — every p-value in the
report comes from an actual test run.

Part of a personal product matrix (TeacherOS → teaching delivery, TutorFlow →
client CRM, Data Analyst Agent → data-science engine).

## What it does

```
CSV upload
   ↓
Profiler          dtype/role inference, missing, duplicates, IQR outliers
   ↓
Quality report    severity-ranked issues + cleaning suggestions
   ↓
Analysis agent    tool-calling loop (mock heuristic pipeline offline,
   ↓               or any OpenAI-compatible LLM choosing tools)
Statistics        normality → t-test / Mann-Whitney / ANOVA / Kruskal-Wallis,
   ↓               chi-square + Cramér's V, Pearson/Spearman + p-values
   ↓
Baseline ML       RandomForest regression/classification + 5-fold CV +
   ↓               feature importances; KMeans clustering with silhouette
   ↓
Report            10-section Markdown report with tables, findings, limitations
```

## Test selection (the differentiator)

| Situation | Method |
| --- | --- |
| 2 groups, both normal | independent t-test (Levene → pooled or Welch) + Cohen's d |
| 2 groups, non-normal/small | Mann-Whitney U |
| 3+ groups, normal | one-way ANOVA |
| 3+ groups, non-normal | Kruskal-Wallis |
| categorical × categorical | chi-square + Cramér's V |
| numeric × numeric | Pearson (normal) / Spearman (non-normal) |

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn main:app --reload --port 8001
```

API (see /docs for interactive):

```bash
curl -F "file=@income.csv" http://127.0.0.1:8001/datasets

# import straight from a database table (read-only, identifier-whitelisted)
curl -X POST http://127.0.0.1:8001/datasets/sql \
     -H "Content-Type: application/json" \
     -d "{\"url\": \"sqlite:///D:/data/demo.db\", \"table\": \"students\"}"

curl -X POST http://127.0.0.1:8001/datasets/<id>/analyze \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"男性和女性的 income 是否存在显著差异？\"}"

# long-running analyses: async job + poll (runs in a worker thread)
curl -X POST http://127.0.0.1:8001/datasets/<id>/analyze/async \
     -H "Content-Type: application/json" -d "{\"question\": \"...\"}"
curl http://127.0.0.1:8001/jobs/<job_id>

# list all jobs (summaries, oldest first; full payload via /jobs/<job_id>)
curl http://127.0.0.1:8001/jobs

# fetch a generated chart (filename as returned in the analyze response)
curl -O -J http://127.0.0.1:8001/datasets/<id>/charts/<chart_file>.png

# list uploaded datasets / list a dataset's analysis history (oldest first)
curl http://127.0.0.1:8001/datasets
curl http://127.0.0.1:8001/datasets/<id>/analyses

# peek at the raw rows (default 20, capped at 500)
curl "http://127.0.0.1:8001/datasets/<id>/data?rows=10"

# release a dataset: drops the in-memory record and its artifacts directory
curl -X DELETE http://127.0.0.1:8001/datasets/<id>

# export a finished report as Word
curl -O -J http://127.0.0.1:8001/datasets/<id>/analyses/<aid>/report.docx

# or grab the canonical Markdown report directly
curl -O -J http://127.0.0.1:8001/datasets/<id>/analyses/<aid>/report.md
```

### API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/datasets` | Upload a CSV, get id + profile back |
| POST | `/datasets/sql` | Import a database table (url/table, `limit` 1..100000) |
| GET | `/datasets` | List uploaded datasets (summaries) |
| GET | `/datasets/{id}` | One dataset's profile |
| GET | `/datasets/{id}/data?rows=N` | Preview raw rows (default 20, capped at 500) |
| DELETE | `/datasets/{id}` | Drop the dataset record and its artifacts directory |
| POST | `/datasets/{id}/analyze` | Run an analysis synchronously (worker thread) |
| POST | `/datasets/{id}/analyze/async` | Start an async analysis job, returns job_id |
| GET | `/jobs` | List job summaries (oldest first) |
| GET | `/jobs/{job_id}` | One job's status / full result |
| GET | `/datasets/{id}/analyses` | Analysis history (oldest first) |
| GET | `/datasets/{id}/analyses/{aid}` | One analysis result (report text included) |
| GET | `/datasets/{id}/analyses/{aid}/report.md` | Download the report as Markdown |
| GET | `/datasets/{id}/analyses/{aid}/report.docx` | Download the report as Word |
| GET | `/datasets/{id}/charts/{filename}` | Download a generated chart PNG |
| GET | `/healthz`, `/` | Liveness probe / app info |

Optional LLM mode (agent picks tools via any OpenAI-compatible API; the
heuristic pipeline remains the fallback):

```bash
copy .env.example .env   # set DAA_LLM_PROVIDER=openai + key
```

## Analysis benchmark

12 synthetic tasks with planted ground truth (gender pay gap, city effect,
experience correlation, churn signal, planted outliers/duplicates/missing):

```bash
python benchmarks/runner.py            # all 12 tasks
python benchmarks/runner.py --strict   # exit 1 on failure
```

Current score: **12/12 passed** — correct test family and significance on every
planted effect. `agent/synth.py` is deterministic, so results are reproducible.

## Layout

```
agent/
├── profiler.py    role inference (numeric/categorical/datetime/text/identifier)
├── quality.py     issue detection + dedup cleaning
├── stats.py       test selection + effect sizes (scipy)
├── ml.py          auto_model (RF + CV), kmeans_profile (silhouette)
├── viz.py         question → chart-type rules, matplotlib rendering
├── report.py      10-section Markdown report builder
├── tools.py       tool registry + Workspace state
├── agent.py       heuristic pipeline / LLM tool-calling loop
├── synth.py       deterministic benchmark data with planted effects
└── llm.py         mock | OpenAI-compatible provider
api/               FastAPI: upload → profile → analyze → charts → report
benchmarks/        tasks.jsonl + runner
tests/             71 tests
```

## Testing

```bash
.venv\Scripts\python -m pytest -q
```

## Roadmap

- [x] Profiler, quality, stats engine, baseline ML, auto charts, report
- [x] Tool-calling agent (mock + OpenAI-compatible)
- [x] Analysis benchmark with planted ground truth
- [x] Async analysis jobs (`analyze/async` + `/jobs/{id}` polling, worker-thread execution)
- [x] SQL table import (`/datasets/sql`, SQLite/MySQL/Postgres URLs, read-only)
- [x] Report export to Word (`report.docx`, charts embedded, inline Markdown stripped)
- [x] Chart download endpoint (`/datasets/{id}/charts/{filename}`), hardened job lifecycle
- [x] Listing endpoints (`GET /datasets`, `GET /datasets/{id}/analyses` — analysis history, oldest first)
- [x] Dataset deletion (`DELETE /datasets/{id}` — frees registry entry and artifacts directory)
- [x] Data preview (`GET /datasets/{id}/data?rows=N` — raw rows, JSON-safe nulls)
- [x] SQL import `limit` bounded (`1..100000`, bound as a query parameter)
- [x] Job listing (`GET /jobs` — summaries oldest first, jobs tagged with `dataset_id`)
- [x] Raw Markdown report download (`report.md`), analysis detail carries `created_at`
- [x] Job retention never evicts running jobs (prune skips `status=running`)
- [ ] Human-in-the-loop hypothesis refinement
- [ ] PDF export (CJK font packaging)
