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
curl -X POST http://127.0.0.1:8001/datasets/<id>/analyze \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"男性和女性的 income 是否存在显著差异？\"}"

# long-running analyses: async job + poll
curl -X POST http://127.0.0.1:8001/datasets/<id>/analyze/async \
     -H "Content-Type: application/json" -d "{\"question\": \"...\"}"
curl http://127.0.0.1:8001/jobs/<job_id>

# export a finished report as Word
curl -O -J http://127.0.0.1:8001/datasets/<id>/analyses/<aid>/report.docx
```

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
api/               FastAPI: upload → profile → analyze → report
benchmarks/        tasks.jsonl + runner
tests/             33 tests
```

## Testing

```bash
.venv\Scripts\python -m pytest -q
```

## Roadmap

- [x] Profiler, quality, stats engine, baseline ML, auto charts, report
- [x] Tool-calling agent (mock + OpenAI-compatible)
- [x] Analysis benchmark with planted ground truth
- [x] Async analysis jobs (`analyze/async` + `/jobs/{id}` polling)
- [x] Report export to Word (`report.docx`, charts embedded)
- [ ] SQL/database sources, concurrent job workers
- [ ] Human-in-the-loop hypothesis refinement
- [ ] PDF export (CJK font packaging)
