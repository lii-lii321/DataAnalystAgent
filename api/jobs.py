import asyncio
import time
import uuid

from agent.agent import run_analysis_sync
from api.store import AnalysisRecord, store_analysis

JOBS: dict = {}
_TASKS: set[asyncio.Task] = set()
MAX_JOBS = 200


def _prune_jobs() -> None:
    if len(JOBS) <= MAX_JOBS:
        return
    evictable = sorted(
        ((job_id, job) for job_id, job in JOBS.items() if job.get("status") != "running"),
        key=lambda kv: kv[1].get("created_at", 0.0),
    )
    overflow = len(JOBS) - MAX_JOBS
    for job_id, _ in evictable[:overflow]:
        JOBS.pop(job_id, None)


async def _execute(
    job_id: str,
    dataset_id: str,
    df,
    question: str,
    artifacts_dir: str,
    created_at: float,
) -> None:
    try:
        ws = await asyncio.to_thread(run_analysis_sync, df, question, None, artifacts_dir)
        analysis = AnalysisRecord(
            question=question,
            report=ws.report_md,
            steps=[{"tool": s.tool, "args": s.args, "summary": s.summary, "ok": s.ok} for s in ws.steps],
            charts=list(ws.chart_files),
        )
        analysis_id = store_analysis(dataset_id, analysis)
        JOBS[job_id] = {
            "status": "done",
            "dataset_id": dataset_id,
            "created_at": created_at,
            "analysis_id": analysis_id,
            "result": {
                "question": question,
                "charts": analysis.charts,
                "steps": analysis.steps,
                "report": analysis.report,
            },
        }
    except Exception as exc:
        JOBS[job_id] = {
            "status": "error",
            "dataset_id": dataset_id,
            "created_at": created_at,
            "error": str(exc),
        }


def start_job(dataset_id: str, df, question: str, artifacts_dir: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    created_at = time.time()
    JOBS[job_id] = {
        "status": "running",
        "dataset_id": dataset_id,
        "created_at": created_at,
    }
    _prune_jobs()
    task = asyncio.create_task(_execute(job_id, dataset_id, df.copy(), question, artifacts_dir, created_at))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return job_id


def list_jobs() -> list[tuple[str, dict]]:
    return sorted(JOBS.items(), key=lambda kv: kv[1].get("created_at", 0.0))


def get_job(job_id: str) -> dict | None:
    return JOBS.get(job_id)
