import asyncio
import uuid

from agent.agent import run_analysis_sync
from api.store import AnalysisRecord, store_analysis

JOBS: dict = {}


async def _execute(job_id: str, dataset_id: str, df, question: str, artifacts_dir: str) -> None:
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
            "analysis_id": analysis_id,
            "result": {
                "question": question,
                "charts": analysis.charts,
                "steps": analysis.steps,
                "report": analysis.report,
            },
        }
    except Exception as exc:
        JOBS[job_id] = {"status": "error", "error": str(exc)}


def start_job(dataset_id: str, df, question: str, artifacts_dir: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running"}
    asyncio.create_task(_execute(job_id, dataset_id, df.copy(), question, artifacts_dir))
    return job_id


def get_job(job_id: str) -> dict | None:
    return JOBS.get(job_id)
