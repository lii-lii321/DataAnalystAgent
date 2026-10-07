import asyncio
import io
import re
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import docx_export
from agent.agent import run_analysis_sync
from agent.config import settings
from agent.profiler import profile_dataframe
from agent.sql_source import load_table, validate_table
from api.jobs import get_job, start_job
from api.store import (
    AnalysisRecord,
    get_analysis,
    get_dataset,
    list_analyses,
    list_datasets,
    store_analysis,
    store_dataset,
)

router = APIRouter()

CHART_FILENAME_RE = re.compile(r"^chart_\d+_[\w\-]+\.png$")


class AnalyzeRequest(BaseModel):
    question: str


class SqlSourceRequest(BaseModel):
    url: str
    table: str
    limit: int = 10000


def _profile_payload(profile) -> dict:
    return {
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "dup_rows": profile.dup_rows,
        "missing_cells_pct": profile.missing_cells_pct,
        "columns": [
            {
                "name": c.name,
                "role": c.role,
                "missing": c.missing,
                "n_unique": c.n_unique,
                "outliers": c.outliers,
            }
            for c in profile.columns
        ],
    }


@router.post("/datasets", status_code=201)
async def upload_dataset(file: UploadFile = File(...)):
    limit_bytes = int(settings.max_upload_mb * 1024 * 1024)
    raw = await file.read(limit_bytes + 1)
    if len(raw) > limit_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file too large (max {settings.max_upload_mb:g} MB)",
        )
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"cannot parse CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=422, detail="empty CSV")
    record = store_dataset(file.filename or "upload.csv", df, profile_dataframe(df))
    return {"id": record.id, "filename": record.filename, "profile": _profile_payload(record.profile)}


@router.post("/datasets/sql", status_code=201)
async def import_sql_dataset(payload: SqlSourceRequest):
    try:
        validate_table(payload.table)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        df = load_table(payload.url, payload.table, payload.limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"cannot read table: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=422, detail="table is empty")
    record = store_dataset(f"{payload.table} (SQL)", df, profile_dataframe(df))
    return {"id": record.id, "filename": record.filename, "profile": _profile_payload(record.profile)}


@router.get("/datasets")
async def list_all_datasets():
    return [
        {
            "id": record.id,
            "filename": record.filename,
            "n_rows": record.profile.n_rows,
            "n_cols": record.profile.n_cols,
        }
        for record in list_datasets()
    ]


@router.get("/datasets/{dataset_id}")
async def get_dataset_profile(dataset_id: str):
    record = get_dataset(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return {"id": record.id, "filename": record.filename, "profile": _profile_payload(record.profile)}


@router.get("/datasets/{dataset_id}/analyses")
async def list_dataset_analyses(dataset_id: str):
    items = list_analyses(dataset_id)
    if items is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return [
        {
            "analysis_id": analysis_id,
            "question": analysis.question,
            "n_steps": len(analysis.steps),
            "n_charts": len(analysis.charts),
            "charts": list(analysis.charts),
            "created_at": analysis.created_at,
        }
        for analysis_id, analysis in items
    ]


@router.get("/datasets/{dataset_id}/charts/{filename}")
async def download_chart(dataset_id: str, filename: str):
    if not CHART_FILENAME_RE.fullmatch(filename):
        raise HTTPException(status_code=404, detail="chart not found")
    if Path(dataset_id).name != dataset_id:
        raise HTTPException(status_code=404, detail="chart not found")
    base_dir = (Path(settings.artifacts_dir) / dataset_id).resolve()
    chart_path = (base_dir / filename).resolve()
    if base_dir not in chart_path.parents or not chart_path.is_file():
        raise HTTPException(status_code=404, detail="chart not found")
    return FileResponse(chart_path, media_type="image/png", filename=filename)


@router.post("/datasets/{dataset_id}/analyze")
async def analyze_dataset(dataset_id: str, payload: AnalyzeRequest):
    record = get_dataset(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    artifacts_dir = str(Path(settings.artifacts_dir) / dataset_id)
    ws = await asyncio.to_thread(run_analysis_sync, record.df, payload.question, None, artifacts_dir)
    analysis = AnalysisRecord(
        question=payload.question,
        report=ws.report_md,
        steps=[{"tool": s.tool, "args": s.args, "summary": s.summary, "ok": s.ok} for s in ws.steps],
        charts=list(ws.chart_files),
    )
    analysis_id = store_analysis(dataset_id, analysis)
    return {
        "analysis_id": analysis_id,
        "dataset_id": dataset_id,
        "charts": analysis.charts,
        "steps": analysis.steps,
        "report": analysis.report,
    }


@router.post("/datasets/{dataset_id}/analyze/async", status_code=202)
async def analyze_dataset_async(dataset_id: str, payload: AnalyzeRequest):
    record = get_dataset(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    artifacts_dir = str(Path(settings.artifacts_dir) / dataset_id)
    job_id = start_job(dataset_id, record.df, payload.question, artifacts_dir)
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, **job}


@router.get("/datasets/{dataset_id}/analyses/{analysis_id}/report.docx")
async def export_report_docx(dataset_id: str, analysis_id: str):
    analysis = get_analysis(dataset_id, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    out_dir = Path(settings.artifacts_dir) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_{analysis_id}.docx"
    docx_export.markdown_to_docx(analysis.report, out_path, image_dir=out_dir)
    return FileResponse(
        out_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"report_{analysis_id}.docx",
    )


@router.get("/datasets/{dataset_id}/analyses/{analysis_id}")
async def get_analysis_result(dataset_id: str, analysis_id: str):
    analysis = get_analysis(dataset_id, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return {
        "analysis_id": analysis_id,
        "question": analysis.question,
        "charts": analysis.charts,
        "steps": analysis.steps,
        "report": analysis.report,
    }
