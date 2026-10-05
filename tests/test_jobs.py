import asyncio
import time

import pytest

from api.store import REGISTRY

CSV = (
    "gender,city,income\n"
    "M,tier1,9000\nF,tier2,7000\nM,tier1,12000\n"
    "F,tier3,6500\nM,tier2,9500\nF,tier1,7200\n"
)


@pytest.fixture(autouse=True)
def clean_registry():
    REGISTRY.clear()
    yield
    REGISTRY.clear()


async def test_async_analysis_job_and_docx_export(client, tmp_path, monkeypatch):
    from agent.config import settings

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))

    upload = await client.post("/datasets", files={"file": ("small.csv", CSV.encode(), "text/csv")})
    dataset_id = upload.json()["id"]

    started = await client.post(
        f"/datasets/{dataset_id}/analyze/async",
        json={"question": "男性和女性的 income 是否存在显著差异？"},
    )
    assert started.status_code == 202
    job_id = started.json()["job_id"]

    payload = None
    for _ in range(100):
        resp = await client.get(f"/jobs/{job_id}")
        payload = resp.json()
        if payload["status"] != "running":
            break
        await asyncio.sleep(0.05)
    assert payload["status"] == "done", payload
    assert "统计检验" in payload["result"]["report"]
    assert payload["result"]["steps"]

    analysis_id = payload["analysis_id"]
    docx = await client.get(f"/datasets/{dataset_id}/analyses/{analysis_id}/report.docx")
    assert docx.status_code == 200
    assert docx.content[:2] == b"PK"
    assert len(docx.content) > 1000


async def test_unknown_job_404(client):
    resp = await client.get("/jobs/doesnotexist")
    assert resp.status_code == 404


async def test_job_error_path_recorded(client, tmp_path, monkeypatch):
    from agent.config import settings
    from api import jobs as jobs_mod

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))

    def boom(*args, **kwargs):
        raise RuntimeError("模拟分析失败")

    monkeypatch.setattr(jobs_mod, "run_analysis_sync", boom)

    upload = await client.post("/datasets", files={"file": ("small.csv", CSV.encode(), "text/csv")})
    dataset_id = upload.json()["id"]
    started = await client.post(
        f"/datasets/{dataset_id}/analyze/async",
        json={"question": "男性和女性的 income 是否存在显著差异？"},
    )
    assert started.status_code == 202
    job_id = started.json()["job_id"]

    payload = None
    for _ in range(100):
        resp = await client.get(f"/jobs/{job_id}")
        payload = resp.json()
        if payload["status"] != "running":
            break
        await asyncio.sleep(0.05)
    assert payload["status"] == "error", payload
    assert "模拟分析失败" in payload["error"]
    assert "created_at" in payload
    jobs_mod.JOBS.clear()


async def test_job_retention_prunes_stale_entries(client, tmp_path, monkeypatch):
    from api import jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "MAX_JOBS", 3)
    jobs_mod.JOBS.clear()
    stale_id = "stale00000001"
    jobs_mod.JOBS[stale_id] = {"status": "done", "created_at": time.time() - 100_000}
    for i in range(2):
        jobs_mod.JOBS[f"filler{i:06d}"] = {"status": "done", "created_at": time.time() - i - 1}

    upload = await client.post("/datasets", files={"file": ("small.csv", CSV.encode(), "text/csv")})
    dataset_id = upload.json()["id"]
    started = await client.post(
        f"/datasets/{dataset_id}/analyze/async",
        json={"question": "男性和女性的 income 是否存在显著差异？"},
    )
    assert started.status_code == 202
    job_id = started.json()["job_id"]

    assert stale_id not in jobs_mod.JOBS
    resp = await client.get(f"/jobs/{stale_id}")
    assert resp.status_code == 404
    assert len(jobs_mod.JOBS) <= 3
    assert job_id in jobs_mod.JOBS
    jobs_mod.JOBS.clear()


async def test_docx_export_missing_analysis_404(client):
    resp = await client.get("/datasets/abc/analyses/xyz/report.docx")
    assert resp.status_code == 404
