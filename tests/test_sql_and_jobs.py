import asyncio
import sqlite3

import pytest

from api.store import REGISTRY


def _make_db(tmp_path) -> str:
    conn = sqlite3.connect(tmp_path / "data.db")
    conn.execute("CREATE TABLE students (name TEXT, gender TEXT, score REAL)")
    rows = [(f"s{i}", "M" if i % 2 else "F", 60.0 + (i % 5) * 7.0) for i in range(24)]
    conn.executemany("INSERT INTO students VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return f"sqlite:///{(tmp_path / 'data.db').as_posix()}"


@pytest.fixture(autouse=True)
def clean_registry():
    REGISTRY.clear()
    yield
    REGISTRY.clear()


async def test_sql_import_then_analyze(client, tmp_path, monkeypatch):
    from agent.config import settings

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    url = _make_db(tmp_path)

    imported = await client.post("/datasets/sql", json={"url": url, "table": "students"})
    assert imported.status_code == 201
    body = imported.json()
    assert body["profile"]["n_rows"] == 24
    assert {c["name"] for c in body["profile"]["columns"]} == {"name", "gender", "score"}

    analyzed = await client.post(
        f"/datasets/{body['id']}/analyze",
        json={"question": "不同 gender 的 score 是否存在显著差异？"},
    )
    assert analyzed.status_code == 200
    assert "统计检验" in analyzed.json()["report"]


async def test_sql_rejects_non_identifier_table(client, tmp_path):
    url = _make_db(tmp_path)
    for table in ("students; DROP TABLE students", "students DELETE", "1", ""):
        resp = await client.post("/datasets/sql", json={"url": url, "table": table})
        assert resp.status_code == 422, table


async def test_sql_missing_table_400(client, tmp_path):
    url = _make_db(tmp_path)
    resp = await client.post("/datasets/sql", json={"url": url, "table": "no_such_table"})
    assert resp.status_code == 400


async def test_sql_bad_url_400(client):
    resp = await client.post("/datasets/sql", json={"url": "sqlite:///:memory:", "table": "students"})
    assert resp.status_code in (400, 422)


def test_sql_remote_dialect_drivers_available():
    from sqlalchemy import create_engine

    create_engine("mysql+pymysql://user:pw@127.0.0.1:3306/db")
    create_engine("postgresql+psycopg2://user:pw@127.0.0.1:5432/db")


async def test_concurrent_jobs_all_complete(client, tmp_path, monkeypatch):
    from agent.config import settings
    from agent.synth import make_income

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    csv = make_income().to_csv(index=False)
    upload = await client.post("/datasets", files={"file": ("income.csv", csv.encode(), "text/csv")})
    dataset_id = upload.json()["id"]

    started = []
    for _ in range(3):
        resp = await client.post(
            f"/datasets/{dataset_id}/analyze/async",
            json={"question": "男性和女性的 income 是否存在显著差异？"},
        )
        assert resp.status_code == 202
        started.append(resp.json()["job_id"])
    assert len(set(started)) == 3

    statuses = {}
    for _ in range(200):
        statuses = {}
        for job_id in started:
            resp = await client.get(f"/jobs/{job_id}")
            statuses[job_id] = resp.json()
        if all(payload["status"] != "running" for payload in statuses.values()):
            break
        await asyncio.sleep(0.05)

    assert all(statuses[j]["status"] == "done" for j in started), statuses
    assert len({statuses[j]["analysis_id"] for j in started}) == 3
