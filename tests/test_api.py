import pytest

from api.store import REGISTRY
from agent.synth import make_income

CSV_TEXT = "gender,city,income\nM,tier1,9000\nF,tier2,7000\nM,tier1,12000\nF,tier3,6500\n"
CSV_TEXT_NAN = "a,b\n1,x\n2,\n3,y\n"


@pytest.fixture(autouse=True)
def clean_registry():
    REGISTRY.clear()
    yield
    REGISTRY.clear()


async def test_upload_and_profile(client):
    resp = await client.post(
        "/datasets",
        files={"file": ("income.csv", CSV_TEXT.encode(), "text/csv")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["profile"]["n_rows"] == 4
    assert data["profile"]["n_cols"] == 3
    assert {c["name"] for c in data["profile"]["columns"]} == {"gender", "city", "income"}


async def test_upload_bad_csv(client):
    resp = await client.post(
        "/datasets",
        files={"file": ("bad.csv", b"this is not a csv at all \xff\xfe", "text/csv")},
    )
    assert resp.status_code in (422, 500)


async def test_upload_rejects_header_only_csv(client):
    resp = await client.post(
        "/datasets",
        files={"file": ("header.csv", b"gender,city,income\n", "text/csv")},
    )
    assert resp.status_code == 422
    assert "empty CSV" in resp.json()["detail"]


async def _upload_and_analyze_income(client, monkeypatch, tmp_path):
    from agent.config import settings

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    df = make_income()
    csv = df.to_csv(index=False)
    upload = await client.post(
        "/datasets",
        files={"file": ("income.csv", csv.encode(), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["id"]
    analyze = await client.post(
        f"/datasets/{dataset_id}/analyze",
        json={"question": "男性和女性的 income 是否存在显著差异？"},
    )
    assert analyze.status_code == 200
    return dataset_id, analyze.json()


async def test_download_chart_png(client, tmp_path, monkeypatch):
    dataset_id, body = await _upload_and_analyze_income(client, monkeypatch, tmp_path)
    charts = body["charts"]
    assert charts
    resp = await client.get(f"/datasets/{dataset_id}/charts/{charts[0]}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")


async def test_download_chart_rejects_unsafe_filename(client, tmp_path, monkeypatch):
    dataset_id, body = await _upload_and_analyze_income(client, monkeypatch, tmp_path)
    chart = body["charts"][0]
    resp = await client.get(f"/datasets/{dataset_id}/charts/%2e%2e%2f{chart}")
    assert resp.status_code == 404
    resp = await client.get(f"/datasets/{dataset_id}/charts/{chart[:-4]}..png")
    assert resp.status_code == 404
    resp = await client.get(f"/datasets/{dataset_id}/charts/{chart[:-4]}.txt")
    assert resp.status_code == 404


async def test_full_analysis_flow(client, tmp_path, monkeypatch):
    from agent.config import settings

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    df = make_income()
    csv = df.to_csv(index=False)

    upload = await client.post(
        "/datasets",
        files={"file": ("income.csv", csv.encode(), "text/csv")},
    )
    dataset_id = upload.json()["id"]

    analyze = await client.post(
        f"/datasets/{dataset_id}/analyze",
        json={"question": "男性和女性的 income 是否存在显著差异？"},
    )
    assert analyze.status_code == 200
    body = analyze.json()
    assert body["report"]
    assert "统计检验" in body["report"]

    fetched = await client.get(f"/datasets/{dataset_id}/analyses/{body['analysis_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["question"].startswith("男性和女性")
    assert fetched.json()["created_at"] > 0


async def test_analyze_missing_dataset(client):
    resp = await client.post("/datasets/doesnotexist/analyze", json={"question": "test"})
    assert resp.status_code == 404


async def test_analyze_rejects_blank_question(client):
    upload = await client.post(
        "/datasets",
        files={"file": ("income.csv", CSV_TEXT.encode(), "text/csv")},
    )
    dataset_id = upload.json()["id"]
    for question in ("", "   \n\t "):
        resp = await client.post(f"/datasets/{dataset_id}/analyze", json={"question": question})
        assert resp.status_code == 422, repr(question)
    resp = await client.post(f"/datasets/{dataset_id}/analyze/async", json={"question": ""})
    assert resp.status_code == 422


async def test_analyze_caps_question_length(client, tmp_path, monkeypatch):
    from agent.config import settings

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    csv = make_income().to_csv(index=False)
    upload = await client.post(
        "/datasets",
        files={"file": ("income.csv", csv.encode(), "text/csv")},
    )
    dataset_id = upload.json()["id"]

    resp = await client.post(f"/datasets/{dataset_id}/analyze", json={"question": "问" * 2001})
    assert resp.status_code == 422

    resp = await client.post(f"/datasets/{dataset_id}/analyze", json={"question": "问" * 2000})
    assert resp.status_code == 200
    assert resp.json()["question"] == "问" * 2000

    resp = await client.post(f"/datasets/{dataset_id}/analyze", json={"question": "  income 差异如何？  "})
    assert resp.status_code == 200
    assert resp.json()["question"] == "income 差异如何？"


async def test_export_report_markdown(client, tmp_path, monkeypatch):
    dataset_id, body = await _upload_and_analyze_income(client, monkeypatch, tmp_path)

    resp = await client.get(f"/datasets/{dataset_id}/analyses/{body['analysis_id']}/report.md")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "attachment" in resp.headers["content-disposition"]
    assert f"report_{body['analysis_id']}.md" in resp.headers["content-disposition"]
    assert resp.text == body["report"]
    assert "统计检验" in resp.text


async def test_export_report_markdown_missing_404(client):
    resp = await client.get("/datasets/abc/analyses/xyz/report.md")
    assert resp.status_code == 404


async def test_list_datasets(client):
    resp = await client.get("/datasets")
    assert resp.status_code == 200
    assert resp.json() == []

    for name in ("one.csv", "two.csv"):
        upload = await client.post(
            "/datasets",
            files={"file": (name, CSV_TEXT.encode(), "text/csv")},
        )
        assert upload.status_code == 201

    body = (await client.get("/datasets")).json()
    assert [item["filename"] for item in body] == ["one.csv", "two.csv"]
    assert body[0]["n_rows"] == 4
    assert body[0]["n_cols"] == 3


async def test_list_analyses_empty_for_fresh_dataset(client):
    upload = await client.post(
        "/datasets",
        files={"file": ("income.csv", CSV_TEXT.encode(), "text/csv")},
    )
    dataset_id = upload.json()["id"]
    resp = await client.get(f"/datasets/{dataset_id}/analyses")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_analyses_returns_history_in_order(client, tmp_path, monkeypatch):
    from agent.config import settings

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    csv = make_income().to_csv(index=False)
    upload = await client.post(
        "/datasets",
        files={"file": ("income.csv", csv.encode(), "text/csv")},
    )
    dataset_id = upload.json()["id"]

    questions = ["男性和女性的 income 是否存在显著差异？", "income 在不同 city 间是否有差异？"]
    first_charts = None
    for question in questions:
        analyze = await client.post(f"/datasets/{dataset_id}/analyze", json={"question": question})
        assert analyze.status_code == 200
        if first_charts is None:
            first_charts = analyze.json()["charts"]

    listing = await client.get(f"/datasets/{dataset_id}/analyses")
    assert listing.status_code == 200
    items = listing.json()
    assert [item["question"] for item in items] == questions
    assert [item["created_at"] for item in items] == sorted(item["created_at"] for item in items)
    assert items[0]["n_steps"] > 0
    assert items[0]["n_charts"] == len(first_charts)
    assert items[0]["charts"] == first_charts

    detail = await client.get(f"/datasets/{dataset_id}/analyses/{items[0]['analysis_id']}")
    assert detail.status_code == 200


async def test_list_analyses_missing_dataset_404(client):
    resp = await client.get("/datasets/doesnotexist/analyses")
    assert resp.status_code == 404


async def test_delete_dataset_removes_record(client):
    upload = await client.post(
        "/datasets",
        files={"file": ("income.csv", CSV_TEXT.encode(), "text/csv")},
    )
    dataset_id = upload.json()["id"]

    resp = await client.delete(f"/datasets/{dataset_id}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": dataset_id}

    assert (await client.get(f"/datasets/{dataset_id}")).status_code == 404
    assert (await client.get("/datasets")).json() == []
    assert (await client.delete(f"/datasets/{dataset_id}")).status_code == 404


async def test_delete_dataset_removes_artifacts(client, tmp_path, monkeypatch):
    from agent.config import settings

    monkeypatch.setattr(settings, "artifacts_dir", str(tmp_path))
    dataset_id, body = await _upload_and_analyze_income(client, monkeypatch, tmp_path)
    chart = body["charts"][0]

    resp = await client.delete(f"/datasets/{dataset_id}")
    assert resp.status_code == 200
    assert not (tmp_path / dataset_id).exists()
    assert (await client.get(f"/datasets/{dataset_id}/charts/{chart}")).status_code == 404
    assert (await client.get(f"/datasets/{dataset_id}/analyses")).status_code == 404


async def test_preview_dataset_rows(client):
    upload = await client.post(
        "/datasets",
        files={"file": ("income.csv", CSV_TEXT.encode(), "text/csv")},
    )
    dataset_id = upload.json()["id"]

    resp = await client.get(f"/datasets/{dataset_id}/data", params={"rows": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == dataset_id
    assert body["n_rows"] == 4
    assert body["columns"] == ["gender", "city", "income"]
    assert body["rows"] == [["M", "tier1", 9000], ["F", "tier2", 7000]]

    full = (await client.get(f"/datasets/{dataset_id}/data", params={"rows": 999})).json()
    assert len(full["rows"]) == 4

    default = (await client.get(f"/datasets/{dataset_id}/data")).json()
    assert len(default["rows"]) == 4


async def test_preview_dataset_renders_missing_as_null(client):
    upload = await client.post(
        "/datasets",
        files={"file": ("nan.csv", CSV_TEXT_NAN.encode(), "text/csv")},
    )
    dataset_id = upload.json()["id"]

    body = (await client.get(f"/datasets/{dataset_id}/data")).json()
    assert body["rows"][1] == [2, None]


async def test_preview_missing_dataset_404(client):
    resp = await client.get("/datasets/doesnotexist/data")
    assert resp.status_code == 404


async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_healthz_reports_load_counters(client):
    from api import jobs as jobs_mod
    from api.store import REGISTRY
    from main import APP_VERSION

    jobs_mod.JOBS.clear()
    before = len(REGISTRY)
    upload = await client.post(
        "/datasets",
        files={"file": ("income.csv", CSV_TEXT.encode(), "text/csv")},
    )
    assert upload.status_code == 201

    body = (await client.get("/healthz")).json()
    assert body["status"] == "ok"
    assert body["version"] == APP_VERSION
    assert body["datasets"] == before + 1
    assert body["jobs_running"] == 0
    assert body["jobs_total"] == 0

    jobs_mod.JOBS["probejob0001"] = {"status": "running", "created_at": 0.0}
    body = (await client.get("/healthz")).json()
    assert body["jobs_running"] == 1
    assert body["jobs_total"] == 1
    jobs_mod.JOBS.clear()


async def test_root_reports_app_metadata(client):
    from main import APP_VERSION

    resp = await client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["app"] == "DataAnalystAgent"
    assert body["version"] == APP_VERSION
    assert body["docs"] == "/docs"


async def test_upload_rejects_oversized_file(client, monkeypatch):
    from agent.config import settings

    monkeypatch.setattr(settings, "max_upload_mb", 0.001)
    big_csv = ("a,b\n" + "1,2\n" * 1000).encode()
    resp = await client.post("/datasets", files={"file": ("big.csv", big_csv, "text/csv")})
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"]

    ok_csv = "a,b\n1,2\n".encode()
    resp = await client.post("/datasets", files={"file": ("ok.csv", ok_csv, "text/csv")})
    assert resp.status_code == 201
