import pytest

from api.store import REGISTRY
from agent.synth import make_income

CSV_TEXT = "gender,city,income\nM,tier1,9000\nF,tier2,7000\nM,tier1,12000\nF,tier3,6500\n"


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


async def test_analyze_missing_dataset(client):
    resp = await client.post("/datasets/doesnotexist/analyze", json={"question": "test"})
    assert resp.status_code == 404
