import asyncio

from agent.agent import run_analysis
from agent.synth import make_income


def test_full_pipeline_on_income_data(tmp_path):
    df = make_income()
    ws = asyncio.run(run_analysis(
        df,
        "男性和女性的 income 是否存在显著差异？",
        artifacts_dir=str(tmp_path),
    ))
    assert ws.profile is not None
    assert ws.profile.n_rows > 500
    assert ws.issues, "quality issues expected (planted)"
    assert any(t.test in ("independent t-test", "mann-whitney U") for t in ws.tests)
    assert ws.report_md
    assert "## 4. 统计检验" in ws.report_md
    assert "## 10. 局限性" in ws.report_md
    assert len(ws.steps) >= 5
    assert all(chart.endswith(".png") for chart in ws.chart_files)
    assert all((tmp_path / chart).exists() for chart in ws.chart_files)


def test_prediction_pipeline(tmp_path):
    df = make_income()
    ws = asyncio.run(run_analysis(df, "预测 income", artifacts_dir=str(tmp_path)))
    assert ws.model is not None
    assert ws.model.task == "regression"
    assert ws.model.target == "income"
    assert "r2" in ws.report_md
