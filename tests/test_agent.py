import asyncio

from agent.agent import run_analysis
from agent.llm import LLMProvider
from agent.synth import make_income


class FakeProvider(LLMProvider):
    name = "openai"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


class BoomProvider(LLMProvider):
    name = "openai"

    async def complete(self, system: str, user: str) -> str:
        raise RuntimeError("LLM 服务不可用")


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


def test_cluster_pipeline_reports_clusters(tmp_path):
    df = make_income()
    ws = asyncio.run(run_analysis(
        df,
        "对 income 和 experience 做聚类分析",
        artifacts_dir=str(tmp_path),
    ))
    assert ws.cluster is not None
    assert ws.cluster.k >= 2
    assert "## 7. 聚类分析" in ws.report_md
    assert "最优 k=" in ws.report_md


def test_llm_loop_executes_chosen_tools(tmp_path):
    replies = [
        '{"tool": "profile_data"}',
        '{"tool": "check_quality"}',
        '{"tool": "make_plot", "args": {"question": "分布 income"}}',
        '{"tool": "finish"}',
    ]
    provider = FakeProvider(replies)
    ws = asyncio.run(
        run_analysis(make_income(), "income 分布如何？", provider=provider, artifacts_dir=str(tmp_path))
    )
    assert provider.calls == 4
    executed = [s.tool for s in ws.steps if s.ok]
    assert executed[:3] == ["profile_data", "check_quality", "make_plot"]
    assert ws.report_md
    assert "## 1. 数据概览" in ws.report_md


def test_llm_failure_falls_back_to_pipeline(tmp_path):
    ws = asyncio.run(
        run_analysis(
            make_income(),
            "男性和女性的 income 是否存在显著差异？",
            provider=BoomProvider(),
            artifacts_dir=str(tmp_path),
        )
    )
    assert ws.profile is not None
    assert ws.tests, "fallback pipeline should run statistical tests"
    assert "## 4. 统计检验" in ws.report_md


def test_llm_unparseable_reply_stops_and_finalizes(tmp_path):
    provider = FakeProvider(["抱歉，我无法决定下一步。"])
    ws = asyncio.run(
        run_analysis(make_income(), "income 分布如何？", provider=provider, artifacts_dir=str(tmp_path))
    )
    assert provider.calls == 1
    assert ws.report_md
    assert "分析报告" in ws.report_md
