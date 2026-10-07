import pandas as pd
import pytest

from agent.profiler import profile_dataframe
from agent.viz import ChartSpec, choose_chart, render_chart


def demo_df() -> pd.DataFrame:
    return pd.DataFrame({
        "gender": ["M", "F"] * 60,
        "city": ["a", "b", "c"] * 40,
        "income": list(range(120)),
        "score": [x * 2 for x in range(120)],
    })


def timed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=60),
        "city": ["a", "b", "c"] * 20,
        "income": list(range(60)),
        "score": [x * 1.5 for x in range(60)],
    })


def spec_for(question, df=None):
    df = demo_df() if df is None else df
    return choose_chart(question, df, profile_dataframe(df))


def test_distribution_question_uses_histogram():
    spec = spec_for("看看 income 的分布")
    assert spec.kind == "hist"
    assert spec.x == "income"


def test_group_question_uses_boxplot():
    spec = spec_for("不同 gender 的 income 差异")
    assert spec.kind == "box"
    assert spec.x == "gender" and spec.y == "income"


def test_relation_question_uses_scatter():
    spec = spec_for("income 与 score 的关系")
    assert spec.kind == "scatter"


def test_count_question_uses_bar():
    spec = spec_for("各 city 的数量")
    assert spec.kind == "bar"


def test_no_plottable_columns_raises():
    df = pd.DataFrame({"joined": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])})
    with pytest.raises(ValueError):
        choose_chart("随便看看", df, profile_dataframe(df))


def test_render_chart_writes_png(tmp_path):
    df = demo_df()
    spec = spec_for("不同 gender 的 income 差异")
    out = tmp_path / "chart.png"
    render_chart(spec, df, out)
    assert out.exists() and out.stat().st_size > 1000


def test_trend_question_with_datetime_uses_line():
    df = timed_df()
    spec = spec_for("income 随时间的趋势", df)
    assert spec.kind == "line"
    assert spec.x == "day"
    assert spec.y == "income"


def test_no_intent_question_defaults_to_histogram():
    spec = spec_for("看看这个数据")
    assert spec.kind == "hist"
    assert spec.x == "income"


@pytest.mark.parametrize(
    "question,kind",
    [
        ("income 的分布", "hist"),
        ("各 city 的数量", "bar"),
        ("income 与 score 的关系", "scatter"),
        ("income 随时间的趋势", "line"),
    ],
)
def test_render_chart_all_kinds(tmp_path, question, kind):
    df = timed_df()
    spec = spec_for(question, df)
    assert spec.kind == kind
    out = tmp_path / f"{kind}.png"
    render_chart(spec, df, out)
    assert out.exists() and out.stat().st_size > 1000


def test_chart_filename_sanitizes_and_matches_download_rule():
    from api.routers import CHART_FILENAME_RE

    spec = ChartSpec(kind="box", x="城市", y="收入(元)")
    name = spec.filename(2)
    assert name == "chart_2_boxplot_城市_收入_元_.png"
    assert CHART_FILENAME_RE.fullmatch(name)
