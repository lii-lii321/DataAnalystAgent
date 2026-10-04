import pandas as pd
import pytest

from agent.profiler import profile_dataframe
from agent.viz import choose_chart, render_chart


def demo_df() -> pd.DataFrame:
    return pd.DataFrame({
        "gender": ["M", "F"] * 60,
        "city": ["a", "b", "c"] * 40,
        "income": list(range(120)),
        "score": [x * 2 for x in range(120)],
    })


def spec_for(question):
    df = demo_df()
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
