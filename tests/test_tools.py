import pandas as pd

from agent.tools import Workspace, run_tool, tool_cluster, tool_group_test, tool_model


def categorical_df() -> pd.DataFrame:
    return pd.DataFrame({"city": ["a", "b", "c"] * 10, "level": ["x", "y"] * 15})


def test_run_tool_dispatches_known_tool():
    ws = Workspace(df=categorical_df(), question="测试")
    assert run_tool(ws, "profile_data", {}) is True
    assert ws.steps[-1].tool == "profile_data"
    assert ws.steps[-1].ok is True
    assert ws.profile is not None


def test_run_tool_logs_unknown_tool():
    ws = Workspace(df=categorical_df(), question="测试")
    assert run_tool(ws, "no_such_tool", {}) is False
    step = ws.steps[-1]
    assert step.tool == "no_such_tool"
    assert step.ok is False


def test_cluster_without_numeric_columns_degrades():
    ws = Workspace(df=categorical_df(), question="聚类")
    summary = tool_cluster(ws)
    assert "2+ numeric" in summary
    assert ws.cluster is None
    assert ws.steps[-1].ok is False


def test_group_test_without_pair_degrades():
    ws = Workspace(df=categorical_df(), question="测试")
    summary = tool_group_test(ws)
    assert "no numeric/categorical pair" in summary
    assert ws.steps[-1].ok is False


def test_model_without_target_degrades():
    ws = Workspace(df=categorical_df(), question="测试")
    summary = tool_model(ws)
    assert "no plausible target" in summary
    assert ws.model is None
    assert ws.steps[-1].ok is False
