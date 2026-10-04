import pandas as pd

from agent.profiler import profile_dataframe
from agent.quality import drop_duplicates, quality_report


def messy_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "age": [25.0, 30.0, None, 41.0, 25.0, 30.0, None, 41.0],
        "salary": [100.0, 200.0, 150.0, 99999.0, 100.0, 200.0, 150.0, 99999.0],
        "city": ["A", "B", "A", "B", "A", "B", "A", "B"],
    })
    return df


def test_quality_flags_duplicates_missing_outliers():
    df = messy_df()
    profile = profile_dataframe(df)
    issues = quality_report(df, profile)
    columns = {(i.column, i.severity) for i in issues}
    assert ("(dataset)", "high") in columns
    assert any(i.column == "age" and "缺失" in i.issue for i in issues)
    assert any(i.column == "salary" and "异常值" in i.issue for i in issues)


def test_drop_duplicates():
    df = messy_df()
    cleaned, removed = drop_duplicates(df)
    assert removed == 4
    assert len(cleaned) == 4
