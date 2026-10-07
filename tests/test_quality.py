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


def test_quality_flags_high_missing_and_constant_columns():
    df = pd.DataFrame({
        "sparse": [None] * 4 + [1.0],
        "constant": ["x"] * 5,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    issues = quality_report(df, profile_dataframe(df))

    per_column: dict = {}
    for issue in issues:
        per_column.setdefault(issue.column, []).append(issue)
    assert any(i.severity == "high" and "缺失率" in i.issue for i in per_column["sparse"])
    assert any(i.severity == "low" and "常量列" in i.issue for i in per_column["constant"])
    assert not any(i.column == "value" for i in issues)


def test_quality_skips_minor_outliers():
    df = pd.DataFrame({
        "v": [10.0 + i for i in range(199)] + [100000.0],
        "g": ["A", "B"] * 100,
    })
    issues = quality_report(df, profile_dataframe(df))
    assert not any(i.column == "v" and "异常值" in i.issue for i in issues)
