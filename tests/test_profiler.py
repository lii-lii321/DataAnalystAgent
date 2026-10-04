import pandas as pd

from agent.profiler import infer_role, profile_dataframe


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": range(1, 32),
        "age": [25, 30, 41, 35, 28] * 6 + [33],
        "income": [100.0, 250.0, 130.0, 110.0, 90.0] * 6 + [999999.0],
        "city": ["北京", "上海", "广州"] * 10 + [None],
        "joined": pd.to_datetime(["2024-01-05"] * 31),
        "joined_str": ["2024-01-05", "2024-02-11", "2024-03-20"] * 10 + ["2024-04-01"],
        "memo": [f"note {i} unique text" for i in range(31)],
    })


def test_role_inference():
    df = sample_df()
    assert infer_role(df["user_id"]) == "numeric"
    assert infer_role(df["city"]) == "categorical"
    assert infer_role(df["joined"]) == "datetime"
    assert infer_role(df["joined_str"]) == "datetime"
    assert infer_role(df["memo"]) == "identifier"
    assert infer_role(pd.Series([0, 1, 1, 0, 1])) == "categorical"


def test_profile_dataframe_summary():
    profile = profile_dataframe(sample_df())
    assert profile.n_rows == 31
    assert profile.n_cols == 7
    assert profile.dup_rows == 0
    city = profile.column("city")
    assert city.missing == 1
    assert city.role == "categorical"


def test_outlier_detection():
    df = sample_df()
    profile = profile_dataframe(df)
    income = profile.column("income")
    assert income.outliers >= 1
    assert income.max == 999999.0


def test_binary_numeric_is_categorical():
    df = pd.DataFrame({"churn": [0, 1, 0, 1, 1] * 10, "fee": [80.5, 60.0, 91.2, 55.5, 77.7] * 10})
    profile = profile_dataframe(df)
    assert profile.column("churn").role == "categorical"
    assert profile.column("fee").role == "numeric"
