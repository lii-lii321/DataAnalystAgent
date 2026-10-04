"""Synthetic benchmark datasets with planted effects (deterministic seeds)."""

import numpy as np
import pandas as pd


def _plant_quality_issues(df: pd.DataFrame, rng, missing_col: str, n_missing: int, n_dupes: int) -> pd.DataFrame:
    idx = rng.choice(len(df), n_missing, replace=False)
    df.loc[idx, missing_col] = np.nan
    return pd.concat([df, df.iloc[:n_dupes]], ignore_index=True)


def make_income(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["M", "F"], n)
    city = rng.choice(["tier1", "tier2", "tier3"], n, p=[0.3, 0.4, 0.3])
    experience = rng.integers(1, 21, n).astype(float)
    noise = rng.normal(0, 1500, n)
    income = 8000 + 1800 * (gender == "M") + 120 * experience + 2000 * (city == "tier1") + noise
    df = pd.DataFrame({
        "gender": gender,
        "city": city,
        "experience": experience,
        "income": income.round(0),
    })
    df = _plant_quality_issues(df, rng, "income", 12, 5)
    outlier_idx = rng.choice(len(df), 8, replace=False)
    df.loc[outlier_idx, "income"] = 50000.0
    return df


def make_churn(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 60, n).astype(float)
    fee = rng.normal(80, 20, n).round(2).clip(10)
    support = rng.integers(0, 8, n).astype(float)
    logit = -1.2 + 0.45 * support - 0.04 * tenure + rng.normal(0, 0.5, n)
    churn = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    df = pd.DataFrame({
        "tenure_months": tenure,
        "monthly_fee": fee,
        "support_calls": support,
        "churn": churn,
    })
    return _plant_quality_issues(df, rng, "monthly_fee", 10, 4)


def make_scores(n: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 20, n).round(1)
    score = (40 + 3.5 * hours + rng.normal(0, 5, n)).clip(0, 100).round(1)
    df = pd.DataFrame({"study_hours": hours, "score": score})
    return _plant_quality_issues(df, rng, "score", 6, 3)


GENERATORS = {
    "income.csv": make_income,
    "churn.csv": make_churn,
    "scores.csv": make_scores,
}
