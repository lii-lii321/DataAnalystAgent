import numpy as np
import pandas as pd
import pytest

from agent.stats import categorical_association, compare_groups, correlation


def groups_df(effect: float, n: int = 150, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "income": np.concatenate([rng.normal(100, 10, n), rng.normal(100 + effect, 10, n)]),
        "gender": ["A"] * n + ["B"] * n,
    })


def test_planted_effect_is_significant():
    result = compare_groups(groups_df(effect=15), "income", "gender")
    assert result.test in ("independent t-test", "mann-whitney U")
    assert result.significant
    assert result.effect is not None and abs(result.effect) > 0.8


def test_no_effect_is_not_significant():
    result = compare_groups(groups_df(effect=0, seed=3), "income", "gender")
    assert not result.significant


def test_three_groups_uses_anova_or_kruskal():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "value": np.concatenate([rng.normal(10, 2, 100), rng.normal(14, 2, 100), rng.normal(18, 2, 100)]),
        "city": ["x"] * 100 + ["y"] * 100 + ["z"] * 100,
    })
    result = compare_groups(df, "value", "city")
    assert result.test in ("one-way ANOVA", "kruskal-wallis")
    assert result.significant


def test_too_few_groups_raises():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0], "g": ["a", "a", "a"]})
    with pytest.raises(ValueError):
        compare_groups(df, "v", "g")


def test_chi_square_planted_association():
    rng = np.random.default_rng(9)
    n = 300
    device = rng.choice(["ios", "android"], n)
    tier = np.where(device == "ios", rng.choice(["high", "low"], n, p=[0.8, 0.2]), rng.choice(["high", "low"], n, p=[0.3, 0.7]))
    df = pd.DataFrame({"device": device, "tier": tier})
    result = categorical_association(df, "device", "tier")
    assert result.test == "chi-square"
    assert result.significant
    assert result.effect_name == "cramers_v"


def test_correlation_strong_linear():
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": np.arange(200, dtype=float), "y": 3.0 * np.arange(200) + rng.normal(0, 5, 200)})
    result = correlation(df, "x", "y")
    assert "correlation" in result.test
    assert result.significant
    assert abs(result.effect) > 0.9
