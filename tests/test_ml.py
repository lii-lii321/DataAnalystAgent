import numpy as np
import pandas as pd
import pytest

from agent.ml import auto_model, detect_task, kmeans_profile


def classification_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(3, 1, n // 2)])
    x2 = rng.normal(0, 1, n)
    y = (np.arange(n) >= n // 2).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "label": y})


def regression_df(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return pd.DataFrame({"x": x, "y": 5 * x + rng.normal(0, 1, n), "group": rng.choice(["a", "b"], n)})


def test_detect_task():
    assert detect_task(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])) == "regression"
    assert detect_task(pd.Series([0, 1] * 10)) == "classification"


def test_classification_accuracy():
    df = classification_df()
    result = auto_model(df, "label")
    assert result.task == "classification"
    assert result.metrics["accuracy"] > 0.85
    assert result.cv["mean"] > 0.85
    assert result.top_features[0][0] == "x1"


def test_regression_r2():
    df = regression_df()
    result = auto_model(df, "y")
    assert result.task == "regression"
    assert result.metrics["r2"] > 0.9
    assert result.top_features[0][0] == "x"


def test_missing_target_raises():
    with pytest.raises(ValueError):
        auto_model(regression_df(), "nope")


def test_kmeans_finds_two_clusters():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "a": np.concatenate([rng.normal(0, 0.4, 100), rng.normal(6, 0.4, 100)]),
        "b": np.concatenate([rng.normal(0, 0.4, 100), rng.normal(6, 0.4, 100)]),
    })
    result = kmeans_profile(df, ["a", "b"], k_range=range(2, 5))
    assert result.k == 2
    assert result.silhouette > 0.6
    assert sum(result.sizes.values()) == 200


def test_kmeans_needs_ten_rows():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
    with pytest.raises(ValueError):
        kmeans_profile(df, ["a"])
