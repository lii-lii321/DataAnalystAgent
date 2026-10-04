from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelResult:
    task: str
    target: str
    metrics: dict
    cv: dict
    top_features: list
    n_train: int
    n_test: int


@dataclass
class ClusterResult:
    k: int
    silhouette: float
    sizes: dict


def detect_task(target: pd.Series) -> str:
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.notna().mean() > 0.9 and target.nunique() > 12:
        return "regression"
    return "classification"


def _preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]), cat_cols),
    ])


def auto_model(df: pd.DataFrame, target_col: str, task: str | None = None) -> ModelResult:
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col!r} not found")
    data = df.dropna(subset=[target_col]).reset_index(drop=True)
    y = data[target_col]
    X = data.drop(columns=[target_col])
    task = task or detect_task(y)

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols and X[c].nunique() <= 50]
    X = X[num_cols + cat_cols]

    pre = _preprocessor(num_cols, cat_cols)
    if task == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        scoring = "r2"
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        scoring = "accuracy"

    pipe = Pipeline([("pre", pre), ("model", model)])

    stratify = y if (task == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    if task == "regression":
        metrics = {
            "r2": round(float(r2_score(y_test, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        }
    else:
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, pred, average="weighted")), 4),
        }
        if y.nunique() == 2:
            try:
                proba = pipe.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
            except ValueError:
                pass

    if task == "classification":
        folds = int(min(5, y.value_counts().min()))
    else:
        folds = int(min(5, len(y)))
    if folds >= 2:
        cv = cross_val_score(pipe, X, y, cv=folds, scoring=scoring)
        cv_result = {"folds": folds, "mean": round(float(cv.mean()), 4), "std": round(float(cv.std()), 4)}
    else:
        cv_result = {"folds": 0, "mean": None, "std": None}

    names = pipe.named_steps["pre"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    top = sorted(zip(names, importances), key=lambda kv: -kv[1])[:10]
    top_features = [(str(n).split("__", 1)[-1], round(float(v), 4)) for n, v in top]

    return ModelResult(
        task=task,
        target=target_col,
        metrics=metrics,
        cv=cv_result,
        top_features=top_features,
        n_train=len(X_train),
        n_test=len(X_test),
    )


def kmeans_profile(df: pd.DataFrame, feature_cols: list | None = None, k_range=range(2, 7)) -> ClusterResult:
    if feature_cols is None:
        feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
    if len(X) < 10:
        raise ValueError("not enough rows for clustering")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    features = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]).fit_transform(X)

    best_k, best_score = 2, -1.0
    for k in k_range:
        if k >= len(X):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(features)
        score = float(silhouette_score(features, labels))
        if score > best_score:
            best_k, best_score = k, score

    labels = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(features)
    sizes = {str(i): int((labels == i).sum()) for i in range(best_k)}
    return ClusterResult(k=best_k, silhouette=round(best_score, 4), sizes=sizes)
