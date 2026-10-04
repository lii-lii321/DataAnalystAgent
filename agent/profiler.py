from dataclasses import dataclass

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    role: str
    missing: int
    missing_pct: float
    n_unique: int
    examples: list
    outliers: int | None = None
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    dup_rows: int
    missing_cells_pct: float
    memory_mb: float
    columns: list[ColumnProfile]

    def role_columns(self, role: str) -> list[str]:
        return [c.name for c in self.columns if c.role == role]

    def column(self, name: str) -> ColumnProfile | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None


def infer_role(series: pd.Series) -> str:
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if non_null.nunique() <= 2:
            return "categorical"
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    values = non_null.astype(str)
    if values.str.contains(r"\d[-/]\d", regex=True).mean() > 0.8:
        parsed = pd.to_datetime(values, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return "datetime"
    parsed_num = pd.to_numeric(values.str.replace(",", "", regex=False), errors="coerce")
    if parsed_num.notna().mean() > 0.9:
        return "numeric"
    n_unique = int(non_null.nunique())
    if n_unique == len(non_null) and len(non_null) > 20:
        return "identifier"
    if n_unique <= max(20, 0.05 * len(non_null)):
        return "categorical"
    return "text"


def _outlier_count(values: pd.Series) -> int:
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((values < low) | (values > high)).sum())


def profile_column(name: str, series: pd.Series) -> ColumnProfile:
    role = infer_role(series)
    non_null = series.dropna()
    cp = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        role=role,
        missing=int(series.isna().sum()),
        missing_pct=round(float(series.isna().mean()) * 100, 2),
        n_unique=int(non_null.nunique()),
        examples=[str(v)[:40] for v in non_null.head(3).tolist()],
    )
    if role == "numeric":
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric):
            cp.outliers = _outlier_count(numeric)
            cp.min = round(float(numeric.min()), 4)
            cp.max = round(float(numeric.max()), 4)
            cp.mean = round(float(numeric.mean()), 4)
            cp.std = round(float(numeric.std()), 4) if len(numeric) > 1 else 0.0
    return cp


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    columns = [profile_column(c, df[c]) for c in df.columns]
    total_cells = len(df) * max(len(df.columns), 1)
    missing_pct = round(float(df.isna().to_numpy().sum()) / total_cells * 100, 2) if total_cells else 0.0
    return DatasetProfile(
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        dup_rows=int(df.duplicated().sum()),
        missing_cells_pct=missing_pct,
        memory_mb=round(float(df.memory_usage(deep=True).sum()) / 1024 / 1024, 2),
        columns=columns,
    )
