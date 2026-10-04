from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

NORMALITY_ALPHA = 0.05
NORMALITY_MAX_N = 5000


@dataclass
class TestResult:
    test: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    detail: str = ""
    effect: float | None = None
    effect_name: str = ""

    @property
    def significant(self) -> bool:
        return bool(self.p_value < self.alpha)


def _is_normal(series: pd.Series) -> bool | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    if n < 8:
        return None
    if n > NORMALITY_MAX_N:
        values = values.sample(NORMALITY_MAX_N, random_state=42)
        result = stats.normaltest(values)
    else:
        result = stats.shapiro(values)
    return bool(result.pvalue > NORMALITY_ALPHA)


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def compare_groups(df: pd.DataFrame, value_col: str, group_col: str) -> TestResult:
    data = df[[value_col, group_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    groups = [g[value_col].astype(float) for _, g in data.groupby(group_col) if len(g) >= 2]
    if len(groups) < 2:
        raise ValueError(f"need at least two groups with 2+ observations in {group_col!r}")

    if len(groups) == 2:
        a, b = groups
        normal_a, normal_b = _is_normal(a), _is_normal(b)
        if normal_a is None or normal_b is None:
            res = stats.mannwhitneyu(a, b, alternative="two-sided")
            return TestResult(
                test="mann-whitney U",
                statistic=float(res.statistic),
                p_value=float(res.pvalue if not np.isnan(res.pvalue) else 1.0),
                detail="样本量过小，无法检验正态性，使用非参数检验",
            )
        if normal_a and normal_b:
            levene = stats.levene(a, b)
            equal_var = bool(levene.pvalue > 0.05)
            res = stats.ttest_ind(a, b, equal_var=equal_var)
            detail = "independent t-test (pooled)" if equal_var else "welch t-test"
            return TestResult(
                test="independent t-test",
                statistic=float(res.statistic),
                p_value=float(res.pvalue),
                detail=detail,
                effect=round(_cohens_d(a, b), 4),
                effect_name="cohen_d",
            )
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        return TestResult(
            test="mann-whitney U",
            statistic=float(res.statistic),
            p_value=float(res.pvalue if not np.isnan(res.pvalue) else 1.0),
            detail="数据非正态，使用非参数检验",
        )

    normals = [_is_normal(g) for g in groups]
    if all(n is True for n in normals):
        res = stats.f_oneway(*groups)
        return TestResult(test="one-way ANOVA", statistic=float(res.statistic), p_value=float(res.pvalue))
    res = stats.kruskal(*groups)
    return TestResult(
        test="kruskal-wallis",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        detail="数据非正态，使用非参数检验",
    )


def categorical_association(df: pd.DataFrame, col_a: str, col_b: str) -> TestResult:
    table = pd.crosstab(df[col_a].astype(str), df[col_b].astype(str))
    if min(table.shape) < 2:
        raise ValueError("crosstab needs at least 2x2 levels")
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = int(table.values.sum())
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if n and min(table.shape) > 1 else 0.0
    return TestResult(
        test="chi-square",
        statistic=float(chi2),
        p_value=float(p),
        detail=f"列联表 {table.shape[0]}x{table.shape[1]}",
        effect=round(cramers_v, 4),
        effect_name="cramers_v",
    )


def correlation(df: pd.DataFrame, col_a: str, col_b: str) -> TestResult:
    data = df[[col_a, col_b]].apply(pd.to_numeric, errors="coerce").dropna()
    a, b = data[col_a], data[col_b]
    if len(a) < 3:
        raise ValueError("correlation needs at least 3 paired observations")
    normal_a, normal_b = _is_normal(a), _is_normal(b)
    if normal_a and normal_b:
        res = stats.pearsonr(a, b)
        name = "pearson"
    else:
        res = stats.spearmanr(a, b)
        name = "spearman"
    statistic = float(res.statistic)
    p_value = float(res.pvalue if not np.isnan(res.pvalue) else 1.0)
    return TestResult(
        test=f"{name} correlation",
        statistic=round(statistic, 4),
        p_value=round(p_value, 6),
        effect=round(statistic, 4),
        effect_name=name,
    )
