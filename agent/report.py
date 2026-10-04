import pandas as pd

from agent.ml import ClusterResult, ModelResult
from agent.profiler import DatasetProfile
from agent.quality import QualityIssue
from agent.stats import TestResult

SECTIONS = [
    "数据概览", "数据质量", "描述统计", "统计检验",
    "关键发现", "建模结果", "聚类分析", "可视化", "结论", "局限性",
]


def _describe_table(df: pd.DataFrame, numeric_cols: list) -> str:
    if not numeric_cols:
        return "_无数值列_"
    desc = df[numeric_cols].apply(pd.to_numeric, errors="coerce").describe().T.round(3)
    header = "| 列 | count | mean | std | min | 25% | 50% | 75% | max |"
    sep = "| --- " * 9 + "|"
    rows = []
    for name, row in desc.iterrows():
        cells = " | ".join(str(row.get(k, "")) for k in ("count", "mean", "std", "min", "25%", "50%", "75%", "max"))
        rows.append(f"| {name} | {cells} |")
    return "\n".join([header, sep, *rows])


def _test_table(tests: list[TestResult]) -> str:
    if not tests:
        return "_本轮未执行统计检验_"
    header = "| 检验方法 | 统计量 | p 值 | 显著(α=0.05) | 效应量 |"
    sep = "| --- | --- | --- | --- | --- |"
    rows = []
    for t in tests:
        effect = f"{t.effect_name}={t.effect}" if t.effect is not None else "-"
        rows.append(f"| {t.test} | {t.statistic:.4g} | {t.p_value:.4g} | {'是' if t.significant else '否'} | {effect} |")
    return "\n".join([header, sep, *rows])


def _overview(profile: DatasetProfile) -> str:
    lines = [
        f"- 样本量：{profile.n_rows} 行 × {profile.n_cols} 列",
        f"- 完全重复行：{profile.dup_rows}",
        f"- 单元格缺失率：{profile.missing_cells_pct}%",
        f"- 内存占用：{profile.memory_mb} MB",
        "",
        "| 列 | 推断角色 | 缺失 | 唯一值 | 示例 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in profile.columns:
        lines.append(f"| {c.name} | {c.role} | {c.missing} | {c.n_unique} | {', '.join(c.examples[:2])} |")
    return "\n".join(lines)


def build_report(
    question: str,
    df: pd.DataFrame,
    profile: DatasetProfile,
    issues: list[QualityIssue],
    tests: list[TestResult],
    findings: list[str],
    model: ModelResult | None = None,
    cluster: ClusterResult | None = None,
    chart_files: list[str] | None = None,
) -> str:
    numeric_cols = profile.role_columns("numeric")
    parts = [f"# 分析报告：{question}", ""]

    parts += ["## 1. 数据概览", _overview(profile), ""]
    parts += ["## 2. 数据质量"]
    if issues:
        for issue in issues:
            parts.append(f"- **[{issue.severity}]** {issue.column}：{issue.issue}（建议：{issue.suggestion}）")
    else:
        parts.append("- 未发现明显数据质量问题")
    parts += ["", "## 3. 描述统计", _describe_table(df, numeric_cols), ""]
    parts += ["## 4. 统计检验", _test_table(tests), ""]
    parts += ["## 5. 关键发现"]
    parts += [f"- {f}" for f in findings] if findings else ["- 暂无"]
    parts += ["", "## 6. 建模结果"]
    if model:
        parts.append(f"- 任务：{model.task}，目标列：`{model.target}`（train={model.n_train}, test={model.n_test}）")
        parts.append(f"- 指标：{model.metrics}")
        parts.append(f"- 交叉验证：{model.cv}")
        parts.append(f"- Top 特征：{model.top_features[:5]}")
    else:
        parts.append("- 本轮未建模（问题未涉及预测/建模）")
    parts += ["", "## 7. 聚类分析"]
    parts.append(f"- 最优 k={cluster.k}（silhouette={cluster.silhouette}），各簇规模：{cluster.sizes}" if cluster else "- 本轮未执行聚类")
    parts += ["", "## 8. 可视化"]
    if chart_files:
        parts += [f"![{f}]({f})" for f in chart_files]
    else:
        parts.append("- 无图表")
    parts += ["", "## 9. 结论"]
    conclusions = [f for f in findings if "显著" in f or "相关" in f] or (findings[:3] if findings else ["暂无可用结论"])
    parts += [f"- {c}" for c in conclusions]
    parts += [
        "",
        "## 10. 局限性",
        "- 相关关系不等于因果关系，结论需结合业务验证",
        "- 缺失值与异常值的处理方式会影响统计结果",
        "- 建模为随机森林基线，未做超参数调优与特征工程",
        "",
    ]
    return "\n".join(parts)
