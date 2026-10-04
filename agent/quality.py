from dataclasses import dataclass

import pandas as pd

from agent.profiler import DatasetProfile


@dataclass
class QualityIssue:
    severity: str
    column: str
    issue: str
    suggestion: str


def quality_report(df: pd.DataFrame, profile: DatasetProfile) -> list[QualityIssue]:
    issues: list[QualityIssue] = []

    if profile.dup_rows > 0:
        issues.append(QualityIssue(
            severity="high",
            column="(dataset)",
            issue=f"发现 {profile.dup_rows} 条完全重复行",
            suggestion="执行 drop_duplicates 去重后再分析",
        ))

    for col in profile.columns:
        if col.missing_pct >= 30:
            issues.append(QualityIssue(
                severity="high",
                column=col.name,
                issue=f"缺失率 {col.missing_pct}%",
                suggestion="考虑删除该列，或按业务含义填充/标注缺失",
            ))
        elif col.missing_pct > 0:
            issues.append(QualityIssue(
                severity="medium",
                column=col.name,
                issue=f"缺失 {col.missing} 个值（{col.missing_pct}%）",
                suggestion="数值列用中位数填充，类别列用众数或单独类别填充",
            ))

        if col.role == "numeric" and col.outliers:
            ratio = col.outliers / max(profile.n_rows - col.missing, 1) * 100
            if ratio >= 1:
                issues.append(QualityIssue(
                    severity="medium",
                    column=col.name,
                    issue=f"疑似异常值 {col.outliers} 个（IQR 法，占 {ratio:.1f}%）",
                    suggestion="核对业务含义：真实极端值保留并说明，错误值应剔除",
                ))

        if col.role not in ("empty",) and col.n_unique <= 1 and profile.n_rows > 1:
            issues.append(QualityIssue(
                severity="low",
                column=col.name,
                issue="常量列（只有 1 个取值）",
                suggestion="对分析无信息量，可从建模特征中移除",
            ))

    return issues


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, len(df) - len(cleaned)
