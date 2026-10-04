import itertools
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from agent import ml, quality, report, stats, viz
from agent.llm import StepLog
from agent.profiler import DatasetProfile, profile_dataframe

CORRELATION_INTENT = ("关系", "相关", "correlation", "relationship")
ASSOCIATION_INTENT = ("关联", "是否有关", "独立性", "association")


@dataclass
class Workspace:
    df: pd.DataFrame
    question: str
    profile: DatasetProfile | None = None
    issues: list = field(default_factory=list)
    tests: list = field(default_factory=list)
    model: ml.ModelResult | None = None
    cluster: ml.ClusterResult | None = None
    chart_files: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    report_md: str = ""
    steps: list = field(default_factory=list)
    dropped_dupes: int = 0


def log(ws: Workspace, tool: str, summary: str, ok: bool = True, tool_args: dict | None = None) -> None:
    ws.steps.append(StepLog(tool=tool, args=tool_args or {}, summary=summary, ok=ok))


def _ensure_profile(ws: Workspace) -> DatasetProfile:
    if ws.profile is None:
        tool_profile(ws)
    return ws.profile


def tool_profile(ws: Workspace) -> str:
    ws.profile = profile_dataframe(ws.df)
    summary = (
        f"{ws.profile.n_rows} rows x {ws.profile.n_cols} cols, "
        f"{ws.profile.dup_rows} duplicate rows, {ws.profile.missing_cells_pct}% missing"
    )
    log(ws, "profile_data", summary)
    return summary


def tool_quality(ws: Workspace) -> str:
    profile = _ensure_profile(ws)
    ws.issues = quality.quality_report(ws.df, profile)
    summary = f"{len(ws.issues)} quality issues found"
    log(ws, "check_quality", summary)
    return summary


def tool_clean(ws: Workspace) -> str:
    ws.df, removed = quality.drop_duplicates(ws.df)
    ws.dropped_dupes = removed
    ws.profile = None
    summary = f"removed {removed} duplicate rows"
    log(ws, "clean_data", summary)
    return summary


def tool_group_test(ws: Workspace, value_col: str | None = None, group_col: str | None = None) -> str:
    profile = _ensure_profile(ws)
    numeric = profile.role_columns("numeric")
    categorical = profile.role_columns("categorical")
    mentioned_num = [c for c in numeric if str(c) in ws.question]
    value = value_col or (mentioned_num[0] if mentioned_num else None) \
        or viz._preferred_numeric(numeric) or (numeric[0] if numeric else None)
    if value is None or not categorical:
        summary = "no numeric/categorical pair available for group test"
        log(ws, "group_test", summary, ok=False)
        return summary
    groups = [group_col] if group_col else categorical[:3]
    ran = []
    for g in groups:
        if g == value:
            continue
        col = profile.column(g)
        if col is None or col.n_unique < 2 or col.n_unique > 8:
            continue
        try:
            result = stats.compare_groups(ws.df, value, g)
        except ValueError as exc:
            log(ws, "group_test", str(exc), ok=False, tool_args={"value_col": value, "group_col": g})
            continue
        ws.tests.append(result)
        ran.append(g)
        log(ws, "group_test", f"{value} by {g}: {result.test}, p={result.p_value:.4g}",
            tool_args={"value_col": value, "group_col": g})
        ws.findings.append(
            f"{value} 在不同 {g} 分组间{'存在' if result.significant else '不存在'}显著差异"
            f"（{result.test}，p={result.p_value:.4g}）"
        )
    return f"group tests on {value} across {ran or 'no valid groups'}"


def tool_correlations(ws: Workspace, max_pairs: int = 4) -> str:
    profile = _ensure_profile(ws)
    numeric = [c for c in profile.role_columns("numeric") if profile.column(c).n_unique > 2]
    pairs = list(itertools.combinations(numeric, 2))[:max_pairs]
    results = []
    for a, b in pairs:
        try:
            result = stats.correlation(ws.df, a, b)
        except ValueError:
            continue
        ws.tests.append(result)
        results.append((a, b, result))
        log(ws, "correlation", f"{a} ~ {b}: {result.test}, r={result.statistic:.4g}, p={result.p_value:.4g}")
    for a, b, result in sorted(results, key=lambda t: abs(t[2].effect or 0), reverse=True)[:2]:
        strength = abs(result.effect or 0)
        if strength >= 0.2:
            label = "强" if strength >= 0.7 else "中等" if strength >= 0.4 else "弱"
            ws.findings.append(
                f"{a} 与 {b} 呈{label}{result.effect_name}相关"
                f"（r={result.effect}，p={result.p_value:.4g}）"
            )
    return f"{len(results)} correlation tests"


def detect_target(ws: Workspace) -> str | None:
    mentioned = [c for c in ws.df.columns if str(c) in ws.question]
    for col in mentioned:
        col_profile = ws.profile.column(col) if ws.profile else None
        if col_profile and col_profile.role in ("numeric", "categorical"):
            return col
    numeric = ws.profile.role_columns("numeric") if ws.profile else []
    return viz._preferred_numeric(numeric)


def tool_model(ws: Workspace, target: str | None = None) -> str:
    _ensure_profile(ws)
    target = target or detect_target(ws)
    if target is None or target not in ws.df.columns:
        summary = f"no plausible target column (tried {target!r})"
        log(ws, "build_model", summary, ok=False)
        return summary
    try:
        ws.model = ml.auto_model(ws.df, target)
    except Exception as exc:
        summary = f"model failed: {exc}"
        log(ws, "build_model", summary, ok=False, tool_args={"target": target})
        return summary
    log(ws, "build_model", f"{ws.model.task} model for {target}: {ws.model.metrics}",
        tool_args={"target": target})
    ws.findings.append(
        f"以 {target} 为目标建立随机森林基线模型（{ws.model.task}），"
        f"测试集指标 {ws.model.metrics}，交叉验证均值 {ws.model.cv['mean']}"
    )
    return f"{ws.model.task} model for {target}"


def tool_cluster(ws: Workspace) -> str:
    profile = _ensure_profile(ws)
    numeric = profile.role_columns("numeric")
    if len(numeric) < 2:
        summary = "need 2+ numeric columns for clustering"
        log(ws, "run_cluster", summary, ok=False)
        return summary
    try:
        ws.cluster = ml.kmeans_profile(ws.df, numeric)
    except Exception as exc:
        summary = f"clustering failed: {exc}"
        log(ws, "run_cluster", summary, ok=False)
        return summary
    log(ws, "run_cluster", f"k={ws.cluster.k}, silhouette={ws.cluster.silhouette}")
    ws.findings.append(f"KMeans 聚类最优 k={ws.cluster.k}（silhouette={ws.cluster.silhouette}），各簇规模 {ws.cluster.sizes}")
    return f"clustered into k={ws.cluster.k}"


def tool_association(ws: Workspace, col_a: str | None = None, col_b: str | None = None) -> str:
    profile = _ensure_profile(ws)
    categorical = profile.role_columns("categorical")
    if col_a and col_b:
        pairs = [(col_a, col_b)]
    else:
        pairs = list(itertools.combinations(categorical[:3], 2))[:2]
    for a, b in pairs:
        try:
            result = stats.categorical_association(ws.df, a, b)
        except ValueError as exc:
            log(ws, "association_test", str(exc), ok=False, tool_args={"col_a": a, "col_b": b})
            continue
        ws.tests.append(result)
        log(ws, "association_test", f"{a} x {b}: chi2, p={result.p_value:.4g}")
        ws.findings.append(
            f"{a} 与 {b} {'存在' if result.significant else '不存在'}显著关联"
            f"（chi-square，p={result.p_value:.4g}，cramers_v={result.effect}）"
        )
    return f"{len(pairs)} association tests"


def tool_plot(ws: Workspace, question: str | None = None, artifacts_dir: str = "./artifacts") -> str:
    profile = _ensure_profile(ws)
    spec = viz.choose_chart(question or ws.question, ws.df, profile)
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = spec.filename(len(ws.chart_files) + 1)
    viz.render_chart(spec, ws.df, out_dir / filename)
    ws.chart_files.append(filename)
    log(ws, "make_plot", f"{spec.kind} chart saved: {filename} ({spec.reason})")
    return f"saved {filename}"


def finalize(ws: Workspace) -> str:
    profile = _ensure_profile(ws)
    if not ws.issues:
        tool_quality(ws)
    ws.report_md = report.build_report(
        question=ws.question,
        df=ws.df,
        profile=profile,
        issues=ws.issues,
        tests=ws.tests,
        findings=list(dict.fromkeys(ws.findings)),
        model=ws.model,
        cluster=ws.cluster,
        chart_files=ws.chart_files,
    )
    log(ws, "build_report", f"report generated, {len(ws.report_md)} chars")
    return ws.report_md


def run_tool(ws: Workspace, name: str, tool_args: dict, artifacts_dir: str = "./artifacts") -> bool:
    if name == "profile_data":
        fn = lambda: tool_profile(ws)
    elif name in ("group_test", "run_group_test"):
        fn = lambda: tool_group_test(ws, tool_args.get("value_col"), tool_args.get("group_col"))
    elif name == "check_quality":
        fn = lambda: tool_quality(ws)
    elif name == "clean_data":
        fn = lambda: tool_clean(ws)
    elif name == "run_correlations":
        fn = lambda: tool_correlations(ws)
    elif name == "association_test":
        fn = lambda: tool_association(ws, tool_args.get("col_a"), tool_args.get("col_b"))
    elif name == "build_model":
        fn = lambda: tool_model(ws, tool_args.get("target"))
    elif name == "run_cluster":
        fn = lambda: tool_cluster(ws)
    elif name == "make_plot":
        fn = lambda: tool_plot(ws, tool_args.get("question"), artifacts_dir)
    else:
        log(ws, name, "unknown tool", ok=False)
        return False
    try:
        fn()
        return True
    except Exception as exc:
        log(ws, name, f"error: {exc}", ok=False, tool_args=dict(tool_args))
        return False


TOOL_SPECS = {
    "profile_data": "数据概览：行列数、重复行、缺失率、每列推断角色",
    "check_quality": "数据质量：缺失/重复/异常值/常量列问题清单",
    "clean_data": "去重：删除完全重复行",
    "run_group_test": "分组差异检验（args: value_col 数值列, group_col 分组列；可自动选择）",
    "run_correlations": "数值列两两相关分析",
    "association_test": "类别列卡方关联检验（args: col_a, col_b 可选）",
    "build_model": "随机森林基线建模（args: target 目标列，可自动推断）",
    "run_cluster": "KMeans 聚类（自动选 k）",
    "make_plot": "自动选图并保存（args: question 可选，影响图表选择）",
    "finish": "结束并生成报告",
}
