from dataclasses import dataclass

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agent.profiler import DatasetProfile  # noqa: E402

NAVY = "#1a365d"
BLUE = "#2563eb"
SLATE = "#334155"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

GROUP_INTENT = ("对比", "比较", "差异", "compare", "difference", " by ", "分组")
DISTRIBUTION_INTENT = ("分布", "distribution", "histogram", "直方图")
RELATION_INTENT = ("关系", "相关", "correlation", "relationship", " vs ", "与")
TREND_INTENT = ("趋势", "trend", "over time", "随时间")
COUNT_INTENT = ("数量", "count", "多少", "占比", "proportion")

PREFERRED_TARGETS = ("income", "revenue", "price", "score", "churn", "sales", "salary", "amount")


KIND_NAMES = {
    "hist": "histogram",
    "box": "boxplot",
    "scatter": "scatter",
    "bar": "bar",
    "line": "line",
}


@dataclass
class ChartSpec:
    kind: str
    x: str | None = None
    y: str | None = None
    reason: str = ""

    def filename(self, index: int) -> str:
        parts = [f"chart_{index}", KIND_NAMES.get(self.kind, self.kind)]
        for col in (self.x, self.y):
            if col:
                parts.append("".join(ch if ch.isalnum() else "_" for ch in col))
        return "_".join(parts) + ".png"


def _mentioned(question: str, columns: list) -> list:
    return [c for c in columns if str(c) in question]


def _preferred_numeric(numeric: list) -> str | None:
    for name in PREFERRED_TARGETS:
        for col in numeric:
            if name in col.lower():
                return col
    return None


def choose_chart(question: str, df: pd.DataFrame, profile: DatasetProfile) -> ChartSpec:
    q = question.lower()
    numeric = [c for c in profile.role_columns("numeric") if c in df.columns]
    categorical = [c for c in profile.role_columns("categorical") if c in df.columns]
    datetimes = [c for c in profile.role_columns("datetime") if c in df.columns]
    mentioned = _mentioned(question, list(df.columns))
    mentioned_num = [c for c in mentioned if c in numeric]
    mentioned_cat = [c for c in mentioned if c in categorical]

    if datetimes and numeric and any(k in q for k in TREND_INTENT):
        return ChartSpec("line", x=datetimes[0], y=mentioned_num[0] if mentioned_num else numeric[0],
                         reason="时间维度 + 趋势类问题 → 折线图")
    if any(k in q for k in DISTRIBUTION_INTENT) and numeric:
        target = mentioned_num[0] if mentioned_num else _preferred_numeric(numeric) or numeric[0]
        return ChartSpec("hist", x=target, reason=f"分布类问题 → 直方图（{target}）")
    if any(k in q for k in GROUP_INTENT) and categorical and numeric:
        group = mentioned_cat[0] if mentioned_cat else categorical[0]
        value = mentioned_num[0] if mentioned_num else _preferred_numeric(numeric) or numeric[0]
        return ChartSpec("box", x=group, y=value, reason=f"分组对比 → 箱线图（{value} by {group}）")
    if any(k in q for k in RELATION_INTENT) and len(numeric) >= 2:
        x = mentioned_num[0] if mentioned_num else numeric[0]
        y = next((c for c in numeric if c != x), None)
        return ChartSpec("scatter", x=x, y=y, reason="两变量关系 → 散点图")
    if categorical and (any(k in q for k in COUNT_INTENT) or not numeric):
        col = mentioned_cat[0] if mentioned_cat else categorical[0]
        return ChartSpec("bar", x=col, reason=f"类别计数 → 柱状图（{col}）")
    if numeric:
        target = mentioned_num[0] if mentioned_num else _preferred_numeric(numeric) or numeric[0]
        return ChartSpec("hist", x=target, reason=f"默认展示数值分布（{target}）")
    raise ValueError("no plottable columns found")


def render_chart(spec: ChartSpec, df: pd.DataFrame, out_path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)

    if spec.kind == "hist":
        values = pd.to_numeric(df[spec.x], errors="coerce").dropna()
        ax.hist(values, bins=30, color=BLUE, edgecolor="white")
        ax.set_xlabel(spec.x)
    elif spec.kind == "bar":
        counts = df[spec.x].astype(str).value_counts().head(15)
        ax.bar(counts.index, counts.values, color=BLUE)
        ax.set_xlabel(spec.x)
        ax.set_ylabel("count")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    elif spec.kind == "box":
        data = df[[spec.x, spec.y]].copy()
        data[spec.y] = pd.to_numeric(data[spec.y], errors="coerce")
        data = data.dropna()
        top_levels = data[spec.x].value_counts().head(12).index
        groups = [data.loc[data[spec.x] == level, spec.y] for level in top_levels]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([str(level) for level in top_levels], rotation=30, ha="right")
        ax.set_ylabel(spec.y)
    elif spec.kind == "scatter":
        data = df[[spec.x, spec.y]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(data[spec.x], data[spec.y], s=14, alpha=0.55, color=BLUE)
        ax.set_xlabel(spec.x)
        ax.set_ylabel(spec.y)
    elif spec.kind == "line":
        data = df[[spec.x, spec.y]].copy()
        data[spec.y] = pd.to_numeric(data[spec.y], errors="coerce")
        data = data.dropna().sort_values(spec.x)
        ax.plot(data[spec.x].astype(str), data[spec.y], color=BLUE, linewidth=1.8)
        ax.set_xlabel(spec.x)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.set_title(f"{spec.kind}: {spec.y or spec.x}", color=NAVY, fontsize=12)
    ax.tick_params(colors=SLATE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)
