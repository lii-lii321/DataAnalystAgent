import json
import re

from agent import tools
from agent.llm import LLMProvider, get_provider

MODEL_INTENT = ("预测", "predict", "model", "建模", "影响因素", "哪些因素")
CLUSTER_INTENT = ("聚类", "分群", "cluster", "segment")

SYSTEM_PROMPT = (
    "You are a rigorous data-analysis agent working on a pandas DataFrame.\n"
    "Choose ONE tool per turn. Reply with ONLY a JSON object:\n"
    '{"tool": "<tool_name>", "args": {...}}\n'
    'When the analysis is complete, reply {"tool": "finish"}.\n'
    "Prefer: profile -> quality -> statistical tests -> model (if prediction is asked) -> plot -> finish.\n"
    "Available tools:\n"
)


def run_pipeline(ws: tools.Workspace, artifacts_dir: str) -> None:
    tools.tool_profile(ws)
    tools.tool_quality(ws)
    if ws.profile.dup_rows > 0:
        tools.tool_clean(ws)
        tools.tool_profile(ws)
        tools.tool_quality(ws)

    q = ws.question
    if any(k in q for k in tools.CORRELATION_INTENT) or len(ws.profile.role_columns("numeric")) >= 2:
        tools.tool_correlations(ws)
    tools.tool_group_test(ws)
    if any(k in q for k in tools.ASSOCIATION_INTENT):
        tools.tool_association(ws)
    if any(k in q for k in MODEL_INTENT):
        tools.tool_model(ws)
    if any(k in q for k in CLUSTER_INTENT):
        tools.tool_cluster(ws)

    try:
        tools.tool_plot(ws, artifacts_dir=artifacts_dir)
    except (ValueError, KeyError):
        pass
    if ws.tests:
        group_logs = [s for s in ws.steps if s.tool == "group_test" and s.ok and s.args.get("group_col")]
        if group_logs:
            last = group_logs[-1]
            try:
                tools.tool_plot(
                    ws,
                    question=f"对比 {last.args['group_col']} 的 {last.args.get('value_col', '')} 差异",
                    artifacts_dir=artifacts_dir,
                )
            except (ValueError, KeyError):
                pass

    tools.finalize(ws)


async def run_llm_loop(ws: tools.Workspace, provider: LLMProvider, artifacts_dir: str, max_steps: int = 8) -> None:
    tool_lines = "\n".join(f"- {name}: {desc}" for name, desc in tools.TOOL_SPECS.items())
    context = (
        f"Question: {ws.question}\n"
        f"Shape: {ws.df.shape}\n"
        f"Columns: {', '.join(map(str, ws.df.columns))}\n\n"
        "What tool should run next?"
    )
    for _ in range(max_steps):
        raw = await provider.complete(SYSTEM_PROMPT + tool_lines, context)
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            break
        data = json.loads(match.group(0))
        tool = str(data.get("tool", "finish")).strip()
        if tool == "finish":
            break
        tools.run_tool(ws, tool, data.get("args") or {}, artifacts_dir)
    tools.finalize(ws)


async def run_analysis(
    df,
    question: str,
    provider: LLMProvider | None = None,
    artifacts_dir: str = "./artifacts",
) -> tools.Workspace:
    ws = tools.Workspace(df=df.copy(), question=question)
    provider = provider or get_provider()
    if provider.name == "mock":
        run_pipeline(ws, artifacts_dir)
        return ws
    try:
        await run_llm_loop(ws, provider, artifacts_dir)
        return ws
    except Exception:
        fallback = tools.Workspace(df=ws.df, question=question)
        run_pipeline(fallback, artifacts_dir)
        return fallback
