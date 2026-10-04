"""Analysis benchmark runner.

Each task in tasks.jsonl runs the full agent pipeline against a synthetic
dataset with planted statistical effects, then checks:
  - the report contains expected substrings,
  - the chosen statistical test family matches the expectation,
  - significance matches the planted ground truth.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent import run_analysis  # noqa: E402
from agent.synth import GENERATORS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
TASKS_PATH = Path(__file__).resolve().parent / "tasks.jsonl"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "benchmarks"


def ensure_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, generator in GENERATORS.items():
        path = DATA_DIR / filename
        if not path.exists():
            generator().to_csv(path, index=False)


def load_tasks() -> list:
    return [json.loads(line) for line in TASKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_task(task: dict, artifacts_dir) -> tuple:
    ensure_data()
    df = pd.read_csv(DATA_DIR / task["dataset"])
    ws = asyncio.run(run_analysis(df, task["question"], artifacts_dir=str(artifacts_dir)))
    expect = task.get("expect", {})
    problems = []

    for needle in expect.get("report_contains", []):
        if needle not in ws.report_md:
            problems.append(f"report missing {needle!r}")

    test_any = expect.get("test_any")
    if test_any:
        matched = [t for t in ws.tests if any(name in t.test.lower() for name in test_any)]
        if not matched:
            problems.append(f"no test from {test_any} (got {[t.test for t in ws.tests]})")
        elif expect.get("significant") is not None:
            if not any(t.significant for t in matched):
                problems.append(f"expected significant result for {test_any}")

    return (not problems), "; ".join(problems)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0, help="run only the first N tasks (0 = all)")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any task fails")
    args = parser.parse_args()

    tasks = load_tasks()
    if args.subset:
        tasks = tasks[: args.subset]

    passed = 0
    failures = []
    for task in tasks:
        ok, detail = run_task(task, ARTIFACTS_DIR)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {task['id']}: {task['question']}" + (f" — {detail}" if detail else ""))
        if ok:
            passed += 1
        else:
            failures.append(task["id"])

    print(f"\nbenchmark: {passed}/{len(tasks)} tasks passed")
    if failures:
        print("failed:", ", ".join(failures))
    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
