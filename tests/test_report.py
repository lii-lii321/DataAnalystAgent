import numpy as np
import pandas as pd

from agent.report import SECTIONS, build_report
from agent.stats import TestResult


def test_report_contains_all_sections():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
    profile = type(
        "P",
        (),
        {
            "n_rows": 4,
            "n_cols": 2,
            "dup_rows": 0,
            "missing_cells_pct": 0.0,
            "memory_mb": 0.0,
            "columns": [],
            "role_columns": lambda self, role: ["x"],
        },
    )()
    tests = [TestResult(test="independent t-test", statistic=2.5, p_value=0.01, effect=0.8, effect_name="cohen_d")]
    report = build_report(
        question="测试问题",
        df=df,
        profile=profile,
        issues=[],
        tests=tests,
        findings=["x 存在显著差异"],
        chart_files=["chart_1.png"],
    )
    for section in SECTIONS:
        assert section in report, section
    assert "independent t-test" in report
    assert "0.01" in report
    assert "chart_1.png" in report
