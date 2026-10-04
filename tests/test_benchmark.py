from benchmarks.runner import load_tasks, run_task


def task_by_id(task_id: str) -> dict:
    return next(t for t in load_tasks() if t["id"] == task_id)


def test_benchmark_gender_gap(tmp_path):
    ok, detail = run_task(task_by_id("income_gender_gap"), tmp_path)
    assert ok, detail


def test_benchmark_data_quality(tmp_path):
    ok, detail = run_task(task_by_id("income_quality"), tmp_path)
    assert ok, detail


def test_benchmark_scores_correlation(tmp_path):
    ok, detail = run_task(task_by_id("scores_corr"), tmp_path)
    assert ok, detail
