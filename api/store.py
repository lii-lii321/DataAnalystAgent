import time
import uuid
from dataclasses import dataclass, field

import pandas as pd

from agent.profiler import DatasetProfile


@dataclass
class AnalysisRecord:
    question: str
    report: str
    steps: list
    charts: list
    created_at: float = field(default_factory=time.time)


@dataclass
class DatasetRecord:
    id: str
    filename: str
    df: pd.DataFrame
    profile: DatasetProfile
    analyses: dict = field(default_factory=dict)


REGISTRY: dict = {}


def store_dataset(filename: str, df: pd.DataFrame, profile: DatasetProfile) -> DatasetRecord:
    record = DatasetRecord(
        id=uuid.uuid4().hex[:12],
        filename=filename,
        df=df,
        profile=profile,
    )
    REGISTRY[record.id] = record
    return record


def store_analysis(dataset_id: str, analysis: AnalysisRecord) -> str:
    analysis_id = uuid.uuid4().hex[:12]
    REGISTRY[dataset_id].analyses[analysis_id] = analysis
    return analysis_id


def get_dataset(dataset_id: str) -> DatasetRecord | None:
    return REGISTRY.get(dataset_id)


def delete_dataset(dataset_id: str) -> bool:
    return REGISTRY.pop(dataset_id, None) is not None


def list_datasets() -> list[DatasetRecord]:
    return list(REGISTRY.values())


def list_analyses(dataset_id: str) -> list[tuple[str, AnalysisRecord]] | None:
    record = REGISTRY.get(dataset_id)
    if record is None:
        return None
    return sorted(record.analyses.items(), key=lambda kv: kv[1].created_at)


def get_analysis(dataset_id: str, analysis_id: str) -> AnalysisRecord | None:
    record = REGISTRY.get(dataset_id)
    if record is None:
        return None
    return record.analyses.get(analysis_id)
