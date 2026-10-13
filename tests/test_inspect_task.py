"""Tests for the Inspect task in sparc/tasks.py (offline model: mockllm)."""

import pytest
from datasets import load_dataset
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from sparc.prompts.single_shot import get_prompt
from sparc.tasks import (
    DATASET_PATH,
    DATASET_REVISION,
    record_to_sample,
    sparc,
    sparc_hard,
    sparc_scorer,
)

pytestmark = [pytest.mark.dataset_download, pytest.mark.huggingface]


@pytest.fixture(scope="module")
def records():
    return load_dataset(DATASET_PATH, "all", split="test", revision=DATASET_REVISION)


def _gold_answer(record) -> str:
    path = record["solutions"][0]["path"]
    return "Reasoning ...\n#### " + ", ".join(f"({p['x']}, {p['y']})" for p in path)


def _metrics(log):
    return {name: m.value for s in log.results.scores for name, m in s.metrics.items()}


def test_record_to_sample(records):
    record = records[0]
    sample = record_to_sample(record)
    prompt = get_prompt(record)
    assert sample.id == record["id"]
    assert [m.role for m in sample.input] == ["system", "user"]
    assert sample.input[0].content == prompt["system"]
    assert sample.input[1].content == prompt["user"]
    assert sample.metadata["difficulty"] == record["difficulty_level"]
    assert sample.metadata["puzzle"]["solutions"] == record["solutions"]


def test_dataset_is_interleaved_by_difficulty():
    task = sparc()
    levels = [s.metadata["difficulty"] for s in task.dataset]
    assert len(levels) == 500
    assert levels[:10] == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]


def test_hard_variant_is_level_5_only():
    task = sparc_hard()
    levels = {s.metadata["difficulty"] for s in task.dataset}
    assert len(task.dataset) == 89
    assert levels == {5}


def test_e2e_default_mock_output_is_incorrect():
    [log] = eval(sparc(), model="mockllm/model", limit=5, display="none")
    assert log.status == "success"
    metrics = _metrics(log)
    assert metrics["all"] == 0.0
    assert metrics["invalid_path_rate"] == 1.0


def test_e2e_gold_answer_is_correct(records):
    record = records[0]
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content(model="mockllm/model", content=_gold_answer(record))],
    )
    [log] = eval(sparc(), model=model, sample_id=record["id"], display="none")
    assert log.status == "success"
    metrics = _metrics(log)
    assert metrics["all"] == 1.0
    assert metrics[f"level_{record['difficulty_level']}"] == 1.0
    assert metrics["invalid_path_rate"] == 0.0


def test_sparc_scorer_is_the_task_scorer():
    assert sparc().scorer[0].__qualname__ == sparc_scorer().__qualname__
