"""SPaRC as an Inspect AI task.

The prompt is the paper's single-shot prompt (`sparc.prompts.single_shot`) and
scoring uses the existing validator (`sparc.validation`): a puzzle is solved
only if the extracted path exactly matches one of its stored solutions.

Run from a checkout:

    inspect eval sparc/tasks.py@sparc --model openai/gpt-4.1-mini
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Target,
    accuracy,
    grouped,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

from sparc.prompts.single_shot import get_prompt
from sparc.validation import analyze_path, extract_solution_path, validate_solution

DATASET_PATH = "lkaesberg/SPaRC"
# Pinned Hugging Face commit of the dataset (2026-08-02).
DATASET_REVISION = "e2d908363190e9a244f931df2654a501facc90a7"

# Fields the prompt builder and validator read from a puzzle record.
PUZZLE_FIELDS = ("id", "grid_size", "puzzle_array", "polyshapes", "solutions")


def _path_text(path: list[dict[str, int]]) -> str:
    return ", ".join(f"({p['x']},{p['y']})" for p in path)


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert one SPaRC puzzle into a Sample using the paper's prompt."""
    prompt = get_prompt(record)
    return Sample(
        id=record["id"],
        input=[
            ChatMessageSystem(content=prompt["system"]),
            ChatMessageUser(content=prompt["user"]),
        ],
        target=_path_text(record["solutions"][0]["path"]),
        metadata={
            "difficulty": int(record["difficulty_level"]),
            "difficulty_score": float(record["difficulty_score"]),
            "solution_count": int(record["solution_count"]),
            "puzzle": {k: record[k] for k in PUZZLE_FIELDS},
        },
    )


def sparc_dataset(
    split: str = "test", difficulty_levels: list[int] | None = None
) -> MemoryDataset:
    """Load the SPaRC `all` config at the pinned revision."""
    dataset = hf_dataset(
        DATASET_PATH,
        name="all",
        split=split,
        revision=DATASET_REVISION,
        sample_fields=record_to_sample,
    )
    samples = list(dataset)
    if difficulty_levels is not None:
        levels = set(difficulty_levels)
        samples = [s for s in samples if s.metadata and s.metadata["difficulty"] in levels]
    name = f"sparc-{split}" + (
        "-level" + "".join(map(str, sorted(difficulty_levels))) if difficulty_levels else ""
    )
    return MemoryDataset(samples, name=name, location=DATASET_PATH)


@metric
def invalid_path_rate() -> Metric:
    """Share of answers that are not a structurally valid path (paper Table 7, "Invalid Path")."""

    def compute(scores: list[SampleScore]) -> float:
        flags = [bool((s.score.metadata or {}).get("invalid_path")) for s in scores]
        return sum(flags) / len(flags) if flags else 0.0

    return compute


@scorer(
    metrics=[
        grouped(accuracy(), "difficulty", all="samples", name_template="level_{group_name}"),
        stderr(),
        invalid_path_rate(),
    ]
)
def sparc_scorer():
    """Exact match of the extracted path against the stored solutions."""

    async def score(state: TaskState, target: Target) -> Score:
        assert state.metadata is not None
        puzzle = state.metadata["puzzle"]
        completion = state.output.completion
        path = extract_solution_path(completion, puzzle)
        solved = validate_solution(path, puzzle)
        analysis = analyze_path(path, puzzle)
        return Score(
            value=CORRECT if solved else INCORRECT,
            answer=_path_text(path) if path else "",
            explanation=completion[-2000:],
            metadata={"invalid_path": not analysis["fully_valid_path"], "analysis": analysis},
        )

    return score


@task
def sparc(difficulty_levels: list[int] | None = None, split: str = "test") -> Task:
    """SPaRC test split (500 puzzles), accuracy overall and per difficulty level.

    Args:
        difficulty_levels: Only keep puzzles of these levels (1-5); default all.
        split: Dataset split ("test" is the benchmark; "train" is for development).
    """
    return Task(
        dataset=sparc_dataset(split, difficulty_levels),
        solver=generate(),
        scorer=sparc_scorer(),
        version="1-A",
    )

