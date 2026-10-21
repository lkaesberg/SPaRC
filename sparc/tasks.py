"""SPaRC as an Inspect AI task.

The prompt is the single-shot prompt (`sparc.prompts.single_shot`) and scoring
uses the existing validator (`sparc.validation`): a puzzle is solved only if the
extracted path exactly matches one of its stored solutions.

Version 2 states the star, polyshape and negative polyshape rules as the stored
solutions follow them. `-T prompt=paper` uses the paper's original wording
(`sparc.prompts.single_shot_paper`), which version 1 used.

Run from a checkout:

    inspect eval sparc/tasks.py@sparc --model openai/gpt-4.1-mini
    inspect eval sparc/tasks.py@sparc_hard --model openai/gpt-4.1-mini
    inspect eval sparc/tasks.py@sparc -T subset=stars -T prompt=paper --model openai/gpt-4.1-mini
"""

from functools import partial
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

from sparc.prompts import single_shot, single_shot_paper
from sparc.validation import analyze_path, extract_solution_path, validate_solution

DATASET_PATH = "lkaesberg/SPaRC"
# Pinned Hugging Face commit of the dataset (2026-08-02).
DATASET_REVISION = "e2d908363190e9a244f931df2654a501facc90a7"

# Fields the prompt builder and validator read from a puzzle record.
PUZZLE_FIELDS = ("id", "grid_size", "puzzle_array", "polyshapes", "solutions")

# Dataset configs. `all` is the benchmark: 500 train and 500 test puzzles that mix all
# rule types. The other configs are separate puzzles with only some rule types (used for
# the paper's per-rule results); none of them appear in `all`.
SUBSETS = (
    "all",
    "dots",
    "dots_stars_polys",
    "gaps",
    "gaps_dots_triangles",
    "polys",
    "polys_ylops",
    "stars",
    "stones",
    "stones_stars",
    "triangles",
)

PROMPTS = {"default": single_shot.get_prompt, "paper": single_shot_paper.get_prompt}


def _path_text(path: list[dict[str, int]]) -> str:
    return ", ".join(f"({p['x']},{p['y']})" for p in path)


def record_to_sample(
    record: dict[str, Any],
    prompt: str = "default",
    system_prompt: str | None = None,
    prompt_template: str | None = None,
) -> Sample:
    """Convert one SPaRC puzzle into a Sample.

    `prompt_template` replaces the user message; it is filled with the fields of
    `sparc.prompts.single_shot.prompt_fields` ({width}, {height}, {start}, {end},
    {grid}, {polyshapes}). `system_prompt` replaces the system message.
    """
    messages = PROMPTS[prompt](record)
    user = messages["user"]
    if prompt_template is not None:
        user = prompt_template.format(**single_shot.prompt_fields(record))
    return Sample(
        id=record["id"],
        input=[
            ChatMessageSystem(content=messages["system"] if system_prompt is None else system_prompt),
            ChatMessageUser(content=user),
        ],
        target=_path_text(record["solutions"][0]["path"]),
        metadata={
            "difficulty": int(record["difficulty_level"]),
            "difficulty_score": float(record["difficulty_score"]),
            "solution_count": int(record["solution_count"]),
            "puzzle": {k: record[k] for k in PUZZLE_FIELDS},
        },
    )


def interleave_by_difficulty(samples: list[Sample]) -> list[Sample]:
    """Order samples round-robin over difficulty levels 1..5.

    Dataset order is kept within each level, so `--limit n` always covers all
    levels evenly and the full run is unaffected.
    """
    by_level: dict[int, list[Sample]] = {}
    for sample in samples:
        assert sample.metadata is not None
        by_level.setdefault(sample.metadata["difficulty"], []).append(sample)
    queues = [by_level[level] for level in sorted(by_level)]
    longest = max(len(q) for q in queues)
    return [q[i] for i in range(longest) for q in queues if i < len(q)]


def sparc_dataset(
    split: str = "test",
    difficulty_levels: list[int] | None = None,
    subset: str = "all",
    prompt: str = "default",
    system_prompt: str | None = None,
    prompt_template: str | None = None,
) -> MemoryDataset:
    """Load one SPaRC config (default `all`) at the pinned revision."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    if prompt not in PROMPTS:
        raise ValueError(f"prompt must be one of {tuple(PROMPTS)}, got {prompt!r}")
    dataset = hf_dataset(
        DATASET_PATH,
        name=subset,
        split=split,
        revision=DATASET_REVISION,
        sample_fields=partial(
            record_to_sample,
            prompt=prompt,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
        ),
    )
    samples = list(dataset)
    if difficulty_levels is not None:
        levels = set(difficulty_levels)
        samples = [s for s in samples if s.metadata and s.metadata["difficulty"] in levels]
    name = "sparc" + ("" if subset == "all" else f"-{subset}") + f"-{split}" + (
        "-level" + "".join(map(str, sorted(difficulty_levels))) if difficulty_levels else ""
    )
    return MemoryDataset(interleave_by_difficulty(samples), name=name, location=DATASET_PATH)


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
def sparc(
    difficulty_levels: list[int] | None = None,
    split: str = "test",
    subset: str = "all",
    prompt: str = "default",
    system_prompt: str | None = None,
    prompt_template: str | None = None,
) -> Task:
    """SPaRC test split (500 puzzles), accuracy overall and per difficulty level.

    Args:
        difficulty_levels: Only keep puzzles of these levels (1-5); default all.
        split: Dataset split ("test" is the benchmark; "train" is for development).
        subset: Dataset config. "all" (default) is the benchmark; the per-rule configs
            ("dots", "gaps", "stars", ...) are separate puzzles, see SUBSETS.
        prompt: "default" for the corrected rule wording, "paper" for the paper's
            original prompt (task version 1).
        system_prompt: Replaces the system message.
        prompt_template: Replaces the user message; filled with {width}, {height},
            {start}, {end}, {grid} and {polyshapes}.
    """
    return Task(
        dataset=sparc_dataset(split, difficulty_levels, subset, prompt, system_prompt, prompt_template),
        solver=generate(),
        scorer=sparc_scorer(),
        metadata={"subset": subset, "prompt": prompt},
        version="2-A",
    )


@task
def sparc_hard(split: str = "test", prompt: str = "default") -> Task:
    """The paper's "hard" subset: the 89 difficulty-level-5 test puzzles."""
    return sparc(difficulty_levels=[5], split=split, prompt=prompt)
