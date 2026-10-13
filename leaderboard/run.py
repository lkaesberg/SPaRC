"""Evaluate one model on SPaRC with the Inspect task and add a leaderboard row.

Examples:

    python leaderboard/run.py --model openrouter/google/gemini-3.8-flash --name "Gemini 3.8 Flash" --limit 10
    python leaderboard/run.py --from-log logs/2026-10-13T10-44-01_sparc_xyz.eval --name "Gemma-3 27B"

The row is appended to leaderboard/results.json with source "rerun" and today's
date. All numbers are computed from the per-sample scores in the Inspect log.
"""

import argparse
import datetime
import json
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, read_eval_log

from sparc.tasks import DATASET_REVISION, sparc

RESULTS = Path(__file__).with_name("results.json")


def pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 1) if den else None


def record_from_log(log: EvalLog, name: str) -> dict:
    """Summarise an Inspect log of the `sparc` task as one leaderboard record."""
    if log.samples is None:
        log = read_eval_log(log.location)
    per_level: dict[int, list[int]] = {}
    solved = invalid = errors = 0
    for sample in log.samples or []:
        level = sample.metadata["difficulty"]
        score = (sample.scores or {}).get("sparc_scorer")
        if score is None:  # the sample errored; count it as unsolved and invalid
            errors += 1
            ok, bad = False, True
        else:
            ok = score.value == "C"
            bad = bool((score.metadata or {}).get("invalid_path", True))
        per_level.setdefault(level, [0, 0])
        per_level[level][0] += ok
        per_level[level][1] += 1
        solved += ok
        invalid += bad
    n = sum(total for _, total in per_level.values())
    hard_solved, hard_n = per_level.get(5, [0, 0])
    config = log.eval.model_generate_config
    return {
        "model": name,
        "model_id": log.eval.model,
        "date": log.eval.created[:10],
        "setting": "single-shot",
        "accuracy_all": pct(solved, n),
        "accuracy_hard": pct(hard_solved, hard_n),
        "invalid_path_rate": pct(invalid, n),
        "accuracy_by_level": {str(k): pct(*per_level[k]) for k in sorted(per_level)},
        "n": n,
        "n_hard": hard_n,
        "errors": errors,
        "source": "rerun",
        "reasoning_effort": config.reasoning_effort,
        "max_tokens": config.max_tokens,
        "inspect_task": f"sparc (version {log.eval.task_version})",
        "dataset_revision": DATASET_REVISION,
        "log": Path(log.location).name,
    }


def append(record: dict, results: Path) -> None:
    rows = json.loads(results.read_text()) if results.exists() else []
    rows.append(record)
    results.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="Inspect model id, e.g. openrouter/google/gemini-3.8-flash")
    parser.add_argument("--name", help="Display name on the leaderboard (default: the model id)")
    parser.add_argument("--limit", type=int, help="Only run the first N puzzles (balanced across levels)")
    parser.add_argument("--reasoning-effort", default="medium", help="Reasoning effort (paper: medium)")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--max-connections", type=int, default=20)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--from-log", type=Path, help="Add a row from an existing log instead of running")
    args = parser.parse_args()

    if args.from_log:
        log = read_eval_log(str(args.from_log))
    else:
        if not args.model:
            parser.error("--model is required unless --from-log is given")
        [log] = inspect_eval(
            sparc(),
            model=args.model,
            limit=args.limit,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            max_connections=args.max_connections,
            log_dir=args.log_dir,
        )
        if log.status != "success":
            raise SystemExit(f"Eval did not finish ({log.status}); no row added. Log: {log.location}")

    record = record_from_log(log, args.name or log.eval.model)
    record["date"] = datetime.date.today().isoformat() if not args.from_log else record["date"]
    append(record, args.results)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
