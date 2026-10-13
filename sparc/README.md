# SPaRC for Inspect AI

This folder contains SPaRC as an [Inspect AI](https://inspect.aisi.org.uk/) task (`tasks.py`). It uses the prompt
from the paper and scores answers with the SPaRC validator.

SPaRC has 1,000 2D grid pathfinding puzzles, and the test split has 500 of them. The model must draw a line from
the start node to the end node that satisfies every rule on the grid. Humans solve 98.0% of the puzzles (94.5%
of the hard ones). o4-mini, the best model in the paper, solves 15.8% (1.1% of the hard ones), and it produces
invalid paths on more than 50% of the puzzles.

## Install

```bash
git clone https://github.com/lkaesberg/SPaRC.git
cd SPaRC
pip install -e .
```

## Run

```bash
# full test split (500 puzzles)
inspect eval sparc --model openai/gpt-4.1-mini

# only the hard puzzles (difficulty level 5, 89 puzzles)
inspect eval sparc_hard --model openai/gpt-4.1-mini

# a quick check: puzzles are ordered round-robin by level, so --limit 50 gives 10 per level
inspect eval sparc --model openai/gpt-4.1-mini --limit 50

# chosen levels only
inspect eval sparc --model openai/gpt-4.1-mini -T 'difficulty_levels=[1,2]'
```

From a checkout you can also name the file: `inspect eval sparc/tasks.py@sparc`.

| Task | Puzzles | Description |
|---|---|---|
| `sparc` | 500 | The full test split. |
| `sparc_hard` | 89 | Only difficulty level 5, which the paper calls "hard". |

Task parameters of `sparc`: `difficulty_levels` (list of levels 1 to 5, default all) and `split` (`test` by
default; `train` is meant for development).

## Dataset

The task loads `lkaesberg/SPaRC` (config `all`, split `test`) from Hugging Face, pinned to revision
`e2d908363190e9a244f931df2654a501facc90a7`. Each sample stores the puzzle and its difficulty level
(`metadata["difficulty"]`), so results can be grouped by level.

## Scoring

The model first explains its reasoning. It then writes `####` and the path as a list of `(x, y)` coordinates.
The scorer takes the coordinates after the last `####` (`sparc.validation.extract_solution_path`). A puzzle counts
as solved only if this path exactly matches one of the puzzle's stored solutions
(`sparc.validation.validate_solution`). The dataset stores every valid solution.

Reported metrics:

* `all`: accuracy over all puzzles, plus `level_1` to `level_5`: accuracy per difficulty level.
* `stderr`: standard error of the overall accuracy.
* `invalid_path_rate`: the share of answers that are not a valid line from start to end. An answer is invalid if
  it has the wrong start or end, is disconnected, crosses itself, or passes through a rule cell
  (`sparc.validation.analyze_path`). This is the "Invalid Path" column of Table 7 in the paper.

We run one generation per puzzle (1 epoch), as in the paper. The task sets no temperature, so each provider
uses its default. The paper does not state a temperature, and the `sparc` CLI defaults to 1.0.

## Evaluation report

We ran the full test split (500 puzzles) on 2026-10-13 with two models from Table 6 of the paper. The
reproduced numbers are next to the paper's numbers, and the paper's numbers are unchanged.

| Model (Inspect id) | Accuracy, paper | Accuracy, reproduced | Invalid path, paper | Invalid path, reproduced |
|---|---|---|---|---|
| Gemma-3 27B (`openrouter/google/gemma-3-27b-it`) | 1.2% | 0.4% (2/500) | 88.0% | 81.2% |
| Qwen 2.5 72B (`openrouter/qwen/qwen-2.5-72b-instruct`) | 0.4% | 0.4% (2/500) | 90.6% | 95.4% |

Both accuracies are within 2 percentage points of the paper. The invalid-path rates differ by 5 to 7 points,
which is plausible because the providers and sampling settings differ from the paper's runs. Both models
solve almost no puzzles, so these runs mainly check that prompting, answer extraction and scoring work end to
end. We could not rerun o4-mini because our API account does not allow it.

Run details: `inspect_ai` 0.3.279, task version `1-A`, 1 epoch, no generation settings changed, dataset revision
as above. Command:

```bash
inspect eval sparc --model openrouter/google/gemma-3-27b-it --max-connections 25
```

### Newer models

These models are not in the paper, so there is no paper number to compare with. They were run on 2026-10-13
with the same task, `--reasoning-effort medium` (the paper's setting for reasoning models) and
`--max-tokens 32000`. Two models were run on a 100-puzzle subset (20 per level) to stay within budget, so their
"hard" value rests on only 20 puzzles.

| Model (Inspect id) | Puzzles | Accuracy | Hard (level 5) | Invalid path |
|---|---|---|---|---|
| GPT-6.1 Sol (`openrouter/openai/gpt-6.1-sol`) | 500 | 83.6% | 49.4% | 7.2% |
| Gemini 3.8 Flash (`openrouter/google/gemini-3.8-flash`) | 100 | 80.0% | 30.0% | 15.0% |
| DeepSeek V4.1 Flash (`openrouter/deepseek/deepseek-v4.1-flash`) | 500 | 41.8% | 4.5% | 39.4% |
| Claude Sonnet 5.5 (`openrouter/anthropic/claude-sonnet-5.5`) | 100 | 34.0% | 10.0% | 16.0% |

For comparison, humans solve 98.0% (94.5% of the hard puzzles), and o4-mini solved 15.8% in the paper. The full
table, with all paper rows, is on the [leaderboard](https://sparc.gipplab.org/#leaderboard).

## Citation

```bibtex
@inproceedings{kaesberg-etal-2025-sparc,
  title     = "{SP}a{RC}: A Spatial Pathfinding Reasoning Challenge",
  author    = "Kaesberg, Lars Benedikt and Wahle, Jan Philip and Ruas, Terry and Gipp, Bela",
  booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
  month     = nov,
  year      = "2025",
  address   = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2025.emnlp-main.526/",
  doi       = "10.18653/v1/2025.emnlp-main.526",
  pages     = "10359--10390"
}
```
