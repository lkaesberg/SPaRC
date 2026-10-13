# NOTES: citation and benchmark-listing work (SPaRC repo)

Work log for the SPaRC / Spatial-Gym citation handoff, written on 2026-10-13. The Spatial-Gym repo has a
matching `NOTES.md` that covers Task D.

## Branches

The branches are stacked: each one starts from the one above it, so `NOTES.md` grows along the stack. All are
pushed to `origin`. No PRs were opened.

| Repo | Branch | Base | Content |
|---|---|---|---|
| SPaRC | `task-a-citations` | `main` | Task A: README citation and related-work block |
| SPaRC | `task-b-inspect` | `task-a-citations` | Task B: Inspect task `sparc/tasks.py`, tests, `sparc/README.md` |
| SPaRC | `task-e-leaderboard` | `task-b-inspect` | Task E: `leaderboard/results.json`, `leaderboard/run.py` |
| SPaRCWebsite | `task-e-leaderboard` | `master` | Task E: sortable leaderboard, citation fix |
| lm-evaluation-harness (fork) | `add-sparc-task` | upstream `main` | Task C: `lm_eval/tasks/sparc/` |
| HF dataset (local clone `paper-boost/SPaRC-hf`) | `main`, 1 local commit | | Task A3: dataset card (not pushed, see below) |

## Code map (handoff §3)

### SPaRC (`lkaesberg/SPaRC`)
- **README**: `README.md`.
- **Puzzle data loader**: there is no separate loader module. `main()` in `sparc/sparc.py:745` loads the data
  with `load_dataset("lkaesberg/SPaRC", "all", split="test")`. Each row has `id`, `difficulty_level` (1 to 5),
  `difficulty_score`, `grid_size`, `polyshapes` (a JSON string), `puzzle_array`, `solution_count`, `solutions`
  (each with a `path` of `{x, y}` points) and `text_visualization`.
- **Path validator**: `sparc/validation.py`.
  - `extract_solution_path` (line 5) reads the text after the last `####`, collects the `(x,y)` pairs and stops
    at the end node.
  - `validate_solution` (line 64) is the validator. A path is correct only if it exactly matches one entry of
    `puzzle["solutions"]`. All 500 test rows have `len(solutions) == solution_count` (1 to 48 solutions).
  - `analyze_path` (line 98) checks the path's shape: right start and end, connected, no self-crossing, no
    rule cell. `fully_valid_path == False` is the paper's "Invalid Path" (Table 7).
- **Prompts**: the paper's single-shot prompt is `sparc/prompts/single_shot.py:22` (`get_prompt`). The registry is
  `sparc/prompts/__init__.py`, and `sparc/prompt.py` holds the older wrapper functions.
- **Existing LLM evaluation script**: the `sparc` CLI, `main()` in `sparc/sparc.py`. The per-puzzle logic is in
  `sparc/process_puzzle.py`:
  - `process_puzzle` (line 66) runs single-shot.
  - `process_puzzle_step_by_step` (line 207) runs gym mode: `observation='SPaRC'`, 100 steps, and no chat history
    (each step sends the system prompt and the current state only).
  - The CLI default is `--temperature 1.0`.
  - `sparc/process_puzzle.py:10` imports `SPaRC_Gym` (PyPI `sparc-gym<0.3`), the old name of Spatial-Gym.

### Spatial-Gym (`lkaesberg/Spatial-Gym`)
See the Spatial-Gym `NOTES.md`. In short: the data loader is `Spatial_Gym/Spatial_Gym.py:77`, and the validator
is the exact-match check in `Spatial_Gym.step` (`outcome_reward`, line 1207). That check follows the same rule
as `validate_solution` here.

## Paper facts used (checked against the ACL Anthology PDF and arXiv on 2026-10-13)

- The per-model results table is **Table 6** (Appendix J.1, p. 10381). Table 1 holds dataset counts. The
  invalid-path rates are in **Table 7** (p. 10381), and the human numbers are in Figure 3 and Table 2.
- "Hard" means difficulty level 5 only. Test puzzles per level: L1 86, L2 118, L3 121, L4 86, L5 89.
- The two OpenAI reasoning models used medium reasoning effort. The paper does not state temperature, maximum
  tokens or seeds.
- All headline numbers in the handoff match the paper. The human 98.0% comes from 100 puzzles, each solved by
  3 people (majority vote), not from the 500 test puzzles. The exact o4-mini invalid-path rate is 59.2%.

## Task B: Inspect task

- `sparc/tasks.py` defines `@task sparc` (all 500 test puzzles; `difficulty_levels` parameter) and
  `@task sparc_hard` (the 89 level-5 puzzles). It uses the paper's prompt (`sparc.prompts.single_shot`) and the
  existing validator (`sparc.validation`). Each sample carries `metadata["difficulty"]`, and the scorer reports
  `all`, `level_1` … `level_5`, `stderr` and `invalid_path_rate`.
- Samples are ordered round-robin by level, so `--limit 50` gives 10 puzzles per level.
- The dataset is pinned to HF revision `e2d908363190e9a244f931df2654a501facc90a7`.
- `tests/test_inspect_task.py` (6 tests, all pass) covers `record_to_sample` on a real record, the ordering, the
  hard subset, a `mockllm` run, and a gold answer that must score 1.0.
- `inspect eval sparc --model mockllm/model --limit 5` runs without errors, from the repo root and from other
  folders.

## Task E: leaderboard

- **E1, website source**: `lkaesberg/SPaRCWebsite`, GitHub Pages from the `master` branch root, CNAME
  `sparc.gipplab.org`.
- **E2, `leaderboard/results.json`**: one record per (model, setting) with `model`, `date`, `setting`,
  `accuracy_all`, `accuracy_hard`, `invalid_path_rate`, plus `accuracy_by_level`, `n`, `source` and `reference`.
  The paper rows were parsed directly from the PDF text, not typed in by hand:
  - 8 rows from the SPaRC paper (Tables 6 and 7, and the human row from Figure 3 and Table 2), `source: "paper"`.
  - 24 rows from the Spatial-Gym paper (Tables 2 to 4: single-shot, step-by-step, and step-by-step with
    backtracking), `source: "paper"`.
- **Existing GippLab runs**: 15 rows with `source: "lab"`, added at your request instead of paying for new runs.
  Every number was recomputed from the per-puzzle result files with `validate_solution` and `analyze_path`
  against the pinned test split. The recomputed flags agreed with the stored flags on every puzzle. Included:
  complete 500-puzzle runs (Qwen 3.5 27B has 498) from `sparcVisualPuzzleEvaluation` (text baselines),
  `Sparc-train` (Qwen 3 base models) and `sparc-gym-analysis` (step-by-step, plus the A* and random baselines).
  Left out: fine-tuned checkpoints, ablations (prompt variants, pass@k, image boards), incomplete files and the
  mixed files listed under Discrepancies.
- **E3, `leaderboard/run.py`**: runs the Inspect task `sparc` for one model and appends a row with
  `source: "rerun"` and today's date. `--from-log` adds a row from an existing Inspect log. `--limit 10` was used
  for the cost pilots (rows written to a scratch file).
- **E4, website**: the leaderboard section of `index.html` is now a sortable table with one tab per setting. It
  loads `leaderboard/results.json` from SPaRC `main` on GitHub. If that fails (for example before the merge), it
  uses the copy bundled with the site. The old static table stays in the HTML as a fallback when JavaScript is
  off. Checked in a browser: sorting, tabs, empty cells sorting last, and no sideways scrolling at 375 px width.
  The citation block on the site now has the right pages.

## Reproduced numbers

### Paper-model check (Task B4)
Inspect task `sparc` (version 1-A), full 500-puzzle test split, run on 2026-10-13 through OpenRouter.

| Model | Accuracy, paper | Accuracy, reproduced | Invalid path, paper | Invalid path, reproduced |
|---|---|---|---|---|
| Gemma-3 27B (`openrouter/google/gemma-3-27b-it`) | 1.2% | 0.4% | 88.0% | 81.2% |
| Qwen 2.5 72B (`openrouter/qwen/qwen-2.5-72b-instruct`) | 0.4% | 0.4% | 90.6% | 95.4% |

Both accuracies are within the 2-point limit. Both models are close to 0%, so this check mainly shows that the
prompt, answer extraction and scoring work end to end. o4-mini (paper: 15.8%) could not be rerun: the
OpenRouter account requires zero data retention, and OpenAI's o4-mini endpoint there does not meet that rule.
This is an account setting, and we did not change it.

Logs: `paper-boost/runs/inspect-full/2026-10-13T10-46-24-00-00_sparc_bYgXdrpjp28dgyieWivC6Q.eval` (Gemma) and
`paper-boost/runs/inspect-full/2026-10-13T10-52-12-00-00_sparc_MJr8B44DVMHrQu6m4hhReT.eval` (Qwen).
A `--limit 50` run with Gemma-3 27B gave 4.0% (2/50) and 80% invalid, without errors.

### Newer models (Task E3)
You chose these after the cost pilots. All went through `leaderboard/run.py` (Inspect task `sparc`,
`--reasoning-effort medium`, `--max-tokens 32000`) on 2026-10-13 and are in `leaderboard/results.json` with
`source: "rerun"`. There are no paper numbers for them.

| Model | Puzzles | Accuracy | Hard (L5) | Invalid path | Per level (L1 to L5) |
|---|---|---|---|---|---|
| GPT-6.1 Sol | 500 | 83.6% | 49.4% | 7.2% | 100.0 / 94.1 / 87.6 / 82.6 / 49.4 |
| Gemini 3.8 Flash | 100 (20 per level) | 80.0% | 30.0% | 15.0% | 100 / 100 / 95 / 75 / 30 |
| DeepSeek V4.1 Flash | 500 | 41.8% | 4.5% | 39.4% | 89.5 / 55.9 / 40.5 / 15.1 / 4.5 |
| Claude Sonnet 5.5 | 100 (20 per level) | 34.0% | 10.0% | 16.0% | 80 / 30 / 20 / 30 / 10 |

Logs are in `paper-boost/runs/modern/`. The 10-puzzle cost pilots are in `paper-boost/runs/pilots/`.
o4-mini and o3-mini are blocked by the account's zero-data-retention setting. QwQ 32B and R1 Distill 70B are no
longer served on OpenRouter.

### lm-evaluation-harness (Task C)
Full test split with Gemma-3 27B (`local-chat-completions` via OpenRouter, `max_gen_toks=8192`):
`acc` 0.8% (paper 1.2%), `invalid_path` 84.2% (paper 88.0%). Per level: 1.2 / 0.8 / 0.8 / 1.2 / 0.0%. Output:
`paper-boost/runs/lmeval-full/`. The `--limit 20` smoke run gave acc 0.0 and invalid_path 0.80 (4 puzzles per
level).

### Spend
All paid runs went through OpenRouter, with a key limit of $50 and a local stop at $44. Total used: **$32.17**.
About $2.35 went to the paper-model checks, smoke tests and the four 10-puzzle pilots. About $29.80 went to the
four newer-model runs, plus the lm-eval full run and the verifiers runs that ran at the same time.

## Commands left for a human

1. **Merge.** Done on 2026-10-13. The SPaRC stack (`task-a-citations`, `task-b-inspect`, `task-e-leaderboard`) and
   the Spatial-Gym stack (`task-a-citations`, `task-d-verifiers`) were fast-forwarded into `main`; Spatial-Gym CI
   passed. The website is live from `SPaRCWebsite` `master`, including the redesign (PR #1) and the collapsible
   leaderboard (`6aa10f4`). The site's leaderboard now loads `leaderboard/results.json` from SPaRC `main`.
2. **PyPI release (A5).** The PyPI page for `sparc-puzzle` (0.5.9) still shows the wrong pages. After the merge:
   ```bash
   gh release create v0.5.10 -R lkaesberg/SPaRC --target main --title v0.5.10 --notes "Fix citation (EMNLP 2025 pages 10359--10390); related-work section; Inspect task"
   ```
3. **HF dataset card (A3).** The stored HF token is read-only (`role: read`), so the upload was refused. The edited
   card is committed in the local clone `paper-boost/SPaRC-hf`. With a write token:
   ```bash
   cd paper-boost/SPaRC-hf && hf upload lkaesberg/SPaRC README.md README.md --repo-type dataset --commit-message "Fix SPaRC citation pages (10359--10390); add task_categories, language, tags"
   ```
4. **Inspect Evals register (B6).** The register uses an issue form; a bot then writes
   `register/sparc/eval.yaml` and opens the PR.
   - Open https://github.com/UKGovernmentBEIS/inspect_evals/issues/new?template=register-submission.yml.
   - arXiv URL: `https://arxiv.org/abs/2505.16686v2`
   - Source URL: `https://github.com/lkaesberg/SPaRC/blob/<SHA>/sparc/tasks.py#L141` (line 141 is the `@task` of
     `sparc`). Use the full SHA of the `main` commit after the merge. Before the merge,
     `42e501dccc40c0f30b4d2ff024a9daf7b41dd9bf` (on `task-b-inspect`) also works.
   - Maintainers: `lkaesberg`. Ask for the eval id `sparc`, and mention that `sparc_hard` is in the same file.
     The register lists one task per entry.
   - When the bot's PR appears, upload the two full logs above at https://logfile-upload.generality.workers.dev/
     and confirm on the PR. The logs contain no local paths, usernames or keys (checked).
5. **lm-evaluation-harness PR (C4).**
   ```bash
   gh pr create -R EleutherAI/lm-evaluation-harness --head lkaesberg:add-sparc-task --base main --title "Add SPaRC (spatial pathfinding) task"
   ```
   Then close your older PR https://github.com/EleutherAI/lm-evaluation-harness/pull/3262 (merge conflicts, no
   reviews; it also changes the unrelated `lm_eval/models/huggingface.py`).
6. **Not for Claude Code (from the handoff)**: Google Scholar, Semantic Scholar and OpenAlex profiles, the
   Awesome-Spatial-Reasoning PR, emails to survey authors, the `prime env push` (see the Spatial-Gym NOTES) and
   the PyPI releases.

## Discrepancies found

1. **SPaRC page numbers.** The ACL Anthology BibTeX and the PDF footer say `pages = "10359--10390"`, as the
   handoff does. The README, the HF dataset card and the website said `10370--10401`. Tasks A and E fix them. The
   Spatial-Gym paper's bibliography has the same error.
2. **README title.** The SPaRC README heading said "SPaRC: Spatial Pathfinding and Reasoning Challenge", which is
   not the paper title. It now says "SPaRC: A Spatial Pathfinding Reasoning Challenge". The `SPaRCWebsite`
   README has the same wrong title. It was left alone because it is not a citation.
3. **The handoff's "Table 1".** The handoff says to compare against Table 1. In the paper, Table 1 holds dataset
   counts, and the accuracy table is Table 6.
4. **The paper does not state the correctness rule.** "Exact match against stored solutions" comes from the code
   (`validation.py`), not the paper text. The paper does say that all solutions are found by brute force and
   that puzzles can have several solutions, which fits the code.
5. **Spatial-Gym's rule checker disagrees with the stored solutions** (242 of 500 ground-truth paths pass it).
   The reward uses exact match and is not affected. Details are in the Spatial-Gym NOTES. The validator was not
   changed.
6. **QwQ 32B.** The per-puzzle file `WitnessDatasetLLM/results/Qwen-QwQ-32B+puzzle_all_test.jsonl` gives 6.0%
   (30/500), while the paper reports 5.8%. The file is dated 2025-06-27, after the first arXiv version, so it is
   probably a later rerun. The leaderboard keeps the paper's 5.8%.
7. **Mixed Spatial-Gym result files.** Two step-by-step numbers in the Spatial-Gym paper (OLMo 3.1 32B 11.4%,
   Nemotron 49B 11.0%) come from files whose last 40 and 80 rows were run with backtracking. Qwen 3 0.6B with
   backtracking has 257 of 500 rows. Details are in the Spatial-Gym NOTES. The leaderboard shows the numbers as
   published.
8. **Prompt text differs between settings.** The backtracking prompt (`sparc/prompts/gym_step_traceback.py`) says
   polyshapes may be rotated. The single-shot and plain step prompts say they may not.
9. **Paper slips** (not changed anywhere):
   - p. 10363 says level 4 has "1 of 89 puzzles" solved, but level 4 has 86 test puzzles.
   - p. 10365 says the pass@k gain is "only 2.3 points for level 4 and 5", but Figure 7 shows +10.4 points for
     level 4.

## Third-party guide differences

- **Task B, folder name.** The handoff asks for an `inspect/` package. A package named `inspect` hides Python's
  standard-library `inspect` module and breaks imports, and the register lint wants the task inside a normal
  package. The task is therefore `sparc/tasks.py` inside the existing `sparc` package, with its README at
  `sparc/README.md`.
- **Task B, `inspect eval sparc`.** The task is registered through `[project.entry-points.inspect_ai]`. With an
  editable install, `inspect eval sparc` works from any folder (checked). The form `inspect eval sparc/sparc`
  only works for a non-editable install, because the distribution is called `sparc-puzzle`. The file form
  `inspect eval sparc/tasks.py@sparc` always works.
- **Task B, dependency.** Inspect 0.3.279 needs `openai>=3.4` for every OpenAI-compatible provider, but SPaRC
  pinned `openai<3`. The pin is now `>=2.7.1,<4.0.0`. The `sparc` CLI was checked with openai 3.29 (2 puzzles,
  no errors), and `poetry.lock` locks openai 3.29.0. `inspect-ai` has a `python_version < "3.15"` marker,
  because one of its dependencies does not support Python 3.15 yet.
- **Task B, submission path.** The register uses an issue form and a bot PR, not a PR written by hand. It also
  wants full logs from two models, so Task B has two full runs instead of one.
- **Task B, model ids.** The handoff says `openai/<model>`. Only an OpenRouter key was available, so the runs use
  `openrouter/...` ids.
- **Task C, layout.** The lm-eval docs suggest a group with one subtask per level (the `mmlu_pro` pattern). That
  layout fails the repo's own `tests/test_tasks.py` with "Duplicate task", because the test loads the group and its
  subtasks together. The task is therefore one `sparc` task that reports `acc`, `acc_level_1` … `acc_level_5` and
  `invalid_path`, as `catalan_bench/cieacova` does. `sparc_hard` is a separate task.
- **Task C, tests.** `pytest tests/test_tasks.py -k sparc` only selects sparc when
  `.github/outputs/tasks_all_changed_and_modified_files.txt` lists the sparc YAMLs, which is how CI selects new
  tasks. With that file in place, 29 tests pass (26 with `-k sparc`). The pre-commit hooks pass.
- **Task E, more rows.** The handoff asks for the paper's results table. The record's `setting` field also fits the
  Spatial-Gym paper, so its Tables 2 to 4 are included too. Each row names its source paper and table.

## Security notes

- `~/Documents/Cursor/WitnessDatasetLLM/config.yaml` and `config-*.yaml` contain plaintext OpenAI and Hugging Face
  keys. Nothing was copied or changed. Rotate them before anything from that folder is published.
- The OpenRouter key used here was pasted into the chat and is stored in `paper-boost/.env`, outside all repos. It
  is not in any commit. Rotate it when this work is finished.
