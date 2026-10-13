# NOTES — citation & benchmark-listing work (SPaRC repo)

Work log for the SPaRC / Spatial-Gym citation-infrastructure handoff, written 2026-10-13. A matching
`NOTES.md` exists in the Spatial-Gym repo.

## Branches

The branches are stacked so `NOTES.md` accumulates. Each one starts from the previous one:

| Branch | Base | Content |
|---|---|---|
| `task-a-citations` | `main` | Task A: README citation, related-work block |
| `task-b-inspect` | `task-a-citations` | Task B: Inspect task in `inspect/` |
| `task-e-leaderboard` | `task-b-inspect` | Task E: `leaderboard/results.json`, `leaderboard/run.py` |

The website change for Task E lives in the separate repo `lkaesberg/SPaRCWebsite` (branch `task-e-leaderboard`).

## Code map (handoff §3)

### SPaRC (`lkaesberg/SPaRC`)
- **README**: `README.md`
- **Puzzle data loader**: there is no dedicated loader module. The data is loaded inline with
  `load_dataset("lkaesberg/SPaRC", "all", split="test")` in `sparc/sparc.py:745` (`main()`).
  Each row has `id`, `difficulty_level` (1–5), `difficulty_score`, `grid_size`, `polyshapes` (a JSON string),
  `puzzle_array`, `solution_count`, `solutions` (each with a `path` list of `{x, y}`), and `text_visualization`.
- **Path validator**: `sparc/validation.py`
  - `extract_solution_path` (`:5`) takes the text after the last `####`, regex-matches `(x,y)` pairs,
    and stops at the end node.
  - `validate_solution` (`:64`) is **the validator**. It does an exact match of the extracted path against
    any entry of `puzzle["solutions"]`. All 500 test rows have `len(solutions) == solution_count` (1–48 solutions).
  - `analyze_path` (`:98`) does structural checks: start/end, connected, non-intersecting, no rule-cell crossing.
    `fully_valid_path == False` is the paper's "Invalid Path" (Table 7).
- **Prompts**: the paper's single-shot prompt is in `sparc/prompts/single_shot.py:22` (`get_prompt`). The
  registry is `sparc/prompts/__init__.py`, and backward-compatible wrappers are in `sparc/prompt.py`.
- **Existing LLM evaluation script**: the `sparc` CLI in `sparc/sparc.py` (`main()`). Per-puzzle logic is in
  `sparc/process_puzzle.py`:
  - `process_puzzle` (`:66`) is single-shot.
  - `process_puzzle_step_by_step` (`:207`) is gym mode: `observation='SPaRC'`, 100 steps, and no chat history
    (each step sends system + current state only).
  - Tables are in `sparc/tables.py`.
  - The CLI default is `--temperature 1.0`.
- `sparc/process_puzzle.py:10` imports `SPaRC_Gym` (PyPI `sparc-gym<0.3`), which is the pre-rename name of
  Spatial-Gym, not `Spatial_Gym`.

### Spatial-Gym (`lkaesberg/Spatial-Gym`)
- **README**: `README.md`. `pyproject.toml` sets `readme = "README.md"`, so the PyPI long description is the
  README. There is no `setup.py`.
- **Puzzle data loader**: `Spatial_Gym/Spatial_Gym.py`
  - `load_dataset(df_name, df_split, split=df_set)` is at `:77` in `__init__`.
  - `_process_puzzles` (`:219`) and `_load_puzzle` (`:95`) build the per-puzzle arrays.
- **Path validator**:
  - The success check is in `Spatial_Gym.step` (`:1111`). On termination or truncation it sets
    `outcome_reward = 1` (`:1207`) only if `self.path` exactly equals one of `solution_paths`. This is the
    same semantics as SPaRC's `validate_solution`.
  - `_is_on_solution_path` (`:1244`) gives the +0.01 shaping reward.
  - `_run_rule_validators` / `_validate_rules` (`:901`, `:941`) fill `info["rule_status"]`. This is
    informational only and not used for reward (see Discrepancies).
- **Renderers**: `Spatial_Gym/render/llm_renderer.py` (pygame window, see Discrepancies) and
  `Spatial_Gym/render/human_renderer.py`.
- **Existing LLM evaluation script**: `llm_testing/llm_host.py`. `run_episode` (`:62`) uses
  `observation='SPaRC'`, `max_steps=100` and `temperature=0.0`. The paper's runs appear to have used SPaRC's
  `sparc --gym` path above.
- **Tests**: `tests/` (43 tests, all pass on `main` as of 2026-10-13).

## Paper facts used (verified against ACL Anthology PDF / arXiv on 2026-10-13)

- The per-model results table is **Table 6** (Appendix J.1, p. 10381), not Table 1, which holds dataset counts.
  The invalid-path rates are in **Table 7** (p. 10381). The human numbers are in Fig. 3 and Table 2.
- "Hard" means difficulty level 5 only. Test counts per level are L1 86, L2 118, L3 121, L4 86, L5 89.
- Reasoning models used medium reasoning effort. The paper does not state temperature, max tokens or seeds.

## Discrepancies found

1. **SPaRC page numbers.** The ACL Anthology BibTeX and the PDF footer give `pages = "10359--10390"`, which
   matches the handoff. The README, the HF dataset card and the website all said `10370--10401`. Task A fixes
   the README and HF card, and Task E fixes the website. The Spatial-Gym paper's bibliography also cites SPaRC as
   pp. 10370–10401 and needs correcting in its next arXiv version (human task).
2. **README title.** The SPaRC README heading said "SPaRC: Spatial Pathfinding and Reasoning Challenge", which
   is not the paper title. It is changed to "SPaRC: A Spatial Pathfinding Reasoning Challenge". The
   `SPaRCWebsite` README has the same wrong title (left as is; it is not a citation).
3. **Spatial-Gym rule checker vs stored solutions.** When each test puzzle's first ground-truth solution is
   replayed in Spatial-Gym, all 500 get `outcome_reward == 1`. However, only 242/500 get
   `rule_status["all_rules_satisfied"] == True`: `poly_ylop_area` fails on 251 and `star_pairing_exact` on 24.
   The paper/validator definition (exact match against stored solutions) is unaffected, and all tasks here use
   it. The rule checker is not "fixed" here, per the handoff rules.
4. **The paper does not state the correctness criterion.** "Exact match against stored solutions" comes from
   the code (`validation.py`), not the paper text. The paper does say all solutions are enumerated by brute force
   and that multiple correct solutions are supported, which is consistent with the code.
5. **Paper slips** (not changed anywhere, just noted):
   - p. 10363 says level 4 is "1 of 89 puzzles", but level 4 has 86 test puzzles.
   - p. 10365 says the pass@k gain is "only 2.3 points for level 4 and 5", but Fig. 7 shows +10.4 points for level 4.

## Reproduced numbers

_(filled in by Tasks B/E)_

## Commands left for a human

### Task A
- **PyPI release (A5).** The PyPI page for `sparc-puzzle` (currently 0.5.9) still shows the wrong pages
  (10370–10401). After `task-a-citations` is merged into `main`, cut a release. The publish workflow takes the
  version from the tag:
  ```bash
  gh release create v0.5.10 -R lkaesberg/SPaRC --target main --title v0.5.10 --notes "Fix citation (EMNLP 2025 pages 10359--10390); related-work section"
  ```
- **HF dataset card (A3).** The stored HF token is read-only (`role: read`), so the upload was refused. The
  edited card is committed locally in `paper-boost/SPaRC-hf` (commit "Task A3: …"). With a write token:
  ```bash
  cd paper-boost/SPaRC-hf && hf upload lkaesberg/SPaRC README.md README.md --repo-type dataset --commit-message "Fix SPaRC citation pages (10359--10390); add task_categories, language, tags"
  ```

## Third-party guide differences

_(filled in per task)_
