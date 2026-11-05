# evpi_props full run on m2b2_v12: EVPI per hidden property (experiment 1, 2026-11-05 to 06)

- Label: "m2b2_v12, posterior checks passed, not accepted, no joint attempt term". 20,000 particles, generator v11.
- Code: the run used evpi-decomposition 2df16a6, dirty 0. The summary used evpi-summary e2ac6e6, dirty 0. Sim: the arm
  sim at e823247 (daily-logs worktree at fe46f6c). The bundle hashes are in each shard's `provenance.txt`.
- Run: `ev_shards.sh 7 class`, then `ev_shards.sh 7 generic`: 14 shards of 1 worker each (units v12-ev-c0..c6 and
  v12-ev-g0..g6, MemoryMax 6G, no swap, huge pages off), `run_step2.sh` for each, in the v12 checks queue at 7 slots.
  Start 2026-11-05 22:04:14, last end 2026-11-06 03:24:38. All 14 EXIT 0. 24.9 CPU-h.
- Records per prior: 5,225 (1,045 cases × 5 settings), 0 errors, 190 artifact records (38 cases).
- Summary: one call per prior (`run_summary_prior.sh class`, `run_summary_prior.sh generic`), in `summary_class/` and
  `summary_generic/` (summary.md, summary.json, fillable.md, provenance.txt). wandb `rpent-pick-place` runs lk4t9ff9
  (class) and mvz31e7m (generic).
- The planned single call over the 14 dirs (`run_summary.sh`) stopped in `load()` before any table: "config differs in
  ['priors_parsed']" (`summary_failed_0326/`). The shards ran one prior at a time. `summarize()` splits the cells by
  prior, so only the bootstrap draws differ from a single call.
- Files: `class/shard_k/` and `generic/shard_k/` (records.jsonl.gz, run_config.json, provenance.txt, run.exit). To use
  `evpi_props summary --dirs` on them, gunzip records.jsonl.gz first. The launch scripts are copies of `logs/evpi/`.
- The pre-registration, the reading rules and the result text: daily-logs
  `researches/property-belief-manipulation/experiments/2026-11-05-evpi-decomposition/README.md`.
