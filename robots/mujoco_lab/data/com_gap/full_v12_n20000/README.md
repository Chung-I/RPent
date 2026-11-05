# com_gap full run on m2b2_v12: the CoM oracle gap at point (b) (TODO 6 step 1, 2026-11-05)

- Label: "m2b2_v12, posterior checks passed, not accepted, no joint attempt term". 20,000 particles, generator v11.
- Code: com-oracle-gap 6f1dbb3, dirty 0. Sim: the arm sim at e823247 (daily-logs worktree at fe46f6c). The bundle
  hashes are in `merge_provenance.txt`.
- Run: `cg_shards.sh 4`, 4 shards of 1 worker each (units v12-cg-s0..s3, MemoryMax 8G, no swap, huge pages off),
  `run_step2.sh` for each. Start 2026-11-05 08:35:41, last end 14:17:36. Each shard's provenance is in
  `provenance.txt`.
- Merge (`com_gap merge`, `merge.log`): 1,045 cases on 95 objects, 6,270 records, no object dropped, 0 reruns.
- Summary (`com_gap summary --wandb`): `summary.md`, `summary.json`, wandb `rpent-pick-place` run 0b4sqop0.
- Verdict (the pre-registered stop rule on H(0.05)): class stop, generic stop. The sensitivity row (errors and
  artifacts at EVPI 0) and the whole-log fit also read stop. No under-coverage flag.
- Files: `records.jsonl.gz` (the merged records), `run_config.json` (the merged config, its cases and "dropped"),
  `merge_provenance.txt` and `summary_provenance.txt` (the two steps), `cg_shards.sh` and `run_step2.sh` (copies of
  the launch scripts in `logs/comgap/`).
- The pre-registration, its amendments and the result text: daily-logs
  `researches/property-belief-manipulation/experiments/2026-11-05-com-oracle-gap/README.md`.
