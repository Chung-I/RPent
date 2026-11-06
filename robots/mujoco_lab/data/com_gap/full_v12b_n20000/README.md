# com_gap full run on m2b2_v12b: the CoM oracle gap at point (b), with post_identify (TODO 6 step 1, rerun, 2026-11-06)

- Label: "m2b2_v12b, accepted (user, 2026-11-06; acceptance 5 pass, 5 undecided, 0 fail), joint attempt term".
  20,000 particles, generator v11.
- Code: com-oracle-gap-v12b b836e52 (with post_identify, Amendment 3), dirty 0. Sim: the arm sim at e823247 (daily-logs
  worktree at fe46f6c). Bundle: RPent-arm-identify `robots/mujoco_lab/data/m2b2_v12b` (manifest.json sha256
  d5e0072c...). The bundle hashes are in `provenance.txt`.
- Run: `cgb_shards.sh 7`, 7 shards of 1 worker each (units v12-cgb-s0..s6, MemoryMax 6G, no swap, huge pages off) in
  the v12 checks queue, `run_step2.sh` for each, with `--post-identify`. Start 2026-11-06 19:36:00, last end 23:19:08.
  Each shard's provenance is in `provenance.txt`.
- Merge and summary: `post_v12b.sh`. Merge (`merge.log`): 1,045 cases on 95 objects, 6,270 records, no object dropped,
  0 reruns. Summary (`com_gap summary --wandb`): `summary.md`, `summary.json`, wandb `rpent-pick-place` run 9436f34e.
- Verdict (the pre-registered stop rule on H(0.05)): class stop, generic stop. The sensitivity row (errors and
  artifacts at EVPI 0) and the whole-log fit also read stop. No under-coverage flag.
- Against `full_v12_n20000`: 6,264 of the 6,270 records are the same in every field (wall times and the post_identify
  fields apart). The other 6 (4 cases, generic prior) differ in the belief at the attempt. EVPI is the same in all
  6,270 records.
- Files: `records.jsonl.gz` (the merged records; sha256 of the uncompressed file a0ee25ec53cbbd28...), `run_config.json`
  (the merged config, its cases and "dropped"), `merge_provenance.txt` and `summary_provenance.txt` (the two steps),
  `cgb_shards.sh`, `run_step2.sh` and `post_v12b.sh` (copies of the launch scripts).
- The pre-registration, its amendments and the result text: daily-logs
  `researches/property-belief-manipulation/experiments/2026-11-05-com-oracle-gap/README.md`.
