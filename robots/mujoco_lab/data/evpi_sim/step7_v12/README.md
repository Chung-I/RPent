# evpi_sim step 7 on m2b2_v12: the simulator check of experiment 1 (2026-11-06)

- Label: "m2b2_v12, posterior checks passed, not accepted, no joint attempt term". Generator v11. Sim: the arm sim at
  e823247 (daily-logs worktree at fe46f6c). The bundle hashes are in each shard's `provenance.txt`.
- Code: evpi-sim 0195dfc, dirty 0, for the sample, the smoke test, the full run and the summary.
- Sample (`sample.json`, `sample.log`): `evpi_sim sample` over experiment 1's 14 shard dirs (the files in
  data/evpi_props/full_v12_n20000 on evpi-decomposition 6c90f0a). Sample seed 20261106, 50 records per cell, 4 cells
  (κ30 A3 and κ100 A10, class and generic priors): 200 records, 156 cases, 49 objects. Drawn 2026-11-06 03:37:13,
  before any simulation. sha256 b8f2b1249439f7871ec483ee0faa8a177a0b42f92d95bb5ff7b13e0041988bc9.
- Smoke test (`smoke/`): the first 5 records of the smoke order, 50 simulations. 0 replay mismatches, 0 errors,
  0 artifacts, 0 fingerprint differences. Projection 6.83 CPU-h (`smoke/project.json`), under the 20 CPU-h stop.
- Full run (`full/shard_0..6/`): `es_shards.sh 7`, 7 shards of 1 worker each (units v12-es-0..6, MemoryMax 6G, no swap,
  huge pages off), `run_step2.sh` for each, in the v12 checks queue at 7 slots. Start 2026-11-06 03:46:05, last end
  05:02:00. All 7 EXIT 0. 6.46 CPU-h. No OOM kill in the kernel log. Memory pressure 0.
- Summary (`summary/`): `evpi_sim summary --dirs full/shard_0..6 --wandb`, unit v12-es-sum, 2026-11-06 05:03. wandb
  `rpent-pick-place` run 9kisihlj. Gates: 200 records, 0 not run, 0 replay mismatches, 0 errors, 10 artifact
  simulations (k30_A3 generic only, on the a_S side: 30 of 550 pairs, 5.5 %, under the 10 % limit).
- The launch scripts are copies of `logs/evpi_sim/`. The pre-registration, the reading rules and the result text:
  daily-logs `researches/property-belief-manipulation/experiments/2026-11-05-evpi-decomposition/README.md`, "Step 7".
