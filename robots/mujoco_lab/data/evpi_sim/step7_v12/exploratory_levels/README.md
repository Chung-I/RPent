# Step 7, exploratory, added after the results: the model at the truth next to the simulator (2026-11-06)

- Label: "m2b2_v12, posterior checks passed, not accepted, no joint attempt term". The coordinator asked for this check
  after the step 7 result, with the reading rule below. It changes none of step 7's pre-stated readings.
- What: on step 7's main-reading records (S = all differs from a_prior, no probe on either side, a kept seed), the
  model's outcome probabilities at the truth particle for the rows that a_prior and a_S chose, next to the simulator's
  outcome (one draw per record), J split into its parts, and the Poisson probability of each observed irreversible
  count. Also the changed records by transition, and a sensitivity without the object of the largest |gain|.
- Code: `robots/mujoco_lab/evpi_sim_levels.py`. The levels run used evpi-sim 1e6f206, and the summary used 1f4f366 (the
  same levels code, plus a gate check from a Codex review). Both are clean. The model only: no simulation.
- Smoke (`smoke/`): hope__tomato_sauce_can, 10 rows, J equal to the stored J at every row.
- Run (`full/shard_0..6/`): `esl_shards.sh 7` in the v12 checks queue (units v12-esl-0..6), 2026-11-06 05:21:21 to
  05:34:22. All 7 EXIT 0. No OOM kill. 194 rows.
- Summary (`summary/`): `evpi_sim_levels summary --levels full/shard_0..6 --runs ../full/shard_0..6 --wandb` (unit
  v12-esl-sum). wandb `rpent-pick-place` run 96atf6xs. J check: 191 rows of the main reading, max |J − the record's J| 0.
- The text and the reading: daily-logs
  `researches/property-belief-manipulation/experiments/2026-11-05-evpi-decomposition/README.md`, "Step 7, exploratory".
