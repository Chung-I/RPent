# E on the 80 gap cases (v11c) at 20,000 particles: the N-sensitivity run

- `python -P -m robots.mujoco_lab.ladder_e run --particles 20000 --chunk 1 --workers 4 --max-tasks-per-worker 1`, then
  `summary` (wandb `rpent-pick-place` run mvukdt07).
- Cases, settings and horizons as in `../e_gap80_v11c/`: 240 jobs and 720 rows (`rows.jsonl.gz`), 0 errors.
- Bundle m2b2_v11c (5-fold cross-fit), generator v11.
- Particles per belief: 20,000 (`--particles`, an override of the bundle's 2,000). Chunk 1. A fresh process per job.
- One launch (`provenance.txt`): ladder-pilot 263b874 (clean), 4 workers in one unit with MemoryMax 9 GiB and no swap,
  NUMPY_MADVISE_HUGEPAGE=0 and a pressure guard. Start 2026-11-04 15:53:01, EXIT 0 at 2026-11-05 05:10:55.
- `overlap.log`: every 60 s, the load, E's memory and pressure, and the python processes outside E's unit.
- Script: `run_e_gap80_n20000.sh`, a copy of daily-logs 599fb648.
- `ncmp.md` and `ncmp.json`: this run (B) against `../e_gap80_v11c/` (A), paired by case (`e_ncmp.py A B out.json`).
  - ncmp prints A as "[None] particles": A's rows are from before the field (belief.N_PARTICLES, 2,000).
  - A ran at chunk 8, with numpy huge pages on and 5 workers of 4 jobs each. So the wall times are not a clean
    particle-count comparison. The tables do not depend on chunk or huge pages (help-ladder README, the timing study).
- `summary.md` was made at 263b874.
