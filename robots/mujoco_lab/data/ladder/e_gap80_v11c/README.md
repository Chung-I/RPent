# E on the 80 gap cases (v11c): item 1 of the help ladder

- `python -P -m robots.mujoco_lab.ladder_e run`, then `summary` (wandb `rpent-pick-place` run casp9c5o).
- Cases: the gap check's 80 seed-13 cases (`data/e1/gap/cases.json`) in k100_A3_abs, k30_A3_abs and k100_A10_abs.
- Horizons 1, 2 and 3: 240 jobs and 720 rows (`rows.jsonl.gz`), 0 errors.
- Bundle m2b2_v11c (5-fold cross-fit), generator v11.
- Particles per belief: 2,000 (belief.N_PARTICLES). The rows and the run configs are from before `--particles`.
- Chunk: 8 (ladder_e.CHUNK before `--chunk`, e0cd9a2). A rerun that must match this run uses `--chunk 8`.
- Launches (`provenance.txt`, one line each):
  - 1: a518158, 8 workers, 11 jobs, stopped to free the box (`run_until_20261104T050014.*`, EXIT 143).
  - 2: a518158, 8 workers, 13 jobs; systemd-oomd killed the user manager (`run_until_20261104T125421.*`, EXIT 137).
  - 3: 993639f, 5 workers, 4 jobs per worker, MemoryMax 9 GiB, no swap, 216 jobs (`run.*`, EXIT 0).
- Scripts: `run_e_gap80.sh` (launch 1) and `run_e_gap80_resume.sh` (launches 2 and 3), copies of daily-logs 0422ade.
- `summary.md` was made at 993639f.
