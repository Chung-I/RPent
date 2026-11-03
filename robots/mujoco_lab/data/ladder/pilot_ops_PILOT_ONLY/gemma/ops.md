# Gemma 4 31B FP8 (pilot): operational numbers only

No choice, outcome or cost is read here. Malformed: a reply with no valid option id (no tool call, or an id not on the list); refusal: a reply with no tool call whose text reads as a refusal (a keyword match).

| rung | runs | errors | call cap | malformed | no tool call | refusals | failed calls | length finishes | resets not at rest | calls per run (mean, max) | tokens in per call (mean, max) | tokens out per call (mean, max) | tokens in per run (mean) | tokens out per run (mean) | model s per call (mean, p90) | minutes per run (mean, p90) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1.17, 3 | 1288, 2379 | 1260, 2150 | 1502 | 1470 | 65.1, 85.0 | 1.29, 1.86 |
| B | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1.17, 3 | 1524, 2822 | 1802, 3585 | 1777 | 2103 | 91.8, 130.4 | 1.81, 2.40 |
| C | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1.08, 2 | 1885, 3366 | 1753, 3434 | 2042 | 1900 | 90.9, 141.0 | 1.66, 2.63 |
| D | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1.08, 2 | 2099, 3308 | 2178, 3702 | 2274 | 2359 | 111.7, 181.7 | 2.03, 3.09 |
| all | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 1.12, 3 | 1688, 3366 | 1740, 3702 | 1899 | 1958 | 89.4, 154.8 | 1.70, 2.65 |

Wall time: 11.3 min for 48 rung runs at [8] workers on one GPU: 0.235 min per rung run. Server start: 28.2 min.
Full ladder at the n50 sizes (rungs A-D x 426 cases = 1704 rung runs, the same workers): about 7.1 GPU-hours for this model (from the wall time, with the pilot's tail).
From the mean minutes per rung run (1.70) at 8 workers: about 6.5 GPU-hours (no tail).
