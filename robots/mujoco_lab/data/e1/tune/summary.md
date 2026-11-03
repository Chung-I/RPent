## E1 pilot tuning (F1/F2, seed 13, 20 cases)

| arm | setting | n | mean cost | placed | abstain | probe | irreversible | overrides (cases) | malformed | failed calls | calls/case (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default_rule | hi_abs | 5 | 2.970 | 0.800 | 0.200 | 1.000 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| default_rule | hi_rec | 5 | 3.070 | 0.800 | 0.000 | 1.000 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| default_rule | lo_abs | 5 | 4.440 | 0.600 | 0.400 | 1.000 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| default_rule | lo_rec | 5 | 2.470 | 0.800 | 0.200 | 1.000 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| default_rule | all | 20 | 3.237 | 0.750 | 0.200 | 1.000 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | hi_abs | 5 | 4.650 | 0.600 | 0.400 | 0.600 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | hi_rec | 5 | 0.610 | 1.000 | 0.000 | 0.200 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | lo_abs | 5 | 2.460 | 0.800 | 0.200 | 0.000 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | lo_rec | 5 | 0.450 | 1.000 | 0.000 | 0.400 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | all | 20 | 2.042 | 0.850 | 0.150 | 0.300 | 0.000 | 0 (0) | 0 | 0 | 0.00 (0) |
| menu_planner | hi_abs | 5 | 3.000 | 0.800 | 0.200 | 1.000 | 0.000 | 0 (0) | 0 | 0 | 2.40 (3) |
| menu_planner | hi_rec | 5 | 3.030 | 0.800 | 0.000 | 0.800 | 0.000 | 0 (0) | 0 | 0 | 2.20 (4) |
| menu_planner | lo_abs | 5 | 2.970 | 0.800 | 0.000 | 0.400 | 0.000 | 4 (2) | 0 | 0 | 1.80 (4) |
| menu_planner | lo_rec | 5 | 0.530 | 1.000 | 0.000 | 0.400 | 0.000 | 1 (1) | 0 | 0 | 1.40 (2) |
| menu_planner | all | 20 | 2.383 | 0.850 | 0.050 | 0.650 | 0.000 | 5 (3) | 0 | 0 | 1.95 (4) |
| rec_planner | hi_abs | 5 | 4.880 | 0.600 | 0.400 | 1.000 | 0.000 | 0 (0) | 0 | 0 | 2.40 (3) |
| rec_planner | hi_rec | 5 | 3.070 | 0.800 | 0.000 | 1.000 | 0.000 | 0 (0) | 0 | 0 | 2.40 (4) |
| rec_planner | lo_abs | 5 | 4.800 | 0.600 | 0.400 | 1.000 | 0.000 | 2 (1) | 0 | 0 | 2.40 (4) |
| rec_planner | lo_rec | 5 | 0.550 | 1.000 | 0.000 | 1.000 | 0.000 | 1 (1) | 0 | 0 | 2.00 (2) |
| rec_planner | all | 20 | 3.325 | 0.750 | 0.200 | 1.000 | 0.000 | 3 (2) | 0 | 0 | 2.30 (4) |
| nomenu_planner | hi_abs | 5 | 2.880 | 0.800 | 0.200 | 0.600 | 0.000 | 1 (1) | 0 | 0 | 1.80 (3) |
| nomenu_planner | hi_rec | 5 | 0.890 | 1.000 | 0.000 | 0.600 | 0.200 | 0 (0) | 0 | 0 | 1.80 (3) |
| nomenu_planner | lo_abs | 5 | 2.980 | 0.800 | 0.000 | 0.200 | 0.000 | 3 (2) | 0 | 0 | 1.60 (3) |
| nomenu_planner | lo_rec | 5 | 0.710 | 1.000 | 0.000 | 0.200 | 0.000 | 1 (1) | 0 | 0 | 1.40 (2) |
| nomenu_planner | all | 20 | 1.865 | 0.900 | 0.050 | 0.400 | 0.050 | 5 (4) | 0 | 0 | 1.65 (3) |

| contrast | setting | n (objects) | mean difference [95 % object-bootstrap interval] | dropped |
|---|---|---|---|---|
| menu_planner - default_rule | hi_abs | 5 (5) | 0.030 [0.000, 0.060] | 0 |
| menu_planner - default_rule | hi_rec | 5 (5) | -0.040 [-0.120, 0.000] | 0 |
| menu_planner - default_rule | lo_abs | 5 (5) | -1.470 [-5.830, 1.600] | 0 |
| menu_planner - default_rule | lo_rec | 5 (5) | -1.940 [-5.770, 0.050] | 0 |
| menu_planner - default_rule | all | 20 (20) | -0.855 [-2.303, 0.240] | 0 |
| menu_planner - oracle_rule | hi_abs | 5 (5) | -1.650 [-5.610, 0.660] | 0 |
| menu_planner - oracle_rule | hi_rec | 5 (5) | 2.420 [0.030, 7.080] | 0 |
| menu_planner - oracle_rule | lo_abs | 5 (5) | 0.510 [0.060, 1.310] | 0 |
| menu_planner - oracle_rule | lo_rec | 5 (5) | 0.080 [-0.050, 0.200] | 0 |
| menu_planner - oracle_rule | all | 20 (20) | 0.340 [-1.225, 1.945] | 0 |
| menu_planner - rec_planner | hi_abs | 5 (5) | -1.880 [-5.720, 0.070] | 0 |
| menu_planner - rec_planner | hi_rec | 5 (5) | -0.040 [-0.120, 0.000] | 0 |
| menu_planner - rec_planner | lo_abs | 5 (5) | -1.830 [-5.830, 0.520] | 0 |
| menu_planner - rec_planner | lo_rec | 5 (5) | -0.020 [-0.120, 0.080] | 0 |
| menu_planner - rec_planner | all | 20 (20) | -0.943 [-2.380, 0.055] | 0 |
| menu_planner - nomenu_planner | hi_abs | 5 (5) | 0.120 [-5.680, 6.230] | 0 |
| menu_planner - nomenu_planner | hi_rec | 5 (5) | 2.140 [-0.600, 6.960] | 0 |
| menu_planner - nomenu_planner | lo_abs | 5 (5) | -0.010 [-0.060, 0.030] | 0 |
| menu_planner - nomenu_planner | lo_rec | 5 (5) | -0.180 [-0.720, 0.180] | 0 |
| menu_planner - nomenu_planner | all | 20 (20) | 0.518 [-1.070, 2.390] | 0 |
| default_rule - oracle_rule | hi_abs | 5 (5) | -1.680 [-5.640, 0.630] | 0 |
| default_rule - oracle_rule | hi_rec | 5 (5) | 2.460 [0.080, 7.120] | 0 |
| default_rule - oracle_rule | lo_abs | 5 (5) | 1.980 [-0.300, 5.980] | 0 |
| default_rule - oracle_rule | lo_rec | 5 (5) | 2.020 [0.030, 5.910] | 0 |
| default_rule - oracle_rule | all | 20 (20) | 1.195 [-0.698, 3.190] | 0 |
