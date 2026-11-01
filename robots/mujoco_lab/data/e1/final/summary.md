## E1 pilot final (F3, seed 12, 80 cases x 2 settings)

| arm | setting | n | mean cost | placed | abstain | probe | irreversible | overrides (cases) | malformed | failed calls | calls/case (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default_rule | hi_abs | 40 | 17.557 | 0.800 | 0.100 | 1.000 | 0.100 | 0 (0) | 0 | 0 | 0.00 (0) |
| default_rule | hi_rec | 40 | 2.801 | 0.800 | 0.150 | 1.000 | 0.075 | 0 (0) | 0 | 0 | 0.00 (0) |
| default_rule | lo_abs | 40 | 3.545 | 0.750 | 0.125 | 1.000 | 0.075 | 0 (0) | 0 | 0 | 0.00 (0) |
| default_rule | lo_rec | 40 | 2.692 | 0.800 | 0.200 | 1.000 | 0.100 | 0 (0) | 0 | 0 | 0.00 (0) |
| default_rule | all | 160 | 6.649 | 0.787 | 0.144 | 1.000 | 0.087 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | hi_abs | 40 | 9.814 | 0.425 | 0.550 | 0.525 | 0.025 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | hi_rec | 40 | 1.973 | 0.875 | 0.050 | 0.350 | 0.025 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | lo_abs | 40 | 2.276 | 0.850 | 0.075 | 0.400 | 0.025 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | lo_rec | 40 | 2.595 | 0.825 | 0.150 | 0.125 | 0.175 | 0 (0) | 0 | 0 | 0.00 (0) |
| oracle_rule | all | 160 | 4.164 | 0.744 | 0.206 | 0.350 | 0.062 | 0 (0) | 0 | 0 | 0.00 (0) |
| menu_planner | hi_abs | 40 | 14.338 | 0.750 | 0.175 | 1.000 | 0.075 | 0 (0) | 0 | 0 | 2.02 (3) |
| menu_planner | hi_rec | 40 | 3.135 | 0.775 | 0.150 | 0.850 | 0.075 | 3 (2) | 0 | 0 | 2.08 (4) |
| menu_planner | lo_abs | 40 | 2.764 | 0.825 | 0.050 | 0.575 | 0.075 | 3 (2) | 0 | 0 | 1.65 (4) |
| menu_planner | lo_rec | 40 | 2.911 | 0.825 | 0.000 | 0.075 | 0.200 | 12 (7) | 0 | 0 | 1.50 (3) |
| menu_planner | all | 160 | 5.787 | 0.794 | 0.094 | 0.625 | 0.106 | 18 (11) | 0 | 0 | 1.81 (4) |
| rec_planner | hi_abs | 40 | 14.510 | 0.725 | 0.200 | 0.975 | 0.075 | 0 (0) | 0 | 0 | 1.98 (3) |
| rec_planner | hi_rec | 40 | 2.642 | 0.825 | 0.100 | 1.000 | 0.075 | 4 (4) | 0 | 0 | 2.20 (4) |
| rec_planner | lo_abs | 40 | 3.225 | 0.800 | 0.050 | 1.000 | 0.100 | 4 (4) | 0 | 0 | 2.08 (4) |
| rec_planner | lo_rec | 40 | 3.020 | 0.825 | 0.000 | 1.000 | 0.150 | 18 (8) | 0 | 0 | 2.38 (4) |
| rec_planner | all | 160 | 5.849 | 0.794 | 0.087 | 0.994 | 0.100 | 26 (16) | 0 | 0 | 2.16 (4) |
| nomenu_planner | hi_abs | 40 | 37.216 | 0.725 | 0.050 | 0.750 | 0.225 | 2 (2) | 1 | 0 | 1.70 (3) |
| nomenu_planner | hi_rec | 40 | 2.089 | 0.900 | 0.050 | 0.575 | 0.175 | 8 (7) | 0 | 0 | 1.85 (3) |
| nomenu_planner | lo_abs | 40 | 6.939 | 0.550 | 0.025 | 0.500 | 0.350 | 5 (3) | 0 | 0 | 1.50 (4) |
| nomenu_planner | lo_rec | 40 | 2.764 | 0.850 | 0.000 | 0.150 | 0.225 | 13 (7) | 0 | 0 | 1.57 (3) |
| nomenu_planner | all | 160 | 12.252 | 0.756 | 0.031 | 0.494 | 0.244 | 28 (19) | 1 | 0 | 1.66 (4) |

| contrast | setting | n (objects) | mean difference [95 % object-bootstrap interval] | dropped |
|---|---|---|---|---|
| menu_planner - default_rule | hi_abs | 39 (15) | -3.303 [-13.055, 1.210] | 1 |
| menu_planner - default_rule | hi_rec | 36 (15) | 0.331 [-0.020, 1.015] | 4 |
| menu_planner - default_rule | lo_abs | 36 (15) | -0.601 [-1.540, -0.007] | 4 |
| menu_planner - default_rule | lo_rec | 39 (15) | 0.228 [-0.590, 1.075] | 1 |
| menu_planner - default_rule | all | 150 (20) | -0.864 [-3.313, 0.494] | 10 |
| menu_planner - oracle_rule | hi_abs | 39 (15) | 4.640 [-3.978, 15.534] | 1 |
| menu_planner - oracle_rule | hi_rec | 36 (15) | 0.936 [0.110, 2.081] | 4 |
| menu_planner - oracle_rule | lo_abs | 36 (15) | -0.300 [-1.032, 0.039] | 4 |
| menu_planner - oracle_rule | lo_rec | 39 (15) | 0.324 [0.080, 0.680] | 1 |
| menu_planner - oracle_rule | all | 150 (20) | 1.443 [-0.835, 4.621] | 10 |
| menu_planner - rec_planner | hi_abs | 39 (15) | -0.177 [-1.578, 0.833] | 1 |
| menu_planner - rec_planner | hi_rec | 36 (15) | 0.281 [-0.023, 1.018] | 4 |
| menu_planner - rec_planner | lo_abs | 36 (15) | -0.511 [-1.518, -0.017] | 4 |
| menu_planner - rec_planner | lo_rec | 39 (15) | -0.108 [-0.218, 0.047] | 1 |
| menu_planner - rec_planner | all | 150 (20) | -0.129 [-0.569, 0.284] | 10 |
| menu_planner - nomenu_planner | hi_abs | 39 (15) | -23.464 [-44.954, -7.965] | 1 |
| menu_planner - nomenu_planner | hi_rec | 36 (15) | 0.450 [-0.779, 2.000] | 4 |
| menu_planner - nomenu_planner | lo_abs | 36 (15) | -5.463 [-7.137, -3.257] | 4 |
| menu_planner - nomenu_planner | lo_rec | 39 (15) | 0.153 [-0.208, 0.844] | 1 |
| menu_planner - nomenu_planner | all | 150 (20) | -7.264 [-11.825, -3.427] | 10 |
| default_rule - oracle_rule | hi_abs | 39 (15) | 7.942 [-3.641, 20.845] | 1 |
| default_rule - oracle_rule | hi_rec | 36 (15) | 0.606 [0.011, 1.564] | 4 |
| default_rule - oracle_rule | lo_abs | 36 (15) | 0.301 [-0.099, 0.958] | 4 |
| default_rule - oracle_rule | lo_rec | 39 (15) | 0.096 [-0.450, 0.747] | 1 |
| default_rule - oracle_rule | all | 150 (20) | 2.308 [-0.655, 6.196] | 10 |
