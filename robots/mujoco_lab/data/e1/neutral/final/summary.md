## E1 pilot post-hoc check: neutral task text (menu and rec planners neutral; rules and nomenu as run)

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
| menu_planner | hi_abs | 40 | 18.348 | 0.725 | 0.175 | 0.975 | 0.100 | 0 (0) | 0 | 0 | 2.00 (3) |
| menu_planner | hi_rec | 40 | 2.886 | 0.800 | 0.075 | 0.425 | 0.075 | 6 (4) | 0 | 0 | 1.68 (4) |
| menu_planner | lo_abs | 40 | 2.392 | 0.850 | 0.050 | 0.350 | 0.050 | 3 (2) | 0 | 0 | 1.45 (4) |
| menu_planner | lo_rec | 40 | 2.875 | 0.825 | 0.025 | 0.000 | 0.200 | 11 (6) | 2 | 0 | 1.48 (4) |
| menu_planner | all | 160 | 6.625 | 0.800 | 0.081 | 0.438 | 0.106 | 20 (12) | 2 | 0 | 1.65 (4) |
| rec_planner | hi_abs | 40 | 11.704 | 0.625 | 0.325 | 0.950 | 0.050 | 0 (0) | 0 | 0 | 1.95 (3) |
| rec_planner | hi_rec | 40 | 2.665 | 0.825 | 0.100 | 1.000 | 0.075 | 5 (4) | 0 | 0 | 2.23 (4) |
| rec_planner | lo_abs | 40 | 3.270 | 0.800 | 0.050 | 1.000 | 0.100 | 6 (5) | 0 | 0 | 2.12 (4) |
| rec_planner | lo_rec | 40 | 3.020 | 0.825 | 0.000 | 1.000 | 0.150 | 18 (8) | 0 | 0 | 2.38 (4) |
| rec_planner | all | 160 | 5.165 | 0.769 | 0.119 | 0.988 | 0.094 | 29 (17) | 0 | 0 | 2.17 (4) |
| nomenu_planner | hi_abs | 40 | 37.216 | 0.725 | 0.050 | 0.750 | 0.225 | 2 (2) | 1 | 0 | 1.70 (3) |
| nomenu_planner | hi_rec | 40 | 2.089 | 0.900 | 0.050 | 0.575 | 0.175 | 8 (7) | 0 | 0 | 1.85 (3) |
| nomenu_planner | lo_abs | 40 | 6.939 | 0.550 | 0.025 | 0.500 | 0.350 | 5 (3) | 0 | 0 | 1.50 (4) |
| nomenu_planner | lo_rec | 40 | 2.764 | 0.850 | 0.000 | 0.150 | 0.225 | 13 (7) | 0 | 0 | 1.57 (3) |
| nomenu_planner | all | 160 | 12.252 | 0.756 | 0.031 | 0.494 | 0.244 | 28 (19) | 1 | 0 | 1.66 (4) |

| contrast | setting | n (objects) | mean difference [95 % object-bootstrap interval] | dropped |
|---|---|---|---|---|
| menu_planner - default_rule | hi_abs | 39 (15) | -3.292 [-13.055, 1.211] | 1 |
| menu_planner - default_rule | hi_rec | 36 (15) | 0.322 [-0.115, 1.146] | 4 |
| menu_planner - default_rule | lo_abs | 36 (15) | -0.586 [-1.515, 0.000] | 4 |
| menu_planner - default_rule | lo_rec | 39 (15) | 0.191 [-0.626, 1.037] | 1 |
| menu_planner - default_rule | all | 150 (20) | -0.870 [-3.324, 0.505] | 10 |
| menu_planner - oracle_rule | hi_abs | 39 (15) | 4.650 [-3.968, 15.553] | 1 |
| menu_planner - oracle_rule | hi_rec | 36 (15) | 0.928 [0.055, 2.169] | 4 |
| menu_planner - oracle_rule | lo_abs | 37 (15) | -0.277 [-0.992, 0.051] | 3 |
| menu_planner - oracle_rule | lo_rec | 39 (15) | 0.287 [0.049, 0.635] | 1 |
| menu_planner - oracle_rule | all | 151 (20) | 1.428 [-0.835, 4.586] | 9 |
| menu_planner - rec_planner | hi_abs | 39 (15) | 2.958 [-1.526, 10.634] | 1 |
| menu_planner - rec_planner | hi_rec | 36 (15) | 0.247 [-0.126, 1.118] | 4 |
| menu_planner - rec_planner | lo_abs | 36 (15) | -0.546 [-1.569, -0.091] | 4 |
| menu_planner - rec_planner | lo_rec | 39 (15) | -0.145 [-0.273, 0.003] | 1 |
| menu_planner - rec_planner | all | 150 (20) | 0.660 [-0.588, 2.745] | 10 |
| menu_planner - nomenu_planner | hi_abs | 39 (15) | -23.454 [-44.957, -7.954] | 1 |
| menu_planner - nomenu_planner | hi_rec | 36 (15) | 0.442 [-0.839, 2.084] | 4 |
| menu_planner - nomenu_planner | lo_abs | 37 (15) | -5.705 [-7.274, -3.701] | 3 |
| menu_planner - nomenu_planner | lo_rec | 39 (15) | 0.115 [-0.277, 0.799] | 1 |
| menu_planner - nomenu_planner | all | 151 (20) | -7.321 [-11.887, -3.428] | 9 |
| default_rule - oracle_rule | hi_abs | 39 (15) | 7.942 [-3.641, 20.845] | 1 |
| default_rule - oracle_rule | hi_rec | 36 (15) | 0.606 [0.011, 1.564] | 4 |
| default_rule - oracle_rule | lo_abs | 36 (15) | 0.301 [-0.099, 0.958] | 4 |
| default_rule - oracle_rule | lo_rec | 39 (15) | 0.096 [-0.450, 0.747] | 1 |
| default_rule - oracle_rule | all | 150 (20) | 2.308 [-0.655, 6.196] | 10 |
| rec_planner - default_rule | hi_abs | 39 (15) | -6.250 [-17.382, 1.712] | 1 |
| rec_planner - default_rule | hi_rec | 36 (15) | 0.075 [0.000, 0.253] | 4 |
| rec_planner - default_rule | lo_abs | 36 (15) | -0.040 [-0.819, 0.514] | 4 |
| rec_planner - default_rule | lo_rec | 39 (15) | 0.336 [-0.470, 1.179] | 1 |
| rec_planner - default_rule | all | 150 (20) | -1.529 [-4.682, 0.552] | 10 |
| menu_planner neutral - v1 | hi_abs | 39 (15) | 0.010 [-0.011, 0.033] | 1 |
| menu_planner neutral - v1 | hi_rec | 36 (15) | -0.008 [-0.097, 0.141] | 4 |
| menu_planner neutral - v1 | lo_abs | 36 (15) | 0.015 [-0.025, 0.064] | 4 |
| menu_planner neutral - v1 | lo_rec | 39 (15) | -0.037 [-0.099, 0.010] | 1 |
| menu_planner neutral - v1 | all | 150 (20) | -0.005 [-0.035, 0.027] | 10 |
| rec_planner neutral - v1 | hi_abs | 39 (15) | -3.124 [-10.638, 0.754] | 1 |
| rec_planner neutral - v1 | hi_rec | 36 (15) | 0.025 [0.000, 0.090] | 4 |
| rec_planner neutral - v1 | lo_abs | 36 (15) | 0.050 [0.000, 0.174] | 4 |
| rec_planner neutral - v1 | lo_rec | 39 (15) | 0.000 [0.000, 0.000] | 1 |
| rec_planner neutral - v1 | all | 150 (20) | -0.794 [-2.919, 0.196] | 10 |

Pairing: 320 neutral rows, 0 without an as-run row, 0 with a different truth, reset or first assessment.
