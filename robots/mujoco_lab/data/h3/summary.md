## H3: recoverability × uncertainty (κ 10, A 10, s 1; seeds [12, 13, 14, 15, 16, 17]; 95 objects; cohort 5845 of 6270 cases (0.932); 2000 joint draws)

| quantity | value [95 % interval] | verdict |
|---|---|---|
| H3: X_D | 0.020 [-0.158, 0.184] | undecided |
| H3-info: X_I | 0.033 [-0.069, 0.136] | undecided |
| H3-term: X_T | -0.132 [-0.295, 0.012] | undecided |
| H3-matched: X_M* | -0.034 [-0.169, 0.085] | undecided |
| X_M (reported) | -0.013 [-0.138, 0.106] | — |
| X_P (reported) | 0.080 [-0.021, 0.192] | — |
| X_pen = X_D − X_T (reported) | 0.152 [-0.130, 0.442] | — |
| X_truth = X_D − X_M*, truth spread at the wide belief (reported) | 0.054 [-0.115, 0.218] | — |
| X_I,T (reported) | -0.172 [-0.371, -0.011] | — |
| X_I,M* (reported) | -0.023 [-0.135, 0.088] | — |

Holm at 0.05 over the four verdicts (sensitivity; the 95 % interval rule is primary):

| verdict | bootstrap p | Holm p | Holm rejects |
|---|---|---|---|
| X_D | 0.8340 | 1.0000 | no |
| X_I | 0.5360 | 1.0000 | no |
| X_T | 0.0800 | 0.3200 | no |
| X_M* | 0.6470 | 1.0000 | no |

X_T by the event that ended the absorbing case: probe 0.003 [0.000, 0.007], attempt -0.135 [-0.298, 0.009], none 0.000 [0.000, 0.000].

| condition | n | D | I | M | P | never cost |
|---|---|---|---|---|---|---|
| absorbing/wide | 5845 | -0.103 [-0.280, 0.054] | 0.143 [0.029, 0.258] | -0.245 [-0.413, -0.114] | 0.035 [-0.077, 0.130] | 6.140 [5.446, 6.879] |
| absorbing/narrow | 5845 | -0.137 [-0.339, 0.031] | 0.087 [-0.054, 0.226] | -0.224 [-0.403, -0.109] | -0.049 [-0.202, 0.075] | 5.831 [5.063, 6.629] |
| rec20/wide | 5845 | -0.142 [-0.184, -0.102] | 0.009 [-0.030, 0.049] | -0.152 [-0.175, -0.130] | -0.012 [-0.032, 0.001] | 5.355 [4.688, 6.104] |
| rec20/narrow | 5845 | -0.157 [-0.213, -0.110] | -0.013 [-0.065, 0.029] | -0.144 [-0.167, -0.121] | -0.016 [-0.059, 0.008] | 5.250 [4.494, 6.046] |
| rec0/wide | 5845 | -0.142 [-0.179, -0.103] | 0.005 [-0.033, 0.044] | -0.146 [-0.170, -0.124] | -0.012 [-0.025, -0.001] | 5.315 [4.644, 6.064] |
| rec0/narrow | 5845 | -0.163 [-0.219, -0.116] | -0.021 [-0.074, 0.022] | -0.142 [-0.164, -0.121] | -0.019 [-0.063, 0.005] | 5.201 [4.453, 5.993] |
| rec_hi/wide | 5845 | 0.013 [-0.205, 0.231] | 0.274 [0.111, 0.460] | -0.261 [-0.504, -0.073] | 0.095 [-0.053, 0.259] | 6.526 [5.691, 7.393] |
| rec_hi/narrow | 5845 | -0.153 [-0.385, 0.061] | 0.046 [-0.141, 0.242] | -0.199 [-0.397, -0.047] | -0.073 [-0.236, 0.056] | 6.320 [5.441, 7.236] |
| absorbing/narrowT_wideB | 5845 | -0.140 [-0.291, 0.003] | 0.100 [-0.006, 0.223] | -0.240 [-0.417, -0.122] | -0.057 [-0.185, 0.040] | 5.847 [5.116, 6.625] |
| rec20/narrowT_wideB | 5845 | -0.126 [-0.154, -0.094] | 0.024 [-0.005, 0.055] | -0.149 [-0.173, -0.129] | -0.013 [-0.028, -0.002] | 5.249 [4.494, 6.058] |

| spread | Q = D(absorbing) − D(rec20) | D(rec20) − D(rec0) |
|---|---|---|
| wide | 0.040 [-0.139, 0.191] | -0.001 [-0.012, 0.008] |
| narrow | 0.020 [-0.150, 0.181] | 0.006 [-0.000, 0.013] |
| narrowT_wideB | -0.014 [-0.159, 0.126] | — |

Matched truth (wide cases inside the narrow ranges: 249, share 0.043, effective n 160.0, weighted by 1 / P(selected | branch)): D(absorbing/wide) 0.012 [-0.185, 0.317], I(absorbing/wide) 0.183 [-0.024, 0.482], Q 0.129 [-0.068, 0.437]; narrow Q 0.020 [-0.150, 0.181].

| condition | probed | same | other row | decline | terminal | no probe |
|---|---|---|---|---|---|---|
| absorbing/wide | 4737 | 0.820 | 0.157 | 0.019 | 0.004 | 1108 |
| absorbing/narrow | 4738 | 0.846 | 0.145 | 0.007 | 0.002 | 1107 |
| rec20/wide | 4737 | 0.827 | 0.148 | 0.021 | 0.004 | 1108 |
| rec20/narrow | 4738 | 0.850 | 0.146 | 0.002 | 0.002 | 1107 |
| rec0/wide | 4737 | 0.826 | 0.151 | 0.019 | 0.004 | 1108 |
| rec0/narrow | 4738 | 0.849 | 0.147 | 0.002 | 0.002 | 1107 |
| rec_hi/wide | 4737 | 0.820 | 0.157 | 0.019 | 0.004 | 1108 |
| rec_hi/narrow | 4738 | 0.846 | 0.145 | 0.007 | 0.002 | 1107 |
| absorbing/narrowT_wideB | 4728 | 0.823 | 0.165 | 0.010 | 0.002 | 1117 |
| rec20/narrowT_wideB | 4728 | 0.841 | 0.151 | 0.006 | 0.002 | 1117 |

| condition / arm | placed per 1,000 | abstain rate | irreversible cases per 1,000 | irreversible events per 1,000 | guard flag share | cost |
|---|---|---|---|---|---|---|
| absorbing/wide/never | 476.3 | 0.484 | 11.3 | 11.3 | 0.018 | 6.140 |
| absorbing/wide/probe_once | 481.1 | 0.480 | 11.5 | 11.5 | 0.018 | 6.243 |
| absorbing/wide/sham_once | 477.7 | 0.481 | 13.5 | 13.5 | 0.018 | 6.385 |
| absorbing/wide/policy | 481.4 | 0.479 | 11.3 | 11.3 | 0.018 | 6.105 |
| absorbing/narrow/never | 507.6 | 0.448 | 11.5 | 11.5 | 0.016 | 5.831 |
| absorbing/narrow/probe_once | 506.6 | 0.445 | 10.8 | 10.8 | 0.016 | 5.969 |
| absorbing/narrow/sham_once | 507.8 | 0.446 | 13.0 | 13.0 | 0.016 | 6.055 |
| absorbing/narrow/policy | 508.1 | 0.445 | 12.1 | 12.1 | 0.016 | 5.881 |
| rec20/wide/never | 512.1 | 0.432 | 23.6 | 42.9 | 0.018 | 5.355 |
| rec20/wide/probe_once | 511.2 | 0.433 | 20.7 | 40.4 | 0.018 | 5.497 |
| rec20/wide/sham_once | 512.2 | 0.431 | 24.6 | 45.7 | 0.018 | 5.507 |
| rec20/wide/policy | 511.7 | 0.432 | 24.6 | 44.5 | 0.018 | 5.367 |
| rec20/narrow/never | 523.4 | 0.405 | 22.6 | 50.1 | 0.016 | 5.250 |
| rec20/narrow/probe_once | 522.7 | 0.404 | 19.0 | 45.7 | 0.016 | 5.407 |
| rec20/narrow/sham_once | 523.9 | 0.406 | 21.7 | 48.8 | 0.016 | 5.394 |
| rec20/narrow/policy | 522.5 | 0.405 | 23.8 | 51.5 | 0.016 | 5.266 |
| rec0/wide/never | 512.1 | 0.431 | 24.0 | 43.8 | 0.018 | 5.315 |
| rec0/wide/probe_once | 511.5 | 0.432 | 21.4 | 42.3 | 0.018 | 5.456 |
| rec0/wide/sham_once | 512.4 | 0.430 | 24.8 | 46.0 | 0.018 | 5.461 |
| rec0/wide/policy | 511.9 | 0.431 | 35.8 | 59.4 | 0.018 | 5.326 |
| rec0/narrow/never | 523.5 | 0.404 | 22.9 | 51.7 | 0.016 | 5.201 |
| rec0/narrow/probe_once | 522.7 | 0.402 | 20.7 | 50.3 | 0.016 | 5.364 |
| rec0/narrow/sham_once | 524.2 | 0.404 | 22.2 | 50.3 | 0.016 | 5.343 |
| rec0/narrow/policy | 522.3 | 0.404 | 25.7 | 55.3 | 0.016 | 5.220 |
| rec_hi/wide/never | 476.8 | 0.490 | 11.3 | 19.0 | 0.018 | 6.526 |
| rec_hi/wide/probe_once | 482.1 | 0.486 | 11.5 | 16.9 | 0.018 | 6.512 |
| rec_hi/wide/sham_once | 478.5 | 0.488 | 13.5 | 21.6 | 0.018 | 6.787 |
| rec_hi/wide/policy | 482.0 | 0.486 | 11.3 | 17.8 | 0.018 | 6.430 |
| rec_hi/narrow/never | 508.3 | 0.452 | 11.5 | 21.2 | 0.016 | 6.320 |
| rec_hi/narrow/probe_once | 507.4 | 0.449 | 10.8 | 20.9 | 0.016 | 6.473 |
| rec_hi/narrow/sham_once | 508.6 | 0.452 | 13.0 | 22.2 | 0.016 | 6.519 |
| rec_hi/narrow/policy | 509.0 | 0.449 | 12.1 | 22.4 | 0.016 | 6.393 |
| absorbing/narrowT_wideB/never | 496.0 | 0.467 | 9.8 | 9.8 | 0.017 | 5.847 |
| absorbing/narrowT_wideB/probe_once | 500.1 | 0.463 | 10.4 | 10.4 | 0.017 | 5.987 |
| absorbing/narrowT_wideB/sham_once | 495.3 | 0.466 | 11.5 | 11.5 | 0.017 | 6.087 |
| absorbing/narrowT_wideB/policy | 498.2 | 0.464 | 10.9 | 10.9 | 0.017 | 5.904 |
| rec20/narrowT_wideB/never | 522.3 | 0.417 | 23.3 | 47.4 | 0.017 | 5.249 |
| rec20/narrowT_wideB/probe_once | 523.0 | 0.419 | 19.2 | 40.7 | 0.017 | 5.375 |
| rec20/narrowT_wideB/sham_once | 522.3 | 0.417 | 22.6 | 45.7 | 0.017 | 5.399 |
| rec20/narrowT_wideB/policy | 521.8 | 0.417 | 24.6 | 49.4 | 0.017 | 5.262 |

Sensitivity: pairwise cohorts X_D -0.011 [-0.180, 0.154], X_I 0.024 [-0.076, 0.126]; artifact cases counted (n 6270) X_D -0.049 [-0.223, 0.122], X_I 0.022 [-0.073, 0.114]; the six conditions' cohort (n 5849, share 0.933) X_D 0.018 [-0.161, 0.182], X_I 0.032 [-0.071, 0.134].

Checks at the verdict seeds (decision 8): first_menus ok, Gr ok, Gs ok, Gp ok, Gt ok (6510 restores, 47356 shams, 58450 pairs, 35070 twins).

Power (pilot): X̂_D -0.408, σ_b² 2.2887, σ_w² 58.8914, k 6 (power target not met). At k: SE(X_T) 0.075 on X̂_T -0.215 (fail, power 0.81), SE(X_M*) 0.147 on X̂_M* -0.168 (undecided, power 0.21).
