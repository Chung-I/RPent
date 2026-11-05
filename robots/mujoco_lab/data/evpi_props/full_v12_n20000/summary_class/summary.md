# evpi_props summary

records 5225, counted 5035, errors 0, artifacts 190; a property matters when the 95 % object-bootstrap lower bound of its mean EVPI is above 0.02

## Mean EVPI per reveal (all cases)

| setting | prior | n | fillable | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|---|
| k100_A10 | class | 1007 | 70 | ≈ +0.1315 [+0.0572, +0.2168] matters | +0.1117 [+0.0523, +0.1803] matters | ≈ +0.0726 [+0.0247, +0.1304] matters | ≈ +0.0323 [+0.0016, +0.0709] no detectable value | +0.3443 [+0.2420, +0.4560] matters |
| k100_A3 | class | 1007 | 70 | ≈ +0.0034 [+0.0002, +0.0085] no detectable value | +0.0030 [+0.0000, +0.0084] no detectable value | ≈ +0.0022 [+0.0000, +0.0063] no detectable value | ≈ +0.0006 [+0.0000, +0.0019] no detectable value | +0.0067 [+0.0011, +0.0144] no detectable value |
| k10_A10 | class | 1007 | 70 | ≈ +0.0811 [+0.0099, +0.1718] no detectable value | +0.0272 [-0.0034, +0.0625] no detectable value | ≈ +0.1683 [+0.0771, +0.2864] matters | ≈ +0.2598 [+0.1279, +0.4238] matters | +0.4323 [+0.2826, +0.6135] matters |
| k1_A10 | class | 1007 | 70 | ≈ +0.0679 [+0.0078, +0.1507] no detectable value | +0.0314 [+0.0059, +0.0567] no detectable value | ≈ +0.1818 [+0.0863, +0.2958] matters | ≈ +0.2034 [+0.0997, +0.3323] matters | +0.3268 [+0.2200, +0.4514] matters |
| k30_A3 | class | 1007 | 70 | ≈ +0.0263 [+0.0136, +0.0420] no detectable value | +0.0231 [+0.0072, +0.0403] no detectable value | ≈ +0.0171 [-0.0052, +0.0403] no detectable value | ≈ +0.0060 [-0.0051, +0.0175] no detectable value | +0.0882 [+0.0605, +0.1176] matters |

≈: the cell holds fillable cases (the fillable column). Their mass, CoM and damage reveals weight the particles by narrow kernels first, so those results are approximate (fillable.md).

## Share of cases with EVPI_S < 0 / > 0 (all cases)

| setting | prior | n | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|
| k100_A10 | class | 1007 | 0.040 / 0.081 | 0.045 / 0.081 | 0.018 / 0.057 | 0.012 / 0.049 | 0.000 / 0.188 |
| k100_A3 | class | 1007 | 0.000 / 0.005 | 0.000 / 0.004 | 0.000 / 0.003 | 0.000 / 0.001 | 0.000 / 0.014 |
| k10_A10 | class | 1007 | 0.057 / 0.108 | 0.064 / 0.108 | 0.053 / 0.124 | 0.028 / 0.186 | 0.002 / 0.295 |
| k1_A10 | class | 1007 | 0.047 / 0.078 | 0.071 / 0.091 | 0.030 / 0.127 | 0.025 / 0.173 | 0.000 / 0.284 |
| k30_A3 | class | 1007 | 0.026 / 0.055 | 0.028 / 0.049 | 0.013 / 0.038 | 0.008 / 0.029 | 0.000 / 0.150 |

## θ-critical cases: EVPI_S median / 90th percentile / share above 0.02 (all cases)

| setting | prior | n critical | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|
| k100_A10 | class | 232 | +0.0000 / +2.8460 / 0.345 | +0.0000 / +2.4429 / 0.349 | +0.0000 / +1.7834 / 0.207 | +0.0000 / +0.4175 / 0.211 | +0.6584 / +3.8715 / 0.815 |
| k100_A3 | class | 14 | +0.0000 / +0.9954 / 0.357 | +0.0000 / +0.7222 / 0.286 | +0.0000 / +0.6905 / 0.214 | +0.0000 / +0.0000 / 0.071 | +0.4493 / +0.9954 / 0.929 |
| k10_A10 | class | 357 | +0.0000 / +0.5196 / 0.261 | +0.0000 / +0.5983 / 0.277 | +0.0000 / +1.4122 / 0.291 | +0.0724 / +1.7143 / 0.521 | +0.2886 / +3.0457 / 0.801 |
| k1_A10 | class | 353 | +0.0000 / +0.3375 / 0.198 | +0.0000 / +0.3927 / 0.238 | +0.0000 / +1.5787 / 0.340 | +0.0000 / +1.3998 / 0.484 | +0.2197 / +2.6096 / 0.790 |
| k30_A3 | class | 196 | +0.0000 / +0.8292 / 0.276 | +0.0000 / +0.7692 / 0.240 | +0.0000 / +0.6160 / 0.194 | +0.0000 / +0.2043 / 0.148 | +0.2411 / +1.1308 / 0.740 |

## Mean EVPI per reveal (cases with ESS >= 100)

| setting | prior | n | fillable | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|---|
| k100_A10 | class | 1007 | 70 | ≈ +0.1315 [+0.0605, +0.2127] matters | +0.1117 [+0.0533, +0.1769] matters | ≈ +0.0726 [+0.0233, +0.1282] matters | ≈ +0.0323 [+0.0008, +0.0682] no detectable value | +0.3443 [+0.2428, +0.4629] matters |
| k100_A3 | class | 1007 | 70 | ≈ +0.0034 [+0.0002, +0.0085] no detectable value | +0.0030 [+0.0000, +0.0084] no detectable value | ≈ +0.0022 [+0.0000, +0.0062] no detectable value | ≈ +0.0006 [+0.0000, +0.0019] no detectable value | +0.0067 [+0.0011, +0.0142] no detectable value |
| k10_A10 | class | 1007 | 70 | ≈ +0.0811 [+0.0118, +0.1730] no detectable value | +0.0272 [-0.0051, +0.0628] no detectable value | ≈ +0.1683 [+0.0771, +0.2858] matters | ≈ +0.2598 [+0.1307, +0.4188] matters | +0.4323 [+0.2888, +0.6066] matters |
| k1_A10 | class | 1007 | 70 | ≈ +0.0679 [+0.0056, +0.1496] no detectable value | +0.0314 [+0.0077, +0.0570] no detectable value | ≈ +0.1818 [+0.0920, +0.2985] matters | ≈ +0.2034 [+0.1046, +0.3254] matters | +0.3268 [+0.2171, +0.4503] matters |
| k30_A3 | class | 1007 | 70 | ≈ +0.0263 [+0.0125, +0.0417] no detectable value | +0.0231 [+0.0068, +0.0416] no detectable value | ≈ +0.0171 [-0.0058, +0.0412] no detectable value | ≈ +0.0060 [-0.0048, +0.0175] no detectable value | +0.0882 [+0.0618, +0.1174] matters |

≈: the cell holds fillable cases (the fillable column). Their mass, CoM and damage reveals weight the particles by narrow kernels first, so those results are approximate (fillable.md).

## Share of cases with EVPI_S < 0 / > 0 (cases with ESS >= 100)

| setting | prior | n | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|
| k100_A10 | class | 1007 | 0.040 / 0.081 | 0.045 / 0.081 | 0.018 / 0.057 | 0.012 / 0.049 | 0.000 / 0.188 |
| k100_A3 | class | 1007 | 0.000 / 0.005 | 0.000 / 0.004 | 0.000 / 0.003 | 0.000 / 0.001 | 0.000 / 0.014 |
| k10_A10 | class | 1007 | 0.057 / 0.108 | 0.064 / 0.108 | 0.053 / 0.124 | 0.028 / 0.186 | 0.002 / 0.295 |
| k1_A10 | class | 1007 | 0.047 / 0.078 | 0.071 / 0.091 | 0.030 / 0.127 | 0.025 / 0.173 | 0.000 / 0.284 |
| k30_A3 | class | 1007 | 0.026 / 0.055 | 0.028 / 0.049 | 0.013 / 0.038 | 0.008 / 0.029 | 0.000 / 0.150 |

## θ-critical cases: EVPI_S median / 90th percentile / share above 0.02 (cases with ESS >= 100)

| setting | prior | n critical | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|
| k100_A10 | class | 232 | +0.0000 / +2.8460 / 0.345 | +0.0000 / +2.4429 / 0.349 | +0.0000 / +1.7834 / 0.207 | +0.0000 / +0.4175 / 0.211 | +0.6584 / +3.8715 / 0.815 |
| k100_A3 | class | 14 | +0.0000 / +0.9954 / 0.357 | +0.0000 / +0.7222 / 0.286 | +0.0000 / +0.6905 / 0.214 | +0.0000 / +0.0000 / 0.071 | +0.4493 / +0.9954 / 0.929 |
| k10_A10 | class | 357 | +0.0000 / +0.5196 / 0.261 | +0.0000 / +0.5983 / 0.277 | +0.0000 / +1.4122 / 0.291 | +0.0724 / +1.7143 / 0.521 | +0.2886 / +3.0457 / 0.801 |
| k1_A10 | class | 353 | +0.0000 / +0.3375 / 0.198 | +0.0000 / +0.3927 / 0.238 | +0.0000 / +1.5787 / 0.340 | +0.0000 / +1.3998 / 0.484 | +0.2197 / +2.6096 / 0.790 |
| k30_A3 | class | 196 | +0.0000 / +0.8292 / 0.276 | +0.0000 / +0.7692 / 0.240 | +0.0000 / +0.6160 / 0.194 | +0.0000 / +0.2043 / 0.148 | +0.2411 / +1.1308 / 0.740 |

## Groups (from the choices)

| setting | prior | n | invariant | critical | abstain-dominated (overlay) | invariant, probe first (overlay) | first choice abstains |
|---|---|---|---|---|---|---|---|
| k100_A10 | class | 1007 | 0.770 | 0.230 | 0.734 | 0.000 | 0.798 |
| k100_A3 | class | 1007 | 0.986 | 0.014 | 0.985 | 0.000 | 0.999 |
| k10_A10 | class | 1007 | 0.645 | 0.355 | 0.440 | 0.000 | 0.478 |
| k1_A10 | class | 1007 | 0.649 | 0.351 | 0.371 | 0.000 | 0.381 |
| k30_A3 | class | 1007 | 0.805 | 0.195 | 0.769 | 0.000 | 0.850 |

## Fillable cases (70 case-prior pairs; per case in fillable.md)

| reveal | n | min kernel ESS | median kernel ESS | below 100 | refreshed |
|---|---|---|---|---|---|
| mass | 70 | 134.6 | 353.7 | 0 | 0 |
| com | 70 | 71.6 | 3175.6 | 6 | 6 |
| damage | 70 | 470.4 | 5233.9 | 0 | 0 |

## Checks

```
{
 "n": 5035,
 "replay": {
  "compared": 3021,
  "mismatch": 30,
  "examples": [
   {
    "case": [
     "hot3d__megaphone",
     0,
     "class",
     "k100_A3"
    ],
    "here": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": {
      "best_cost": null,
      "recommended_cost": null
     }
    },
    "com_gap": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": null
    }
   },
   {
    "case": [
     "hot3d__megaphone",
     0,
     "class",
     "k30_A3"
    ],
    "here": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": {
      "best_cost": null,
      "recommended_cost": null
     }
    },
    "com_gap": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": null
    }
   },
   {
    "case": [
     "hot3d__megaphone",
     0,
     "class",
     "k100_A10"
    ],
    "here": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": {
      "best_cost": null,
      "recommended_cost": null
     }
    },
    "com_gap": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": null
    }
   },
   {
    "case": [
     "hot3d__megaphone",
     1,
     "class",
     "k100_A3"
    ],
    "here": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": {
      "best_cost": null,
      "recommended_cost": null
     }
    },
    "com_gap": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": null
    }
   },
   {
    "case": [
     "hot3d__megaphone",
     1,
     "class",
     "k30_A3"
    ],
    "here": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": {
      "best_cost": null,
      "recommended_cost": null
     }
    },
    "com_gap": {
     "recommendation": null,
     "best_option": null,
     "guard_flag": false,
     "first_menu": null
    }
   }
  ]
 },
 "evpi_all_negative": {
  "n": 2,
  "guard_under_truth": 0
 },
 "a_all_is_min": {
  "off": 0,
  "examples": []
 }
}
```
