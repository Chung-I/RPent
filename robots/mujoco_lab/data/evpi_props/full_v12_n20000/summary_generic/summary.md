# evpi_props summary

records 5225, counted 5035, errors 0, artifacts 190; a property matters when the 95 % object-bootstrap lower bound of its mean EVPI is above 0.02

## Mean EVPI per reveal (all cases)

| setting | prior | n | fillable | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|---|
| k100_A10 | generic | 1007 | 70 | ≈ +0.2798 [+0.1436, +0.4398] matters | +0.1311 [+0.0623, +0.2141] matters | ≈ +0.2348 [+0.1121, +0.3780] matters | ≈ +0.0280 [-0.0001, +0.0576] no detectable value | +0.6556 [+0.4544, +0.8856] matters |
| k100_A3 | generic | 1007 | 70 | ≈ +0.0018 [+0.0000, +0.0042] no detectable value | -0.0021 [-0.0064, +0.0000] no detectable value | ≈ +0.0004 [+0.0000, +0.0013] no detectable value | ≈ +0.0000 [+0.0000, +0.0000] no detectable value | +0.0078 [+0.0015, +0.0158] no detectable value |
| k10_A10 | generic | 1007 | 70 | ≈ +0.1210 [+0.0487, +0.2056] matters | +0.0232 [-0.0162, +0.0647] no detectable value | ≈ +0.1689 [+0.0642, +0.2749] matters | ≈ +0.2557 [+0.1276, +0.4112] matters | +0.6285 [+0.4573, +0.8287] matters |
| k1_A10 | generic | 1007 | 70 | ≈ +0.0673 [+0.0195, +0.1310] no detectable value | +0.0327 [-0.0191, +0.1039] no detectable value | ≈ +0.1656 [+0.0848, +0.2602] matters | ≈ +0.2072 [+0.1054, +0.3322] matters | +0.4399 [+0.3140, +0.5809] matters |
| k30_A3 | generic | 1007 | 70 | ≈ +0.0693 [+0.0358, +0.1070] matters | +0.0171 [+0.0031, +0.0368] no detectable value | ≈ +0.0420 [+0.0074, +0.0817] no detectable value | ≈ +0.0034 [+0.0000, +0.0105] no detectable value | +0.1601 [+0.1101, +0.2140] matters |

≈: the cell holds fillable cases (the fillable column). Their mass, CoM and damage reveals weight the particles by narrow kernels first, so those results are approximate (fillable.md).

## Share of cases with EVPI_S < 0 / > 0 (all cases)

| setting | prior | n | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|
| k100_A10 | generic | 1007 | 0.015 / 0.115 | 0.033 / 0.055 | 0.035 / 0.088 | 0.002 / 0.009 | 0.000 / 0.227 |
| k100_A3 | generic | 1007 | 0.000 / 0.004 | 0.003 / 0.000 | 0.000 / 0.001 | 0.000 / 0.000 | 0.000 / 0.016 |
| k10_A10 | generic | 1007 | 0.071 / 0.187 | 0.083 / 0.116 | 0.077 / 0.203 | 0.033 / 0.184 | 0.003 / 0.395 |
| k1_A10 | generic | 1007 | 0.048 / 0.108 | 0.046 / 0.063 | 0.067 / 0.168 | 0.037 / 0.149 | 0.005 / 0.351 |
| k30_A3 | generic | 1007 | 0.013 / 0.088 | 0.022 / 0.031 | 0.017 / 0.069 | 0.001 / 0.002 | 0.000 / 0.193 |

## θ-critical cases: EVPI_S median / 90th percentile / share above 0.02 (all cases)

| setting | prior | n critical | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|
| k100_A10 | generic | 264 | +0.0000 / +4.7998 / 0.439 | +0.0000 / +2.6882 / 0.201 | +0.0000 / +4.5956 / 0.333 | +0.0000 / +0.0000 / 0.034 | +1.7591 / +5.6960 / 0.867 |
| k100_A3 | generic | 18 | +0.0000 / +0.4548 / 0.222 | +0.0000 / +0.0000 / 0.000 | +0.0000 / +0.0000 / 0.056 | +0.0000 / +0.0000 / 0.000 | +0.3948 / +0.9497 / 0.833 |
| k10_A10 | generic | 452 | +0.0000 / +0.9921 / 0.358 | +0.0000 / +0.2007 / 0.195 | +0.0000 / +1.7677 / 0.394 | +0.0000 / +1.2219 / 0.407 | +0.5372 / +3.9060 / 0.858 |
| k1_A10 | generic | 417 | +0.0000 / +0.7951 / 0.237 | +0.0000 / +0.0751 / 0.122 | +0.0000 / +1.4742 / 0.379 | +0.0000 / +1.1168 / 0.353 | +0.5054 / +2.7943 / 0.835 |
| k30_A3 | generic | 219 | +0.0000 / +1.2572 / 0.406 | +0.0000 / +0.2718 / 0.142 | +0.0000 / +1.1645 / 0.311 | +0.0000 / +0.0000 / 0.009 | +0.7224 / +1.4337 / 0.872 |

## Mean EVPI per reveal (cases with ESS >= 100)

| setting | prior | n | fillable | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|---|
| k100_A10 | generic | 1000 | 63 | ≈ +0.2818 [+0.1441, +0.4460] matters | +0.1320 [+0.0611, +0.2085] matters | ≈ +0.2364 [+0.1129, +0.3766] matters | ≈ +0.0282 [+0.0002, +0.0586] no detectable value | +0.6602 [+0.4623, +0.9029] matters |
| k100_A3 | generic | 1000 | 63 | ≈ +0.0018 [+0.0000, +0.0041] no detectable value | -0.0021 [-0.0064, +0.0000] no detectable value | ≈ +0.0004 [+0.0000, +0.0013] no detectable value | ≈ +0.0000 [+0.0000, +0.0000] no detectable value | +0.0079 [+0.0018, +0.0158] no detectable value |
| k10_A10 | generic | 1000 | 63 | ≈ +0.1219 [+0.0431, +0.2002] matters | +0.0233 [-0.0152, +0.0654] no detectable value | ≈ +0.1701 [+0.0667, +0.2856] matters | ≈ +0.2575 [+0.1333, +0.4119] matters | +0.6329 [+0.4541, +0.8239] matters |
| k1_A10 | generic | 1000 | 63 | ≈ +0.0678 [+0.0186, +0.1355] no detectable value | +0.0329 [-0.0181, +0.1046] no detectable value | ≈ +0.1667 [+0.0901, +0.2639] matters | ≈ +0.2086 [+0.1062, +0.3274] matters | +0.4430 [+0.3137, +0.5881] matters |
| k30_A3 | generic | 1000 | 63 | ≈ +0.0697 [+0.0360, +0.1074] matters | +0.0172 [+0.0027, +0.0380] no detectable value | ≈ +0.0423 [+0.0087, +0.0788] no detectable value | ≈ +0.0034 [+0.0000, +0.0104] no detectable value | +0.1613 [+0.1079, +0.2189] matters |

≈: the cell holds fillable cases (the fillable column). Their mass, CoM and damage reveals weight the particles by narrow kernels first, so those results are approximate (fillable.md).

## Share of cases with EVPI_S < 0 / > 0 (cases with ESS >= 100)

| setting | prior | n | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|
| k100_A10 | generic | 1000 | 0.015 / 0.116 | 0.033 / 0.055 | 0.035 / 0.089 | 0.002 / 0.009 | 0.000 / 0.229 |
| k100_A3 | generic | 1000 | 0.000 / 0.004 | 0.003 / 0.000 | 0.000 / 0.001 | 0.000 / 0.000 | 0.000 / 0.016 |
| k10_A10 | generic | 1000 | 0.072 / 0.188 | 0.084 / 0.117 | 0.078 / 0.204 | 0.033 / 0.185 | 0.003 / 0.398 |
| k1_A10 | generic | 1000 | 0.048 / 0.109 | 0.046 / 0.063 | 0.067 / 0.169 | 0.037 / 0.150 | 0.005 / 0.353 |
| k30_A3 | generic | 1000 | 0.013 / 0.089 | 0.022 / 0.031 | 0.017 / 0.069 | 0.001 / 0.002 | 0.000 / 0.194 |

## θ-critical cases: EVPI_S median / 90th percentile / share above 0.02 (cases with ESS >= 100)

| setting | prior | n critical | mass | mu | com | damage | all |
|---|---|---|---|---|---|---|---|
| k100_A10 | generic | 264 | +0.0000 / +4.7998 / 0.439 | +0.0000 / +2.6882 / 0.201 | +0.0000 / +4.5956 / 0.333 | +0.0000 / +0.0000 / 0.034 | +1.7591 / +5.6960 / 0.867 |
| k100_A3 | generic | 18 | +0.0000 / +0.4548 / 0.222 | +0.0000 / +0.0000 / 0.000 | +0.0000 / +0.0000 / 0.056 | +0.0000 / +0.0000 / 0.000 | +0.3948 / +0.9497 / 0.833 |
| k10_A10 | generic | 452 | +0.0000 / +0.9921 / 0.358 | +0.0000 / +0.2007 / 0.195 | +0.0000 / +1.7677 / 0.394 | +0.0000 / +1.2219 / 0.407 | +0.5372 / +3.9060 / 0.858 |
| k1_A10 | generic | 417 | +0.0000 / +0.7951 / 0.237 | +0.0000 / +0.0751 / 0.122 | +0.0000 / +1.4742 / 0.379 | +0.0000 / +1.1168 / 0.353 | +0.5054 / +2.7943 / 0.835 |
| k30_A3 | generic | 219 | +0.0000 / +1.2572 / 0.406 | +0.0000 / +0.2718 / 0.142 | +0.0000 / +1.1645 / 0.311 | +0.0000 / +0.0000 / 0.009 | +0.7224 / +1.4337 / 0.872 |

## Groups (from the choices)

| setting | prior | n | invariant | critical | abstain-dominated (overlay) | invariant, probe first (overlay) | first choice abstains |
|---|---|---|---|---|---|---|---|
| k100_A10 | generic | 1007 | 0.738 | 0.262 | 0.744 | 0.000 | 0.886 |
| k100_A3 | generic | 1007 | 0.982 | 0.018 | 0.984 | 0.000 | 0.999 |
| k10_A10 | generic | 1007 | 0.551 | 0.449 | 0.384 | 0.000 | 0.418 |
| k1_A10 | generic | 1007 | 0.586 | 0.414 | 0.320 | 0.000 | 0.325 |
| k30_A3 | generic | 1007 | 0.783 | 0.217 | 0.777 | 0.000 | 0.934 |

## Fillable cases (70 case-prior pairs; per case in fillable.md)

| reveal | n | min kernel ESS | median kernel ESS | below 100 | refreshed |
|---|---|---|---|---|---|
| mass | 70 | 4.1 | 180.2 | 10 | 10 |
| com | 70 | 86.6 | 1089.9 | 1 | 1 |
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
     "generic",
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
     "generic",
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
     "generic",
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
     "generic",
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
     "generic",
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
  "n": 8,
  "guard_under_truth": 0
 },
 "a_all_is_min": {
  "off": 0,
  "examples": []
 }
}
```
