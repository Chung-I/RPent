# Experiment 1, step 7: why the realized gain is above the predicted one (exploratory, added after the results)

The main reading's records (S = all, non-probe pairs). A row's outcome is one draw per record: its label is the same at every seed. The model's P is its prediction at the truth particle, which J prices. J check: 191 rows, max |J - the record's J| 0.

## Outcomes: the model's mean P at the truth and the expected count, next to the observed count

| cell | prior | side | rows | outcome | mean P | expected count | observed | Poisson P(observed or fewer) |
|---|---|---|---|---|---|---|---|---|
| k30_A3 | class | a_prior | 18 | placed_stable | 0.9228 | 16.61 | 16 |  |
| k30_A3 | class | a_prior | 18 | tipped | 0.0153 | 0.28 | 0 |  |
| k30_A3 | class | a_prior | 18 | missed | 0.0023 | 0.04 | 0 |  |
| k30_A3 | class | a_prior | 18 | dropped_low | 0.0104 | 0.19 | 0 |  |
| k30_A3 | class | a_prior | 18 | dropped_high | 0.0098 | 0.18 | 1 | 0.986 |
| k30_A3 | class | a_prior | 18 | rolled_off | 0.0016 | 0.03 | 0 | 0.972 |
| k30_A3 | class | a_prior | 18 | not_lifted | 0.0374 | 0.67 | 1 |  |
| k30_A3 | class | a_prior | 18 | crushed | 0.0003 | 0.01 | 0 | 0.994 |
| k30_A3 | class | a_prior | 18 | irreversible (any) | 0.0117 | 0.21 | 1 | 0.981 |
| k30_A3 | class | a_S | 40 | placed_stable | 0.9352 | 37.41 | 39 |  |
| k30_A3 | class | a_S | 40 | tipped | 0.0167 | 0.67 | 1 |  |
| k30_A3 | class | a_S | 40 | missed | 0.0033 | 0.13 | 0 |  |
| k30_A3 | class | a_S | 40 | dropped_low | 0.0065 | 0.26 | 0 |  |
| k30_A3 | class | a_S | 40 | dropped_high | 0.0071 | 0.28 | 0 | 0.752 |
| k30_A3 | class | a_S | 40 | rolled_off | 0.0022 | 0.09 | 0 | 0.914 |
| k30_A3 | class | a_S | 40 | not_lifted | 0.0280 | 1.12 | 0 |  |
| k30_A3 | class | a_S | 40 | crushed | 0.0007 | 0.03 | 0 | 0.972 |
| k30_A3 | class | a_S | 40 | broken | 0.0000 | 0.00 | 0 | 0.999 |
| k30_A3 | class | a_S | 40 | spilled | 0.0002 | 0.01 | 0 | 0.993 |
| k30_A3 | class | a_S | 40 | irreversible (any) | 0.0102 | 0.41 | 0 | 0.664 |
| k30_A3 | generic | a_prior | 3 | placed_stable | 0.4935 | 1.48 | 2 |  |
| k30_A3 | generic | a_prior | 3 | tipped | 0.0079 | 0.02 | 0 |  |
| k30_A3 | generic | a_prior | 3 | missed | 0.0014 | 0.00 | 0 |  |
| k30_A3 | generic | a_prior | 3 | dropped_low | 0.0451 | 0.14 | 1 |  |
| k30_A3 | generic | a_prior | 3 | dropped_high | 0.0142 | 0.04 | 0 | 0.958 |
| k30_A3 | generic | a_prior | 3 | rolled_off | 0.0007 | 0.00 | 0 | 0.998 |
| k30_A3 | generic | a_prior | 3 | not_lifted | 0.4372 | 1.31 | 0 |  |
| k30_A3 | generic | a_prior | 3 | irreversible (any) | 0.0149 | 0.04 | 0 | 0.956 |
| k30_A3 | generic | a_S | 35 | placed_stable | 0.9346 | 32.71 | 33 |  |
| k30_A3 | generic | a_S | 35 | tipped | 0.0185 | 0.65 | 2 |  |
| k30_A3 | generic | a_S | 35 | missed | 0.0034 | 0.12 | 0 |  |
| k30_A3 | generic | a_S | 35 | dropped_low | 0.0071 | 0.25 | 0 |  |
| k30_A3 | generic | a_S | 35 | dropped_high | 0.0070 | 0.25 | 0 | 0.783 |
| k30_A3 | generic | a_S | 35 | rolled_off | 0.0016 | 0.06 | 0 | 0.944 |
| k30_A3 | generic | a_S | 35 | not_lifted | 0.0268 | 0.94 | 0 |  |
| k30_A3 | generic | a_S | 35 | crushed | 0.0008 | 0.03 | 0 | 0.973 |
| k30_A3 | generic | a_S | 35 | broken | 0.0000 | 0.00 | 0 | 0.999 |
| k30_A3 | generic | a_S | 35 | spilled | 0.0002 | 0.01 | 0 | 0.994 |
| k30_A3 | generic | a_S | 35 | irreversible (any) | 0.0096 | 0.34 | 0 | 0.715 |
| k100_A10 | class | a_prior | 16 | placed_stable | 0.9277 | 14.84 | 16 |  |
| k100_A10 | class | a_prior | 16 | tipped | 0.0233 | 0.37 | 0 |  |
| k100_A10 | class | a_prior | 16 | missed | 0.0028 | 0.04 | 0 |  |
| k100_A10 | class | a_prior | 16 | dropped_low | 0.0072 | 0.11 | 0 |  |
| k100_A10 | class | a_prior | 16 | dropped_high | 0.0094 | 0.15 | 0 | 0.860 |
| k100_A10 | class | a_prior | 16 | rolled_off | 0.0017 | 0.03 | 0 | 0.973 |
| k100_A10 | class | a_prior | 16 | not_lifted | 0.0278 | 0.44 | 0 |  |
| k100_A10 | class | a_prior | 16 | crushed | 0.0002 | 0.00 | 0 | 0.997 |
| k100_A10 | class | a_prior | 16 | irreversible (any) | 0.0113 | 0.18 | 0 | 0.835 |
| k100_A10 | class | a_S | 33 | placed_stable | 0.9236 | 30.48 | 32 |  |
| k100_A10 | class | a_S | 33 | tipped | 0.0215 | 0.71 | 1 |  |
| k100_A10 | class | a_S | 33 | missed | 0.0046 | 0.15 | 0 |  |
| k100_A10 | class | a_S | 33 | dropped_low | 0.0074 | 0.24 | 0 |  |
| k100_A10 | class | a_S | 33 | dropped_high | 0.0081 | 0.27 | 0 | 0.766 |
| k100_A10 | class | a_S | 33 | rolled_off | 0.0021 | 0.07 | 0 | 0.932 |
| k100_A10 | class | a_S | 33 | not_lifted | 0.0322 | 1.06 | 0 |  |
| k100_A10 | class | a_S | 33 | crushed | 0.0006 | 0.02 | 0 | 0.981 |
| k100_A10 | class | a_S | 33 | irreversible (any) | 0.0108 | 0.36 | 0 | 0.700 |
| k100_A10 | generic | a_prior | 8 | placed_stable | 0.9375 | 7.50 | 8 |  |
| k100_A10 | generic | a_prior | 8 | tipped | 0.0243 | 0.19 | 0 |  |
| k100_A10 | generic | a_prior | 8 | missed | 0.0019 | 0.02 | 0 |  |
| k100_A10 | generic | a_prior | 8 | dropped_low | 0.0075 | 0.06 | 0 |  |
| k100_A10 | generic | a_prior | 8 | dropped_high | 0.0108 | 0.09 | 0 | 0.917 |
| k100_A10 | generic | a_prior | 8 | rolled_off | 0.0009 | 0.01 | 0 | 0.993 |
| k100_A10 | generic | a_prior | 8 | not_lifted | 0.0171 | 0.14 | 0 |  |
| k100_A10 | generic | a_prior | 8 | irreversible (any) | 0.0117 | 0.09 | 0 | 0.911 |
| k100_A10 | generic | a_S | 38 | placed_stable | 0.9322 | 35.42 | 36 |  |
| k100_A10 | generic | a_S | 38 | tipped | 0.0201 | 0.76 | 1 |  |
| k100_A10 | generic | a_S | 38 | missed | 0.0036 | 0.14 | 0 |  |
| k100_A10 | generic | a_S | 38 | dropped_low | 0.0080 | 0.30 | 0 |  |
| k100_A10 | generic | a_S | 38 | dropped_high | 0.0073 | 0.28 | 0 | 0.759 |
| k100_A10 | generic | a_S | 38 | rolled_off | 0.0027 | 0.10 | 0 | 0.904 |
| k100_A10 | generic | a_S | 38 | not_lifted | 0.0255 | 0.97 | 1 |  |
| k100_A10 | generic | a_S | 38 | crushed | 0.0006 | 0.02 | 0 | 0.978 |
| k100_A10 | generic | a_S | 38 | broken | 0.0000 | 0.00 | 0 | 0.999 |
| k100_A10 | generic | a_S | 38 | spilled | 0.0002 | 0.01 | 0 | 0.993 |
| k100_A10 | generic | a_S | 38 | irreversible (any) | 0.0107 | 0.41 | 0 | 0.666 |

## J split into its parts: the mean over the rows, the model at the truth / the simulator

| cell | prior | side | rows | irreversible | recoverable | A × P(not placed) | time | total |
|---|---|---|---|---|---|---|---|---|
| k30_A3 | class | a_prior | 18 | +1.760 / +8.333 | +0.047 / +0.028 | +0.232 / +0.333 | +0.406 / +0.406 | +2.444 / +9.100 |
| k30_A3 | class | a_S | 40 | +1.537 / +0.000 | +0.041 / +0.025 | +0.194 / +0.075 | +0.407 / +0.407 | +2.180 / +0.508 |
| k30_A3 | generic | a_prior | 3 | +2.233 / +0.000 | +0.273 / +0.333 | +1.519 / +1.000 | +0.467 / +0.467 | +4.492 / +1.800 |
| k30_A3 | generic | a_S | 35 | +1.441 / +0.000 | +0.042 / +0.057 | +0.196 / +0.171 | +0.417 / +0.417 | +2.096 / +0.646 |
| k100_A10 | class | a_prior | 16 | +5.632 / +0.000 | +0.047 / +0.000 | +0.723 / +0.000 | +0.447 / +0.447 | +6.849 / +0.447 |
| k100_A10 | class | a_S | 33 | +5.401 / +0.000 | +0.050 / +0.030 | +0.764 / +0.303 | +0.479 / +0.479 | +6.694 / +0.812 |
| k100_A10 | generic | a_prior | 8 | +5.840 / +0.000 | +0.042 / +0.000 | +0.625 / +0.000 | +0.425 / +0.425 | +6.932 / +0.425 |
| k100_A10 | generic | a_S | 38 | +5.345 / +0.000 | +0.044 / +0.039 | +0.678 / +0.526 | +0.479 / +0.479 | +6.546 / +1.045 |

## The changed records by transition: count and mean realized gain

| cell | prior | decline → row | row → decline | row → row |
|---|---|---|---|---|
| k30_A3 | class | 25, +2.426 | 3, +48.433 | 15, +0.237 |
| k30_A3 | generic | 34, +2.347 | 2, -0.550 | 1, +0.100 |
| k100_A10 | class | 18, +8.933 | 1, -9.500 | 15, -0.063 |
| k100_A10 | generic | 31, +8.842 | 1, -9.600 | 7, -0.114 |

## Sensitivity: S = all without the object of the largest |gain| (report only; the readings do not change)

| cell | prior | object left out (its records) | records (objects) | mean realized gain [95 %] | all records |
|---|---|---|---|---|---|
| k30_A3 | class | hope__tomato_sauce_can (1) | 42 (31) | +1.407 [+0.851, +2.008] | 43, +4.872 |
| k30_A3 | generic | hope__canned_tuna (2) | 35 (23) | +2.100 [+1.556, +2.486] | 37, +2.130 |
| k100_A10 | class | hope__mayonnaise_bottle (2) | 32 (23) | +4.102 [+2.018, +5.991] | 34, +4.422 |
| k100_A10 | generic | hope__popcorn_box (2) | 37 (27) | +6.605 [+4.574, +8.430] | 39, +6.762 |
