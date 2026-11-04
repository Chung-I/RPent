A: [None] particles, B: [20000] particles. Regret = the default rule's first choice minus the lowest score on E's first table. diff = B - A per case (object-bootstrap 95 % interval).

| setting | H | cases | regret A | regret B | diff B - A | 95 % interval | cases with diff != 0 | above 0 A / B | E's first choice changed | lowest score shift: mean, max abs | table shift: mean, max | E s: A, B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k100_A10_abs | 1 | 77 | 1.089 | 1.512 | +0.423 | [-0.030, +1.307] | 19 | 24 / 25 | 1 | -0.0045, 0.2880 | 1.7330, 11.6631 | 0.0, 0.0 |
| k100_A10_abs | 2 | 77 | 1.089 | 1.512 | +0.423 | [-0.030, +1.307] | 18 | 25 / 25 | 0 | -0.0012, 0.2754 | 1.7334, 11.6631 | 1.3, 10.6 |
| k100_A10_abs | 3 | 77 | 1.089 | 1.512 | +0.423 | [-0.031, +1.307] | 18 | 25 / 25 | 0 | -0.0013, 0.2754 | 1.7335, 11.6631 | 111.3, 710.1 |
| k100_A3_abs | 1 | 77 | 2.029 | 2.527 | +0.498 | [-0.028, +1.530] | 21 | 35 / 35 | 0 | +0.0004, 0.0242 | 1.7117, 11.5007 | 0.0, 0.0 |
| k100_A3_abs | 2 | 77 | 2.029 | 2.527 | +0.498 | [-0.029, +1.532] | 21 | 35 / 35 | 0 | +0.0003, 0.0246 | 1.7113, 11.5007 | 1.4, 10.7 |
| k100_A3_abs | 3 | 77 | 2.029 | 2.527 | +0.498 | [-0.027, +1.553] | 21 | 35 / 35 | 0 | +0.0003, 0.0246 | 1.7113, 11.5007 | 116.8, 709.9 |
| k30_A3_abs | 1 | 77 | 0.414 | 0.542 | +0.128 | [-0.011, +0.399] | 22 | 29 / 29 | 0 | +0.0002, 0.0459 | 0.5195, 3.4991 | 0.0, 0.0 |
| k30_A3_abs | 2 | 77 | 0.413 | 0.543 | +0.129 | [-0.009, +0.400] | 20 | 29 / 29 | 0 | +0.0003, 0.0471 | 0.5201, 3.4991 | 1.4, 10.5 |
| k30_A3_abs | 3 | 77 | 0.413 | 0.543 | +0.129 | [-0.010, +0.401] | 20 | 29 / 29 | 0 | +0.0002, 0.0471 | 0.5201, 3.4991 | 113.9, 711.6 |

- k100_A10_abs: 77 paired, 3 left out, menu rows differ on 0; the default rule's first choice changed on 1, the oracle rule's on 1.
  - default: fruits_veggies__lemon1/1: probe_squeeze -> commit_std
  - oracle rule: ycb__tuna_can/9: firm_grip -> probe_weigh
  - E at H = 1: ycb__tuna_can/9: firm_grip (9.838 -> 9.782) -> probe_weigh (9.878 -> 9.550)
- k100_A3_abs: 77 paired, 3 left out, menu rows differ on 0; the default rule's first choice changed on 1, the oracle rule's on 0.
  - default: fruits_veggies__lemon1/1: probe_squeeze -> commit_std
- k30_A3_abs: 77 paired, 3 left out, menu rows differ on 0; the default rule's first choice changed on 1, the oracle rule's on 0.
  - default: fruits_veggies__lemon1/1: probe_squeeze -> commit_std
