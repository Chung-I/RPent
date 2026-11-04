# E: the receding-horizon reference planner

E scores every row the agents may choose with a lookahead of H attempts on the true menu, and replans at each real menu. H = 1 is the menu's own prices. The default rule's regret is its first choice's score on E's table minus the lowest score of that table (mean, object-bootstrap 95 % interval; n50: e1_gap.power_n, the cases that show half of this gap at 80 % power). A case whose reset did not reach rest in any of its runs is left out of the regrets, the horizon changes and the two lists in Checks; the H = 1 comparison, the table agreement and the wall times count every case.

20000 particles per belief.

## E at H = 1 against the oracle rule (first decision)

| setting | compared | match | mismatch by reason |
|---|---|---|---|
| k100_A10_abs | 80 | 80 | 0 |
| k100_A3_abs | 80 | 80 | 0 |
| k30_A3_abs | 80 | 80 | 0 |

## The default rule's first-decision regret against E

| setting | H | cases (objects) | regret | 95 % interval | n50 | cases above 0 | oracle rule's regret |
|---|---|---|---|---|---|---|---|
| k100_A10_abs | 1 | 77 (77) | 1.512 | [+0.514, +2.827] | 191 | 25 | +0.000 [+0.000, +0.000] |
| k100_A10_abs | 2 | 77 (77) | 1.512 | [+0.528, +2.774] | 191 | 25 | +0.000 [+0.000, +0.000] |
| k100_A10_abs | 3 | 77 (77) | 1.512 | [+0.563, +2.809] | 191 | 25 | +0.000 [+0.000, +0.000] |
| k100_A3_abs | 1 | 77 (77) | 2.527 | [+1.211, +4.114] | 109 | 35 | +0.000 [+0.000, +0.000] |
| k100_A3_abs | 2 | 77 (77) | 2.527 | [+1.255, +4.087] | 109 | 35 | +0.000 [+0.000, +0.000] |
| k100_A3_abs | 3 | 77 (77) | 2.527 | [+1.236, +4.073] | 109 | 35 | +0.000 [+0.000, +0.000] |
| k30_A3_abs | 1 | 77 (77) | 0.542 | [+0.223, +0.966] | 154 | 29 | +0.000 [+0.000, +0.000] |
| k30_A3_abs | 2 | 77 (77) | 0.543 | [+0.215, +0.983] | 154 | 29 | +0.000 [+0.000, +0.000] |
| k30_A3_abs | 3 | 77 (77) | 0.543 | [+0.236, +0.948] | 154 | 29 | +0.000 [+0.000, +0.000] |

## How often a longer horizon changes E's choice

| setting | H | first decisions | changed | changes | every decision of the H = 3 run | changed |
|---|---|---|---|---|---|---|
| k100_A10_abs | 2 | 77 | 0 | none | 83 | 0 |
| k100_A10_abs | 3 | 77 | 0 | none | 83 | 0 |
| k100_A3_abs | 2 | 77 | 0 | none | 77 | 0 |
| k100_A3_abs | 3 | 77 | 0 | none | 77 | 0 |
| k30_A3_abs | 2 | 77 | 0 | none | 78 | 0 |
| k30_A3_abs | 3 | 77 | 0 | none | 78 | 0 |

## Checks

- k100_A10_abs: 80 cases, 0 without a menu, 0 incomplete, 3 reset artifacts; the first tables agree across the runs on 80 of 80.
  - A policy's first choice below E's on E's own table: 0.
  - The lowest score rises with H: 2; hope__granola_bars/5 H 2 9.639 > 9.630; hot3d__frozen_vegetable_block/8 H 3 5.718 > 5.718.
- k100_A3_abs: 80 cases, 0 without a menu, 0 incomplete, 3 reset artifacts; the first tables agree across the runs on 80 of 80.
  - A policy's first choice below E's on E's own table: 0.
  - The lowest score rises with H: 0.
- k30_A3_abs: 80 cases, 0 without a menu, 0 incomplete, 3 reset artifacts; the first tables agree across the runs on 80 of 80.
  - A policy's first choice below E's on E's own table: 0.
  - The lowest score rises with H: 0.

## Wall time per case (seconds)

| setting | H | E's own scoring: mean, p90, max | the whole run: mean, p90, max |
|---|---|---|---|
| k100_A10_abs | 1 | 0.0, 0.0, 0.0 | 13.8, 25.0, 51.1 |
| k100_A10_abs | 2 | 10.8, 18.1, 22.6 | 24.5, 39.6, 66.0 |
| k100_A10_abs | 3 | 723.6, 1314.6, 1445.7 | 748.2, 1352.7, 1482.1 |
| k100_A3_abs | 1 | 0.0, 0.0, 0.0 | 11.4, 15.9, 19.5 |
| k100_A3_abs | 2 | 10.8, 17.4, 26.9 | 22.3, 30.8, 45.2 |
| k100_A3_abs | 3 | 723.6, 1313.0, 1457.3 | 745.7, 1343.2, 1495.1 |
| k30_A3_abs | 1 | 0.0, 0.0, 0.0 | 12.9, 20.3, 39.2 |
| k30_A3_abs | 2 | 10.7, 17.8, 22.6 | 23.5, 36.4, 56.6 |
| k30_A3_abs | 3 | 725.2, 1315.6, 1451.1 | 748.6, 1348.5, 1487.9 |
