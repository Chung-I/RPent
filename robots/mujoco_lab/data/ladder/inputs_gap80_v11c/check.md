# Ladder inputs: the lowest-value check

The policy that always picks the lowest D value (expected cost at the true setting, decline at A) against the one-step rule at the true setting, on every first decision. A mismatch's reason: tie (the rule's row shows the same value), guard (the novelty guard declined a cheaper row), probe_margin (a probe row is the lowest, but it beats the best row by less than EVOI_Z Monte Carlo errors), other (a ladder bug).

| setting | first decisions | no menu | match | mismatch by reason | reset artifacts | gap-check oracle: compared, menu differs, first choice differs |
|---|---|---|---|---|---|---|
| k100_A3_abs | 80 | 0 | 80 | 0 | 3 | 80, 0, 0 |
| k30_A3_abs | 80 | 0 | 79 | {"probe_margin": 1} | 3 | 80, 0, 0 |
| k100_A10_abs | 80 | 0 | 79 | {"probe_margin": 1} | 3 | 80, 0, 0 |

B and C are the same under every setting for 80 of 80 cases.

- k30_A3_abs hope__butter/0: probe_margin: lowest probe_squeeze (2.901), rule light_grip (2.916)
- k100_A10_abs hope__cream_cheese/0: probe_margin: lowest probe_squeeze (8.001), rule light_grip (8.052)
