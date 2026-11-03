# Ladder inputs: the lowest-value check

The policy that always picks the lowest D value (expected cost at the true setting, decline at A) against the one-step rule at the true setting, on every first decision. A mismatch's reason: tie (the rule's row shows the same value), guard (the novelty guard declined a cheaper row), probe_margin (a probe row is the lowest, but it beats the best row by less than EVOI_Z Monte Carlo errors), other (a ladder bug).

| setting | first decisions | no menu | match | mismatch by reason | reset artifacts | gap-check oracle: compared, menu differs, first choice differs |
|---|---|---|---|---|---|---|
| k100_A3_abs | 80 | 0 | 80 | 0 | 3 | 80, 0, 0 |
| k30_A3_abs | 80 | 0 | 79 | {"probe_margin": 1} | 3 | 80, 0, 0 |
| k100_A10_abs | 80 | 0 | 79 | {"probe_margin": 1} | 3 | 80, 0, 0 |

B and C are the same under every setting for 42 of 80 cases (they differ for handal__red_serving_spoon/10, handal__hammer_6/1, handal__ladle/6, handal__measuring_spoon/8, hope__cream_cheese/0, ycb__soft_scrub/2, hot3d__foam_roller/10, hot3d__frozen_vegetable_block/8, hot3d__dumbbell/7, hot3d__computer_mouse/7, hot3d__soup_can/4, hope__mustard_bottle/9, ycb__dry_erase_marker/10, handal__hammer_4/4, ycb__spam_can/6, hope__orange_juice_carton/10, ycb__tomato_soup_can/2, ycb__tuna_can/9, hope__mayonnaise_bottle/1, hope__pitted_cherries/2, handal__hammer_7/3, hot3d__lizard_figurine/5, handal__hammer_8/0, ycb__brick/1, hope__butter/0, hot3d__glasses/3, hope__spaghetti/1, hot3d__mug/2, hot3d__smartphone/9, handal__hammer_3/9, ycb__jello/0, hope__canned_tuna/0, hope__tomato_sauce_can/7, hot3d__coffee_pot/0, handal__hammer/6, hope__raisin_box/7, hope__peas_and_carrots/8, ycb__scissors/2).

- k30_A3_abs hope__butter/0: probe_margin: lowest probe_squeeze (2.901), rule light_grip (2.916)
- k100_A10_abs hope__cream_cheese/0: probe_margin: lowest probe_squeeze (8.001), rule light_grip (8.052)
