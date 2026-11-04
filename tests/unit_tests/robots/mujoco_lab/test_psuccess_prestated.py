# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""The pre-stated test of the v12 spread check: the CoM share of the spread removed, the per-object Spearman, the far
grasps' |dP|, and the contrasts by a joint object bootstrap over the objects both arms have."""

import numpy as np
import pytest


def _rec(obj, stage="weigh", sd=(0.2, 0.1), sdc=(0.1, 0.08), p=(0.5, 0.6), s=0.5, com=0.5, kind="measured", key=None):
    """A spread record: commit_std's (or the attempted row's) P(placed_stable) at the prior and after ``stage``."""
    c = {"belief": {"prior": {"sd": sd[0], "mean": p[0]}, stage: {"sd": sd[1], "mean": p[1]}},
         "com_true": {"prior": {"sd": sdc[0], "mean": p[0]}, stage: {"sd": sdc[1], "mean": p[1]}},
         "all_true": {"prior": {"sd": 0.0, "mean": 0.7}, stage: {"sd": 0.0, "mean": 0.7}}}
    r = {"stage": stage, "prior": "class", "object": obj, "key": key or f"{obj}__{s}", "truth": {"com": com}, "kind": kind}
    if stage == "attempt":
        return {**r, "row": {"placed": c}, "s": s}
    return {**r, "commit_std": {"placed": c}, "s_std": s}


def test_case_values_read_commit_std_for_a_probe_and_the_attempted_row_for_an_attempt():
    from robots.mujoco_lab.psuccess_prestated import case_values

    v = case_values(_rec("a", "identify", s=0.3, com=0.55))
    assert v["sd_prior"] == 0.2 and v["sd_after"] == 0.1 and v["sdc_prior"] == 0.1 and v["sdc_after"] == 0.08
    assert v["dp"] == pytest.approx(0.1) and v["d"] == pytest.approx(0.25) and v["p_all_true"] == 0.7
    w = case_values(_rec("a", "attempt", s=0.9, com=0.5))
    assert w["d"] == pytest.approx(0.4)
    with pytest.raises(KeyError):                         # a record without the row's pinch (the 2,000-particle v11c ones)
        case_values({k: v for k, v in _rec("a").items() if k != "s_std"})


def test_the_com_share_of_the_spread_removed_is_a_ratio_of_sums():
    from robots.mujoco_lab.psuccess_prestated import case_values, r_com, split_com_after

    V = [case_values(_rec("a", sd=(0.2, 0.1), sdc=(0.1, 0.08))), case_values(_rec("b", sd=(0.3, 0.1), sdc=(0.1, 0.1)))]
    # D_com prior: 0.1 + 0.2 = 0.3; after: 0.02 + 0.0 = 0.02; removed: 0.5 - 0.2 = 0.3
    assert r_com(V) == pytest.approx((0.3 - 0.02) / 0.3)
    assert split_com_after(V) == pytest.approx(1.0 - 0.18 / 0.2)


def test_the_per_object_shares_and_their_spearman():
    from robots.mujoco_lab.psuccess_prestated import case_values, object_shares, spearman

    V = [case_values(_rec(o, sd=(0.2, a), sdc=(c, 0.05), key=f"{o}{i}")) for i, (o, a, c) in
         enumerate((("a", 0.18, 0.18), ("a", 0.18, 0.18), ("b", 0.1, 0.1), ("c", 0.02, 0.02)))]
    sh = object_shares(V)
    assert sh["a"] == pytest.approx((1 - 0.36 / 0.4, 1 - 0.36 / 0.4)) and sh["c"] == pytest.approx((0.9, 0.9))
    assert spearman(V) == pytest.approx(1.0)            # a larger prior CoM share goes with a larger share removed
    V2 = [case_values(_rec(o, sd=(0.2, a), sdc=(c, 0.05))) for o, a, c in (("a", 0.18, 0.02), ("b", 0.1, 0.1), ("c", 0.02, 0.18))]
    assert spearman(V2) == pytest.approx(-1.0)


def test_far_grasps_move_p_more_than_near_grasps():
    from robots.mujoco_lab.psuccess_prestated import FAR, case_values, far_near

    V = [case_values(_rec("a", p=(0.5, 0.8), s=0.5 + FAR, com=0.5)),      # d = 0.15: far
         case_values(_rec("b", p=(0.5, 0.3), s=0.9, com=0.5)),            # far, a fall
         case_values(_rec("c", p=(0.5, 0.55), s=0.55, com=0.5))]          # near
    f = far_near(V)
    assert f["n_far"] == 2 and f["n_near"] == 1
    assert f["F"] == pytest.approx((0.3 + 0.2) / 2 - 0.05)
    assert f["far_signed_dp"] == pytest.approx((0.3 - 0.2) / 2)
    assert f["truth_gap_change"] == pytest.approx(np.mean([abs(0.8 - 0.7) - 0.2, abs(0.3 - 0.7) - 0.2, abs(0.55 - 0.7) - 0.2]))


def test_a_contrast_resamples_the_shared_objects_jointly():
    from robots.mujoco_lab.psuccess_prestated import contrast, decision

    A = [_rec(o, sd=(0.2, 0.05 + 0.02 * j), sdc=(0.1 - 0.01 * j, 0.02), p=(0.5, 0.5 + 0.05 * i + 0.01 * j), s=(0.5, 0.8)[i],
              key=f"{o}{i}") for j, o in enumerate("abcdef") for i in range(2)]          # objects differ; far and near
    c = contrast(A, A, n_boot=200, seed=0)
    for stat in ("r_com", "rho", "F"):
        assert c[stat]["diff"] == pytest.approx(0.0, abs=1e-12) and c[stat]["ci"] == pytest.approx([0.0, 0.0], abs=1e-12)
        assert decision(c[stat]["ci"]) == "undecided"
    B = A + [_rec("z", sd=(0.2, 0.0), sdc=(0.2, 0.0))]                   # an object only one arm has is left out
    assert contrast(A, B, n_boot=50, seed=0)["n_objects"] == 6
    assert decision([0.01, 0.2]) == "supported" and decision([-0.2, -0.01]) == "contradicted"


def test_a_lower_com_spread_after_gives_a_positive_r_com_contrast():
    from robots.mujoco_lab.psuccess_prestated import contrast

    rng = np.random.default_rng(1)
    A, B = [], []
    for o in "abcdefgh":
        for i in range(3):
            x = rng.uniform(0.15, 0.25)
            A.append(_rec(o, sd=(x, 0.05), sdc=(0.05, 0.045), key=f"{o}{i}"))     # most of the CoM part removed
            B.append(_rec(o, sd=(x, 0.10), sdc=(0.05, 0.005), key=f"{o}{i}"))     # little of it removed
    c = contrast(A, B, n_boot=300, seed=0)
    assert c["r_com"]["diff"] > 0 and c["r_com"]["ci"][0] > 0
