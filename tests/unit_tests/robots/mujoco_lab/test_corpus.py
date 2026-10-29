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

"""Corpus generator: rollouts with hidden properties, destinations and labels (needs the sim)."""

import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RPENT_MUJOCO_SIM_DIR"), reason="sim dir not set")

ROW_KEYS = {
    "key", "object", "theta", "com_end", "truth", "seed", "destination", "s", "force", "v_lift",
    "v_transport", "release_height", "width", "label", "max_z", "drop_z", "loss_phase", "loss_step", "fixture",
    "loss_t", "loss_speed", "zone", "release", "planned", "geometry", "features", "pose", "com_obj_true", "readouts",
    "reset", "rev_steps", "deepest_mm", "artifact",
    "a_transport", "entry", "theta_frag", "fill", "spill_angle_deg", "meter", "motion_label", "damage",      # corpus v9
}
OUTCOME_KEYS = {"label", "max_z", "drop_z", "loss_phase", "loss_step", "loss_t", "loss_speed", "zone", "release", "planned", "readouts",
                "rev_steps", "deepest_mm", "artifact", "meter", "motion_label", "damage"}
DECLINED_KEYS = (ROW_KEYS - OUTCOME_KEYS) | {"declined"}       # a pinch the skill never offers: no rollout


def test_draw_theta_ranges():
    from robots.mujoco_lab.corpus import draw_theta

    rng = np.random.default_rng(0)
    draws = [draw_theta(rng, 0.2) for _ in range(500)]
    scales = [t["mass_scale"] for t, _ in draws]
    assert 0.5 <= min(scales) and max(scales) <= 3.0
    assert all(0.1 <= t["mu"] <= 0.9 for t, _ in draws)
    assert {t["v_lift"] for t, _ in draws} == {0.05, 0.15, 0.4, 0.8}
    assert {e for _, e in draws} == {-1, 1}
    com = [t["com_s"] for t, _ in draws]
    assert 0.3 < sum(c is None for c in com) / len(com) < 0.7
    assert all(0.3 <= c <= 0.8 for c in com if c is not None)


def test_corpus_two_objects_two_draws(tmp_path):
    from robots.mujoco_lab.corpus import main
    from rpent.skills.pick_place.protocol import OUTCOMES

    out = tmp_path / "c.jsonl"
    n = main(["--out", str(out), "--draws", "2", "--workers", "1", "--objects", "ycb__banana,ycb__mug"])
    rows = [json.loads(line) for line in out.open()]
    assert n == 4 and len(rows) == 4
    ran = [r for r in rows if "declined" not in r]
    assert [r["key"] for r in rows if "declined" in r] == ["ycb__mug__d0"]          # its s 0.3 pinch is 9.09 cm
    assert all(set(r) == ROW_KEYS for r in ran) and all(set(r) == DECLINED_KEYS for r in rows if "declined" in r)
    assert all(set(r["reset"]) == {"settle_s", "at_rest", "zone"} for r in rows)
    assert {r["label"] for r in ran} <= set(OUTCOMES)
    assert all(set(r["theta"]) == {"mu", "mass_scale", "com_s", "v_lift"} for r in rows)
    assert all(r["planned"] for r in ran) and all(r["truth"]["mass"] > 0 for r in rows)
    assert all(r["destination"] in ("table_centre", "table_edge", "table_gap", "mouse_pad", "plate", "box", "bin") for r in rows)
    # resume: nothing new is written
    assert main(["--out", str(out), "--draws", "2", "--workers", "1", "--objects", "ycb__banana,ycb__mug", "--resume"]) == 0
    assert len([line for line in out.open()]) == 4


def test_draws_differ_across_objects_at_the_same_index():
    from robots.mujoco_lab.corpus import case_rng, draw_action, draw_theta

    a = draw_theta(case_rng(0, "ycb__banana", 3), 0.2)[0]
    b = draw_theta(case_rng(0, "ycb__mug", 3), 0.2)[0]
    assert a != b
    assert draw_action(case_rng(0, "ycb__banana", 3), 10.0) != draw_action(case_rng(0, "ycb__mug", 3), 10.0)
    assert draw_theta(case_rng(0, "ycb__banana", 3), 0.2)[0] == a       # deterministic


def test_corpus_row_has_pose_readouts_and_true_com():
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import rollout_row

    env = MuJoCoLabEnv()
    row = rollout_row(env, "ycb__banana", 1, 0)
    assert {"pose", "com_obj_true", "readouts"} <= set(row)
    assert len(row["pose"]["R"]) == 9 and len(row["com_obj_true"]) == 3 and {"pivot", "nominal_mass", "pt", "yaw"} <= set(row["pose"])
    assert row["readouts"]["rose"] == (row["label"] != "not_lifted")
    env.reset("ycb__banana", row["theta"], seed=row["seed"] * 1000 + 1, com_end=row["com_end"], fixture=row["fixture"])
    prop = next(p for p in env.proposals() if abs(p.s - row["s"]) < 1e-9)
    assert np.allclose(row["pose"]["pt"], prop.point) and abs(row["pose"]["yaw"] - prop.yaw) < 1e-9
    assert np.allclose(row["com_obj_true"], env.truth()["com_obj"])


def test_a_pinch_wider_than_the_gripper_is_declined_without_a_rollout():
    """cheez_it d0: the drawn s 0.7 pinch is 17 cm across the box, the pads open 9.4 cm. The row keeps the draw,
    the geometry and the features, and says why it did not run."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import rollout_row

    env = MuJoCoLabEnv()

    def execute(*a, **k):
        raise AssertionError("a declined pinch must not run")
    env.execute = execute
    row = rollout_row(env, "ycb__cheez_it", 0, 0)
    assert set(row) == DECLINED_KEYS and row["declined"] == "wider than the gripper opening"
    assert row["s"] == 0.7 and row["width"] > env.max_pinch_width and row["features"]["width"] == row["width"]


def test_trainable_rows_ran_and_were_offered():
    from robots.mujoco_lab.corpus import trainable

    rows = ({"error": "boom"}, {"declined": "wider than the gripper opening"}, {"label": "placed_stable", "artifact": True},
            {"label": "placed_stable", "artifact": False}, {"label": "placed_stable"})
    assert [trainable(r) for r in rows] == [False, False, False, True, True]


def test_draw_damage_ranges():
    """The damage stream: a fillable object is empty with p 0.5, else filled to U[0.3, 0.9] with the container's mass
    scale on [0.8, 1.25]; the force is log-uniform on [0.4, 2.5] f_std and the carry acceleration one of 0.5, 1, 2;
    every draw keeps three standard normals for its fragility."""
    from robots.mujoco_lab.corpus import apply_damage, damage_rng, draw_damage
    from robots.mujoco_lab.fragility import THRESHOLDS_V0, Thresholds, entry_of, load_classes

    classes, th = load_classes(), Thresholds.from_json(THRESHOLDS_V0)
    mug, _ = entry_of("ycb__mug", classes)
    d = [draw_damage(damage_rng(0, "ycb__mug", i), mug, th, 10.0) for i in range(400)]
    filled = [x for x in d if x["fill"] > 0]
    assert 0.4 < len(filled) / len(d) < 0.6 and all(0.3 <= x["fill"] <= 0.9 for x in filled)
    assert all(0.8 <= x["mass_scale"] <= 1.25 for x in filled) and all("mass_scale" not in x for x in d if x["fill"] == 0)
    assert all(x["spill_angle_deg"] is None for x in d if x["fill"] == 0) and all(0 < x["spill_angle_deg"] < 90 for x in filled)
    assert all(4.0 <= x["force"] <= 25.0 for x in d) and {x["a_transport"] for x in d} == {0.5, 1.0, 2.0}
    assert all(len(x["theta_frag"]["z"]) == 3 and x["theta_frag"]["v_break_ms"] > 0 for x in d)
    brick, _ = entry_of("ycb__brick", classes)
    assert all(draw_damage(damage_rng(0, "ycb__brick", i), brick, th, 10.0)["fill"] == 0.0 for i in range(50))
    theta, act = apply_damage({"mass_scale": 2.5, "mu": 0.4, "com_s": 0.6, "v_lift": 0.15}, {"force": 10.0, "s": 0.5}, filled[0])
    assert theta["mass_scale"] == filled[0]["mass_scale"] and theta["com_s"] is None and theta["mu"] == 0.4
    assert act == {"force": filled[0]["force"], "s": 0.5, "a_transport": filled[0]["a_transport"]}
    assert draw_damage(damage_rng(0, "ycb__mug", 3), mug, th, 10.0) == d[3]          # deterministic


def test_control_rows_reproduce_v8():
    """Three rows of corpus v8 rerun as control rows (v8's draws and actions, a_transport 2.0, no liquid): the meter and
    the fragility change no motion, so the motion labels are v8's."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import rollout_row

    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "robots", "mujoco_lab", "data", "corpus_v8.jsonl")
    want = ("ycb__banana__d1", "ycb__brick__d1", "hot3d__mug__d2")         # placed_stable, not_lifted, dropped_low in v8
    v8 = {r["key"]: r for r in map(json.loads, open(os.path.normpath(path))) if r["key"] in want}
    env = MuJoCoLabEnv()
    for key in want:
        oid, idx = key.rsplit("__d", 1)
        row = rollout_row(env, oid, int(idx), v8[key]["seed"], mode="control")
        assert row["a_transport"] == 2.0 and row["fill"] == 0.0 and row["force"] == v8[key]["force"]
        assert row["motion_label"] == v8[key]["label"], key


def test_m2_rows_are_v8_rows():
    """Mode m2 is milestone 2a's world: v8's draws and actions, no liquid and no fragility, so nothing can be damaged
    and the label is v8's (a crushable object squeezed at v8's force, and a v8 row)."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv
    from robots.mujoco_lab.corpus import rollout_row

    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "robots", "mujoco_lab", "data", "corpus_v8.jsonl")
    v8 = next(r for r in map(json.loads, open(os.path.normpath(path))) if r["key"] == "ycb__banana__d1")
    env = MuJoCoLabEnv()
    for oid, idx, seed in (("ycb__jello", 0, 0), ("ycb__banana", 1, v8["seed"])):
        row = rollout_row(env, oid, idx, seed, mode="m2")
        assert row["fill"] == 0.0 and row["theta_frag"] is None and row["damage"] is None and row["a_transport"] == 2.0
        assert row["label"] == row["motion_label"]
    assert row["label"] == v8["label"] and row["force"] == v8["force"]
