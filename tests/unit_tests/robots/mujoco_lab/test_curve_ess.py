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

"""The second menus' ESS counts of the success-damage curve's Stage 2 (curve_ess; added after the v12b item result, before
any curve reading, not pre-registered)."""

import json

import pytest

from robots.mujoco_lab import curve as cv
from robots.mujoco_lab import curve_ess as ce

K0, K0B, K4 = cv.dkey("R0", 10.0, 1.0), cv.dkey("R0", 10.0, 2.0), cv.dkey("R4", 10.0, None)
LK = {"R0": {"knob": 1.0, "key": K0}, "R4": {"knob": None, "key": K4}}


def _sim(first, label, menu2=None, seed_k=0):
    return {"seed_k": seed_k, "first": first, "label": label, "artifact": False, "error": None, "menu2": menu2, "second": {}}


def _m2(ess, status="ok"):
    m = {"status": status, "rows": [], "decisions": {K0: "give_up", K0B: "give_up", K4: "give_up"}}
    return dict(m, ess=ess) if ess is not None else m


def _s2(o, dec, sims, side, **kw):
    return dict({"object": o, "idx": 0, "side": side, "thin": False, "decisions": dec, "sims": sims, "dropped": False,
                 "skipped": None, "error": None, "stop": None}, **kw)


def _recs():
    return [
        _s2("a", {K0: "commit_std", K0B: "commit_std", K4: "firm_grip"},
            [_sim("commit_std", "tipped", _m2(50.0)), _sim("firm_grip", "placed_stable")], "validation"),
        _s2("b", {K0: "commit_std", K0B: "give_up", K4: "commit_std"},
            [_sim("commit_std", "tipped", _m2(1.0), seed_k=1), _sim("commit_std", "tipped", _m2(500.0))], "validation"),
        _s2("c", {K0: "commit_std", K0B: "commit_std", K4: "commit_std"}, [_sim("commit_std", "tipped", _m2(2.0))], "validation",
            dropped=True),
        _s2("d", {K0: "give_up", K0B: "commit_std", K4: "commit_std"},
            [_sim("commit_std", "tipped", _m2(None, status="unreachable"))], "test"),
        _s2("e", {K0: "firm_grip", K0B: "firm_grip", K4: "firm_grip"}, [_sim("firm_grip", "crushed")], "test"),
    ]


def test_second_menu_is_the_menu_the_rules_episode_reaches():
    a, b, d, e = (r for r in _recs() if r["object"] != "c")
    assert cv.episode(a["decisions"], a["sims"], K0)["labels"] == ["tipped"]  # the episode reaches a second decision (a give-up)
    assert ce.second_menu(a["decisions"], a["sims"], K0)["ess"] == 50.0
    assert ce.second_menu(a["decisions"], a["sims"], K4) is None               # the first attempt placed the object
    assert ce.second_menu(d["decisions"], d["sims"], K0) is None               # a give-up at the first decision
    assert ce.second_menu(e["decisions"], e["sims"], K0) is None               # an irreversible first attempt
    assert ce.second_menu(b["decisions"], b["sims"], K4)["ess"] == 500.0       # the main seed's simulation, not seed 1's


def test_ess_counts_per_side_and_rule_at_the_locked_knob_over_the_kept_cases():
    c = ce.ess_counts(_recs(), LK, ("R0", "R4"))
    v, t = c["validation"], c["test"]
    assert v["cases"] == 2 and t["cases"] == 2                                 # case c is dropped
    assert v["rules"]["R0"] == {"knob": 1.0, "second_menus": 2, "ess_low": 1, "no_ess": 0}
    assert v["rules"]["R4"] == {"knob": None, "second_menus": 1, "ess_low": 0, "no_ess": 0}
    assert t["rules"]["R0"]["second_menus"] == 0                               # its other knob (K0B) is not the locked one
    assert t["rules"]["R4"] == {"knob": None, "second_menus": 1, "ess_low": 0, "no_ess": 1}
    assert v["distinct"] == {"second_menus": 2, "ess_low": 1, "no_ess": 0}     # seed 1's menu is not counted
    assert t["distinct"] == {"second_menus": 1, "ess_low": 0, "no_ess": 1}
    assert ce.ess_counts(_recs(), LK, ("R0", "R4"), threshold=600.0)["validation"]["rules"]["R0"]["ess_low"] == 2


def _write(tmp_path, recs):
    d = tmp_path / "s2"
    d.mkdir()
    (d / "run_config.json").write_text(json.dumps({"rules": ["R0", "R4"], "code_revision": "abc"}))
    (d / "run_metrics.json").write_text("{}")
    for side in cv.SIDES:
        (d / f"records_{side}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs if r["side"] == side))
    s = tmp_path / "sum"
    s.mkdir()
    sha = {f"{d} {side}": cv._sha(cv.records_path(str(d), side)) for side in cv.SIDES}
    rep = {"lock": LK, "provenance": {"records_sha256": sha, "locked": "lock/locked.json (sha256 x, commit c0ffee)"}}
    (s / "stage2_summary.json").write_text(json.dumps(rep))
    return str(d), str(s)


def test_main_counts_only_after_the_summary_on_the_records_it_read(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    d, s = _write(tmp_path, _recs())
    with pytest.raises(SystemExit, match="summary"):
        ce.main(["--runs", d, "--summary", str(tmp_path / "nosum"), "--out", str(tmp_path / "x")])
    out = ce.main(["--runs", d, "--summary", s, "--out", str(tmp_path / "ess")])
    assert out["counts"]["validation"]["rules"]["R0"]["ess_low"] == 1 and out["threshold"] == cv.ESS_APART
    md = (tmp_path / "ess" / "ess.md").read_text()
    assert "Added after the v12b item result, before any curve reading (not pre-registered)" in md
    assert "It changes no lock, no margin and no reading." in md and "commit c0ffee" in md
    with pytest.raises(SystemExit, match="exists"):
        ce.main(["--runs", d, "--summary", s, "--out", str(tmp_path / "ess")])
    with open(f"{d}/records_test.jsonl", "a") as f:                            # a record the summary did not read
        f.write(json.dumps(_s2("f", {K0: "give_up", K0B: "give_up", K4: "give_up"}, [], "test")) + "\n")
    with pytest.raises(SystemExit, match="records"):
        ce.main(["--runs", d, "--summary", s, "--out", str(tmp_path / "ess2")])
