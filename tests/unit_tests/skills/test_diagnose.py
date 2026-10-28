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

"""The failed-attempt rule table: one case per rule, and the belief line on every diagnosis."""

from rpent.skills.pick_place.diagnose import diagnose

BEFORE = {"mass_median": 0.4, "mu_median": 0.5, "p_com_shifted": 0.5, "ess": 2000.0}
AFTER = {"mass_median": 0.45, "mu_median": 0.18, "p_com_shifted": 0.7, "ess": 640.0}
ACT = {"force": 10.0, "v_lift": 0.15}


def _r(**kw):
    r = {"rose": True, "fz_hold_n": None, "fz_slip_max_n": None, "fz_at_loss_n": None, "rot_deg": 0.0, "heavy_side": 0,
         "loss_phase": None, "loss_t": None, "loss_since_lift_s": None, "v_lift": 0.15}
    r.update(kw)
    return r


def test_diagnose_rules():
    d = diagnose("not_lifted", _r(rose=False, fz_slip_max_n=3.0), ACT, BEFORE, AFTER)
    assert d[0] == "never left the table: too heavy or too slippery for this squeeze"
    assert any("friction about 0.15" in s for s in d)                     # 3 N over 2 x 10 N
    assert not any("friction about" in s for s in diagnose("not_lifted", _r(rose=False), ACT, BEFORE, AFTER))

    d = diagnose("dropped_low", _r(loss_phase="hold", fz_at_loss_n=4.0, rot_deg=3.0, loss_since_lift_s=1.2), ACT, BEFORE, AFTER)
    assert d[0] == "rose then slid without rotating: slippery or heavy at this squeeze"
    assert any("friction about 0.20" in s for s in d)                     # 4 N over 2 x 10 N

    d = diagnose("dropped_high", _r(loss_phase="transport", rot_deg=25.0, heavy_side=-1, loss_since_lift_s=2.0), ACT, BEFORE, AFTER)
    assert d[0] == "rotated 25° before the fall: the CoM is off the pinch toward the − end"
    d = diagnose("dropped_low", _r(loss_phase="lift", rot_deg=12.0, heavy_side=1, loss_since_lift_s=0.5), ACT, BEFORE, AFTER)
    assert d[0] == "rotated 12° before the fall: the CoM is off the pinch toward the + end"
    assert not any(s.startswith("rose then slid") for s in d)

    fast = dict(ACT, v_lift=0.5)
    d = diagnose("dropped_low", _r(loss_phase="lift", rot_deg=2.0, loss_since_lift_s=0.1), fast, BEFORE, AFTER)
    assert "lost at the start of a fast lift: lift slower" in d
    assert "lost at the start of a fast lift: lift slower" not in diagnose(
        "dropped_low", _r(loss_phase="lift", rot_deg=2.0, loss_since_lift_s=0.1), ACT, BEFORE, AFTER)

    for label in ("tipped", "missed"):
        assert f"set-down: {label}" in diagnose(label, _r(), ACT, BEFORE, AFTER)

    for label in ("placed_stable", "not_lifted", "tipped", "dropped_high"):
        line = diagnose(label, _r(), ACT, BEFORE, AFTER)[-1]
        assert line.startswith("belief: ") and "friction median 0.50 → 0.18" in line and "mass median 0.400 → 0.450 kg" in line
    assert diagnose("placed_stable", _r(), ACT, BEFORE, AFTER)[:-1] == []


def test_diagnose_tolerates_missing_readouts():
    """A substrate without readouts (the milestone 1 fakes) still gets the label rules and the belief line."""
    d = diagnose("not_lifted", {}, ACT, BEFORE, AFTER)
    assert d[0].startswith("never left the table") and d[-1].startswith("belief: ")
    assert diagnose("tipped", {}, ACT, BEFORE, AFTER)[0] == "set-down: tipped"


def test_diagnose_corrects_the_friction_reading_with_the_fitted_law():
    """The lab's slip and loss forces read low (the fitted median log ratio is about -1 for slip, -2.3 for
    a loss); with the readout model's laws the friction number is corrected by exp(-b)."""
    laws = {"slip": (-1.0, 1.0), "loss": (-2.0, 1.0)}
    d = diagnose("not_lifted", _r(rose=False, fz_slip_max_n=3.0), ACT, BEFORE, AFTER, laws=laws)
    assert any("friction about 0.41" in s and "fitted" in s for s in d)          # 0.15 x e^1.0
    d = diagnose("dropped_low", _r(loss_phase="hold", fz_at_loss_n=4.0, rot_deg=3.0), ACT, BEFORE, AFTER, laws=laws)
    assert any("friction about 1.48" in s for s in d)                           # 0.20 x e^2.0
    d = diagnose("not_lifted", _r(rose=False, fz_slip_max_n=3.0), ACT, BEFORE, AFTER, laws={"slip": None})
    assert any("friction about 0.15" in s and "fitted" not in s for s in d)


def test_no_friction_number_from_an_unusable_force():
    """A slip or loss force at or below 0 N gives no friction (and no weight bound); a positive force
    whose friction lies outside the belief's range gives no number either."""
    d = diagnose("not_lifted", _r(rose=False, fz_slip_max_n=-5.0), ACT, BEFORE, AFTER)
    assert not any("friction about" in s or "-5.0" in s for s in d) and any("not usable" in s for s in d)
    d = diagnose("dropped_low", _r(loss_phase="hold", fz_at_loss_n=0.0, rot_deg=3.0), ACT, BEFORE, AFTER)
    assert not any("friction about" in s for s in d) and any("not usable" in s for s in d)
    d = diagnose("not_lifted", _r(rose=False, fz_slip_max_n=39.0), ACT, BEFORE, AFTER, mu_range=(0.1, 0.9))   # 1.95
    assert not any("friction about" in s for s in d) and any("outside the belief" in s for s in d)


def test_friction_number_carries_its_one_reading_range():
    """3 N over 2 x 10 N is 0.15, corrected by e^1 (b -1) to 0.41; with the law's sd 1 the 80 % range from
    this one reading is 0.41 x e^(-/+1.28): 0.11 to 1.47."""
    d = diagnose("not_lifted", _r(rose=False, fz_slip_max_n=3.0), ACT, BEFORE, AFTER, laws={"slip": (-1.0, 1.0)},
                 mu_range=(0.1, 0.9))
    line = next(s for s in d if "friction about" in s)
    assert "friction about 0.41" in line and "0.11 to 1.47" in line
