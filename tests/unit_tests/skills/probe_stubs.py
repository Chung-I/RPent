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

"""Stub probe models and an outcome model whose best row depends on the friction, for the value-of-
information tests (menu and tools)."""

import numpy as np

from rpent.skills.pick_place.destinations import DESTINATIONS, edge_distance
from rpent.skills.pick_place.probes import PROBE_KINDS
from rpent.skills.pick_place.protocol import OUTCOMES, Prediction

G = 9.81
CENTRE_EDGE = float(edge_distance(DESTINATIONS["table_centre"]))


class PerfectPush:
    """A push that tells exactly whether mu is above ``cut``; P(tipped) ``p_tip``, P(thrown) ``p_thrown``."""

    probe = "push"
    classes = PROBE_KINDS["push"]

    def __init__(self, cut: float = 0.3, p_tip: float = 0.0, p_thrown: float = 0.0):
        self.cut, self.p_tip, self.p_thrown = cut, p_tip, p_thrown

    def p_kind(self, X):
        row = np.zeros(len(self.classes))
        row[0], row[1] = 1.0 - self.p_tip, self.p_tip
        return np.tile(row, (len(X), 1))

    def risk(self, X):
        return np.full(len(X), float(self.p_thrown))

    def terminal_risk(self, X):
        return self.risk(X)

    def sample(self, x_row, particle, force, rng, p=None):
        return {"probe": "push", "kind": "slid", "reading": {"high": float(particle["mu"] > self.cut)}, "thrown": False}

    def loglik(self, obs, X, arrays, force, temper=None, terms=None, P=None):
        hi = np.asarray(arrays["mu"], dtype=float) > self.cut
        ok = hi if obs["reading"]["high"] else ~hi
        return np.where(ok, 0.0, -np.inf), {"used": ["kind", "reading"], "rejected": []}


class FlatWeigh:
    """A weigh probe that tells nothing."""

    probe = "weigh"
    classes = PROBE_KINDS["weigh"]

    def p_kind(self, X):
        return np.tile([1.0, 0.0, 0.0], (len(X), 1))

    def risk(self, X):
        return np.zeros(len(X))

    def terminal_risk(self, X):
        return self.risk(X)

    def sample(self, x_row, particle, force, rng, p=None):
        return {"probe": "weigh", "kind": "measured", "reading": {"m_hat": 0.3}, "thrown": False}

    def loglik(self, obs, X, arrays, force, temper=None, terms=None, P=None):
        return np.zeros(len(X)), {"used": ["kind"], "rejected": []}


class FlatSqueeze(FlatWeigh):
    """A squeeze probe (milestone 2b) that tells nothing and never damages."""

    probe = "squeeze"
    classes = ("measured",)

    def p_kind(self, X):
        return np.ones((len(X), 1))

    def sample(self, x_row, particle, force, rng, p=None):
        return {"probe": "squeeze", "kind": "measured", "reading": {"k_hat": 2.0}, "thrown": False}


class MuGate:
    """firm_grip (twice the standard squeeze ``f_std``) places only above mu 0.3, else drops high;
    slow_lift always places; the other rows drop high; with ``inboard`` a row to the table centre
    always places."""

    kind = "classifier"

    def __init__(self, f_std: float, inboard: bool = False):
        self.f_std, self.inboard = f_std, inboard

    def predict_many(self, feats):
        out = []
        for f in feats:
            force = f["mass"] * G / (2 * f["mu"] * f["load_ratio"])
            if self.inboard and abs(f["dest_edge_dist"] - CENTRE_EDGE) < 1e-9:
                ok = True
            elif f["v_lift"] < 0.1:
                ok = True
            elif force > 1.5 * self.f_std:
                ok = f["mu"] > 0.3
            else:
                ok = False
            p = {k: 0.0 for k in OUTCOMES}
            p["placed_stable" if ok else "dropped_high"] = 1.0
            out.append(Prediction(p=p))
        return out
