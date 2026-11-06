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


"""The success-damage curve (daily-logs researches/property-belief-manipulation/designs/2026-11-06-success-damage-curve-design.md,
approved by the user on 2026-11-07; the experiment's README is experiments/2026-11-07-success-damage-curve).

Stage 0 (``stage0_points``): a first look at the m4 sweep (bundle m2b2, seed 12, 1,045 cases per cell, s = 1), descriptive
only. Per (care kappa, abstain cost A) cell, the policy (it may probe, up to POLICY_ATTEMPTS attempts) and commit-only (one
attempt at the best row by J; R4 picks by P(placed) instead) as points: task success against irreversible damage, with
r = A / (5 kappa). Rates are over the counted cases (calibrate._counted: no error, no artifact), give-ups included:
  success       the case ended placed;
  irreversible  an attempt ended in an irreversible label, or a probe damaged the object;
  probe_damage  a probe damaged the object (part of irreversible);
  give_up       the case ended abstained, no_row or unreachable.

    python -P -m robots.mujoco_lab.curve stage0 --m4 DIR --out DIR
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from typing import Any

from rpent.skills.pick_place.protocol import IRREVERSIBLE

M4_PROTOCOLS = ("commit", "policy")
GIVE_UP = ("abstained", "no_row", "unreachable")
M4_NAME = re.compile(r"^(?P<protocol>[a-z_]+)_k(?P<care>[0-9.]+)_A(?P<A>[0-9.]+)_s(?P<s>[0-9.]+)_seed(?P<seed>\d+)\.json\.gz$")


# ---------------------------------------------------------------- Stage 0: the m4 sweep's points

def m4_point(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Success, irreversible damage, probe damage, give-up and second-attempt rates over the counted cases (see the
    module docstring)."""
    rs = [r for r in rows if "error" not in r and not r.get("artifact")]
    n = len(rs)

    def probe_dmg(r: dict[str, Any]) -> bool:
        return any(p.get("kind") == "damaged" or p.get("damage") for p in r.get("probes") or [])

    def irr(r: dict[str, Any]) -> bool:
        return any(a.get("label") in IRREVERSIBLE for a in r.get("attempts") or []) or probe_dmg(r)

    rate = lambda k: (k / n) if n else None  # noqa: E731
    return {"n": n, "success": rate(sum(r["ended"] == "placed" for r in rs)), "irreversible": rate(sum(irr(r) for r in rs)),
            "probe_damage": rate(sum(probe_dmg(r) for r in rs)), "give_up": rate(sum(r["ended"] in GIVE_UP for r in rs)),
            "second_attempt": rate(sum(len(r.get("attempts") or []) > 1 for r in rs))}


def stage0_points(m4_dir: str, s: float = 1.0, seed: int = 12) -> list[dict[str, Any]]:
    """One point per M4_PROTOCOLS file at time scale ``s`` and ``seed``, sorted by (protocol, kappa, A)."""
    pts = []
    for name in sorted(os.listdir(m4_dir)):
        m = M4_NAME.match(name)
        if not m or m["protocol"] not in M4_PROTOCOLS or float(m["s"]) != float(s) or int(m["seed"]) != int(seed):
            continue
        with gzip.open(os.path.join(m4_dir, name), "rt") as f:
            rows = json.load(f)["rows"]
        care, A = float(m["care"]), float(m["A"])
        pts.append(dict(m4_point(rows), protocol=m["protocol"], care=care, A=A, r=A / (5.0 * care), file=name))
    return sorted(pts, key=lambda p: (p["protocol"], p["care"], p["A"]))


def stage0_markdown(pts: list[dict[str, Any]]) -> str:
    pct = lambda x: "-" if x is None else f"{100.0 * x:.1f}"  # noqa: E731
    by = {(p["protocol"], p["care"], p["A"]): p for p in pts}
    cells = sorted({(p["care"], p["A"]) for p in pts}, key=lambda c: (c[1] / (5.0 * c[0]), c))
    out = ["| kappa | A | r | policy success % | policy irreversible % (probe) | policy give-up % | commit success % | "
           "commit irreversible % | commit give-up % | n (policy, commit) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for care, A in cells:
        p, c = by.get(("policy", care, A)), by.get(("commit", care, A))
        f = lambda x, k: "-" if x is None else pct(x[k])  # noqa: E731
        out.append(f"| {care:g} | {A:g} | {A / (5.0 * care):.3g} | {f(p, 'success')} | {f(p, 'irreversible')} "
                   f"({f(p, 'probe_damage')}) | {f(p, 'give_up')} | {f(c, 'success')} | {f(c, 'irreversible')} | {f(c, 'give_up')} | "
                   f"{p['n'] if p else '-'}, {c['n'] if c else '-'} |")
    return "\n".join(out) + "\n"


def _label_groups(pts: list[tuple[float, float, float]], dx: float = 0.15, dy: float = 0.6) -> list[tuple[tuple[float, float], list[float]]]:
    """Points (x, y, r) within (dx, dy) of each other (single linkage) share one label at the group's highest point, its
    r values in increasing order: overlapping labels are unreadable."""
    groups: list[list[tuple[float, float, float]]] = []
    for p in sorted(pts):
        near = [g for g in groups if any(abs(p[0] - q[0]) <= dx and abs(p[1] - q[1]) <= dy for q in g)]
        merged = [p] + [q for g in near for q in g]
        groups = [g for g in groups if g not in near] + [merged]
    out = []
    for g in groups:
        top = max(g, key=lambda q: (q[1], q[0]))
        out.append(((top[0], top[1]), sorted(q[2] for q in g)))
    return sorted(out)


def stage0_figure(pts: list[dict[str, Any]], path: str) -> None:
    """Success against irreversible damage: the policy (blue, labelled with r) and commit-only (orange)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"policy": "#2a78d6", "commit": "#eb6834"}
    names = {"policy": "policy (may probe, up to 3 attempts)", "commit": "commit-only (best row by J, 1 attempt)"}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for proto in ("commit", "policy"):
        ps = [p for p in pts if p["protocol"] == proto]
        ax.scatter([100 * p["irreversible"] for p in ps], [100 * p["success"] for p in ps], s=36, color=colours[proto],
                   edgecolors="white", linewidths=1.5, zorder=3, label=names[proto])
        if proto == "policy":
            groups = sorted(_label_groups([(100 * p["irreversible"], 100 * p["success"], p["r"]) for p in ps]), key=lambda g: -g[0][1])
            y_lab = 1e9
            for xy, rs in groups:              # one label column in the empty middle, top down, 1.8 points apart, leader lines
                y_lab = min(xy[1], y_lab - 1.8)
                ax.annotate("r " + ", ".join(f"{r:.3g}" for r in rs), xy, xytext=(3.4, y_lab), textcoords="data", fontsize=7,
                            color="#52514e", va="center", arrowprops={"arrowstyle": "-", "color": "#c3c2b7", "lw": 0.6})
    ax.set_xlabel("irreversible damage (% of cases)")
    ax.set_ylabel("task success (% of cases)")
    ax.set_title("m4 sweep (m2b2, s = 1): where the tested settings sit", fontsize=10)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def stage0(a: argparse.Namespace) -> list[dict[str, Any]]:
    os.makedirs(a.out, exist_ok=True)
    pts = stage0_points(a.m4)
    with open(os.path.join(a.out, "stage0_points.json"), "w") as f:
        json.dump(pts, f, indent=1)
    with open(os.path.join(a.out, "stage0_points.md"), "w") as f:
        f.write(stage0_markdown(pts))
    stage0_figure(pts, os.path.join(a.out, "stage0_points.png"))
    print(stage0_markdown(pts))
    return pts


def main(argv: list[str] | None = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s0 = sub.add_parser("stage0", help="the m4 sweep's points (descriptive)")
    s0.add_argument("--m4", required=True)
    s0.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"stage0": stage0}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
