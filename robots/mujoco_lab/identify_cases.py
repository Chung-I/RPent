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


"""Identify cases for the v12 checks (option A, the user's decision of 2026-11-04): one identify probe per scene-B weigh
case of a bundle's probe corpus, in the arm scene, as gate R1 ran it.

A case is the weigh case's scene B (``probe_corpus.probe_rows``: the same draws from the case and damage streams, the
reset seed, no fixture), its mid pinch (``probe_corpus._state``) and the force max(the weigh's force, f_std) ("a firm
grasp", R1). The probe is ``adapter.probe_identify`` at 6 s (the R1 decision), the best of 40 candidates. Each row is a
probe-corpus row (key tag Bidentify, probe identify, its kind, the floored fit as the reading, None unless measured)
plus ``weigh_check``: whether its draws, pinch and pose equal the weigh row's. identify_fit.identify_cases maps the rows
onto the weigh cases (a weigh case without a row reads ``missing``), so a check stage draws the weigh's particles.

    python -P -m robots.mujoco_lab.identify_cases run --bundle B --classes C --thresholds T --workers 3 --out DIR [--limit 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

DURATION_S = 6.0          # the identify probe's excitation (R1 decision, 2026-11-04)
TAG = "Bidentify"


def firm_force(weigh: float, f_std: float) -> float:
    """Gate R1's identify grip: max(the case's own force, the standard force)."""
    return max(float(weigh), float(f_std))


def identify_row(env, object_id: str, idx: int, seed: int, classes: Any, thresholds: Any, duration: float = DURATION_S) -> dict[str, Any]:
    """Scene B of one draw, as probe_corpus.probe_rows sets it, then one identify probe at the firm force."""
    from robots.mujoco_lab.adapter import object_props
    from robots.mujoco_lab.corpus import apply_damage, case_rng, damage_rng, draw_damage, draw_theta, key_tag, lab_generator, world
    from robots.mujoco_lab.fragility import FLAGS, entry_of
    from robots.mujoco_lab.probe_corpus import _state
    from rpent.skills.pick_place.probes import weigh_force

    generator = lab_generator()
    rng = case_rng(seed, object_id, idx)
    props = object_props(object_id, generator)
    theta, com_end = draw_theta(rng, props["mass"], generator)
    f_std = float(env.s["t3_make"].std_force(props["mass"]))
    C, th = world(classes, thresholds)
    entry, _ = entry_of(object_id, C)
    env.classes = C
    d = draw_damage(damage_rng(seed, object_id, idx), entry, th, f_std)
    theta, _ = apply_damage(theta, {}, d)
    weigh = weigh_force(f_std)
    force = firm_force(weigh, f_std)
    env.slosh_zeta, env.slosh_scale = th.slosh_zeta, th.slosh_scale
    env.reset(object_id, theta, seed=seed * 1000 + idx, com_end=com_end, fixture=None, fill=d["fill"], frag=d["theta_frag"])
    start = dict(env.reset_info)
    truth = env.truth()
    key = f"{object_id}__{key_tag(generator)}s{seed}__d{idx}__{TAG}"
    common = {"key": key, "object": object_id, "draw": idx, "seed": seed, "generator": generator, "theta": theta, "com_end": com_end,
              "force": force, "weigh_force": weigh, "f_std": f_std, "com_obj_true": truth["com_obj"],
              "truth": {k: truth[k] for k in ("mass", "mu", "com_s")}, "entry": {k: entry.get(k) for k in FLAGS},
              "theta_frag": d["theta_frag"], "fill": d["fill"], "spill_angle_deg": d["spill_angle_deg"], "scene": "B", "order": 0}
    st = _state(env, force)
    if isinstance(st, str):
        return {**common, "declined": st, "reset": start}
    prop, pose, base = st
    t0 = time.time()
    res = env.probe_identify(prop, force, duration=duration)
    return {**common, "s": prop.s, "pose": pose, "base": base, **res.to_data(), "reset": start, "wall_s": time.time() - t0}


def _maxdiff(a: Any, b: Any) -> float:
    """The largest absolute difference between two nested JSON values (inf for a structure mismatch)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return max([_maxdiff(a[k], b[k]) if k in b else float("inf") for k in a] + [float("inf") for k in b if k not in a] + [0.0])
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return float("inf") if len(a) != len(b) else max([_maxdiff(x, y) for x, y in zip(a, b)] + [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def weigh_check(row: dict[str, Any], weigh: dict[str, Any]) -> dict[str, Any]:
    """Whether an identify row's scene is its weigh case's: the draws, the pinch, the pose and the standard force."""
    return {"draws_equal": all(_maxdiff(row.get(k), weigh.get(k)) == 0.0 for k in ("theta", "com_end", "fill", "theta_frag")),
            "s_equal": "s" in row and abs(float(row["s"]) - float(weigh["s"])) < 1e-12,
            "pose_maxdiff": _maxdiff(row["pose"], weigh["pose"]) if "pose" in row else None,
            "f_std_equal": abs(float(row["f_std"]) - float(weigh["f_std"])) < 1e-9}


def _object_job(spec: dict[str, Any]) -> str:
    """One object's cases in one process (a fresh env): its rows, written whole to DIR/rows/<object>.jsonl."""
    from robots.mujoco_lab.adapter import MuJoCoLabEnv

    env = MuJoCoLabEnv()
    if "ident" not in env.s:
        raise RuntimeError("the identify probe needs the arm port's sim (RPENT_MUJOCO_SIM_DIR) with joint sensing")
    rows = []
    for w in spec["weigh"]:
        t0 = time.time()
        try:
            r = identify_row(env, spec["object"], int(w["draw"]), int(w["seed"]), spec["classes"], spec["thresholds"], spec["duration"])
            r["weigh_check"] = weigh_check(r, w)
        except Exception as exc:  # noqa: BLE001  one bad draw must not lose the object's other cases
            r = {"key": w["key"].rsplit("__", 1)[0] + "__" + TAG, "object": spec["object"], "draw": int(w["draw"]),
                 "error": f"{type(exc).__name__}: {exc}"[:300]}
        r["case_wall_s"] = time.time() - t0
        rows.append(r)
        print(f"{r['key']}: {r.get('kind') or r.get('declined') or r.get('error')} {r['case_wall_s']:.1f} s", flush=True)
    path = os.path.join(spec["out"], "rows", f"{spec['object']}.jsonl")
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_plain) + "\n")
    os.replace(path + ".tmp", path)
    return path


def _plain(x: Any) -> Any:
    """json's fallback for numpy values in a probe's records."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON serializable: {type(x).__name__}")


def pick(cases: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Every case, or ``limit`` of them spread evenly over the sorted cases (the timing run)."""
    if not limit or limit >= len(cases):
        return cases
    return [cases[i] for i in np.linspace(0, len(cases) - 1, limit).round().astype(int)]


def summarize(rows: list[dict[str, Any]], n_weigh: int) -> dict[str, Any]:
    """Counts by kind (declined and error apart), the wall time per case, and the weigh checks."""
    kinds = Counter(r.get("kind") or ("declined" if "declined" in r else "error") for r in rows)
    wall = np.array([r["case_wall_s"] for r in rows if "case_wall_s" in r])
    ck = [r["weigh_check"] for r in rows if "weigh_check" in r]
    pm = [c["pose_maxdiff"] for c in ck if c["pose_maxdiff"] is not None]
    return {"n_rows": len(rows), "n_weigh_cases": n_weigh, "kinds": dict(sorted(kinds.items())),
            "wall_s": {"total": float(wall.sum()), "median": float(np.median(wall)), "p90": float(np.percentile(wall, 90)),
                       "max": float(wall.max())} if len(wall) else None,
            "checks": {"draws_unequal": sum(not c["draws_equal"] for c in ck), "s_unequal": sum(not c["s_equal"] for c in ck),
                       "f_std_unequal": sum(not c["f_std_equal"] for c in ck), "pose_maxdiff": max(pm) if pm else None}}


def git_state(path: str | None) -> dict[str, Any] | None:
    """A folder's last commit and whether its tracked files changed since (the code's and the sim's provenance)."""
    if not path:
        return None
    g = ["git", "--no-optional-locks", "-C", path]
    commit = subprocess.run(g + ["log", "-1", "--format=%H", "--", "."], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(g + ["status", "--porcelain", "--untracked-files=no", "--", "."], capture_output=True,
                                text=True).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run(a: argparse.Namespace) -> dict[str, Any]:
    from robots.mujoco_lab.corpus import lab_generator
    from robots.mujoco_lab.estimation_check import probe_rows, stage_cases

    os.makedirs(os.path.join(a.out, "rows"), exist_ok=True)
    man = json.load(open(os.path.join(a.bundle, "manifest.json")))
    for name, path in (("classes", a.classes), ("thresholds", a.thresholds)):
        want = (man.get(name) or {}).get("sha256")
        if want and _sha256(path) != want:
            raise SystemExit(f"--{name} {path} is not the bundle's ({want[:12]}...)")
    weigh = [c[0] for c in stage_cases(probe_rows(a.bundle), "weigh")]
    objects = set(a.objects.split(",")) if a.objects else None
    todo = pick([w for w in weigh if objects is None or w["object"] in objects], a.limit)
    by: dict[str, list[dict[str, Any]]] = {}
    for w in todo:
        by.setdefault(w["object"], []).append(w)
    specs = [{"object": o, "weigh": ws, "classes": a.classes, "thresholds": a.thresholds, "duration": a.duration, "out": a.out}
             for o, ws in sorted(by.items()) if not os.path.exists(os.path.join(a.out, "rows", f"{o}.jsonl"))]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "--no-optional-locks", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    sim = os.environ.get("RPENT_MUJOCO_SIM_DIR")
    cfg = {**{k: v for k, v in vars(a).items() if k != "func"}, "code_revision": rev, "dirty": dirty, "generator": lab_generator(),
           "sim_dir": sim, "sim_git": git_state(sim), "torque_noise_env": os.environ.get("RPENT_LAB_TORQUE_NOISE"),
           "manifest_sensor": man.get("sensor"), "n_weigh_cases": len(weigh), "n_cases": len(todo), "n_objects": len(by),
           "objects_todo": len(specs)}
    with open(os.path.join(a.out, f"run_config_{time.strftime('%Y%m%dT%H%M%S')}.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    t0 = time.time()
    if a.workers <= 1:
        for s in specs:
            print(_object_job(s), flush=True)
    else:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
            for path in ex.map(_object_job, specs):
                print(path, flush=True)
    rows = []
    for o in sorted(by):
        with open(os.path.join(a.out, "rows", f"{o}.jsonl")) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: (r["object"], int(r["draw"])))
    with open(os.path.join(a.out, "identify_rows.jsonl.tmp"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(os.path.join(a.out, "identify_rows.jsonl.tmp"), os.path.join(a.out, "identify_rows.jsonl"))
    s = {**summarize(rows, len(todo)), "wall_clock_s": time.time() - t0, "workers": a.workers}
    from robots.mujoco_lab import tracking

    s["wandb"] = tracking.log_run("identify_cases", cfg, {k: v for k, v in s.items() if k != "wandb"})
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s, indent=1), flush=True)
    return s


def main(argv: list[str] | None = None) -> Any:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True, help="the bundle whose probe corpus gives the scene-B weigh cases")
    r.add_argument("--classes", required=True, help="the fragility class file the probe corpus was built with")
    r.add_argument("--thresholds", required=True, help="the thresholds file the probe corpus was built with")
    r.add_argument("--objects", default=None)
    r.add_argument("--limit", type=int, default=None, help="this many cases spread over the sorted cases (the timing run)")
    r.add_argument("--duration", type=float, default=DURATION_S)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    main()
