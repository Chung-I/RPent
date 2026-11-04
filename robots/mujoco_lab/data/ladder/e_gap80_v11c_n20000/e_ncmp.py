#!/usr/bin/env python3
"""E at two particle counts (A, then B), paired by case. Per setting and H: the default rule's first-decision regret at
each count (ladder_e.summarize's rule: the first tables of the H = 3 run, the reset-artifact cases left out), the paired
difference B - A with an object-bootstrap 95 % interval (e1.object_bootstrap's method), how often E's, the default
rule's and the oracle rule's first choices change, and how far the first tables move. Usage: e_ncmp.py A B out.json"""
import gzip
import json
import sys
import zlib
from collections import Counter

import numpy as np

TOL = 1e-9  # ladder_e.TOL


def load(path):
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def group(rows):
    g = {}
    for r in rows:
        g.setdefault((r["object"], int(r["idx"]), int(r["seed"]), r["setting"]), {})[int(r["H"])] = r
    return g


def usable(runs, Hs):
    """None if incomplete or without a menu or a reset artifact (ladder_e.summarize's exclusions), else the H = max run."""
    if set(runs) != set(Hs):
        return None
    full = runs[Hs[-1]]
    if not full.get("e_log") or any(not runs[H].get("e_log") for H in Hs):
        return None
    if any(runs[H].get("reset_artifact") for H in Hs) or not (full.get("default") or {}).get("reset_at_rest", True):
        return None
    return full


def boot(values, objects, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    names = sorted(set(objects))
    inv = np.array([names.index(o) for o in objects])
    sums = np.bincount(inv, weights=v, minlength=len(names))
    counts = np.bincount(inv, minlength=len(names)).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(names), size=(n_boot, len(names)))
    b = sums[draws].sum(1) / np.maximum(counts[draws].sum(1), 1.0)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def compare(rows_a, rows_b):
    ga, gb = group(rows_a), group(rows_b)
    Hs = sorted({int(r["H"]) for r in rows_a})
    out = {"horizons": Hs, "particles": [sorted({r.get("particles") for r in rows_a}, key=str),
                                         sorted({r.get("particles") for r in rows_b}, key=str)], "settings": {}}
    for key in sorted(set(ga) & set(gb)):
        name = key[3]
        s = out["settings"].setdefault(name, {"paired": 0, "left_out": 0, "menu_rows_differ": 0, "default_changed": [],
                                              "oracle_changed": [], "per_H": {}})
        fa, fb = usable(ga[key], Hs), usable(gb[key], Hs)
        if fa is None or fb is None:
            s["left_out"] += 1
            continue
        s["paired"] += 1
        tag = f"{key[0]}/{key[1]}"
        dca, dcb = (fa.get("default") or {}).get("choice"), (fb.get("default") or {}).get("choice")
        oca, ocb = fa["first"].get("oracle_choice"), fb["first"].get("oracle_choice")
        if dca != dcb:
            s["default_changed"].append(f"{tag}: {dca} -> {dcb}")
        if oca != ocb:
            s["oracle_changed"].append(f"{tag}: {oca} -> {ocb}")
        Ta = {H: fa["e_log"][0]["tables"][str(H)] for H in Hs}
        Tb = {H: fb["e_log"][0]["tables"][str(H)] for H in Hs}
        if set(Ta[Hs[0]]) != set(Tb[Hs[0]]):
            s["menu_rows_differ"] += 1
        for H in Hs:
            p = s["per_H"].setdefault(str(H), {"ra": [], "rb": [], "obj": [], "e_changed": Counter(), "e_cases": [],
                                               "low_shift": [], "table_shift": [], "e_s": [[], []]})
            ea, eb = ga[key][H]["e_log"][0]["choices"][str(H)], gb[key][H]["e_log"][0]["choices"][str(H)]
            if ea != eb:
                p["e_changed"][f"{ea} -> {eb}"] += 1
                p["e_cases"].append(f"{tag}: {ea} ({Ta[H][ea]:.3f} -> {Tb[H][ea]:.3f}) -> {eb} "
                                    f"({Ta[H].get(eb, float('nan')):.3f} -> {Tb[H][eb]:.3f})")
            p["e_s"][0].append(float(ga[key][H]["e_seconds"]))
            p["e_s"][1].append(float(gb[key][H]["e_seconds"]))
            lo_a, lo_b = min(Ta[H].values()), min(Tb[H].values())
            p["low_shift"].append(lo_b - lo_a)
            common = set(Ta[H]) & set(Tb[H])
            p["table_shift"].append(max(abs(Tb[H][o] - Ta[H][o]) for o in common))
            if dca in Ta[H] and dcb in Tb[H]:
                p["ra"].append(Ta[H][dca] - lo_a)
                p["rb"].append(Tb[H][dcb] - lo_b)
                p["obj"].append(key[0])
    for name, s in out["settings"].items():
        for H, p in s["per_H"].items():
            ra, rb, d = np.array(p["ra"]), np.array(p["rb"]), np.array(p["rb"]) - np.array(p["ra"])
            seed = zlib.crc32(f"{name}/H{H}/B-A".encode())
            m, lo, hi = boot(d, p["obj"], seed=seed) if len(d) else (None, None, None)
            s["per_H"][H] = {
                "n": len(d), "regret_a": float(ra.mean()) if len(ra) else None, "regret_b": float(rb.mean()) if len(rb) else None,
                "diff": m, "diff_ci": [lo, hi], "diff_nonzero": int((np.abs(d) > TOL).sum()),
                "above_0": [int((ra > TOL).sum()), int((rb > TOL).sum())],
                "e_changed": int(sum(p["e_changed"].values())), "e_pairs": dict(p["e_changed"]), "e_cases": p["e_cases"],
                "low_shift_mean": float(np.mean(p["low_shift"])), "low_shift_absmax": float(np.max(np.abs(p["low_shift"]))),
                "table_shift_mean": float(np.mean(p["table_shift"])), "table_shift_max": float(np.max(p["table_shift"])),
                "e_seconds_mean": [float(np.mean(p["e_s"][0])), float(np.mean(p["e_s"][1]))]}
    return out


def report(c):
    na, nb = c["particles"]
    L = [f"A: {na} particles, B: {nb} particles. Regret = the default rule's first choice minus the lowest score on E's "
         "first table. diff = B - A per case (object-bootstrap 95 % interval).", "",
         "| setting | H | cases | regret A | regret B | diff B - A | 95 % interval | cases with diff != 0 | above 0 A / B | "
         "E's first choice changed | lowest score shift: mean, max abs | table shift: mean, max | E s: A, B |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in c["settings"].items():
        for H, p in s["per_H"].items():
            L.append(f"| {name} | {H} | {p['n']} | {p['regret_a']:.3f} | {p['regret_b']:.3f} | {p['diff']:+.3f} | "
                     f"[{p['diff_ci'][0]:+.3f}, {p['diff_ci'][1]:+.3f}] | {p['diff_nonzero']} | {p['above_0'][0]} / "
                     f"{p['above_0'][1]} | {p['e_changed']} | {p['low_shift_mean']:+.4f}, {p['low_shift_absmax']:.4f} | "
                     f"{p['table_shift_mean']:.4f}, {p['table_shift_max']:.4f} | {p['e_seconds_mean'][0]:.1f}, "
                     f"{p['e_seconds_mean'][1]:.1f} |")
    L.append("")
    for name, s in c["settings"].items():
        L.append(f"- {name}: {s['paired']} paired, {s['left_out']} left out, menu rows differ on {s['menu_rows_differ']}; "
                 f"the default rule's first choice changed on {len(s['default_changed'])}, the oracle rule's on "
                 f"{len(s['oracle_changed'])}.")
        for x in s["default_changed"]:
            L.append(f"  - default: {x}")
        for x in s["oracle_changed"]:
            L.append(f"  - oracle rule: {x}")
        for H, p in s["per_H"].items():
            for x in p["e_cases"]:
                L.append(f"  - E at H = {H}: {x}")
    return "\n".join(L)


if __name__ == "__main__":
    c = compare(load(sys.argv[1]), load(sys.argv[2]))
    with open(sys.argv[3], "w") as f:
        json.dump(c, f, indent=1)
    print(report(c))
