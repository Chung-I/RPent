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

"""The model lock of Stage 2c and oracle-c (daily-logs designs/2026-11-09-stage2-contacts-rerun-design.md, section 2: the
model files' sha256 must equal the published run's, checked before the first case). The published runs' provenance
(run_curve.sh) holds the sha256 of the bundle's manifests only, so ``write`` locks the whole bundle in two steps: every
manifest a published provenance names must hash as it did, and no bundle file may have changed after the earliest
published start; then the lock records the sha256 of every file. ``check`` compares the bundle with the lock (a run
calls it before its first case)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from typing import Any

_HASH = re.compile(r"^([0-9a-f]{64})  (\S.*)$")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bundle_files(bundle: str) -> dict[str, str]:
    """The sha256 of every file under ``bundle``, by its path relative to the bundle, in sorted order."""
    out = {}
    for root, dirs, files in os.walk(bundle):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            out[os.path.relpath(p, bundle)] = sha256(p)
    return dict(sorted(out.items()))


def read_provenance(path: str) -> dict[str, Any]:
    """A run_curve.sh provenance file: the bundle path, its hashed files ({relative path: sha256}) and the start time."""
    bundle, hashes, start = None, {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("bundle "):
            bundle = line[len("bundle "):].rstrip(":")
        elif line.startswith("start "):
            start = line[len("start "):].strip()
        elif bundle is not None and _HASH.match(line):
            h, rel = _HASH.match(line).groups()
            hashes[rel] = h
    if bundle is None or not hashes:
        raise SystemExit(f"{path}: no bundle lines")
    if start is None:
        raise SystemExit(f"{path}: no start line")
    return {"bundle": bundle, "hashes": hashes, "start": start}


def _epoch(stamp: str) -> float:
    return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))


def write(bundle: str, provenances: list[str], out: str) -> dict[str, Any]:
    """Lock ``bundle`` against the published runs' ``provenances`` (refuses: a provenance of another bundle, a manifest
    whose sha256 differs from a provenance's, a file modified after the earliest start, an existing ``out``)."""
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: pick a new path")
    real = os.path.realpath(bundle)
    files = bundle_files(real)
    provs = {p: read_provenance(p) for p in provenances}
    bad: list[str] = []
    for p, pv in provs.items():
        if os.path.realpath(pv["bundle"]) != real:
            raise SystemExit(f"{p}: the provenance's bundle {pv['bundle']} is not {real}")
        for rel, h in pv["hashes"].items():
            if files.get(rel) != h:
                bad.append(f"{rel} ({p}: {h[:12]}, now {(files.get(rel) or 'missing')[:12]})")
    if bad:
        raise SystemExit("the bundle's files differ from the published runs': " + "; ".join(bad))
    earliest = min((pv["start"] for pv in provs.values()), key=_epoch)
    late = [rel for rel in files if os.path.getmtime(os.path.join(real, rel)) > _epoch(earliest)]
    if late:
        raise SystemExit(f"{len(late)} bundle files changed after the earliest published start {earliest}: " + ", ".join(late))
    lock = {"bundle": real, "n_files": len(files), "files": files, "earliest_start": earliest,
            "provenance_sha256": {p: sha256(p) for p in provenances}}
    with open(out, "w") as f:
        json.dump(lock, f, indent=1)
    return lock


def check(bundle: str, lock_path: str) -> dict[str, Any]:
    """The lock when ``bundle`` holds exactly its files with the same sha256; else SystemExit naming each file."""
    lock = json.load(open(lock_path))
    real = os.path.realpath(bundle)
    if lock["bundle"] != real:
        raise SystemExit(f"the model lock is for the bundle {lock['bundle']}, not {real}")
    now, want = bundle_files(real), lock["files"]
    changed = sorted(k for k in set(now) & set(want) if now[k] != want[k])
    added, removed = sorted(set(now) - set(want)), sorted(set(want) - set(now))
    if changed or added or removed:
        raise SystemExit(f"the bundle {real} differs from the model lock {lock_path}: changed {changed}, added {added}, "
                         f"removed {removed}")
    return lock


def main(argv: list[str] | None = None) -> Any:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="lock the bundle against the published runs' provenance files")
    w.add_argument("--bundle", required=True)
    w.add_argument("--provenance", required=True, help="comma-separated run_curve.sh provenance.txt files")
    w.add_argument("--out", required=True)
    c = sub.add_parser("check", help="compare the bundle with a lock")
    c.add_argument("--bundle", required=True)
    c.add_argument("--lock", required=True)
    a = ap.parse_args(argv)
    lock = write(a.bundle, a.provenance.split(","), a.out) if a.cmd == "write" else check(a.bundle, a.lock)
    print(json.dumps({k: v for k, v in lock.items() if k != "files"}, indent=1), flush=True)
    return lock


if __name__ == "__main__":
    main()
