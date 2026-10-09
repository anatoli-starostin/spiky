"""Host-vs-host check of the HARDWARE-INDEPENDENT quantities in two profile_cartridges.py results. FAILS (exit 1) on any
mismatch -- a mismatch is a bug in the harness or the config, never something to average away.

Compared per arm: the whole "static" record (read MACs/token, read bytes/token, index dtype, param dtype, #params,
param+buffer bytes, soft-sign tail), and from the "memory" pass the saved-for-backward inventory (count, total bytes,
the top-12 shapes/dtypes/MiB). Not compared (hardware-dependent by design): timings, peak/reserved memory, held-after-
forward bytes (they include allocator / cuBLAS-workspace effects). An arm that is OOM or errored on one side is
reported as skipped, not as a mismatch (OOM is hardware-dependent); an n/a status must match (it is a library fact).
The two results must have been run with the same library commit, geometry and token counts (checked first).

usage: check_hw_indep.py <A/cartridges.json> <B/cartridges.json> [--label-a 5090 --label-b H100]
"""
from __future__ import annotations

import argparse
import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args(argv)
    A, B = load(args.a), load(args.b)
    la, lb = args.label_a, args.label_b
    bad, skipped, ok = [], [], 0

    for key, get in (("library commit", lambda r: r["env"].get("lutorch_ex_src_commit")),
                     ("geometry", lambda r: r["geometry"]), ("tokens", lambda r: r["args"]["tokens"]),
                     ("lib features", lambda r: r["env"].get("lib_features"))):
        if get(A) != get(B):
            bad.append(f"[setup] {key}: {la}={get(A)} {lb}={get(B)}")
    if bad:
        print("\n".join(bad))
        print(f"FAIL: the two results are not comparable ({len(bad)} setup mismatch(es))")
        return 1

    def cmp(arm, what, x, y):
        nonlocal ok
        if x == y:
            ok += 1
        else:
            bad.append(f"[{arm}] {what}: {la}={json.dumps(x, default=str)[:300]}  {lb}={json.dumps(y, default=str)[:300]}")

    for arm in sorted(set(A.get("static", {})) | set(B.get("static", {}))):
        sa, sb = A.get("static", {}).get(arm), B.get("static", {}).get(arm)
        if sa is None or sb is None:
            bad.append(f"[{arm}] static: present only on {la if sb is None else lb}")
            continue
        if "OOM" in (sa.get("status"), sb.get("status")) or "error" in (sa.get("status"), sb.get("status")):
            skipped.append(f"[{arm}] static: {la}={sa.get('status')} {lb}={sb.get('status')}")
            continue
        for k in sorted(set(sa) | set(sb)):
            cmp(arm, f"static.{k}", sa.get(k), sb.get(k))

    for arm in sorted(set(A.get("memory", {})) | set(B.get("memory", {}))):
        ma, mb = A.get("memory", {}).get(arm), B.get("memory", {}).get(arm)
        if ma is None or mb is None:
            bad.append(f"[{arm}] memory: present only on {la if mb is None else lb}")
            continue
        sa_, sb_ = ma.get("status"), mb.get("status")
        if sa_ != "ok" or sb_ != "ok":
            if sa_ == "n/a" or sb_ == "n/a":
                cmp(arm, "memory.status", sa_, sb_)
            else:
                skipped.append(f"[{arm}] memory: {la}={sa_} {lb}={sb_} (hardware-dependent, not compared)")
            continue
        for k in ("tokens", "n_saved", "saved_total_bytes", "saved_top"):
            cmp(arm, f"memory.{k}", ma.get(k), mb.get(k))

    for s in skipped:
        print("skipped:", s)
    for b in bad:
        print("MISMATCH:", b)
    if bad:
        print(f"FAIL: {len(bad)} hardware-independent mismatch(es) between {la} and {lb} ({ok} fields matched)")
        return 1
    print(f"OK: {ok} hardware-independent fields identical between {la} and {lb} ({len(skipped)} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
