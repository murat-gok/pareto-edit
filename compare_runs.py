#!/usr/bin/env python3
r"""
What actually differs between the two runs, and what does a clean
two-objective ablation look like?

Two jobs:

 1. Diagnose. If layer2_full_aggot.json were layer2_full_direct.json with a
    different off-target model, then for any design present in both the
    efficiency and purity values would be identical, and the per-locus maximum
    purity would match exactly. This checks that. If other fields differ, the
    aggot file is a different configuration and cannot be relabelled as an
    ablation of the direct run.

 2. Produce the ablation you actually want, from the direct run alone: recompute
    each locus on efficiency and purity only, and report what changes. No re-run,
    no oracle calls, same models, same candidates - the only thing removed is the
    off-target objective.

Usage
    python compare_runs.py layer2_full_direct.json [layer2_full_aggot.json]
"""

import json
import sys
import os
import statistics as stats

EFF_KEYS = ("efficiency", "eff", "on_target", "ontarget", "activity", "pred_efficiency")
PUR_KEYS = ("purity", "pur", "bystander_purity", "product_purity", "pred_purity")
OFF_KEYS = ("offtarget", "off_target", "off", "offtarget_risk", "off_target_risk",
            "cfd", "cfd_score", "specificity")
ED_KEYS = ("editor", "be", "base_editor", "enzyme")
PAM_KEYS = ("pam", "pam_variant", "cas", "cas9")
POS_KEYS = ("position", "pos", "target_position", "t", "protospacer_pos")
LIST_KEYS = ("front", "designs", "pareto", "pareto_front", "solutions",
             "non_dominated", "members")

PURITY_THRESHOLD = 0.50


def _pick(d, c):
    low = {k.lower(): k for k in d}
    for x in c:
        if x in low:
            return low[x]
    return None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(path):
    obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, dict):
        for w in ("loci", "results", "data", "layer2", "output"):
            if w in obj and isinstance(obj[w], (dict, list)):
                obj = obj[w]
                break
    items = (list(obj.items()) if isinstance(obj, dict)
             else [(r.get("locus") or r.get("id") or f"locus_{i}", r)
                   for i, r in enumerate(obj) if isinstance(r, dict)])
    out = {}
    for lid, val in items:
        lst = val if isinstance(val, list) else (
            val.get(_pick(val, LIST_KEYS)) if isinstance(val, dict)
            and _pick(val, LIST_KEYS) else None)
        if not isinstance(lst, list):
            continue
        ds = []
        for d in lst:
            if not isinstance(d, dict):
                continue
            ke, kp, ko = _pick(d, EFF_KEYS), _pick(d, PUR_KEYS), _pick(d, OFF_KEYS)
            if not (ke and kp and ko):
                continue
            e, p, o = _f(d[ke]), _f(d[kp]), _f(d[ko])
            if None in (e, p, o):
                continue
            ked, kpa, kpo = _pick(d, ED_KEYS), _pick(d, PAM_KEYS), _pick(d, POS_KEYS)
            ds.append({
                "eff": e, "pur": p, "off": o,
                "ed": str(d.get(ked, "?")) if ked else "?",
                "pam": str(d.get(kpa, "?")) if kpa else "?",
                "pos": str(d.get(kpo, "?")) if kpo else "?",
            })
        if ds:
            out[str(lid)] = ds
    return out


def key(d):
    return (d["ed"], d["pam"], d["pos"])


def dom2(a, b):
    return ((a["eff"] >= b["eff"] and a["pur"] >= b["pur"])
            and (a["eff"] > b["eff"] or a["pur"] > b["pur"]))


def front2(ds):
    return [d for d in ds if not any(dom2(o, d) for o in ds if o is not d)]


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    direct = load(sys.argv[1])
    O = []
    w = O.append
    w("# Run comparison and two-objective ablation\n")

    # ---------- part 2 first: the ablation (this is the deliverable) ----------
    n3 = n2 = 0
    best_pur3, best_pur2, best_eff3, best_eff2 = [], [], [], []
    low3 = low2 = 0
    sizes3, sizes2 = [], []
    for lid, ds in direct.items():
        f2 = front2(ds)
        n3 += len(ds); n2 += len(f2)
        sizes3.append(len(ds)); sizes2.append(len(f2))
        b3, b2 = max(d["pur"] for d in ds), max(d["pur"] for d in f2)
        best_pur3.append(b3); best_pur2.append(b2)
        best_eff3.append(max(d["eff"] for d in ds))
        best_eff2.append(max(d["eff"] for d in f2))
        low3 += b3 < PURITY_THRESHOLD
        low2 += b2 < PURITY_THRESHOLD

    L = len(direct)
    w(f"## Dropping the off-target objective ({os.path.basename(sys.argv[1])}, {L} loci)\n")
    w("Same candidates, same efficiency and purity models. The only change is that "
      "dominance is computed on two objectives instead of three.\n")
    w("\n| Quantity | Three objectives | Two objectives |")
    w("| --- | --- | --- |")
    w(f"| Designs on the set | {n3} | {n2} |")
    w(f"| Mean set size | {n3 / L:.2f} | {n2 / L:.2f} |")
    w(f"| Median set size | {stats.median(sizes3):.1f} | {stats.median(sizes2):.1f} |")
    w(f"| Mean per-locus best purity | {stats.mean(best_pur3):.4f} | {stats.mean(best_pur2):.4f} |")
    w(f"| Mean per-locus best efficiency | {stats.mean(best_eff3):.4f} | {stats.mean(best_eff2):.4f} |")
    w(f"| Loci with no design at purity > {PURITY_THRESHOLD} | "
      f"{low3} ({100 * low3 / L:.0f}%) | {low2} ({100 * low2 / L:.0f}%) |")

    same = (low3 == low2 and
            all(abs(a - b) < 1e-12 for a, b in zip(best_pur3, best_pur2)))
    w("\n" + ("**Per-locus best purity is identical under both, so the "
              "no-clean-correction result is unchanged by construction.** This is not a "
              "coincidence: the highest-purity design at a locus can only be dominated by "
              "another design of equal purity, so the maximum purity always survives on the "
              "two-objective set. The headline does not depend on the third objective, and "
              "this can be stated as a property rather than an empirical check."
              if same else
              "Per-locus best purity differs between the two, which should not happen - "
              "check for ties being dropped in the dominance filter."))

    # ---------- part 1: the diagnosis ----------
    if len(sys.argv) > 2:
        agg = load(sys.argv[2])
        w(f"\n## Is {os.path.basename(sys.argv[2])} the same run with another "
          "off-target model?\n")
        shared_loci = set(direct) & set(agg)
        w(f"- Loci: {len(direct)} direct, {len(agg)} aggregate, "
          f"{len(shared_loci)} shared")

        offs = {d["off"] for ds in agg.values() for d in ds}
        w(f"- Distinct off-target values in the aggregate run: **{len(offs)}**"
          + (f" (constant {next(iter(offs))})" if len(offs) == 1 else ""))

        keyed = sum(1 for ds in direct.values() for d in ds if d["ed"] != "?")
        if not keyed:
            w("\nDesigns carry no editor/PAM/position fields in this file, so they cannot "
              "be matched one to one. Compare per-locus maxima instead:")
        matched = mism_e = mism_p = 0
        for lid in shared_loci:
            a = {key(d): d for d in direct[lid]}
            b = {key(d): d for d in agg[lid]}
            for k in set(a) & set(b):
                if k == ("?", "?", "?"):
                    continue
                matched += 1
                if abs(a[k]["eff"] - b[k]["eff"]) > 1e-9:
                    mism_e += 1
                if abs(a[k]["pur"] - b[k]["pur"]) > 1e-9:
                    mism_p += 1
        if matched:
            w(f"- Designs present in both by (editor, PAM, position): {matched}")
            w(f"- Of those, efficiency differs at {mism_e}; purity differs at {mism_p}")

        dp = [max(d["pur"] for d in direct[l]) for l in sorted(shared_loci)]
        ap = [max(d["pur"] for d in agg[l]) for l in sorted(shared_loci)]
        diff = sum(abs(x - y) > 1e-9 for x, y in zip(dp, ap))
        w(f"- Per-locus maximum purity differs at **{diff}/{len(shared_loci)} loci** "
          f"(mean {stats.mean(dp):.4f} direct against {stats.mean(ap):.4f} aggregate)")

        w("")
        if diff == 0 and mism_e == 0 and mism_p == 0:
            w("**The two runs agree everywhere except off-target.** The aggregate file is "
              "then the same configuration with a broken off-target column, and nothing is "
              "lost by dropping it: the ablation above already says what it was meant to "
              "say, exactly and from one run.")
        else:
            w("**The two runs differ in more than the off-target column.** The aggregate "
              "file is a different configuration, so it cannot be relabelled as an ablation "
              "of the direct run, and its numbers should not be reported alongside the "
              "direct run as if only the off-target model had changed. Either rebuild it "
              "from the direct configuration, or drop it and report the ablation above, "
              "which is exact and needs no re-run.")

    rep = "\n".join(O)
    print(rep)
    out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "run_comparison.md")
    open(out, "w", encoding="utf-8").write(rep)
    print(f"\n[written to {out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
