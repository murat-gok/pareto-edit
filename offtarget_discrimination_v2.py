#!/usr/bin/env python3
r"""
Does the off-target objective do REAL work, or does it inflate the front on
numerically negligible differences?

v2 fixes a sign error in v1's tolerance test (the off-target advantage was
computed backwards, which made the tolerance filter a no-op) and adds three
diagnostics v1 lacked:

  * an epsilon sweep, showing how fast the "off-target-justified" designs
    disappear as the tolerance grows
  * the distribution of winning margins for those designs
  * a within-locus vs between-locus variance split for the off-target score,
    which tells you whether a constant off-target column is a modelling fact
    or a computation bug

Usage
    python offtarget_discrimination_v2.py layer2_full_direct.json [layer2_full_aggot.json]
"""

import json
import sys
import os
import statistics as stats

EFF_KEYS = ("efficiency", "eff", "on_target", "ontarget", "activity", "pred_efficiency")
PUR_KEYS = ("purity", "pur", "bystander_purity", "product_purity", "pred_purity")
OFF_KEYS = ("offtarget", "off_target", "off", "offtarget_risk", "off_target_risk",
            "cfd", "cfd_score", "specificity")
DESIGN_LIST_KEYS = ("front", "designs", "pareto", "pareto_front", "solutions",
                    "non_dominated", "members")
CANDIDATE_KEYS = ("candidates", "all_designs", "enumerated", "feasible", "all")

EPS_GRID = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def _pick(d, cands):
    low = {k.lower(): k for k in d}
    for c in cands:
        if c in low:
            return low[c]
    return None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_designs(lst):
    out = []
    for d in lst or []:
        if not isinstance(d, dict):
            continue
        ke, kp, ko = _pick(d, EFF_KEYS), _pick(d, PUR_KEYS), _pick(d, OFF_KEYS)
        if not (ke and kp and ko):
            continue
        e, p, o = _f(d[ke]), _f(d[kp]), _f(d[ko])
        if None in (e, p, o):
            continue
        out.append({"eff": e, "pur": p, "off": o, "raw": d})
    return out


def extract(obj):
    """Yield (locus_id, front_designs, candidate_designs_or_None)."""
    if isinstance(obj, dict):
        for wrap in ("loci", "results", "data", "layer2", "output"):
            if wrap in obj and isinstance(obj[wrap], (dict, list)):
                obj = obj[wrap]
                break

    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, list):
        items = [(r.get("locus") or r.get("id") or f"locus_{i}", r)
                 for i, r in enumerate(obj) if isinstance(r, dict)]
    else:
        return

    for lid, val in items:
        front = cands = None
        if isinstance(val, list):
            front = _parse_designs(val)
        elif isinstance(val, dict):
            k = _pick(val, DESIGN_LIST_KEYS)
            if k:
                front = _parse_designs(val[k])
            kc = _pick(val, CANDIDATE_KEYS)
            if kc and isinstance(val[kc], list):
                cands = _parse_designs(val[kc])
        if front:
            yield str(lid), front, cands


def dom2(a, b):
    """a dominates b on (efficiency, purity), both maximized."""
    return ((a["eff"] >= b["eff"] and a["pur"] >= b["pur"])
            and (a["eff"] > b["eff"] or a["pur"] > b["pur"]))


def off_justified(designs, eps):
    """
    Designs kept on the front ONLY by off-target, at tolerance eps.

    d qualifies when (a) something 2-dominates it, so it is not on the
    two-objective front, and (b) it beats every one of those dominators on
    off-target by more than eps. Off-target is minimized, so d's advantage
    over o is o.off - d.off.

    eps = 0 reproduces F3 \ F2 exactly. Raising eps demands a larger real
    margin, so the count falls monotonically.
    """
    out = []
    for d in designs:
        doms = [o for o in designs if o is not d and dom2(o, d)]
        if not doms:
            continue                      # already on the two-objective front
        if all((o["off"] - d["off"]) > eps for o in doms):
            out.append(d)
    return out


def best_margin(d, designs):
    """Smallest off-target win d holds over anything that 2-dominates it."""
    m = [o["off"] - d["off"] for o in designs if o is not d and dom2(o, d)]
    return min(m) if m else None


def spearman(xs, ys):
    n = len(xs)
    if n < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def q(v, f):
    s = sorted(v)
    return s[min(len(s) - 1, int(f * (len(s) - 1)))]


def analyse(path):
    obj = json.load(open(path, encoding="utf-8"))
    loci = list(extract(obj))
    if not loci:
        raise SystemExit(f"No per-locus design lists found in {path}.")

    L = len(loci)
    total = sum(len(f) for _, f, _ in loci)
    rng = {"eff": [], "pur": [], "off": []}
    all_e, all_p, all_o, locus_mean_off = [], [], [], []
    cand_e, cand_p, n_cand_loci = [], [], 0

    for lid, front, cands in loci:
        for k in rng:
            v = [d[k] for d in front]
            rng[k].append(max(v) - min(v))
        for d in front:
            all_e.append(d["eff"]); all_p.append(d["pur"]); all_o.append(d["off"])
        locus_mean_off.append(stats.mean(d["off"] for d in front))
        if cands:
            n_cand_loci += 1
            for d in cands:
                cand_e.append(d["eff"]); cand_p.append(d["pur"])

    O = []
    w = O.append
    w(f"# Off-target discrimination, v2 — {os.path.basename(path)}\n")
    w(f"Loci {L} · front designs {total} · mean front {total / L:.2f}\n")

    # ---- 1. scale
    w("\n## 1. Scale of each axis within a front\n")
    w("| Objective | Median range | IQR | Relative to efficiency |")
    w("| --- | --- | --- | --- |")
    me = stats.median(rng["eff"])
    for k, lab in (("eff", "Efficiency"), ("pur", "Purity"), ("off", "Off-target")):
        m = stats.median(rng[k])
        w(f"| {lab} | {m:.4f} | {q(rng[k], .25):.4f}–{q(rng[k], .75):.4f} | "
          f"{(m / me if me else float('nan')):.3f}x |")

    # ---- 2. epsilon sweep (the headline)
    w("\n## 2. How much of the front survives a real off-target margin?\n")
    w("At each tolerance, a design counts as off-target-justified only if it beats "
      "everything that outranks it on efficiency and purity by at least that margin.\n")
    w("\n| Tolerance | Loci affected | % loci | Designs justified | % of front |")
    w("| --- | --- | --- | --- | --- |")
    sweep = {}
    for eps in EPS_GRID:
        nl = nd = 0
        for _, front, _ in loci:
            if len(front) < 2:
                continue
            j = off_justified(front, eps)
            if j:
                nl += 1
                nd += len(j)
        sweep[eps] = (nl, nd)
        w(f"| {eps:g} | {nl} | {100 * nl / L:.0f}% | {nd} | {100 * nd / total:.0f}% |")

    # ---- 3. margins
    margins = []
    for _, front, _ in loci:
        for d in off_justified(front, 0.0):
            m = best_margin(d, front)
            if m is not None:
                margins.append(m)
    w("\n## 3. How big are the winning margins?\n")
    if margins:
        w(f"- Median margin: **{stats.median(margins):.4f}**")
        w(f"- Quartiles: {q(margins, .25):.4f} / {q(margins, .75):.4f}")
        w(f"- Below 0.01: {100 * sum(m < .01 for m in margins) / len(margins):.0f}% · "
          f"below 0.02: {100 * sum(m < .02 for m in margins) / len(margins):.0f}%")
        w("\nCompare these to the efficiency and purity ranges in section 1. A median "
          "margin two orders of magnitude smaller means the third axis is splitting "
          "hairs, not expressing a trade-off a designer could act on.")
    else:
        w("No designs are justified by off-target at any tolerance.")

    # ---- 4. variance split (diagnoses a constant column)
    w("\n## 4. Where does off-target variation live?\n")
    within = stats.mean(rng["off"])
    between = (stats.pstdev(locus_mean_off) if len(locus_mean_off) > 1 else 0.0)
    w(f"- Mean within-front range: {within:.6f}")
    w(f"- Between-locus SD of the per-locus mean: {between:.6f}")
    flat = sum(r < 1e-12 for r in rng["off"])
    w(f"- Loci with a *constant* off-target across the whole front: "
      f"**{flat}/{L} ({100 * flat / L:.0f}%)**")
    if flat == L and between > 1e-9:
        w("\n**Diagnosis: the score varies between loci but is identical within every "
          "locus.** That is the signature of a score computed per target site rather "
          "than per guide. A guide-level specificity score must change when the "
          "protospacer moves. Check this before reporting anything from this run — "
          "as it stands this is a two-objective run with a decorative third column.")
    elif flat == L:
        w("\n**Diagnosis: the score is constant everywhere** — most likely a fallback "
          "value written when a lookup failed. Check the oracle.")

    # ---- 5. correlations
    w("\n## 5. Efficiency-purity relationship\n")
    rf = spearman(all_e, all_p)
    w(f"- Over front members only: {('n/a' if rf is None else f'{rf:+.3f}')}  "
      "— **do not report this number**")
    if cand_e:
        rc = spearman(cand_e, cand_p)
        w(f"- Over the full candidate set ({n_cand_loci} loci): "
          f"{('n/a' if rc is None else f'{rc:+.3f}')}  — report this one")
    else:
        w("- Over the full candidate set: **not available in this file**")
        w("\nThe front is selected *on* efficiency and purity, so a correlation measured "
          "across front members is a selection artifact and can carry any sign. Your "
          "paper's premise is that these two objectives conflict; the estimator for that "
          "is the correlation across all enumerated candidates, which this file does not "
          "contain. Dump the candidate sets and measure it — it is a one-line addition "
          "to the Results and it is the number a reviewer will want.")
    ro = spearman(all_o, all_e)
    w(f"- Off-target vs efficiency: {('n/a (no variance)' if ro is None else f'{ro:+.3f}')}")

    # ---- 6. verdict
    w("\n## 6. Verdict\n")
    n01 = sweep[0.01][0]
    pct01 = 100 * n01 / L
    if flat == L:
        w("This run cannot support a three-objective claim: the off-target column does "
          "not vary within any locus. Fix or drop it.")
    elif pct01 < 10:
        w(f"At a 0.01 margin the third objective survives at only {pct01:.0f}% of loci. "
          "Treat the work as two-objective and keep off-target as a reported attribute "
          "or a hard constraint.")
    elif pct01 < 40:
        w(f"At a 0.01 margin the third objective survives at {pct01:.0f}% of loci, down "
          f"from {100 * sweep[0.0][0] / L:.0f}% at exact domination. Keep it, but report "
          "the sweep in section 2 — it shows you checked, and pre-empts the reviewer.")
    else:
        w(f"The third objective holds up at {pct01:.0f}% of loci even at a 0.01 margin. "
          "Report the sweep as evidence for the three-objective framing.")
    return "\n".join(O)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rep = "\n\n---\n\n".join(analyse(p) for p in sys.argv[1:])
    print(rep)
    out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])),
                       "offtarget_report_v2.md")
    open(out, "w", encoding="utf-8").write(rep)
    print(f"\n[written to {out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
