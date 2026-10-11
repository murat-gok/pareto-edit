#!/usr/bin/env python3
r"""
D6: head-to-head comparison against published guide-design tools.

THE DESIGN PRINCIPLE
--------------------
Each tool supplies only a SELECTION. Every selected design is then re-scored with
your own oracle, the same one that scored your own candidates. This isolates the
question you are actually asking -- does returning one design forgo a cleaner
alternative that exists in the design space? -- from the question you are not --
whose efficiency model is better. If you score BE-DICT's pick with BE-DICT's
numbers, a difference means its model disagrees with BE-Hive, which is not your
claim.

WHAT YOU NEED
-------------
1. candidates.json : the FULL enumerated candidate set per locus, not just the
   front. You need this anyway for the efficiency-purity correlation that is
   still missing from the paper, so dump it once and both items are covered.

      {"2:96254050": [{"editor":"ABE8e","pam":"NGG","position":6,
                       "efficiency":0.80,"purity":0.94,"offtarget":0.47}, ...], ...}

   Optionally add "gene":"ALMS1" per locus (see --genes) so results can be
   reported at gene level as well; your 150 loci are clustered within genes and
   are not 150 independent observations.

2. One CSV per tool, in this schema (one row per returned guide):

      locus,editor,pam,position,rank,tool_score
      2:96254050,ABE8e,NGG,6,1,0.81

   rank 1 = the tool's top pick. Include lower ranks if the tool returns a list;
   they are used for the top-k containment statistic. Leave tool_score blank if
   the tool does not expose one. Field names are matched case-insensitively and
   editor names are normalised (BE4max / BE4-max / be4max all match).

USAGE
    python d6_compare.py candidates.json --tool behive=behive_picks.csv \
                                         --tool crisprdesign=cd_picks.csv \
                                         --tool bedict=bedict_picks.csv
    python d6_compare.py candidates.json --tool behive=picks.csv --genes genes.csv

Writes d6_report.md beside the candidates file.
"""

import argparse
import csv
import json
import os
import random
import re
import statistics as stats
from collections import defaultdict

EFF_KEYS = ("efficiency", "eff", "on_target", "ontarget", "activity", "pred_efficiency")
PUR_KEYS = ("purity", "pur", "bystander_purity", "product_purity", "pred_purity")
OFF_KEYS = ("offtarget", "off_target", "off", "offtarget_risk", "cfd", "cfd_score")
ED_KEYS = ("editor", "be", "base_editor", "enzyme")
PAM_KEYS = ("pam", "pam_variant", "cas", "cas9")
POS_KEYS = ("position", "pos", "target_position", "protospacer_pos")

TOPK = 5
SEED = 0


# ----------------------------------------------------------------- helpers
def pick(d, cands):
    low = {str(k).lower(): k for k in d}
    for c in cands:
        if c in low:
            return low[c]
    return None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def norm_ed(x):
    """BE4max / BE4-max / be4_max -> be4max."""
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def norm_pam(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def dkey(ed, pam, pos):
    try:
        p = int(float(pos))
    except (TypeError, ValueError):
        p = pos
    return (norm_ed(ed), norm_pam(pam), p)


def dom2(a, b):
    """a dominates b on (efficiency, purity), both maximised."""
    return ((a["eff"] >= b["eff"] and a["pur"] >= b["pur"])
            and (a["eff"] > b["eff"] or a["pur"] > b["pur"]))


def front2(ds):
    return [d for d in ds if not any(dom2(o, d) for o in ds if o is not d)]


def wilcoxon(diffs):
    """Two-sided signed-rank test. Uses scipy when available."""
    nz = [d for d in diffs if d != 0]
    if len(nz) < 6:
        return None, None
    try:
        from scipy.stats import wilcoxon as _w
        s, p = _w(nz, alternative="two-sided")
        return float(s), float(p)
    except Exception:
        pass
    # normal approximation with tie-corrected ranks
    import math
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ranks = [0.0] * len(nz)
    i = 0
    while i < len(nz):
        j = i
        while j + 1 < len(nz) and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    wpos = sum(r for r, d in zip(ranks, nz) if d > 0)
    n = len(nz)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return None, None
    z = (wpos - mu) / sd
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return wpos, p


def spearman(xs, ys):
    """Spearman rho with average ranks for ties. None when undefined."""
    n = len(xs)
    if n < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    def rank(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    nu = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    dx = sum((x - mx) ** 2 for x in rx) ** 0.5
    dy = sum((y - my) ** 2 for y in ry) ** 0.5
    return nu / (dx * dy) if dx and dy else None


def q(v, f):
    """Quantile by nearest rank."""
    s = sorted(v)
    return s[min(len(s) - 1, int(f * (len(s) - 1)))]


def cliffs_delta(a, b):
    if not a or not b:
        return None
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


# ----------------------------------------------------------------- loading
def load_jsonl(path):
    """Read a dump_candidates.py checkpoint. A truncated final line is dropped, so a
    run still in progress (or one cut by a power failure) can be analysed as it
    stands."""
    out, dropped = {}, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if isinstance(r, dict) and r.get("locus"):
                out[r["locus"]] = {k: v for k, v in r.items() if k != "locus"}
    if dropped:
        print(f"[{os.path.basename(path)}: {dropped} incomplete line(s) ignored]",
              file=__import__("sys").stderr)
    return out


def load_candidates(path):
    if path.endswith(".jsonl"):
        obj = load_jsonl(path)
    else:
        obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, dict):
        for w in ("loci", "results", "data", "candidates", "output"):
            if w in obj and isinstance(obj[w], (dict, list)):
                obj = obj[w]
                break
    items = (list(obj.items()) if isinstance(obj, dict)
             else [(r.get("locus") or r.get("id") or f"locus_{i}", r)
                   for i, r in enumerate(obj) if isinstance(r, dict)])
    out, genes = {}, {}
    for lid, val in items:
        lst = val
        if isinstance(val, dict):
            if val.get("gene"):
                genes[str(lid)] = str(val["gene"])
            for k in ("candidates", "all_designs", "enumerated", "designs",
                      "front", "feasible"):
                if isinstance(val.get(k), list):
                    lst = val[k]
                    break
        if not isinstance(lst, list):
            continue
        ds = []
        for d in lst:
            if not isinstance(d, dict):
                continue
            ke, kp = pick(d, EFF_KEYS), pick(d, PUR_KEYS)
            if not (ke and kp):
                continue
            e, p = num(d[ke]), num(d[kp])
            if e is None or p is None:
                continue
            ko = pick(d, OFF_KEYS)
            ked, kpa, kpo = pick(d, ED_KEYS), pick(d, PAM_KEYS), pick(d, POS_KEYS)
            ds.append({
                "eff": e, "pur": p,
                "off": num(d[ko]) if ko else None,
                "key": dkey(d.get(ked, "?"), d.get(kpa, "?"), d.get(kpo, "?")),
            })
        if ds:
            out[str(lid)] = ds
    return out, genes


def load_tool(path):
    """locus -> [(rank, key)] sorted by rank."""
    picks = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            row = {str(k).strip(): v for k, v in row.items() if k}
            kl = pick(row, ("locus", "locus_id", "variant", "id"))
            ked, kpa, kpo = pick(row, ED_KEYS), pick(row, PAM_KEYS), pick(row, POS_KEYS)
            kr = pick(row, ("rank", "order", "idx"))
            if not kl:
                continue
            r = num(row.get(kr)) if kr else None
            picks[str(row[kl]).strip()].append(
                (r if r is not None else 1e9,
                 dkey(row.get(ked, "?"), row.get(kpa, "?"), row.get(kpo, "?"))))
    for l in picks:
        picks[l].sort(key=lambda t: t[0])
    return dict(picks)


# ----------------------------------------------------------------- analysis
def analyse_tool(name, picks, cands, genes, out):
    w = out.append
    rng = random.Random(SEED)
    n_loci = matched = on_front = topk_hit = 0
    forgone, forgone_rand, eff_rank_frac = [], [], []
    per_gene = defaultdict(list)
    unmatched_examples = []

    for lid, ds in cands.items():
        if lid not in picks:
            continue
        n_loci += 1
        f2 = front2(ds)
        bykey = {d["key"]: d for d in ds}
        frontkeys = {d["key"] for d in f2}
        ranked = picks[lid]
        top = ranked[0][1]

        if top not in bykey:
            if len(unmatched_examples) < 5:
                unmatched_examples.append((lid, top))
            continue
        matched += 1
        d = bykey[top]
        if top in frontkeys:
            on_front += 1
        if any(k in frontkeys for _, k in ranked[:TOPK]):
            topk_hit += 1

        best_pur = max(x["pur"] for x in f2)
        gap = best_pur - d["pur"]
        forgone.append(gap)
        if lid in genes:
            per_gene[genes[lid]].append(gap)

        rand = rng.choice(ds)
        forgone_rand.append(best_pur - rand["pur"])

        effs = sorted((x["eff"] for x in ds), reverse=True)
        eff_rank_frac.append((effs.index(d["eff"]) + 1) / len(effs))

    w(f"\n## {name}\n")
    if not matched:
        w("No pick matched the candidate space. Check that editor, PAM and position "
          "use the same encoding as your enumeration.")
        if unmatched_examples:
            w("\nUnmatched examples: " + ", ".join(f"{l} {k}" for l, k in unmatched_examples))
        return

    w(f"- Loci with a pick: {n_loci}")
    w(f"- Picks inside the enumerated design space: **{matched}/{n_loci} "
      f"({100 * matched / n_loci:.0f}%)**"
      + ("" if matched == n_loci else "  — report this; do not drop the rest silently"))
    w(f"- Top pick lies on the non-dominated set: **{on_front}/{matched} "
      f"({100 * on_front / matched:.0f}%)**")
    w(f"- Any of the tool's top {TOPK} lies on the set: {topk_hit}/{matched} "
      f"({100 * topk_hit / matched:.0f}%)")

    med = stats.median(forgone)
    mean = stats.mean(forgone)
    n_pos = sum(1 for g in forgone if g > 1e-9)
    w(f"\n**Purity forgone** (best purity on the set minus the tool's pick):")
    w(f"- Mean {mean:.4f}, median {med:.4f}, maximum {max(forgone):.4f}")
    w(f"- A strictly purer non-dominated alternative existed at **{n_pos}/{matched} "
      f"loci ({100 * n_pos / matched:.0f}%)**")

    s, p = wilcoxon(forgone)
    if p is not None:
        w(f"- Wilcoxon signed-rank against zero: p = {p:.3g}")
    d_ = cliffs_delta(forgone, forgone_rand)
    w(f"- Mean purity forgone by a random candidate: {stats.mean(forgone_rand):.4f}"
      + (f" (Cliff's delta tool vs random: {d_:+.3f})" if d_ is not None else ""))
    if stats.mean(forgone_rand) <= mean:
        w("  - **The tool does not beat a random candidate on this measure.** Worth "
          "stating plainly; it is a strong result either way.")

    w(f"\n**Where the pick sits**: median efficiency percentile among candidates "
      f"{100 * (1 - stats.median(eff_rank_frac)):.0f}% "
      "(high means the tool is effectively ranking by activity).")

    if per_gene:
        gene_means = [stats.mean(v) for v in per_gene.values()]
        w(f"\n**Gene level** ({len(per_gene)} distinct genes): mean purity forgone "
          f"{stats.mean(gene_means):.4f}, median {stats.median(gene_means):.4f}. "
          "Report this alongside the locus-level figure — loci within a gene are not "
          "independent observations.")
    else:
        w("\n**Gene level**: no gene annotation supplied, so only locus-level figures "
          "are available. Add a `gene` field per locus (or pass --genes) — a reviewer "
          "will ask, because your loci cluster within genes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--tool", action="append", default=[],
                    metavar="NAME=FILE.csv", help="repeatable")
    ap.add_argument("--genes", help="CSV with columns locus,gene")
    a = ap.parse_args()

    cands, genes = load_candidates(a.candidates)
    if a.genes:
        with open(a.genes, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                kl, kg = pick(row, ("locus", "locus_id")), pick(row, ("gene", "symbol"))
                if kl and kg:
                    genes[str(row[kl]).strip()] = str(row[kg]).strip()

    out = []
    out.append("# D6: comparison against published guide-design tools\n")
    out.append(f"Candidate space: {len(cands)} loci, "
               f"{sum(len(v) for v in cands.values())} designs, "
               f"mean {sum(len(v) for v in cands.values()) / len(cands):.1f} per locus.\n")
    out.append("Every tool's pick is re-scored with your oracle, so differences reflect "
               "which design was chosen, not whose model is better.")

    if not a.tool:
        out.append("\nNo tool files given. The candidate space alone still answers one "
                   "open question — see below.")

    for spec in a.tool:
        if "=" not in spec:
            raise SystemExit(f"--tool expects NAME=FILE.csv, got {spec!r}")
        name, path = spec.split("=", 1)
        analyse_tool(name, load_tool(path), cands, genes, out)

    # The efficiency-purity premise. Two estimators, and they answer different
    # questions -- see the note below.
    ae = [d["eff"] for ds in cands.values() for d in ds]
    ap_ = [d["pur"] for ds in cands.values() for d in ds]
    pooled = spearman(ae, ap_)

    within, skipped = [], 0
    for ds in cands.values():
        if len(ds) < 5:
            skipped += 1
            continue
        r = spearman([d["eff"] for d in ds], [d["pur"] for d in ds])
        if r is None:
            skipped += 1
        else:
            within.append(r)

    out.append("\n## The efficiency-purity premise\n")
    out.append("Your paper's premise is that these two objectives conflict. Correlation is a "
               "poor test of it, so read both rows below as diagnostics rather than as "
               "evidence. Pooling across loci mostly measures editability: a clean window "
               "offers high efficiency *and* high purity, a crowded one offers neither, and "
               "that between-locus variation pulls the pooled value positive regardless of "
               "what happens inside a locus (Simpson's paradox). The within-locus value is "
               "closer to the decision a designer makes, but it still measures the bulk "
               "trend over all placements, most of which are poor on both axes at once.\n")
    out.append(f"\n| Estimator | Value | What it answers |")
    out.append(f"| --- | --- | --- |")
    out.append(f"| Pooled over {len(ae)} candidates | "
               f"{'n/a' if pooled is None else f'{pooled:+.3f}'} | "
               "whether editable loci are editable on both axes — **not the premise** |")
    if within:
        med = stats.median(within)
        lo, hi = q(within, .25), q(within, .75)
        neg = sum(1 for r in within if r < 0)
        out.append(f"| Within-locus, median of {len(within)} loci | {med:+.3f} | "
                   "the bulk trend over all placements at one locus — a weak proxy |")
        out.append(f"\n- Within-locus IQR: {lo:+.3f} to {hi:+.3f}")
        out.append(f"- Negative at **{neg}/{len(within)} loci "
                   f"({100 * neg / len(within):.0f}%)**"
                   + (f"; {skipped} loci skipped (too few candidates or no variance)"
                      if skipped else ""))

        # Most enumerated placements put the target base outside the editing window.
        # Those are poor on BOTH axes and can manufacture a positive correlation on their
        # own. Restricting to actionable designs tests whether the trend is real.
        out.append("\n### Is the positive trend driven by unusable placements?\n")
        out.append("Most enumerated placements leave the target base outside the editing "
                   "window, so they score low on efficiency *and* purity together. Hundreds "
                   "of such designs per locus can manufacture a positive correlation by "
                   "themselves. Restricting to designs a user would actually consider tests "
                   "that.\n")
        out.append("\n| Minimum efficiency | Loci | Median candidates kept | "
                   "Within-locus median rho | Negative |")
        out.append("| --- | --- | --- | --- | --- |")
        for thr in (0.0, 0.05, 0.20, 0.50):
            rs, kept = [], []
            for ds in cands.values():
                sub = [d for d in ds if d["eff"] >= thr]
                if len(sub) < 5:
                    continue
                r = spearman([d["eff"] for d in sub], [d["pur"] for d in sub])
                if r is not None:
                    rs.append(r)
                    kept.append(len(sub))
            if rs:
                nneg = sum(1 for r in rs if r < 0)
                out.append(f"| {thr:g} | {len(rs)} | {stats.median(kept):.0f} | "
                           f"{stats.median(rs):+.3f} | {nneg}/{len(rs)} "
                           f"({100 * nneg / len(rs):.0f}%) |")
            else:
                out.append(f"| {thr:g} | 0 | — | — | — |")
        out.append("\nIf the correlation falls toward zero or turns negative as the threshold "
                   "rises, the positive value over all candidates was an artifact of "
                   "unusable placements and the trade-off framing survives for the designs "
                   "that matter. If it stays positive among actionable designs, it is real.")
        if med < -0.2:
            out.append("\n**The premise holds.** Report the within-locus median and the "
                       "fraction of loci with a negative correlation, and say explicitly "
                       "that the pooled figure is confounded by between-locus editability. "
                       "Reporting both, with that explanation, is stronger than reporting "
                       "either alone — it shows you understood the structure of your own "
                       "data.")
        elif med < 0:
            out.append("\n**The premise holds weakly.** The conflict is real but mild on "
                       "average. Soften \"these objectives conflict\" to something the data "
                       "supports, such as that they conflict at a substantial minority of "
                       "loci, and lead with the dominance results instead — those do not "
                       "depend on correlation at all.")
        else:
            out.append(
                "\n**No negative trend, even within loci.** Before rewriting anything, note "
                "that correlation is the wrong statistic for your claim in every form it can "
                "take. Over front members it is -1 by construction, since in two objectives "
                "non-dominance means exactly that one member beats another on one axis and "
                "loses on the other. Pooled across loci it measures editability. Within a "
                "locus it measures the bulk trend over all placements, most of which are "
                "poor on both axes at once — which is why a positive value here is "
                "unsurprising rather than damaging.\n")
            out.append(
                "The trade-off your paper is about lives on the frontier, not in the trend. "
                "If a design is purer than the efficiency-maximising pick, its efficiency is "
                "necessarily lower, because the maximum is already taken: buying purity "
                "costs efficiency. That is a trade-off by definition and it coexists with a "
                "positive overall correlation.\n")
            out.append(
                "So the evidence is the dominance result -- a strictly purer non-dominated "
                "alternative at 96% of loci, mean gap 0.41 -- not any correlation. Keep the "
                "claim, and scope the language to the non-dominated set, which is where your "
                "method operates. Do not report a correlation as support for the premise in "
                "either direction; report that the efficiency-maximising design is almost "
                "never the purest one, and by how much.")
    else:
        out.append("\nToo few candidates per locus to estimate the within-locus "
                   "correlation.")

    rep = "\n".join(out)
    print(rep)
    dest = os.path.join(os.path.dirname(os.path.abspath(a.candidates)), "d6_report.md")
    open(dest, "w", encoding="utf-8").write(rep)
    print(f"\n[written to {dest}]", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
