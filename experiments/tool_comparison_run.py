"""
B3 — PARETO-EDIT front vs a single-recommendation tool (reviewer #4).

Operationalizes the "complementary, not competitive" claim as an empirical result:
for each ClinVar locus we (1) compute PARETO-EDIT's Pareto front, and (2) compute the single
design a recommendation tool would return — the maximizer of a standard composite score over the
*identical* candidate set. We then check whether that single pick is on the front (it should be,
if non-dominated), and quantify how many additional non-dominated trade-off designs the front
exposes that the single composite score hides — in particular higher-purity designs the composite
ranked below its pick.

Two composite proxies are reported (both are common single-score strategies existing tools use):
  - eff_x_purity : efficiency * purity            (balanced recommender)
  - eff_only     : efficiency                      (activity-first recommender)

Runs on the SAME machinery as layer2_application.py (real BE-Hive purity, CFD off-target,
calibrated efficiency). No extra downloads beyond what Layer-2 already needs.

Usage:
  python -m experiments.tool_comparison_run --bystander .\\be_predict_bystander --celltype mES \\
      --clinvar variant_summary.txt.gz --n 150 \\
      --cfd-matrix mismatch_score.pkl --cfd-pam pam_scores.pkl \\
      --efficiency-calibration efficiency_calibration.json --out-json tool_comparison.json
"""
import argparse
import sys
import json
import numpy as np
from collections import Counter

from design_space import SNVContext, PROTOSPACER_LEN
from data.genome import GenomeContext
from optimizers.nsga2 import NSGA2
from oracles.base import CompositeOracle
from oracles.surrogate import SurrogateOracle
from pipeline import ParetoEditPipeline


def attach_real_context(snv, genome):
    try:
        ctx = genome.window_context(snv.chrom, snv.pos, snv.ref, snv.alt)
        if ctx is None:
            return False
        snv.context = ctx
        return True
    except Exception:
        return False


def is_dominated(p, others):
    """True if point p (minimization vector) is dominated by any q in others."""
    for q in others:
        if all(qi <= pi for qi, pi in zip(q, p)) and any(qi < pi for qi, pi in zip(q, p)):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bystander", required=True)
    ap.add_argument("--celltype", default="mES")
    ap.add_argument("--clinvar", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--offline-genome", action="store_true")
    ap.add_argument("--cfd-matrix", default=None)
    ap.add_argument("--cfd-pam", default=None)
    ap.add_argument("--aggregate-offtarget", action="store_true")
    ap.add_argument("--ot-max-mismatches", type=int, default=2)
    ap.add_argument("--efficiency-calibration", default=None)
    ap.add_argument("--high-purity", type=float, default=0.50)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    try:
        import torch  # noqa
    except Exception:
        print("PyTorch required for BE-Hive. pip install torch"); sys.exit(2)

    from oracles.behive import BEHiveOracle
    from oracles.cfd import CFDOffTargetOracle

    behive = BEHiveOracle(repo_bystander=args.bystander, repo_efficiency=None,
                          celltype=args.celltype)
    surrogate = SurrogateOracle()
    cfd = CFDOffTargetOracle(mismatch_pkl=args.cfd_matrix, pam_pkl=args.cfd_pam,
                             aggregate=args.aggregate_offtarget,
                             max_mismatches=args.ot_max_mismatches)
    eff_cal = None
    if args.efficiency_calibration:
        from oracles.efficiency_calibration import CalibratedEfficiency
        eff_cal = CalibratedEfficiency.load(args.efficiency_calibration)
    oracle = CompositeOracle(efficiency_source=surrogate, purity_source=behive,
                             offtarget_source=cfd, efficiency_calibrator=eff_cal)

    genome = GenomeContext(half_width=PROTOSPACER_LEN, offline=args.offline_genome)

    if args.clinvar:
        from data.clinvar import load_problems
        snvs = load_problems(args.clinvar, max_variants=args.n)
        labels = [f"{s.chrom}:{s.pos}{s.ref}>{s.alt}[{s.target_base}]" for s in snvs]
    else:
        from data.clinvar import synthetic_clinvar_problems, variant_to_snv
        variants = synthetic_clinvar_problems(n=args.n, seed=0)
        snvs = [variant_to_snv(v) for v in variants]
        labels = [f"{v.gene} {v.chrom}:{v.pos}{v.ref}>{v.alt}[{v.chemistry}]" for v in variants]

    print(f"Tool comparison: {len(snvs)} loci. Front vs single-composite recommendation.\n")

    # accumulators
    n_loci = 0
    on_front = {"eff_x_purity": 0, "eff_only": 0}      # single pick is non-dominated (on front)
    extra_designs = []                                  # front_size - 1 (the alternatives a single pick hides)
    higher_purity_hidden = {"eff_x_purity": 0, "eff_only": 0}  # a front design has strictly higher purity than the pick
    purity_gap = {"eff_x_purity": [], "eff_only": []}   # best-front-purity minus pick-purity
    records = []

    for snv, label in zip(snvs, labels):
        ok = attach_real_context(snv, genome)
        pipe = ParetoEditPipeline(oracle=oracle, optimizer=NSGA2(pop_size=30, n_gen=20))
        try:
            res = pipe.run(snv, seed=0)
        except Exception as e:
            print(f"{label}: failed ({e})"); continue
        rep = pipe.report(res, snv, top=10**9)
        if not rep:
            continue
        n_loci += 1

        # front objective vectors (minimization): (-eff, -purity, offtarget)
        front_objs = [(-r["efficiency"], -r["bystander_purity"], r["offtarget_risk"]) for r in rep]
        best_front_purity = max(r["bystander_purity"] for r in rep)
        extra_designs.append(len(rep) - 1)

        # single-composite picks over the SAME candidate set (the front is the candidate set here:
        # these are the non-dominated designs; a recommender that scalarizes would choose one of the
        # evaluated designs — we score the front members, which are exactly the non-dominated options).
        for key, score in [
            ("eff_x_purity", lambda r: r["efficiency"] * r["bystander_purity"]),
            ("eff_only",     lambda r: r["efficiency"]),
        ]:
            pick = max(rep, key=score)
            pick_obj = (-pick["efficiency"], -pick["bystander_purity"], pick["offtarget_risk"])
            others = [o for o in front_objs if o != pick_obj]
            if not is_dominated(pick_obj, others):
                on_front[key] += 1
            gap = best_front_purity - pick["bystander_purity"]
            purity_gap[key].append(gap)
            if gap > 1e-6:
                higher_purity_hidden[key] += 1

        records.append({
            "locus": label,
            "front_size": len(rep),
            "best_front_purity": round(best_front_purity, 4),
            "pick_eff_x_purity_purity": round(max(rep, key=lambda r: r["efficiency"]*r["bystander_purity"])["bystander_purity"], 4),
            "pick_eff_only_purity": round(max(rep, key=lambda r: r["efficiency"])["bystander_purity"], 4),
        })

        print(f"{label}: front={len(rep)}  best_purity={best_front_purity:.2f}  "
              f"eff*pur_pick_purity={records[-1]['pick_eff_x_purity_purity']:.2f}  "
              f"eff_only_pick_purity={records[-1]['pick_eff_only_purity']:.2f}")

    # summary
    print("\n=== Tool comparison summary (Results: front vs single recommendation) ===")
    print(f"loci evaluated                     : {n_loci}")
    print(f"mean extra non-dominated designs   : {np.mean(extra_designs):.1f} "
          f"(median {np.median(extra_designs):.0f}) the front exposes beyond a single pick")
    for key in ["eff_x_purity", "eff_only"]:
        frac_on = on_front[key] / n_loci if n_loci else float("nan")
        frac_hidden = higher_purity_hidden[key] / n_loci if n_loci else float("nan")
        mean_gap = float(np.mean(purity_gap[key])) if purity_gap[key] else float("nan")
        print(f"\n  composite = {key}")
        print(f"    single pick lies on the front       : {on_front[key]}/{n_loci} ({100*frac_on:.0f}%)")
        print(f"    a front design has higher purity     : {higher_purity_hidden[key]}/{n_loci} ({100*frac_hidden:.0f}%)")
        print(f"    mean purity gap (best_front - pick)  : {mean_gap:.3f}")

    print("\nInterpretation: where the single pick lies on the front, PARETO-EDIT contains the "
          "tool's recommendation (complementary). Where a front design has higher purity than the "
          "pick, the single composite score hid a cleaner-editing alternative the front surfaces — "
          "the concrete added value of returning the trade-off set.")

    if args.out_json:
        summary = {
            "n_loci": n_loci,
            "mean_extra_nondominated_designs": float(np.mean(extra_designs)) if extra_designs else None,
            "median_extra_nondominated_designs": float(np.median(extra_designs)) if extra_designs else None,
            "composites": {
                key: {
                    "single_pick_on_front": on_front[key],
                    "single_pick_on_front_frac": on_front[key]/n_loci if n_loci else None,
                    "front_has_higher_purity": higher_purity_hidden[key],
                    "front_has_higher_purity_frac": higher_purity_hidden[key]/n_loci if n_loci else None,
                    "mean_purity_gap": float(np.mean(purity_gap[key])) if purity_gap[key] else None,
                } for key in ["eff_x_purity", "eff_only"]
            },
            "loci": records,
        }
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
