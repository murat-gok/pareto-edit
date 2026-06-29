"""
experiments/layer2_application.py
=================================
Layer 2 (the biological application): produce real Pareto-optimal editing designs for real
pathogenic loci, using REAL genomic context (Ensembl GRCh38) and the REAL BE-Hive bystander
model for the bystander-purity objective.

Pipeline per locus:
    ClinVar SNV -> fetch GRCh38 flank (Ensembl, cached) -> attach as context ->
    optimize with NSGA-II against CompositeOracle(eff=surrogate, purity=BE-Hive, off=surrogate)
    -> report the Pareto front.

This is distinct from Layer 1 (the surrogate-only optimizer benchmark). See APPROACH_A.md.

Run (needs torch + BE-Hive repos + internet for Ensembl, first time only):
    python experiments/layer2_application.py \
        --bystander ./be_predict_bystander --celltype mES \
        --clinvar variant_summary.txt.gz --n 10

Without --clinvar it uses a few synthetic pathogenic SNVs (still fetches real GRCh38 flanks).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_space import SNVContext, PROTOSPACER_LEN
from oracles.base import CompositeOracle
from oracles.surrogate import SurrogateOracle
from optimizers.nsga2 import NSGA2
from pipeline import ParetoEditPipeline
from data.genome import GenomeContext


def attach_real_context(snv: SNVContext, genome: GenomeContext) -> bool:
    """Fetch the real GRCh38 window centred on the variant and attach it as snv.context so that
    window_context() slices real sequence. half_width must equal PROTOSPACER_LEN so the variant
    base lands at index PROTOSPACER_LEN (what window_context expects). Returns False if the
    flank could not be obtained (offline + uncached)."""
    window = genome.fetch_window(snv.chrom, snv.pos, ref=snv.ref if snv.ref in "ACGT" else None)
    if window is None:
        return False
    # window has length 2*half_width+1, centre index == half_width. window_context expects the
    # centre (target base) at index PROTOSPACER_LEN, so half_width must be PROTOSPACER_LEN.
    snv.context = window
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bystander", required=True)
    ap.add_argument("--celltype", default="mES")
    ap.add_argument("--clinvar", default=None, help="variant_summary.txt(.gz); else synthetic")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--offline-genome", action="store_true",
                    help="use only cached genome windows (no Ensembl calls)")
    ap.add_argument("--cfd-matrix", default=None,
                    help="path to Doench 2016 mismatch_score.pkl (real CFD matrix)")
    ap.add_argument("--cfd-pam", default=None,
                    help="path to Doench 2016 pam_scores.pkl (optional)")
    ap.add_argument("--high-purity", type=float, default=0.50,
                    help="purity threshold for flagging loci with no high-purity option")
    ap.add_argument("--direct-only", action="store_true",
                    help="restrict the catalogue to editors with a direct (non-proxy) BE-Hive "
                         "model (excludes YE1, CGBE1); use for the primary analysis")
    ap.add_argument("--out-json", default=None,
                    help="write machine-readable summary + per-locus fronts to this JSON")
    ap.add_argument("--efficiency-calibration", default=None,
                    help="path to efficiency_calibration.json (from oracles.efficiency_calibration "
                         "fit); if given, efficiency is calibrated to BE-dataHIVE measurements")
    ap.add_argument("--aggregate-offtarget", action="store_true",
                    help="use the genome-aware-style aggregate-CFD specificity score (Hsu 2013, "
                         "seed-region near-cognate aggregation) instead of single-mismatch "
                         "propensity for the off-target objective")
    ap.add_argument("--ot-max-mismatches", type=int, default=2,
                    help="max mismatches for aggregate off-target enumeration (default 2)")
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
    # off-target from CFD (Doench 2016). Provide --cfd-matrix for the real published matrix;
    # otherwise CFD runs in its documented position-weighted fallback mode.
    cfd = CFDOffTargetOracle(mismatch_pkl=args.cfd_matrix, pam_pkl=args.cfd_pam,
                             aggregate=args.aggregate_offtarget,
                             max_mismatches=args.ot_max_mismatches)
    if args.aggregate_offtarget:
        print(f"Off-target: aggregate-CFD specificity score (Hsu 2013 family, seed-region, "
              f"<={args.ot_max_mismatches}mm)")
    else:
        print("Off-target: single-mismatch CFD propensity (pass --aggregate-offtarget for "
              "genome-aware-style aggregate scoring)")
    eff_cal = None
    if args.efficiency_calibration:
        from oracles.efficiency_calibration import CalibratedEfficiency
        eff_cal = CalibratedEfficiency.load(args.efficiency_calibration)
        print("Efficiency: surrogate calibrated to BE-dataHIVE measurements "
              f"({args.efficiency_calibration})")
    else:
        print("Efficiency: surrogate (uncalibrated; pass --efficiency-calibration to calibrate)")
    oracle = CompositeOracle(efficiency_source=surrogate,  # efficiency: calibrated surrogate
                             purity_source=behive,          # purity: REAL BE-Hive  (novel)
                             offtarget_source=cfd,          # off-target: CFD (Doench 2016)
                             efficiency_calibrator=eff_cal)
    print(f"Off-target oracle: CFD ({'REAL matrix' if cfd.using_real_matrix else 'fallback'})")

    # half_width MUST equal PROTOSPACER_LEN for alignment with window_context()
    genome = GenomeContext(half_width=PROTOSPACER_LEN, offline=args.offline_genome)

    # gather loci
    if args.clinvar:
        from data.clinvar import load_problems
        snvs = load_problems(args.clinvar, max_variants=args.n)
        labels = [f"{s.chrom}:{s.pos}{s.ref}>{s.alt}[{s.target_base}]" for s in snvs]
    else:
        from data.clinvar import synthetic_clinvar_problems, variant_to_snv
        variants = synthetic_clinvar_problems(n=args.n, seed=0)
        snvs = [variant_to_snv(v) for v in variants]
        labels = [f"{v.gene} {v.chrom}:{v.pos}{v.ref}>{v.alt}[{v.chemistry}]" for v in variants]

    import json
    import numpy as np
    from collections import Counter
    from oracles.behive import editor_is_proxy, BEHIVE_EDITOR_IS_PROXY

    proxy_editors = tuple(e for e, p in BEHIVE_EDITOR_IS_PROXY.items() if p)
    exclude = proxy_editors if args.direct_only else ()
    mode_note = ("DIRECT-MODEL editors only (proxy editors excluded: "
                 f"{', '.join(proxy_editors)})" if args.direct_only
                 else "all editors (proxy-mapped editors included, flagged in output)")
    print(f"Layer-2 application: {len(snvs)} loci, real GRCh38 context, real BE-Hive purity")
    print(f"Editor set: {mode_note}\n")

    front_sizes = []
    calls_list = []
    n_real_ctx = 0
    n_no_high_purity = 0
    editor_counts = Counter()
    pos_counts = Counter()
    n_proxy_designs = 0          # how many front designs use a proxy-mapped editor
    n_total_designs = 0
    best_purity_per_locus = []
    best_eff_per_locus = []
    records = []
    HIGH_PURITY = args.high_purity

    for snv, label in zip(snvs, labels):
        ok = attach_real_context(snv, genome)
        n_real_ctx += int(ok)
        ctx_note = "real GRCh38" if ok else "SYNTHETIC (flank unavailable)"
        pipe = ParetoEditPipeline(oracle=oracle, optimizer=NSGA2(pop_size=30, n_gen=20),
                                  exclude_editors=exclude)
        try:
            res = pipe.run(snv, seed=0)
        except Exception as e:
            print(f"{label}: optimization failed ({e})")
            continue
        rep = pipe.report(res, snv, top=10**9)
        front_sizes.append(len(res.objectives))
        calls_list.append(res.n_evals)
        purities = [r["bystander_purity"] for r in rep]
        effs = [r["efficiency"] for r in rep]
        best_purity_per_locus.append(max(purities) if purities else float("nan"))
        best_eff_per_locus.append(max(effs) if effs else float("nan"))
        if purities and max(purities) < HIGH_PURITY:
            n_no_high_purity += 1
        for r in rep:
            editor_counts[r["editor"]] += 1
            pos_counts[r["target_pos"]] += 1
            n_total_designs += 1
            is_px = editor_is_proxy(r["editor"])
            r["proxy_editor"] = is_px
            n_proxy_designs += int(is_px)
        records.append({"locus": label, "context": ctx_note,
                        "front_size": len(res.objectives), "be_hive_calls": res.n_evals,
                        "max_purity": max(purities) if purities else None,
                        "max_efficiency": max(effs) if effs else None,
                        "front": rep[:10]})

        if len(records) <= 10:
            print(f"### {label}  [{ctx_note}]  front={len(res.objectives)} "
                  f"BE-Hive calls={res.n_evals}")
            for r in rep[:6]:
                px = "  (proxy)" if r.get("proxy_editor") else ""
                print(f"     {r['editor']:10s} {r['pam_variant']:12s} pos {r['target_pos']:>2d} | "
                      f"eff {r['efficiency']:.3f}  purity {r['bystander_purity']:.3f}  "
                      f"offtgt {r['offtarget_risk']:.3f}{px}")
            print()

    n = len(front_sizes)
    if n:
        summary = {
            "editor_mode": "direct_only" if args.direct_only else "all_editors",
            "proxy_editors_excluded": list(proxy_editors) if args.direct_only else [],
            "n_loci_attempted": len(snvs),
            "n_loci_solved": n,
            "n_real_context": n_real_ctx,
            "front_size_mean": float(np.mean(front_sizes)),
            "front_size_median": float(np.median(front_sizes)),
            "front_size_min": int(np.min(front_sizes)),
            "front_size_max": int(np.max(front_sizes)),
            "be_hive_calls_mean": float(np.mean(calls_list)),
            "high_purity_threshold": HIGH_PURITY,
            "n_loci_no_high_purity": n_no_high_purity,
            "frac_loci_no_high_purity": n_no_high_purity / n,
            "mean_best_purity_per_locus": float(np.nanmean(best_purity_per_locus)),
            "mean_best_efficiency_per_locus": float(np.nanmean(best_eff_per_locus)),
            "n_front_designs_total": n_total_designs,
            "n_front_designs_proxy": n_proxy_designs,
            "frac_front_designs_proxy": (n_proxy_designs / n_total_designs
                                         if n_total_designs else 0.0),
            "editor_distribution": dict(editor_counts.most_common()),
            "position_distribution": dict(sorted(pos_counts.items())),
        }
        print("\n=== Layer-2 summary (for manuscript Results 3.4) ===")
        print(f"editor mode                 : {summary['editor_mode']}")
        print(f"loci solved                 : {n}/{len(snvs)}  (real context: {n_real_ctx})")
        print(f"front size  mean/median/range: {summary['front_size_mean']:.1f} / "
              f"{summary['front_size_median']:.0f} / "
              f"{summary['front_size_min']}–{summary['front_size_max']}")
        print(f"BE-Hive calls per locus (mean): {summary['be_hive_calls_mean']:.0f}")
        print(f"loci with NO high-purity option (<{HIGH_PURITY}): "
              f"{n_no_high_purity}/{n} ({100*n_no_high_purity/n:.0f}%)")
        print(f"mean best purity / efficiency : {summary['mean_best_purity_per_locus']:.3f} / "
              f"{summary['mean_best_efficiency_per_locus']:.3f}")
        if not args.direct_only:
            print(f"front designs using proxy editors: {n_proxy_designs}/{n_total_designs} "
                  f"({100*summary['frac_front_designs_proxy']:.0f}%)")
        print(f"editor usage on fronts        : {summary['editor_distribution']}")
        if args.out_json:
            with open(args.out_json, "w") as fh:
                json.dump({"summary": summary, "loci": records}, fh, indent=2)
            print(f"\nWrote machine-readable results to {args.out_json}")

    print("\nThese fronts are the paper's biological output: real loci, real genomic context, "
          "real BE-Hive bystander-purity. (Layer 1 surrogate benchmarks are reported separately.)")


if __name__ == "__main__":
    main()
