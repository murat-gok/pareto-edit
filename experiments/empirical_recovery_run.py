"""
experiments/empirical_recovery_run.py
=====================================
Held-out empirical recovery (guide-level) against BE-dataHIVE.

For each ClinVar locus: build the PARETO-EDIT front with the real composition (BE-Hive purity,
CFD off-target, calibrated efficiency, real GRCh38 context), then look up each front design's
protospacer in BE-dataHIVE's measured library and check whether the predicted front recovers the
empirically best-measured guide (top-k), with predicted-vs-measured rank correlation and regret.

This is the paper's strongest validation: predictions scored against real measurements at guide
resolution. Needs torch + BE-Hive repos + reachable be-datahive.com (or PARETO_EDIT_OFFLINE=1 for
a synthetic smoke test of the plumbing only).

Run:
    python experiments/empirical_recovery_run.py \
        --bystander ./be_predict_bystander --celltype mES \
        --clinvar variant_summary.txt.gz --n 20 \
        --cfd-matrix mismatch_score.pkl --cfd-pam pam_scores.pkl
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from design_space import PROTOSPACER_LEN
from oracles.base import CompositeOracle
from oracles.surrogate import SurrogateOracle
from optimizers.nsga2 import NSGA2
from pipeline import ParetoEditPipeline
from data.genome import GenomeContext
from data.be_datahive import BEDataHiveClient
from experiments.benchmark import empirical_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bystander", required=True)
    ap.add_argument("--celltype", default="mES")
    ap.add_argument("--clinvar", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cfd-matrix", default=None)
    ap.add_argument("--cfd-pam", default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--offline-genome", action="store_true")
    ap.add_argument("--be-datahive-file", default=None,
                    help="local BE-dataHIVE parquet/pkl from fetch_be_datahive.py (preferred)")
    args = ap.parse_args()

    offline = os.environ.get("PARETO_EDIT_OFFLINE", "0") == "1"
    if not offline:
        try:
            import torch  # noqa
        except Exception:
            print("PyTorch required for BE-Hive. pip install torch"); sys.exit(2)
        from oracles.behive import BEHiveOracle
        from oracles.cfd import CFDOffTargetOracle
        behive = BEHiveOracle(repo_bystander=args.bystander, repo_efficiency=None,
                              celltype=args.celltype)
        surrogate = SurrogateOracle()
        cfd = CFDOffTargetOracle(mismatch_pkl=args.cfd_matrix, pam_pkl=args.cfd_pam)
        oracle = CompositeOracle(efficiency_source=surrogate, purity_source=behive,
                                 offtarget_source=cfd)
    else:
        oracle = SurrogateOracle()  # plumbing-only smoke test

    genome = GenomeContext(half_width=PROTOSPACER_LEN, offline=args.offline_genome)
    client = BEDataHiveClient(local_file=args.be_datahive_file)
    print(f"BE-dataHIVE: {client.source}")

    # loci
    if args.clinvar:
        from data.clinvar import load_problems
        snvs = load_problems(args.clinvar, max_variants=args.n)
        labels = [f"{s.chrom}:{s.pos}{s.ref}>{s.alt}" for s in snvs]
    else:
        from data.clinvar import synthetic_clinvar_problems, variant_to_snv
        vs = synthetic_clinvar_problems(n=args.n, seed=0)
        snvs = [variant_to_snv(v) for v in vs]
        labels = [f"{v.gene} {v.chrom}:{v.pos}{v.ref}>{v.alt}" for v in vs]

    rows = []
    total_measured = recovered = 0
    corrs, regrets = [], []
    for snv, label in zip(snvs, labels):
        genome.fetch_window(snv.chrom, snv.pos, ref=snv.ref if snv.ref in "ACGT" else None)
        win = genome._cache.get(genome._key(snv.chrom, snv.pos))
        if win:
            snv.context = win
        pipe = ParetoEditPipeline(oracle=oracle, optimizer=NSGA2(pop_size=30, n_gen=20))
        try:
            res = pipe.run(snv, seed=0)
        except Exception as e:
            print(f"{label}: optimization failed ({e})"); continue
        rec = empirical_recovery(res.designs, snv, oracle=oracle, client=client, top_k=args.top_k)
        n_m = rec["n_measured"]
        if n_m == 0:
            print(f"{label}: no BE-dataHIVE measurement for any front guide (skipped)")
            continue
        total_measured += 1
        recovered += int(bool(rec["recovered_top_k"]))
        if rec["rank_corr"] == rec["rank_corr"]:
            corrs.append(rec["rank_corr"])
        if rec["regret"] == rec["regret"]:
            regrets.append(rec["regret"])
        eb = rec["empirical_best"]
        print(f"{label}: measured={n_m} top{args.top_k}_recovered={rec['recovered_top_k']} "
              f"rank_corr={rec['rank_corr']:.3f} regret={rec['regret']:.3f} "
              f"emp_best={eb[0]}")

    print("\n=== Empirical recovery summary ===")
    if total_measured == 0:
        print("No loci had front guides present in BE-dataHIVE. Report this overlap limitation "
              "honestly; recovery cannot be computed where measured guides do not coincide with "
              "the ClinVar-derived protospacers.")
    else:
        print(f"loci with >=1 measured front guide : {total_measured}")
        print(f"top-{args.top_k} recovery rate        : {recovered}/{total_measured} "
              f"({100*recovered/total_measured:.0f}%)")
        if corrs:
            print(f"mean predicted-vs-measured rank corr: {np.mean(corrs):+.3f}")
        if regrets:
            print(f"mean regret (measured eff units)    : {np.mean(regrets):.3f}")
    print("\nNote: overlap between ClinVar-derived protospacers and BE-dataHIVE's measured guides "
          "bounds how many loci are scorable; state this in the paper.")


if __name__ == "__main__":
    main()
