"""
experiments/phase2_demo.py
==========================
Phase-2 integration demo: real-data plumbing end-to-end.

Flow:
  ClinVar pathogenic SNV  ->  SNVContext problem  ->  PARETO-EDIT (NSGA-II)  ->  Pareto front
                                                  ->  empirical recovery vs BE-dataHIVE

Runs OFFLINE by default (PARETO_EDIT_OFFLINE=1 synthetic stand-ins) so it is testable without
the multi-hundred-MB ClinVar download or live be-datahive.com access. To run for real:

  # 1. download ClinVar once
  wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
  # 2. ensure be-datahive.com is reachable (unset the offline flag)
  python experiments/phase2_demo.py --clinvar variant_summary.txt.gz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.clinvar import (
    load_problems, stream_variant_summary, synthetic_clinvar_problems, variant_to_snv,
)
from data.be_datahive import BEDataHiveClient
from oracles.surrogate import SurrogateOracle
from optimizers.nsga2 import NSGA2
from pipeline import ParetoEditPipeline
from experiments.benchmark import empirical_recovery


def main(clinvar_path: str | None, n: int, offline: bool):
    if offline:
        os.environ["PARETO_EDIT_OFFLINE"] = "1"

    # 1) obtain problems
    if clinvar_path:
        print(f"Loading base-editable pathogenic SNVs from {clinvar_path} ...")
        problems = load_problems(clinvar_path, max_variants=n)
        snvs = problems
        labels = [f"{s.chrom}:{s.pos}{s.ref}>{s.alt}" for s in snvs]
    else:
        print("No ClinVar file given -> using synthetic base-editable pathogenic SNVs.")
        variants = synthetic_clinvar_problems(n=n, seed=0)
        snvs = [variant_to_snv(v) for v in variants]
        labels = [f"{v.gene} {v.chrom}:{v.pos}{v.ref}>{v.alt} [{v.chemistry}]" for v in variants]

    oracle = SurrogateOracle()
    client = BEDataHiveClient()
    print(f"BE-dataHIVE client: {'OFFLINE (synthetic)' if client.offline else 'LIVE'}")

    print(f"\n{'locus':42s} {'front':>5s} {'emp.best':>10s} {'recov':>6s} {'top3':>5s}")
    print("-" * 75)
    n_recovered = 0
    for snv, label in zip(snvs, labels):
        pipe = ParetoEditPipeline(oracle=oracle, optimizer=NSGA2(pop_size=40, n_gen=25))
        res = pipe.run(snv, seed=0)
        rec = empirical_recovery(res.designs, snv, client=client, objective="efficiency")
        recovered = rec["recovered"]
        n_recovered += int(bool(recovered))
        print(f"{label[:42]:42s} {len(res.objectives):>5d} "
              f"{str(rec['empirical_best_editor'])[:10]:>10s} "
              f"{str(recovered):>6s} {str(rec['recovered_top_k']):>5s}")

    print(f"\nRecovered empirical-best editor in front for {n_recovered}/{len(snvs)} loci "
          f"({'offline synthetic — illustrative only' if client.offline else 'LIVE data'}).")
    print("\nNOTE: offline numbers exercise the plumbing only. Real recovery requires the "
          "ClinVar download + live BE-dataHIVE, and guide-level matching (Phase 3).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinvar", type=str, default=None, help="path to variant_summary.txt(.gz)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--live", action="store_true", help="attempt live BE-dataHIVE (else offline)")
    args = ap.parse_args()
    main(args.clinvar, args.n, offline=not args.live)
