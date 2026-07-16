"""
experiments/run_example.py
==========================
Minimal, dependency-light reproducible example for PARETO-EDIT.

Reads a small CSV of SNVs (see examples/example_snvs.csv), runs the surrogate (offline)
oracle so no downloads / GPU / BE-Hive repos are needed, and writes the Pareto front of
non-dominated (editor, PAM, protospacer) designs per locus to a CSV.

Usage (from repository root):
    python -m experiments.run_example --input examples/example_snvs.csv --out out.csv

Input columns : chrom,pos,ref,alt,target_base,required_edit
Output columns: locus,editor,pam_variant,target_pos,efficiency,bystander_purity,offtarget_risk

The real-model (BE-Hive) fronts reported in the paper are produced by
experiments/layer2_application.py; this script exists so the pipeline can be run and
verified end to end from a tiny, self-contained input.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_space import SNVContext
from oracles.surrogate import SurrogateOracle
from oracles.base import CompositeOracle
from optimizers.nsga2 import NSGA2
from pipeline import ParetoEditPipeline


def parse_edit(s: str):
    # "A>G" -> ("A","G")
    s = (s or "").strip().upper()
    if ">" in s:
        a, b = s.split(">", 1)
        return (a.strip(), b.strip())
    return ("", "")


def main():
    ap = argparse.ArgumentParser(description="Minimal PARETO-EDIT example (surrogate oracle).")
    ap.add_argument("--input", default="examples/example_snvs.csv",
                    help="CSV with columns chrom,pos,ref,alt,target_base,required_edit")
    ap.add_argument("--out", default="out.csv", help="output CSV path")
    ap.add_argument("--min-efficiency", type=float, default=0.05,
                    help="drop designs below this predicted efficiency (default 0.05)")
    args = ap.parse_args()

    oracle = CompositeOracle(
        efficiency_source=SurrogateOracle(),
        purity_source=SurrogateOracle(),
        offtarget_source=SurrogateOracle(),
    )

    rows_out = []
    with open(args.input, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            snv = SNVContext(
                chrom=r["chrom"], pos=int(r["pos"]), ref=r["ref"].upper(), alt=r["alt"].upper(),
                target_base=r.get("target_base", "").strip().upper(),
                required_edit=parse_edit(r.get("required_edit", "")),
            )
            pipe = ParetoEditPipeline(oracle=oracle, optimizer=NSGA2(pop_size=30, n_gen=20))
            res = pipe.run(snv, seed=0)
            rep = pipe.report(res, snv, top=10 ** 9)
            if args.min_efficiency > 0:
                rep = [x for x in rep if x["efficiency"] >= args.min_efficiency]
            locus = f'{snv.chrom}:{snv.pos}{snv.ref}>{snv.alt}'
            print(f"{locus}: Pareto front = {len(rep)} non-dominated designs")
            for x in rep:
                rows_out.append({
                    "locus": locus, "editor": x["editor"], "pam_variant": x["pam_variant"],
                    "target_pos": x["target_pos"],
                    "efficiency": round(x["efficiency"], 3),
                    "bystander_purity": round(x["bystander_purity"], 3),
                    "offtarget_risk": round(x["offtarget_risk"], 3),
                })

    cols = ["locus", "editor", "pam_variant", "target_pos",
            "efficiency", "bystander_purity", "offtarget_risk"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {len(rows_out)} designs across all loci to {args.out}")


if __name__ == "__main__":
    main()
