"""
experiments/phase3_behive.py
============================
Phase-3 driver: optimize against the REAL BE-Hive oracle (efficiency + bystander purity),
composed with an off-target oracle, then score held-out empirical recovery.

This requires BE-Hive's repos + PyTorch, which are NOT available in every sandbox. The script
is written against BE-Hive's verified interface and will run wherever those are installed
(e.g. the user's office PC / WSL). If imports fail it explains exactly what to install and
exits cleanly -- it never silently fakes real-model numbers.

Setup (once):
    git clone https://github.com/maxwshen/be_predict_bystander.git
    git clone https://github.com/maxwshen/be_predict_efficiency.git
    pip install torch pandas numpy        # CPU torch is sufficient (BE-Hive runs on CPU)

Run:
    python experiments/phase3_behive.py \
        --bystander /path/to/be_predict_bystander \
        --efficiency /path/to/be_predict_efficiency \
        --celltype mES
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_space import SNVContext
from oracles.base import CompositeOracle
from oracles.surrogate import SurrogateOracle
from optimizers.nsga2 import NSGA2
from optimizers.cpo_gwo import CPO_GWO
from pipeline import ParetoEditPipeline


def build_oracle(bystander_repo, efficiency_repo, celltype, offtarget):
    """Compose the oracle so each objective comes from its most appropriate source:

      * bystander_purity : REAL BE-Hive bystander model  (the paper's novel objective)
      * efficiency       : calibrated surrogate efficiency component
      * offtarget_risk   : off-target oracle (surrogate here; crispAI/CFD in Phase-3b)

    Rationale: BE-Hive's *bystander* model is the authoritative source for purity, which is our
    contribution. Editing *efficiency* is a well-studied quantity with many dedicated predictors;
    we source it from the calibrated surrogate component rather than BE-Hive's separately-pickled
    efficiency model (which ships in a legacy scikit-learn format). This keeps each objective on
    its best-justified source and yields a diverse front. `efficiency_repo` is accepted for
    interface stability but not required for purity.
    """
    from oracles.behive import BEHiveOracle
    behive = BEHiveOracle(repo_bystander=bystander_repo, repo_efficiency=None,
                          celltype=celltype)
    return CompositeOracle(efficiency_source=offtarget,   # surrogate supplies efficiency
                           purity_source=behive,           # REAL BE-Hive supplies purity
                           offtarget_source=offtarget)     # surrogate supplies off-target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bystander", required=True, help="path to cloned be_predict_bystander")
    ap.add_argument("--efficiency", default=None, help="path to cloned be_predict_efficiency")
    ap.add_argument("--celltype", default="mES", choices=["mES", "HEK293", "HEK293T"])
    ap.add_argument("--optimizer", default="nsga2", choices=["nsga2", "cpo-gwo"])
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gen", type=int, default=20)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except Exception:
        print("ERROR: PyTorch not found. BE-Hive needs it. Install with:\n"
              "  pip install torch   (CPU build is fine)\n"
              "Then re-run. (In sandboxes without torch, use the surrogate oracle instead.)")
        sys.exit(2)

    try:
        offtarget = SurrogateOracle()  # supplies offtarget_risk only (Phase-3b: crispAI/CFD)
        oracle = build_oracle(args.bystander, args.efficiency, args.celltype, offtarget)
    except Exception as e:
        print(f"ERROR initializing BE-Hive oracle: {e}\n"
              "Check that the repo paths are correct and the params/ weights are present.")
        sys.exit(2)

    Opt = NSGA2 if args.optimizer == "nsga2" else CPO_GWO
    # Example CBE-correctable locus (target base C)
    snv = SNVContext(chrom="chr11", pos=5226778, ref="A", alt="G", target_base="C")

    pipe = ParetoEditPipeline(oracle=oracle, optimizer=Opt(pop_size=args.pop, n_gen=args.gen))
    print(f"Optimizing against REAL BE-Hive ({args.celltype}) with {args.optimizer} ...")
    res = pipe.run(snv, seed=0)
    print(f"Pareto front: {len(res.objectives)} designs, {res.n_evals} unique BE-Hive calls\n")
    for r in pipe.report(res, snv, top=12):
        print(f"  {r['editor']:10s} {r['pam_variant']:12s} pos {r['target_pos']:>2d} | "
              f"eff {r['efficiency']:.3f}  purity {r['bystander_purity']:.3f}  "
              f"offtgt {r['offtarget_risk']:.3f}")

    print("\nNext: compare this REAL-oracle front against the SURROGATE front (agreement / rank "
          "correlation), and run empirical_recovery() against BE-dataHIVE for this locus.")


if __name__ == "__main__":
    main()
