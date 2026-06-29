"""
experiments/ablation.py
=======================
Ablation studies for PARETO-EDIT (B5), addressing two reviewer requests:

1. OBJECTIVE-DROPPING ablation — does treating bystander purity as an explicit objective actually
   matter? We compare the full 3-objective optimization against a 2-objective variant that drops
   purity (optimizing only efficiency and off-target), then measure the *true* purity of each
   variant's selected designs. If the 2-objective front systematically yields lower-purity
   designs, the purity objective demonstrably adds value — directly supporting the paper's
   central novelty claim.

2. HYPERPARAMETER SENSITIVITY — how stable is NSGA-II to population size and generation count?
   We sweep (pop_size, n_gen) and report front quality (hypervolume) to show the reported results
   are not a artefact of one lucky setting.

Both run on the surrogate by default (fast, no external models); pass --bystander/--celltype to
run the objective-dropping ablation against the real BE-Hive purity, which is the version to
report. Writes a JSON summary.

Run (surrogate, fast):
    python -m experiments.ablation --n 30 --out-json ablation_results.json

Run (real BE-Hive purity for the objective ablation):
    python -m experiments.ablation --n 30 --bystander ./be_predict_bystander --celltype mES \
        --cfd-matrix mismatch_score.pkl --cfd-pam pam_scores.pkl --out-json ablation_real.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from design_space import PROTOSPACER_LEN
from objectives import evaluate
from oracles.surrogate import SurrogateOracle
from optimizers.nsga2 import NSGA2
from pipeline import ParetoEditPipeline
from metrics import hypervolume


def _build_oracle(args):
    """Surrogate by default; real composition if --bystander is given."""
    if not args.bystander:
        return SurrogateOracle(), None
    from oracles.base import CompositeOracle
    from oracles.behive import BEHiveOracle
    from oracles.cfd import CFDOffTargetOracle
    behive = BEHiveOracle(repo_bystander=args.bystander, repo_efficiency=None,
                          celltype=args.celltype)
    surrogate = SurrogateOracle()
    cfd = CFDOffTargetOracle(mismatch_pkl=args.cfd_matrix, pam_pkl=args.cfd_pam)
    genome = None
    if not args.no_genome:
        from data.genome import GenomeContext
        genome = GenomeContext(half_width=PROTOSPACER_LEN, offline=args.offline_genome)
    return CompositeOracle(efficiency_source=surrogate, purity_source=behive,
                           offtarget_source=cfd), genome


def _attach_ctx(snv, genome):
    if genome is None:
        return False
    w = genome.fetch_window(snv.chrom, snv.pos, ref=snv.ref if snv.ref in "ACGT" else None)
    if w:
        snv.context = w
        return True
    return False


def objective_dropping(snvs, oracle, genome, seed=0):
    """Compare full 3-objective vs purity-dropped 2-objective optimization, measuring the true
    purity achieved by each. Returns per-locus and aggregate stats."""
    rows = []
    for snv in snvs:
        _attach_ctx(snv, genome)
        # full 3-objective
        full = ParetoEditPipeline(oracle=oracle, optimizer=NSGA2(pop_size=30, n_gen=20),
                                  objective_mask=(0, 1, 2))
        try:
            r_full = full.run(snv, seed=seed)
        except Exception:
            continue
        # 2-objective: drop purity (index 1); optimizer sees only (efficiency, offtarget)
        drop = ParetoEditPipeline(oracle=oracle, optimizer=NSGA2(pop_size=30, n_gen=20),
                                  objective_mask=(0, 2))
        try:
            r_drop = drop.run(snv, seed=seed)
        except Exception:
            continue
        # measure TRUE purity (full 3-obj) of each front's designs
        def true_purity(designs):
            ps = []
            for d in designs:
                f = evaluate(d, snv, oracle)
                ps.append(-f[1])  # purity = -neg_purity
            return ps
        pur_full = true_purity(r_full.designs)
        pur_drop = true_purity(r_drop.designs)
        rows.append({
            "locus": f"{snv.chrom}:{snv.pos}",
            "best_purity_full": float(max(pur_full)) if pur_full else float("nan"),
            "best_purity_drop": float(max(pur_drop)) if pur_drop else float("nan"),
            "mean_purity_full": float(np.mean(pur_full)) if pur_full else float("nan"),
            "mean_purity_drop": float(np.mean(pur_drop)) if pur_drop else float("nan"),
            "front_full": len(r_full.designs), "front_drop": len(r_drop.designs),
        })
    # aggregate
    bf = np.array([r["best_purity_full"] for r in rows], dtype=float)
    bd = np.array([r["best_purity_drop"] for r in rows], dtype=float)
    valid = np.isfinite(bf) & np.isfinite(bd)
    agg = {
        "n_loci": int(valid.sum()),
        "mean_best_purity_full": float(np.nanmean(bf[valid])) if valid.any() else None,
        "mean_best_purity_drop": float(np.nanmean(bd[valid])) if valid.any() else None,
        "loci_where_full_higher_purity": int(np.sum(bf[valid] > bd[valid] + 1e-9)),
        "loci_where_equal": int(np.sum(np.abs(bf[valid] - bd[valid]) <= 1e-9)),
    }
    # Wilcoxon if enough points
    if valid.sum() >= 5:
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(bf[valid], bd[valid])
            agg["wilcoxon_p"] = float(p)
        except Exception:
            agg["wilcoxon_p"] = None
    return {"aggregate": agg, "per_locus": rows}


def hyperparameter_sensitivity(snvs, oracle, genome, seed=0):
    """Sweep (pop_size, n_gen) and report mean hypervolume of the full 3-objective front."""
    settings = [(20, 15), (30, 20), (40, 30), (60, 40)]
    out = []
    for pop, gen in settings:
        hvs = []
        for snv in snvs[:min(len(snvs), 15)]:  # cap for speed
            _attach_ctx(snv, genome)
            pipe = ParetoEditPipeline(oracle=oracle, optimizer=NSGA2(pop_size=pop, n_gen=gen))
            try:
                res = pipe.run(snv, seed=seed)
                # reference point for HV: objectives are minimization; worst-case [0,0,1] in the
                # negated space corresponds to (eff=0, purity=0, offtarget=1)
                ref = np.array([0.0, 0.0, 1.0])
                hvs.append(hypervolume(res.objectives, ref))
            except Exception:
                pass
        out.append({"pop_size": pop, "n_gen": gen,
                    "mean_hv": float(np.mean(hvs)) if hvs else None,
                    "std_hv": float(np.std(hvs)) if hvs else None,
                    "n": len(hvs)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--bystander", default=None)
    ap.add_argument("--celltype", default="mES")
    ap.add_argument("--cfd-matrix", default=None)
    ap.add_argument("--cfd-pam", default=None)
    ap.add_argument("--clinvar", default=None)
    ap.add_argument("--no-genome", action="store_true")
    ap.add_argument("--offline-genome", action="store_true")
    ap.add_argument("--out-json", default="ablation_results.json")
    args = ap.parse_args()

    oracle, genome = _build_oracle(args)

    # build SNVs
    if args.clinvar:
        from data.clinvar import load_problems
        snvs = load_problems(args.clinvar, max_variants=args.n)
    else:
        from data.clinvar import synthetic_clinvar_problems, variant_to_snv
        snvs = [variant_to_snv(v) for v in synthetic_clinvar_problems(n=args.n, seed=0)]

    print(f"Ablation on {len(snvs)} loci "
          f"({'real BE-Hive' if args.bystander else 'surrogate'} oracle)\n")

    print("[1/2] Objective-dropping ablation (3-obj vs purity-dropped 2-obj) ...")
    od = objective_dropping(snvs, oracle, genome)
    a = od["aggregate"]
    print(f"   loci compared              : {a['n_loci']}")
    print(f"   mean best purity (3-obj)   : {a['mean_best_purity_full']}")
    print(f"   mean best purity (2-obj)   : {a['mean_best_purity_drop']}")
    print(f"   loci where 3-obj higher    : {a['loci_where_full_higher_purity']}/{a['n_loci']}")
    if "wilcoxon_p" in a:
        print(f"   Wilcoxon p (full vs drop)  : {a['wilcoxon_p']}")

    print("\n[2/2] Hyperparameter sensitivity (pop_size x n_gen) ...")
    hs = hyperparameter_sensitivity(snvs, oracle, genome)
    for s in hs:
        print(f"   pop={s['pop_size']:>2d} gen={s['n_gen']:>2d}: "
              f"mean HV={s['mean_hv']:.4f} (±{s['std_hv']:.4f}, n={s['n']})"
              if s['mean_hv'] is not None else
              f"   pop={s['pop_size']} gen={s['n_gen']}: no data")

    with open(args.out_json, "w") as fh:
        json.dump({"objective_dropping": od, "hyperparameter_sensitivity": hs}, fh, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
