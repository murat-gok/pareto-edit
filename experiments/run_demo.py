"""
experiments/run_demo.py
=======================
Reproducible PARETO-EDIT demonstration.

1. Defines an example pathogenic-SNV instance.
2. Runs NSGA-II and CPO-GWO through the pipeline (surrogate oracle).
3. Compares against baselines (weighted-sum, random search) and the exhaustive ground truth.
4. Reports quality indicators (HV, IGD, spacing) and prints the Pareto front of designs.
5. Saves figures: 3D objective scatter + 2D projections, and HV/archive convergence.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from design_space import SNVContext
from oracles.surrogate import SurrogateOracle
from optimizers.nsga2 import NSGA2
from optimizers.cpo_gwo import CPO_GWO
from pipeline import ParetoEditPipeline
from metrics import (
    exhaustive_front,
    hypervolume,
    igd,
    spacing,
    baseline_weighted_sum,
    baseline_random_search,
)
from objectives import to_human

OUT = os.path.dirname(os.path.abspath(__file__))
REF = np.array([0.0, 0.0, 1.2])  # hypervolume reference (worse than any achievable objective)


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    # Example instance: a C>T-correctable pathogenic SNV (CBE chemistry, target base C).
    snv = SNVContext(chrom="chr11", pos=5226778, ref="A", alt="G", target_base="C")

    orc = SurrogateOracle()

    banner("GROUND TRUTH (exhaustive enumeration)")
    gt_d, gt_F = exhaustive_front(snv, orc)
    hv_gt = hypervolume(gt_F, REF, seed=1)
    print(f"Exhaustive Pareto-front size : {len(gt_F)}")
    print(f"Exhaustive hypervolume       : {hv_gt:.4f}")

    results = {}
    banner("OPTIMIZERS")
    for Opt in (NSGA2, CPO_GWO):
        pipe = ParetoEditPipeline(oracle=orc, optimizer=Opt(pop_size=40, n_gen=30))
        res = pipe.run(snv, seed=0)
        hv = hypervolume(res.objectives, REF, seed=1)
        ig = igd(res.objectives, gt_F)
        sp = spacing(res.objectives)
        results[res.algo] = res
        print(f"{res.algo:9s} | front={len(res.objectives):3d} | uniq oracle calls={res.n_evals:3d} "
              f"| HV={hv:.4f} ({100*hv/hv_gt:5.1f}% of GT) | IGD={ig:.4f} | spacing={sp:.4f}")

    banner("BASELINES")
    ws_d, ws_F = baseline_weighted_sum(snv, orc)
    print(f"Weighted-sum (single pick)   : HV={hypervolume(ws_F, REF, seed=1):.4f}  "
          f"-> {to_human(ws_F[0])}")
    budget = max(r.n_evals for r in results.values())
    rs_d, rs_F = baseline_random_search(snv, orc, n_evals=budget, seed=0)
    print(f"Random search (same budget)  : front={len(rs_F)}  HV={hypervolume(rs_F, REF, seed=1):.4f}")

    banner("PARETO FRONT  (NSGA-II)  -- editor / PAM / placement -> outcomes")
    pipe = ParetoEditPipeline(oracle=orc, optimizer=NSGA2())
    rows = pipe.report(results["NSGA-II"], snv, top=12)
    hdr = f"{'editor':10s} {'PAM':12s} {'pos':>3s} | {'eff':>5s} {'purity':>6s} {'offtgt':>6s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['editor']:10s} {r['pam_variant']:12s} {r['target_pos']:>3d} | "
              f"{r['efficiency']:.3f} {r['bystander_purity']:.3f} {r['offtarget_risk']:.3f}")

    # ----------------------------- figures -------------------------------- #
    _figure_front(results, gt_F, rs_F)
    _figure_convergence(results)
    print(f"\nFigures written to {OUT}/fig_front.png and {OUT}/fig_convergence.png")


def _figure_front(results, gt_F, rs_F):
    fig = plt.figure(figsize=(13, 4.2))

    # 3D
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(-gt_F[:, 0], -gt_F[:, 1], gt_F[:, 2], c="lightgray", s=60, label="ground truth", depthshade=False)
    colors = {"NSGA-II": "#1f77b4", "CPO-GWO": "#d62728"}
    for algo, res in results.items():
        F = res.objectives
        ax.scatter(-F[:, 0], -F[:, 1], F[:, 2], s=22, c=colors.get(algo), label=algo, depthshade=False)
    ax.set_xlabel("efficiency"); ax.set_ylabel("bystander purity"); ax.set_zlabel("off-target risk")
    ax.set_title("Pareto front (3D)"); ax.legend(loc="upper left", fontsize=7)

    # eff vs purity
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(-rs_F[:, 0], -rs_F[:, 1], c="0.8", s=18, label="random")
    for algo, res in results.items():
        F = res.objectives
        ax2.scatter(-F[:, 0], -F[:, 1], s=22, c=colors.get(algo), label=algo)
    ax2.set_xlabel("efficiency"); ax2.set_ylabel("bystander purity")
    ax2.set_title("Efficiency vs purity"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # eff vs offtarget
    ax3 = fig.add_subplot(1, 3, 3)
    for algo, res in results.items():
        F = res.objectives
        ax3.scatter(-F[:, 0], F[:, 2], s=22, c=colors.get(algo), label=algo)
    ax3.set_xlabel("efficiency"); ax3.set_ylabel("off-target risk")
    ax3.set_title("Efficiency vs off-target"); ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    fig.suptitle("PARETO-EDIT: non-dominated editing designs (surrogate oracle)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_front.png"), dpi=150)
    plt.close(fig)


def _figure_convergence(results):
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, res in results.items():
        ax.plot(range(1, len(res.history) + 1), res.history, marker="o", ms=3, label=algo)
    ax.set_xlabel("generation"); ax.set_ylabel("non-dominated set size")
    ax.set_title("Archive / front growth per generation"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_convergence.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
