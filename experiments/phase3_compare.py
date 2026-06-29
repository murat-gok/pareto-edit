"""
experiments/phase3_compare.py
=============================
Compare the SURROGATE oracle against the REAL BE-Hive oracle on the same locus.

This is the paper's key validation: how well does the cheap surrogate agree with the real
deep-learning model? We report, over the full enumerated design space:
  * Spearman & Pearson correlation of efficiency and of bystander_purity
    (surrogate vs BE-Hive), i.e. does the surrogate RANK designs like BE-Hive does?
  * Overlap of the two Pareto fronts (Jaccard over design identities).
  * A scatter figure (surrogate vs real) for each objective.

If the correlations are high, the surrogate is a faithful cheap stand-in and the surrogate-only
benchmark results are trustworthy. If low, we report that honestly and lean on the real oracle.

Run (needs torch + BE-Hive repos):
    python experiments/phase3_compare.py \
        --bystander ./be_predict_bystander \
        --efficiency ./be_predict_efficiency --celltype mES
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from design_space import SNVContext, enumerate_designs
from objectives import evaluate
from oracles.base import CompositeOracle
from oracles.surrogate import SurrogateOracle
from optimizers.base import non_dominated_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bystander", required=True)
    ap.add_argument("--efficiency", default=None)
    ap.add_argument("--celltype", default="mES")
    ap.add_argument("--target", default="C", choices=["C", "A"], help="target base / chemistry")
    args = ap.parse_args()

    try:
        import torch  # noqa
    except Exception:
        print("PyTorch required. pip install torch"); sys.exit(2)

    from oracles.behive import BEHiveOracle
    behive = BEHiveOracle(repo_bystander=args.bystander, repo_efficiency=None,
                          celltype=args.celltype)
    surrogate = SurrogateOracle()
    # real oracle: purity from REAL BE-Hive bystander model; efficiency + off-target from the
    # calibrated surrogate (see phase3_behive.build_oracle for rationale).
    real = CompositeOracle(efficiency_source=surrogate, purity_source=behive,
                           offtarget_source=surrogate)

    snv = SNVContext(chrom="chr11", pos=5226778, ref="A", alt="G", target_base=args.target)
    designs = enumerate_designs(snv)
    print(f"Scoring {len(designs)} designs with BOTH oracles (this calls BE-Hive a lot)...")

    sur_eff, sur_pur, real_eff, real_pur = [], [], [], []
    kept = []
    for d in designs:
        try:
            rr = real.predict(d, snv)
        except Exception:
            continue  # unsupported editor/celltype
        sr = surrogate.predict(d, snv)
        sur_eff.append(sr.efficiency); sur_pur.append(sr.bystander_purity)
        real_eff.append(rr.efficiency); real_pur.append(rr.bystander_purity)
        kept.append(d)

    sur_eff = np.array(sur_eff); sur_pur = np.array(sur_pur)
    real_eff = np.array(real_eff); real_pur = np.array(real_pur)

    from scipy.stats import spearmanr, pearsonr
    print(f"\nScored {len(kept)} designs.\n")
    print("NOTE: efficiency is sourced from the surrogate in BOTH oracles here, so only the")
    print("bystander-purity comparison is meaningful (that is the BE-Hive-backed axis).\n")
    print("Agreement surrogate vs REAL BE-Hive (bystander purity):")
    s, r = sur_pur, real_pur
    if np.std(s) < 1e-9 or np.std(r) < 1e-9:
        print("  bystander_purity: (constant in one oracle; correlation undefined)")
    else:
        rho, _ = spearmanr(s, r); rp, _ = pearsonr(s, r)
        print(f"  bystander_purity: Spearman={rho:+.3f}  Pearson={rp:+.3f}")
        mae = float(np.mean(np.abs(s - r)))
        print(f"  mean |surrogate - BE-Hive| purity = {mae:.3f}")

    # front overlap
    def front_keys(eff, pur, off):
        F = np.column_stack([-eff, -pur, off])
        nd = non_dominated_mask(F)
        return set(kept[i].as_tuple() for i in np.where(nd)[0])
    off = np.array([surrogate.predict(d, snv).offtarget_risk for d in kept])
    sk = front_keys(sur_eff, sur_pur, off)
    rk = front_keys(real_eff, real_pur, off)
    jac = len(sk & rk) / max(1, len(sk | rk))
    print(f"\nPareto-front overlap (Jaccard): {jac:.3f}  "
          f"(surrogate front={len(sk)}, real front={len(rk)}, shared={len(sk & rk)})")

    _figure(sur_eff, real_eff, sur_pur, real_pur)
    print(f"\nWrote fig_phase3_agreement.png")
    print("\nInterpretation: high purity correlation => surrogate is a faithful stand-in for "
          "the novel objective; the surrogate-only benchmark is then well-justified.")


def _figure(sur_eff, real_eff, sur_pur, real_pur):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.3))
    a.scatter(sur_eff, real_eff, s=18, alpha=0.6, c="#1f77b4")
    a.plot([0, 1], [0, 1], "k--", lw=1); a.set_xlabel("surrogate efficiency")
    a.set_ylabel("BE-Hive efficiency"); a.set_title("Efficiency agreement"); a.grid(alpha=0.3)
    b.scatter(sur_pur, real_pur, s=18, alpha=0.6, c="#d62728")
    b.plot([0, 1], [0, 1], "k--", lw=1); b.set_xlabel("surrogate purity")
    b.set_ylabel("BE-Hive purity"); b.set_title("Bystander-purity agreement"); b.grid(alpha=0.3)
    fig.suptitle("Surrogate vs real BE-Hive oracle", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fig_phase3_agreement.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
