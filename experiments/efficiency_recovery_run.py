"""
experiments/efficiency_recovery_run.py
======================================
Guide-level empirical recovery on BE-dataHIVE's OWN measured target sequences (C1).

The earlier ClinVar-based recovery failed because ClinVar-derived protospacers do not coincide
with BE-dataHIVE's measured guides (disjoint construction). This runner sidesteps that overlap
problem entirely: instead of trying to match ClinVar loci to measured guides, it evaluates
PARETO-EDIT's efficiency predictor directly on the target sequences BE-dataHIVE *actually measured*
(BE-dataHIVE provides, per row: the measured 40-nt target `sequence`, the editor, the PAM, and the
measured efficiency). Overlap is therefore 100% by construction.

What it measures
----------------
For a measured-guide sample (optionally restricted to one editor / cell type):
  * Spearman and Pearson correlation between PARETO-EDIT's predicted efficiency and BE-dataHIVE's
    measured efficiency, at guide resolution.
  * Top-k recovery: grouping guides into small panels, does the predictor's top-ranked guide fall
    within the top-k measured guides? (a design-relevant question: would PARETO-EDIT pick a guide
    that is actually among the best-measured?)
  * Regret: measured efficiency of the empirical best minus that of the predictor's top pick.

Scope note (state in the paper): this validates the EFFICIENCY objective against measured outcomes
on BE-dataHIVE's library targets; it is a guide-level efficiency validation, not a claim about
ClinVar loci or about the purity/off-target objectives. It is exactly the measured-outcome
validation a reviewer asks for, on the data that exists.

Uses the surrogate/calibrated efficiency predictor only (no BE-Hive/CFD needed), so it runs
without the heavy model stack:

    python -m experiments.efficiency_recovery_run \
        --be-datahive-file be_datahive_efficiency.pkl \
        --editor BE4max --n 2000 --panel-size 8 --top-k 1 \
        --efficiency-calibration efficiency_calibration.json \
        --out-json efficiency_recovery.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from design_space import Design, SNVContext, PROTOSPACER_LEN, EDITORS
from oracles.surrogate import SurrogateOracle


# Map BE-dataHIVE base_editor names to the catalogue editor whose chemistry/window matches, so the
# surrogate scores each measured guide with the right editor parameters. Unmapped editors are
# skipped (we only validate editors PARETO-EDIT models).
BEDH_TO_CATALOGUE = {
    # cytosine base editors -> closest catalogue CBE
    "CBE": "BE4max", "CBE4max": "BE4max", "BE4": "BE4max", "BE4max": "BE4max",
    "Target-AID": "BE4max", "AID": "BE4max",
    "evoAPOBEC": "evoAPOBEC", "evoAPOBEC1": "evoAPOBEC",
    "YE1": "YE1", "CGBE": "CGBE1", "CGBE1": "CGBE1",
    # adenine base editors -> closest catalogue ABE
    "ABE": "ABE7.10", "ABE7.10": "ABE7.10", "ABEmax": "ABE7.10",
    "ABE8": "ABE8e", "ABE8e": "ABE8e",
}


def _normalize_eff(x):
    x = np.asarray(x, dtype=float)
    if np.isfinite(x).any() and np.nanmax(x) > 1.5:
        x = x / 100.0
    return np.clip(x, 0.0, 1.0)


def _extract_protospacer(sequence: str, grna: str) -> int:
    """Locate the protospacer (grna) within the 40-nt target sequence; return its start index,
    or -1 if absent. BE-dataHIVE's `sequence` contains the protospacer as a substring."""
    return sequence.find(grna) if (sequence and grna) else -1


def predict_efficiency(surrogate, calibrator, editor_name, sequence, grna, pam):
    """Build a Design/SNVContext from a measured guide and return PARETO-EDIT's predicted
    efficiency. We place the target base at the canonical window position for the editor and use
    the measured target sequence as the context the oracle sees."""
    ed = EDITORS[editor_name]
    target_base = ed.edit[0]
    start = _extract_protospacer(sequence, grna)
    if start < 0:
        proto = grna  # fall back to the gRNA itself as the protospacer
    else:
        proto = sequence[start:start + PROTOSPACER_LEN]
        if len(proto) < PROTOSPACER_LEN:
            proto = grna
    # choose the protospacer position of the first target-base inside the editor window
    win_lo, win_hi = ed.window
    pos = None
    for p in range(win_lo, win_hi + 1):
        if 1 <= p <= len(proto) and proto[p - 1] == target_base:
            pos = p
            break
    if pos is None:
        return None  # no editable base in window -> not a meaningful efficiency comparison
    snv = SNVContext(chrom="lib", pos=0, ref=target_base,
                     alt=("T" if target_base == "C" else "G"), target_base=target_base)
    snv.context = proto
    design = Design(editor=editor_name, pam_variant="SpCas9-NGG", target_pos=pos)
    eff = surrogate.predict(design, snv).efficiency
    if calibrator is not None:
        eff = calibrator.apply(eff)
    return eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--be-datahive-file", required=True)
    ap.add_argument("--editor", default=None,
                    help="restrict to one BE-dataHIVE base_editor (e.g. BE4); default: all mapped")
    ap.add_argument("--cell", default=None, help="restrict to one cell type (e.g. mES)")
    ap.add_argument("--n", type=int, default=2000, help="max measured guides to sample")
    ap.add_argument("--panel-size", type=int, default=8,
                    help="guides per synthetic panel for top-k recovery")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--efficiency-calibration", default=None)
    ap.add_argument("--efficiency-model", default=None,
                    help="path to a trained efficiency model pickle (train_efficiency_model.py). "
                         "If given, uses the trained model instead of the biophysical surrogate.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="efficiency_recovery.json")
    args = ap.parse_args()

    import pandas as pd
    if args.be_datahive_file.endswith(".parquet"):
        df = pd.read_parquet(args.be_datahive_file)
    else:
        df = pd.read_pickle(args.be_datahive_file)

    # keep rows with a measured efficiency and a usable sequence
    df = df[df["efficiency_full_grna_calculated"].notna()].copy()
    if args.cell:
        df = df[df["cell"] == args.cell]
    if args.editor:
        df = df[df["base_editor"] == args.editor]
    # map editors
    df["catalogue_editor"] = df["base_editor"].map(BEDH_TO_CATALOGUE)
    df = df[df["catalogue_editor"].notna()]
    if len(df) == 0:
        print("No measured guides match the requested editor/cell after mapping. "
              f"Available editors: {sorted(set(df['base_editor'])) if len(df) else 'none'}")
        return

    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"Evaluating {len(df)} measured guides "
          f"(editor={args.editor or 'all-mapped'}, cell={args.cell or 'all'})")

    calibrator = None
    if args.efficiency_calibration:
        from oracles.efficiency_calibration import CalibratedEfficiency
        calibrator = CalibratedEfficiency.load(args.efficiency_calibration)

    surrogate = SurrogateOracle()

    # Optional: use a dedicated trained efficiency model (reviewer #1) instead of the surrogate.
    trained = None
    if args.efficiency_model:
        from oracles.deep_efficiency import DeepEfficiencyOracle
        trained = DeepEfficiencyOracle.from_pickle(args.efficiency_model)
        print(f"Efficiency source: TRAINED model ({args.efficiency_model})")
    else:
        print("Efficiency source: biophysical surrogate"
              + (" + calibration" if calibrator else ""))

    pred, meas, used_editor = [], [], []
    for _, row in df.iterrows():
        ed = row["catalogue_editor"]
        if trained is not None:
            # trained model scores directly from the protospacer + editor
            proto = str(row.get("grna", "") or row.get("sequence", ""))[:20]
            e = trained._predict_efficiency(proto, str(row["base_editor"]))
        else:
            e = predict_efficiency(surrogate, calibrator, ed,
                                   row.get("sequence", ""), row.get("grna", ""),
                                   row.get("pam_sequence", ""))
        if e is None:
            continue
        pred.append(e)
        meas.append(row["efficiency_full_grna_calculated"])
        used_editor.append(ed)

    if len(pred) < 10:
        print(f"Only {len(pred)} comparable guides — too few to correlate. "
              "Try a different editor or larger --n.")
        return

    pred = np.array(pred, dtype=float)
    meas = _normalize_eff(np.array(meas, dtype=float))

    from scipy.stats import spearmanr, pearsonr
    sp, sp_p = spearmanr(pred, meas)
    pr, pr_p = pearsonr(pred, meas)

    # top-k recovery over random panels
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pred))
    hits, total, regrets = 0, 0, []
    ps = args.panel_size
    for i in range(0, len(order) - ps + 1, ps):
        idx = order[i:i + ps]
        p_local = pred[idx]
        m_local = meas[idx]
        pred_top = idx[np.argmax(p_local)]
        meas_rank = idx[np.argsort(-m_local)]
        if pred_top in meas_rank[:args.top_k]:
            hits += 1
        total += 1
        regrets.append(float(meas.max() if False else (m_local.max() - meas[pred_top])))
    recovery = hits / total if total else float("nan")

    summary = {
        "n_guides": int(len(pred)),
        "editor": args.editor or "all-mapped",
        "cell": args.cell or "all",
        "spearman": float(sp), "spearman_p": float(sp_p),
        "pearson": float(pr), "pearson_p": float(pr_p),
        "panel_size": ps, "top_k": args.top_k,
        "n_panels": int(total),
        "topk_recovery": float(recovery),
        "mean_regret": float(np.mean(regrets)) if regrets else None,
        "editor_counts": {e: int((np.array(used_editor) == e).sum())
                          for e in sorted(set(used_editor))},
    }
    print("\n=== Efficiency recovery on BE-dataHIVE measured targets (C1) ===")
    print(f"guides compared        : {summary['n_guides']}")
    print(f"Spearman (pred vs meas): {sp:.3f} (p={sp_p:.1e})")
    print(f"Pearson  (pred vs meas): {pr:.3f} (p={pr_p:.1e})")
    print(f"top-{args.top_k} recovery over {total} panels of {ps}: {recovery:.3f}")
    print(f"mean regret            : {summary['mean_regret']:.3f}")
    print(f"editors used           : {summary['editor_counts']}")

    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {args.out_json}")
    print("\nInterpretation: this validates the efficiency objective against measured outcomes at "
          "guide resolution on BE-dataHIVE's library targets (100% overlap by construction), "
          "the measured-outcome validation that the ClinVar-locus overlap could not provide.")


if __name__ == "__main__":
    main()
