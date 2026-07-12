"""
B2 — Measured-outcome purity validation (reviewer #2), using Arbab-2020 genotype-level data.

Input: the per-editor .pkl files from figshare (10.6084/m9.figshare.10678097), each a dict:
    { guide_name : DataFrame of measured outcome genotypes }
where guide_name encodes the target edit position ("...editpos6"), each DataFrame row is one
observed genotype with a signed-position substrate column per window position (e.g. C6, C5, C-1;
the letter is the wildtype base, the number the protospacer position), plus a Frequency column.

MEASURED PURITY per guide (BE-Hive's precision definition, computed from measured frequencies):
    purity = sum(freq of genotypes where the TARGET substrate is edited AND no other substrate
                 position in the window is edited)
             / sum(freq of genotypes where the TARGET substrate is edited)
For a CBE (C->T) the substrate base is C; for an ABE (A->G) it is A. The target column is the
substrate column at the edit position parsed from the guide name.

VALIDATION (the measured analogue of the predicted-purity ablation):
    We pool guides across editor files that target the SAME sequence context, giving, per target,
    several (editor, measured_purity, measured_efficiency) candidates — exactly the choice
    PARETO-EDIT faces. We compare the MEASURED purity achieved by
      - the efficiency-first choice   (max measured efficiency)   vs
      - the purity-aware choice        (max efficiency*measured_purity, a balanced selector)
    and report the measured-purity gain, Wilcoxon, and Cliff's delta. If the purity-aware choice
    achieves higher MEASURED purity, optimizing purity (as PARETO-EDIT does) selects guides that
    are measured — not merely predicted — to be cleaner.

Usage (point at the folder of .pkl files):
  python -m experiments.purity_recovery_run --pkl-dir . --cell HEK293T --out-json purity_recovery.json
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd


def parse_pos_cols(cols):
    out = {}
    for c in cols:
        m = re.fullmatch(r'([ACGT])(-?\d+)', str(c))
        if m:
            out[c] = (m.group(1), int(m.group(2)))
    return out


def edit_position_from_name(name):
    m = re.search(r'editpos(-?\d+)', str(name))
    return int(m.group(1)) if m else None


def substrate_base_for_editor(fname):
    # CBE family edits C; ABE family edits A. Infer from filename.
    f = fname.lower()
    if "ctot" in f or "ctoga" in f or "ctog" in f or "be4" in f or "cda" in f or "a3a" in f or "apobec" in f or "aid" in f:
        return "C"
    if "atog" in f or "abe" in f:
        return "A"
    return "C"


def freq_col(df):
    for c in ["Frequency", "Y"]:
        if c in df.columns:
            return c
    # fall back to any column literally named Frequency*
    for c in df.columns:
        if str(c).startswith("Frequency"):
            return c
    return None


def measured_purity_for_guide(df, guide_name, substrate):
    pos_cols = parse_pos_cols(df.columns)
    tpos = edit_position_from_name(guide_name)
    if tpos is None:
        return None, None
    sub_cols = {c: (wt, p) for c, (wt, p) in pos_cols.items() if wt == substrate}
    target_col = next((c for c, (wt, p) in sub_cols.items() if p == tpos), None)
    if target_col is None:
        return None, None
    other = [c for c in sub_cols if c != target_col]
    fc = freq_col(df)
    if fc is None:
        return None, None
    freq = pd.to_numeric(df[fc], errors="coerce").fillna(0.0)
    target_edited = df[target_col] != substrate
    no_bystander = (df[other] == substrate).all(axis=1) if other else pd.Series(True, index=df.index)
    denom = freq[target_edited].sum()
    if denom <= 0:
        return None, float(denom)
    numer = freq[target_edited & no_bystander].sum()
    return float(numer / denom), float(denom)  # purity, and total edited freq (~efficiency proxy)


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((x > y) for x in a for y in b); lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir", required=True, help="folder containing the per-editor .pkl files")
    ap.add_argument("--cell", default=None, help="only use files whose name contains this (e.g. HEK293T)")
    ap.add_argument("--glob", default="*.pkl")
    ap.add_argument("--min-candidates", type=int, default=2, help="min editors per target to compare")
    ap.add_argument("--out-json", default="purity_recovery.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.pkl_dir, args.glob)))
    if args.cell:
        files = [f for f in files if args.cell.lower() in os.path.basename(f).lower()]
    if not files:
        print("No .pkl files matched."); return
    print(f"Using {len(files)} editor files:")
    for f in files:
        print("  ", os.path.basename(f))

    # Build per-(target context, editor) measured purity + efficiency.
    # We key targets by the guide's core name minus the editor tag, so the SAME target across
    # editor files groups together. The guide names look like "<id>_<EDITOR>_...editpos<k>".
    records = {}  # target_key -> list of dicts(editor, purity, eff)
    for f in files:
        editor = os.path.basename(f).replace(".pkl", "")
        substrate = substrate_base_for_editor(editor)
        try:
            d = pd.read_pickle(f)
        except Exception as e:
            print(f"  ! could not read {f}: {e}"); continue
        if not isinstance(d, dict):
            print(f"  ! {f} is not a dict of guides; skipping"); continue
        for gname, gdf in d.items():
            if not hasattr(gdf, "columns"):
                continue
            purity, edited_freq = measured_purity_for_guide(gdf, gname, substrate)
            if purity is None:
                continue
            # target key: strip a leading numeric id and the editor token; keep the structural tail
            key = re.sub(r'^\d+_', '', str(gname))
            key = key.replace(editor, "").replace("__", "_")
            records.setdefault(key, []).append(
                {"editor": editor, "purity": purity, "eff": edited_freq})

    # Compare, per target, purity-aware vs efficiency-first choice on MEASURED purity.
    mp_purity_choice, mp_eff_choice = [], []
    n_groups = 0
    rows = []
    for key, cands in records.items():
        if len(cands) < args.min_candidates:
            continue
        n_groups += 1
        eff_first = max(cands, key=lambda c: c["eff"])
        purity_aware = max(cands, key=lambda c: c["eff"] * c["purity"])
        mp_purity_choice.append(purity_aware["purity"])
        mp_eff_choice.append(eff_first["purity"])
        rows.append({
            "target": key[:50],
            "n_candidates": len(cands),
            "measured_purity_purity_choice": round(purity_aware["purity"], 4),
            "measured_purity_eff_choice": round(eff_first["purity"], 4),
            "gain": round(purity_aware["purity"] - eff_first["purity"], 4),
            "purity_choice_editor": purity_aware["editor"],
            "eff_choice_editor": eff_first["editor"],
        })

    a = np.array(mp_purity_choice); b = np.array(mp_eff_choice)
    from scipy.stats import wilcoxon
    try:
        w_p = wilcoxon(a, b).pvalue if np.any(a != b) else float("nan")
    except Exception:
        w_p = float("nan")
    delta = cliffs_delta(a, b)

    print("\n=== Measured-purity validation (reviewer #2) ===")
    print(f"editor files                        : {len(files)}")
    print(f"guides with computable measured purity across all files : "
          f"{sum(len(v) for v in records.values())}")
    print(f"target contexts with >= {args.min_candidates} editor candidates : {n_groups}")
    if n_groups:
        print(f"mean MEASURED purity, purity-aware choice : {np.mean(a):.3f}")
        print(f"mean MEASURED purity, efficiency-first     : {np.mean(b):.3f}")
        print(f"mean gain (purity-aware - eff-first)       : {np.mean(a-b):.3f}")
        print(f"targets where purity-aware is cleaner      : {int((a>b).sum())}/{n_groups} "
              f"({100*(a>b).mean():.0f}%)")
        print(f"Wilcoxon p                                 : {w_p:.2e}")
        print(f"Cliff's delta                              : {delta:+.3f}")
    print("\nAlso reporting the raw measured-purity distribution (single-editor view):")
    all_pur = [c["purity"] for v in records.values() for c in v]
    if all_pur:
        print(f"  n guides={len(all_pur)}  mean={np.mean(all_pur):.3f}  "
              f"median={np.median(all_pur):.3f}  frac>=0.5={np.mean(np.array(all_pur)>=0.5):.2f}")

    summary = {
        "n_editor_files": len(files),
        "n_guides_with_purity": sum(len(v) for v in records.values()),
        "n_groups": n_groups,
        "mean_measured_purity_purity_choice": float(np.mean(a)) if n_groups else None,
        "mean_measured_purity_eff_choice": float(np.mean(b)) if n_groups else None,
        "mean_gain": float(np.mean(a - b)) if n_groups else None,
        "frac_purity_choice_cleaner": float((a > b).mean()) if n_groups else None,
        "wilcoxon_p": float(w_p),
        "cliffs_delta": float(delta),
        "guide_purity_mean": float(np.mean(all_pur)) if all_pur else None,
        "guide_purity_median": float(np.median(all_pur)) if all_pur else None,
        "groups": rows[:1000],
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
