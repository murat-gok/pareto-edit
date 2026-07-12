"""
B1 (option 4) — Train a dedicated efficiency model on BE-dataHIVE, no external download.

Replaces the biophysical efficiency predictor (Spearman 0.09 vs measured) with a model trained
directly on BE-dataHIVE's measured efficiencies, using a STRICT held-out split so the recovery
test is honest (reviewer #1 + the data-leakage concern, reviewer's Experimental Design point).

Design:
  - Features: k-mer (default k=3) spectrum of the protospacer + position-wise one-hot of a fixed
    window + editor one-hot. Purely sequence-derived; no measured label leaks into features.
  - Model: gradient-boosted trees (HistGradientBoostingRegressor) — fast, strong on tabular k-mer
    features, no GPU. Falls back to RandomForest if needed.
  - Split: GROUP split by protospacer so the same guide can't appear in both train and test
    (prevents the leakage the reviewer flagged). Reports test-set Spearman/Pearson.
  - Output: a pickled predictor exposing predict_efficiency(protospacer, editor)->[0,1], consumed
    by oracles/deep_efficiency.py at Layer-2 time and by efficiency_recovery_run.py.

Usage:
  python -m experiments.train_efficiency_model --be-datahive-file be_datahive_efficiency.pkl \\
      --cell HEK293T --out trained_efficiency.pkl
Then re-run recovery with the trained model (see REVISION_EXPERIMENTS.md).
"""
import argparse
import pickle
import numpy as np
import pandas as pd


NTS = "ACGT"


def kmer_features(seq, k=3):
    seq = (seq or "").upper()
    idx = {"".join(p): i for i, p in enumerate(_all_kmers(k))}
    v = np.zeros(len(idx), dtype=np.float32)
    if len(seq) >= k:
        for i in range(len(seq) - k + 1):
            km = seq[i:i + k]
            if km in idx:
                v[idx[km]] += 1.0
        v /= max(1, len(seq) - k + 1)
    return v


def _all_kmers(k):
    from itertools import product
    return list(product(NTS, repeat=k))


def onehot_window(seq, width=20):
    seq = (seq or "").upper()[:width].ljust(width, "A")
    m = np.zeros((width, 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = NTS.find(ch)
        if j >= 0:
            m[i, j] = 1.0
    return m.ravel()


def featurize(protospacers, editors, k=3):
    editor_levels = sorted(set(editors))
    ed_idx = {e: i for i, e in enumerate(editor_levels)}
    rows = []
    for s, e in zip(protospacers, editors):
        km = kmer_features(s, k)
        ow = onehot_window(s, 20)
        ed = np.zeros(len(editor_levels), dtype=np.float32)
        ed[ed_idx[e]] = 1.0
        rows.append(np.concatenate([km, ow, ed]))
    return np.vstack(rows), editor_levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--be-datahive-file", required=True)
    ap.add_argument("--cell", default="HEK293T")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="trained_efficiency.pkl")
    args = ap.parse_args()

    df = pd.read_pickle(args.be_datahive_file)
    df = df[df["efficiency_full_grna_calculated"].notna()].copy()
    if args.cell:
        df = df[df["cell"] == args.cell].copy()
    # protospacer: use 'grna' (20-nt) if present else derive from 'sequence'
    seq_col = "grna" if "grna" in df.columns else "sequence"
    df = df[df[seq_col].notna()].copy()
    df["_seq"] = df[seq_col].astype(str).str.upper()
    df["_eff"] = df["efficiency_full_grna_calculated"].astype(float)
    # normalize efficiency to [0,1] if it looks like a percentage
    if df["_eff"].max() > 1.5:
        df["_eff"] = df["_eff"] / 100.0
    df["_eff"] = df["_eff"].clip(0, 1)
    df["_editor"] = df["base_editor"].astype(str)

    print(f"Training rows: {len(df)}  | cell={args.cell}  | editors={sorted(df._editor.unique())}")

    # GROUP split by protospacer (no guide in both splits) -> leakage-free
    rng = np.random.default_rng(args.seed)
    groups = df["_seq"].values
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    n_test = int(len(uniq) * args.test_frac)
    test_groups = set(uniq[:n_test].tolist())
    is_test = df["_seq"].isin(test_groups).values
    print(f"Unique guides: {len(uniq)}  | test guides: {len(test_groups)}  "
          f"| train rows: {(~is_test).sum()}  test rows: {is_test.sum()}")

    X, editor_levels = featurize(df["_seq"].tolist(), df["_editor"].tolist(), k=args.k)
    y = df["_eff"].values

    Xtr, ytr = X[~is_test], y[~is_test]
    Xte, yte = X[is_test], y[is_test]

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                              max_depth=None, l2_regularization=1.0,
                                              random_state=args.seed)
        model_name = "HistGradientBoostingRegressor"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300, random_state=args.seed, n_jobs=-1)
        model_name = "RandomForestRegressor"

    model.fit(Xtr, ytr)
    pred_te = np.clip(model.predict(Xte), 0, 1)

    from scipy.stats import spearmanr, pearsonr
    sp = spearmanr(pred_te, yte).correlation
    pr = pearsonr(pred_te, yte)[0]
    mae = float(np.mean(np.abs(pred_te - yte)))
    print(f"\n=== Held-out efficiency model ({model_name}) ===")
    print(f"test Spearman : {sp:.3f}")
    print(f"test Pearson  : {pr:.3f}")
    print(f"test MAE      : {mae:.3f}")
    print(f"(compare against biophysical surrogate's guide-level Spearman 0.09)")

    payload = {
        "model": model, "editor_levels": editor_levels, "k": args.k,
        "cell": args.cell, "metrics": {"spearman": float(sp), "pearson": float(pr), "mae": mae},
        "feature_spec": "kmer(k)+onehot_window(20)+editor_onehot",
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nWrote {args.out}  (load in oracles/deep_efficiency.py)")


if __name__ == "__main__":
    main()
