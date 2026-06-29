"""
experiments/make_tables.py
==========================
Generate paper-ready LaTeX tables from the benchmark CSV outputs.
Produces:
  table_main.tex      -- HV / IGD / spacing / evals / recovery per method x regime (mean +/- std)
  table_wilcoxon.tex  -- Wilcoxon signed-rank summary (NSGA-II vs others)
"""

from __future__ import annotations

import os
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
METHODS = ["NSGA-II", "CPO-GWO", "WeightedSum", "Random"]
PRETTY = {"NSGA-II": "NSGA-II", "CPO-GWO": "CPO--GWO",
          "WeightedSum": "Weighted-sum", "Random": "Random search"}


def main():
    df = pd.read_csv(os.path.join(OUT, "benchmark_raw.csv"))

    # ---- main table: aggregate across problems within each regime ---- #
    rows = []
    for regime in ["small", "scaled"]:
        sub = df[df.regime == regime]
        for m in METHODS:
            s = sub[sub.method == m]
            rows.append({
                "regime": regime, "method": PRETTY[m],
                "HV": f"{s.hv.mean():.4f} $\\pm$ {s.hv.std():.4f}",
                "IGD": f"{s.igd.mean():.4f} $\\pm$ {s.igd.std():.4f}" if s.igd.notna().any() else "--",
                "Spacing": f"{s.spacing.mean():.4f}",
                "Oracle calls": f"{s.n_evals.mean():.0f}",
                "Recovery": f"{s.front_recovery.mean():.3f}" if s.front_recovery.notna().any() else "--",
            })
    main_df = pd.DataFrame(rows)

    with open(os.path.join(OUT, "table_main.tex"), "w") as f:
        f.write("% PARETO-EDIT main benchmark table (auto-generated)\n")
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Multi-objective benchmark over 6 SNV instances $\\times$ 30 seeds. "
                "\\emph{Small} regime: exact Pareto front enumerable (recovery = fraction of "
                "true front found). \\emph{Scaled} regime: enlarged design space ($\\sim$1296 "
                "designs) under a capped evaluation budget; IGD computed against a proxy "
                "reference front. HV: higher is better; IGD/Spacing: lower is better; oracle "
                "calls: lower is cheaper.}\n")
        f.write("\\label{tab:benchmark}\n")
        f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        f.write("Regime & Method & HV $\\uparrow$ & IGD $\\downarrow$ & Spacing $\\downarrow$ "
                "& Oracle calls $\\downarrow$ & Recovery $\\uparrow$ \\\\\n\\midrule\n")
        prev = None
        for _, r in main_df.iterrows():
            reg = r["regime"] if r["regime"] != prev else ""
            prev = r["regime"]
            f.write(f"{reg} & {r['method']} & {r['HV']} & {r['IGD']} & {r['Spacing']} "
                    f"& {r['Oracle calls']} & {r['Recovery']} \\\\\n")
            if r["method"] == PRETTY["Random"]:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # ---- wilcoxon table ---- #
    w = pd.read_csv(os.path.join(OUT, "benchmark_wilcoxon.csv"))
    wr = []
    for metric in ["hv", "igd"]:
        sub = w[w.metric == metric]
        for method in ["CPO-GWO", "WeightedSum", "Random"]:
            mm = sub[sub.method == method]
            wr.append({
                "Metric": metric.upper(),
                "NSGA-II vs": PRETTY[method],
                "Sig. (p<0.05)": f"{(mm.p_value < 0.05).sum()}/{len(mm)}",
                "Mean median diff.": f"{mm.median_diff.mean():+.4f}",
            })
    wdf = pd.DataFrame(wr)
    with open(os.path.join(OUT, "table_wilcoxon.tex"), "w") as f:
        f.write("% PARETO-EDIT Wilcoxon table (auto-generated)\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\caption{Wilcoxon signed-rank tests across seeds, NSGA-II vs.\\ each "
                "competitor, aggregated over all problem/regime cases (12 total). For HV a "
                "positive median difference favours NSGA-II; for IGD a negative one does.}\n")
        f.write("\\label{tab:wilcoxon}\n\\begin{tabular}{llcc}\n\\toprule\n")
        f.write("Metric & NSGA-II vs & Significant & Mean median diff. \\\\\n\\midrule\n")
        for _, r in wdf.iterrows():
            f.write(f"{r['Metric']} & {r['NSGA-II vs']} & {r['Sig. (p<0.05)']} "
                    f"& {r['Mean median diff.']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print("Wrote table_main.tex and table_wilcoxon.tex")
    print("\n--- main table preview ---")
    print(main_df.to_string(index=False))
    print("\n--- wilcoxon preview ---")
    print(wdf.to_string(index=False))


if __name__ == "__main__":
    main()
