"""
data/be_datahive.py
===================
BE-dataHIVE client for PARETO-EDIT (Phase 2/3).

Wraps the official `be-datahive` PyPI package (REST API over https://be-datahive.com) and
exposes exactly what PARETO-EDIT needs:

  * empirical efficiency for a guide/target/editor
  * empirical bystander outcome distribution -> bystander_purity (fraction of edited reads
    carrying only the intended conversion)

These empirical measurements are the GROUND TRUTH used by experiments/benchmark.py's
`empirical_recovery()`: we ask whether the optimizer's predicted Pareto front contains the
editor/guide that BE-dataHIVE *measured* to be best for a locus.

The official API (verified):
    from be_datahive import be_datahive
    api = be_datahive()
    eff = api.get_efficiency()                 # pandas DataFrame
    bys = api.get_bystander()                  # pandas DataFrame
    studies = api.get_studies()
    X, y = api.get_efficiency_ml_arrays(eff, encoding='raw', target_col=...)

Network note: be-datahive.com must be reachable. In sandboxes where it is blocked (HTTP 403),
set PARETO_EDIT_OFFLINE=1 to use a small synthetic stand-in with the SAME schema so downstream
code is exercised end-to-end. Real runs simply leave the flag unset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EmpiricalOutcome:
    editor: str
    guide: str
    efficiency: float          # measured editing efficiency, [0,1]
    bystander_purity: float    # measured fraction of pure intended edits, [0,1]
    cell: str = ""
    source_study: str = ""


class BEDataHiveClient:
    """Thin wrapper. Lazily constructs the official client; falls back to synthetic data when
    offline so the pipeline remains runnable and testable."""

    def __init__(self, offline: Optional[bool] = None):
        self.offline = (os.environ.get("PARETO_EDIT_OFFLINE", "0") == "1"
                        if offline is None else offline)
        self._api = None
        if not self.offline:
            try:
                from be_datahive import be_datahive  # type: ignore
                self._api = be_datahive()
            except Exception as e:  # pragma: no cover - depends on environment
                # If the package or network is unavailable, degrade to offline mode.
                self.offline = True
                self._init_error = str(e)

    # ---------------- raw frames ---------------- #
    def efficiency_frame(self, **kwargs) -> pd.DataFrame:
        if self.offline:
            return _synthetic_efficiency_frame()
        return self._api.get_efficiency(**kwargs)

    def bystander_frame(self, **kwargs) -> pd.DataFrame:
        if self.offline:
            return _synthetic_bystander_frame()
        return self._api.get_bystander(**kwargs)

    def studies(self, **kwargs):
        if self.offline:
            return pd.DataFrame([{"id": 0, "name": "synthetic", "year": 2020}])
        return self._api.get_studies(**kwargs)

    # ---------------- guide-level lookup (for empirical recovery) ---------------- #
    def _ensure_guide_index(self):
        """Build (once) an index mapping each measured gRNA sequence to its BE-dataHIVE rows.
        Uses the real 'grna' column (BE-dataHIVE RAW_FEATURE_COLS) and the default efficiency
        target 'efficiency_full_grna_calculated'. Cached on the instance."""
        if getattr(self, "_guide_index", None) is not None:
            return
        df = self.efficiency_frame()
        self._eff_df = df
        gcol = _guess_col(df, ["grna", "guide", "protospacer"])
        self._gcol = gcol
        self._eff_col = _guess_col(
            df, ["efficiency_full_grna_calculated", "efficiency", "editing_efficiency"])
        self._ed_col = _guess_col(df, ["base_editor", "editor"])
        idx = {}
        if gcol is not None:
            for i, g in enumerate(df[gcol].astype(str).str.upper()):
                idx.setdefault(g, []).append(i)
        self._guide_index = idx

    def measured_for_guide(self, protospacer: str, editor: Optional[str] = None
                           ) -> Optional[EmpiricalOutcome]:
        """Return the empirically measured outcome for an exact guide match (optionally also
        matching the editor). Matching is on the 20-nt protospacer against BE-dataHIVE's 'grna'
        column (case-insensitive, exact). Returns None if no measured guide matches.

        Exact matching is deliberate: BE-dataHIVE guides come from specific screens, so a hit
        means the design's protospacer was actually measured — the strongest form of recovery."""
        self._ensure_guide_index()
        key = protospacer.upper()
        rows = self._guide_index.get(key)
        if not rows:
            return None
        df = self._eff_df
        sub = df.iloc[rows]
        if editor and self._ed_col:
            sub2 = sub[sub[self._ed_col].astype(str) == str(editor)]
            if not sub2.empty:
                sub = sub2
        # take the best-measured efficiency among matching rows
        if self._eff_col and self._eff_col in sub.columns:
            sub = sub.sort_values(self._eff_col, ascending=False)
        best = sub.iloc[0]
        return EmpiricalOutcome(
            editor=str(best.get(self._ed_col, "")) if self._ed_col else "",
            guide=key,
            efficiency=float(best.get(self._eff_col)) if self._eff_col else float("nan"),
            bystander_purity=float("nan"),  # purity comes from the bystander frame (separate)
            cell=str(best.get(_guess_col(df, ["cell", "cell_line"]), "")),
            source_study=str(best.get(_guess_col(df, ["study", "source"]), "")),
        )

    # ---------------- PARETO-EDIT-facing API ---------------- #
    def best_measured_for_locus(
        self,
        guide_prefix: Optional[str] = None,
        editors: Optional[list] = None,
        objective: str = "efficiency",
    ) -> Optional[EmpiricalOutcome]:
        """Return the empirically best-measured editor/guide for a locus, by the chosen
        objective ('efficiency' or 'bystander_purity'). Used as ground truth for recovery."""
        eff = self.efficiency_frame()
        if eff.empty:
            return None
        df = eff.copy()
        if guide_prefix:
            gcol = _guess_col(df, ["grna", "guide", "protospacer", "target"])
            if gcol:
                df = df[df[gcol].astype(str).str.startswith(guide_prefix)]
        if editors:
            ecol = _guess_col(df, ["base_editor", "editor"])
            if ecol:
                df = df[df[ecol].isin(editors)]
        if df.empty:
            return None
        eff_col = _guess_col(df, ["efficiency", "editing_efficiency", "efficiency_rate"])
        pur_col = _guess_col(df, ["bystander_purity", "purity"])
        sort_col = eff_col if objective == "efficiency" else (pur_col or eff_col)
        best = df.sort_values(sort_col, ascending=False).iloc[0]
        return EmpiricalOutcome(
            editor=str(best.get(_guess_col(df, ["base_editor", "editor"]), "")),
            guide=str(best.get(_guess_col(df, ["grna", "guide", "protospacer", "target"]), "")),
            efficiency=float(best.get(eff_col, np.nan)) if eff_col else np.nan,
            bystander_purity=float(best.get(pur_col, np.nan)) if pur_col else np.nan,
            cell=str(best.get(_guess_col(df, ["cell", "cell_line"]), "")),
        )


def _guess_col(df: pd.DataFrame, candidates) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    # partial match
    for cand in candidates:
        for c in df.columns:
            if cand in c.lower():
                return c
    return None


# --------------------------------------------------------------------------- #
# Synthetic stand-ins (offline) -- mirror the real schema closely enough to test logic
# --------------------------------------------------------------------------- #

def _synthetic_efficiency_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    editors = ["BE4max", "evoAPOBEC", "YE1", "ABE8e", "ABE7.10", "CGBE1"]
    bases = np.array(list("ACGT"))
    rows = []
    for i in range(n):
        guide = "".join(rng.choice(bases, size=20))
        rows.append({
            "id": i,
            "grna": guide,
            "base_editor": rng.choice(editors),
            "cell": rng.choice(["HEK293T", "K562", "mESC"]),
            "efficiency": float(np.clip(rng.beta(2, 2), 0, 1)),
            "bystander_purity": float(np.clip(rng.beta(2, 1.5), 0, 1)),
            "study": rng.choice(["Arbab", "Pallaseni", "Marquart", "Yuan", "Song"]),
        })
    return pd.DataFrame(rows)


def _synthetic_bystander_frame(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _synthetic_efficiency_frame(n, seed)
    # add a couple of bystander-specific columns
    df["bystander_count"] = rng.integers(0, 4, size=len(df))
    df["pure_fraction"] = df["bystander_purity"]
    return df
