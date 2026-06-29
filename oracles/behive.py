"""
oracles/behive.py
=================
Real BE-Hive oracle adapter (Phase 3).

Wraps Max Shen's BE-Hive models:
  * be_predict_bystander  (https://github.com/maxwshen/be_predict_bystander)
  * be_predict_efficiency (https://github.com/maxwshen/be_predict_efficiency)

into PARETO-EDIT's Oracle interface, so NSGA-II / CPO-GWO optimize against REAL predicted
editing outcomes instead of the surrogate. The optimizer code is unchanged.

Verified BE-Hive interface (from the repos):
    import sys; sys.path.append('/path/to/be_predict_bystander')
    from be_predict_bystander import predict as bystander_model
    bystander_model.init_model(base_editor='BE4', celltype='mES')
    pred_df, stats = bystander_model.predict(seq)         # seq is 50-nt
    pred_df = bystander_model.add_genotype_column(pred_df, stats)
    # pred_df: rows = genotypes, col 'Predicted frequency', col 'Genotype'
    # stats['Assumed protospacer sequence'] = seq[20:40]

Sequence convention (critical):
    - 50-nt window; protospacer occupies seq[20:40] (positions 1..20 w.r.t. gRNA).
    - Internally BE-Hive uses p0idx = 19 so that gRNA position p maps to seq index 19 + p.
    - Editing-window substrate columns are defined per editor in editor_profiles.csv.

How PARETO-EDIT quantities are derived from BE-Hive:
    efficiency        : from be_predict_efficiency (a scalar base-editing efficiency / fraction
                        of edited reads). If only the bystander model is available, we proxy
                        efficiency by (1 - predicted frequency of the wild-type/unedited
                        genotype), i.e. total edited probability mass.
    bystander_purity  : from the bystander genotype distribution -> the summed predicted
                        frequency of genotypes in which ONLY the intended target position is
                        converted (all other substrate positions retain their reference base),
                        divided by the total edited probability mass.  [the novel objective]
    offtarget_risk    : BE-Hive does NOT model off-target. We keep this objective sourced from a
                        dedicated off-target oracle (crispAI / CFD); see CompositeOracle below.

Editor-name mapping: PARETO-EDIT design_space names -> BE-Hive 'Public base editor' names.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from design_space import EDITORS, PROTOSPACER_LEN, Design, SNVContext
from oracles.base import Oracle, OracleReading

# Map our catalogue editor keys to BE-Hive public base-editor names (models.csv).
# NOTE: not every (editor, celltype) combination exists in BE-Hive. The adapter checks the
# real models.csv at init and skips designs whose editor is unavailable in the chosen celltype
# (rather than crashing). Mappings below are the nearest shipped editor; proxies are documented
# and must be stated as a limitation in the paper.
#
# Verified against models.csv:
#   mES      ships: ABE, ABE-CP1040, ABE8, BE4, BE4-CP1028, AID, CDA, eA3A, evoAPOBEC,
#                   BE4-H47ES48A, eA3A-T31A, eA3A-T31AT44A, and CG-* CGBEs (CG-689, CG-APOBEC1, ...)
#   HEK293T  ships: ABE, ABE-CP1040, BE4, BE4-CP1028, AID, CDA, eA3A, evoAPOBEC,
#                   eA3A-T44DS45A, BE4-H47ES48A, CG-eA3A
# Whether each catalogue editor is backed by a direct/near-exact BE-Hive model or by a documented
# proxy (a different editor used as a stand-in because the exact one is not shipped). This lets the
# analysis separate primary results (direct-model editors) from proxy-mapped editors, which a
# reviewer rightly flags as approximate.
#   direct  : exact or same-family closest variant (BE4max->BE4, evoAPOBEC, ABE8e->ABE8, ABE7.10->ABE)
#   proxy   : no model for this editor; a different editor stands in (YE1->BE4, CGBE1->a CGBE)
BEHIVE_EDITOR_IS_PROXY = {
    "BE4max":    False,
    "evoAPOBEC": False,
    "ABE8e":     False,
    "ABE7.10":   False,
    "YE1":       True,    # represented by BE4 (a non-narrowed CBE) — approximate
    "CGBE1":     True,    # represented by a CGBE that is not CGBE1 itself — approximate
}


def editor_is_proxy(design_editor: str) -> bool:
    """True if this catalogue editor is served by a proxy BE-Hive model (approximate)."""
    return BEHIVE_EDITOR_IS_PROXY.get(design_editor, False)


BEHIVE_EDITOR_MAP = {
    "BE4max":    "BE4",        # BE4 is the shipped CBE closest to BE4max
    "evoAPOBEC": "evoAPOBEC",  # exact
    "YE1":       "BE4",        # YE1 not in BE-Hive; nearest narrow-window CBE proxy (documented)
    "ABE8e":     "ABE8",       # ABE8 (mES) is the closest to ABE8e; falls back to ABE elsewhere
    "ABE7.10":   "ABE",        # base ABE
    "CGBE1":     "CG-eA3A",    # CGBE; only in HEK293T -> see celltype-specific fallback below
}

# Celltype-specific fallbacks for editors whose primary mapping is absent in that celltype.
# Looked up only when the primary BEHIVE_EDITOR_MAP name is not available for the chosen cell.
BEHIVE_CELLTYPE_FALLBACK = {
    # editor key -> {celltype -> available BE-Hive name}
    "ABE8e":  {"HEK293T": "ABE", "HEK293": "ABE"},   # ABE8 is mES-only
    "CGBE1":  {"mES": "CG-689"},                       # CG-eA3A is HEK293T-only; use a mES CGBE
}

P0IDX = 19          # BE-Hive internal: gRNA position p -> seq index 19 + p
SEQ_LEN = 50
PROTO_START = 20    # protospacer occupies seq[20:40]


class UnsupportedDesign(Exception):
    """Raised when a design's editor has no BE-Hive model in the chosen celltype."""


def _install_sklearn_legacy_shim():
    """BE-Hive's efficiency models were pickled with scikit-learn < 0.22, where several
    submodules lived at old paths (e.g. sklearn.ensemble.gradient_boosting) that were renamed
    with a leading underscore in 0.22+ (sklearn.ensemble._gb). Map the old import paths to the
    new modules so the legacy pickles load on modern scikit-learn. No effect if already present.
    """
    import importlib

    # (old_dotted_path -> new_module_name). Covers the modules GradientBoosting pickles touch.
    aliases = {
        "sklearn.ensemble.gradient_boosting": "sklearn.ensemble._gb",
        "sklearn.ensemble.forest": "sklearn.ensemble._forest",
        "sklearn.ensemble.base": "sklearn.ensemble._base",
        "sklearn.ensemble.weight_boosting": "sklearn.ensemble._weight_boosting",
        "sklearn.tree.tree": "sklearn.tree._tree",
        "sklearn.tree._tree": "sklearn.tree._tree",
        "sklearn.tree": "sklearn.tree",
        "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor": "sklearn.ensemble._gb",
    }
    for old, new in aliases.items():
        if old in sys.modules:
            continue
        try:
            sys.modules[old] = importlib.import_module(new)
        except Exception:
            # If a particular new module doesn't exist on this sklearn version, skip it;
            # the pickle may not need it.
            pass


import contextlib
import io


@contextlib.contextmanager
def _suppress_stdout():
    """Silence BE-Hive's verbose per-call prints ('Model successfully initialized', etc.)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


class BEHiveOracle(Oracle):
    """Real BE-Hive-backed oracle for efficiency + bystander purity.

    off-target is NOT provided here (BE-Hive doesn't model it); compose with an off-target
    oracle via CompositeOracle. If `repo_bystander` / `repo_efficiency` paths are not importable
    (e.g. torch missing, repos not cloned), construction raises -- callers should catch and fall
    back to the surrogate, or use offline=True for a structural dry-run.
    """

    name = "be-hive"

    def __init__(self, repo_bystander: str, repo_efficiency: Optional[str] = None,
                 celltype: str = "mES", offline: bool = False):
        self.celltype = celltype
        self.offline = offline
        self._bystander = None
        self._efficiency = None
        self._cur_editor = None
        self._supported = set()   # set of (public_editor, celltype) available in BE-Hive
        if offline:
            return
        # import the real packages
        sys.path.append(repo_bystander)
        from be_predict_bystander import predict as bystander_model  # type: ignore
        self._bystander = bystander_model
        if repo_efficiency:
            sys.path.append(repo_efficiency)
            _install_sklearn_legacy_shim()  # let old (sklearn<0.22) pickles load
            try:
                from be_predict_efficiency import predict as efficiency_model  # type: ignore
                self._efficiency = efficiency_model
            except Exception:
                self._efficiency = None
        # load the real list of supported (editor, celltype) combos from models.csv
        self._load_supported(repo_bystander)

    def _load_supported(self, repo_bystander: str):
        """Read models.csv so we only ask BE-Hive for combinations it actually ships."""
        import csv
        path = os.path.join(repo_bystander, "models.csv")
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    ed = row.get("Public base editor", "").strip()
                    ct = row.get("Celltype", "").strip()
                    if ed and ct:
                        self._supported.add((ed, ct))
        except FileNotFoundError:
            # If unreadable, leave empty -> resolver will just trust the static map.
            pass

    def resolve_editor(self, design_editor: str) -> Optional[str]:
        """Map our editor name to a BE-Hive public editor available in the current celltype.
        Returns None if no supported mapping exists (caller should skip the design)."""
        primary = BEHIVE_EDITOR_MAP.get(design_editor)
        # if we have the supported set, honor it; otherwise trust the static map
        def ok(name):
            return (not self._supported) or ((name, self.celltype) in self._supported)
        if primary and ok(primary):
            return primary
        # try celltype-specific fallback
        fb = BEHIVE_CELLTYPE_FALLBACK.get(design_editor, {}).get(self.celltype)
        if fb and ok(fb):
            return fb
        return None

    # ----------------------------------------------------------------- #
    def _ensure_editor(self, behive_editor: str):
        """(Re)initialize BE-Hive models when the editor changes (BE-Hive uses global state)."""
        if self._cur_editor == behive_editor:
            return
        with _suppress_stdout():
            self._bystander.init_model(base_editor=behive_editor, celltype=self.celltype)
            if self._efficiency is not None:
                self._efficiency.init_model(base_editor=behive_editor, celltype=self.celltype)
        self._cur_editor = behive_editor

    def _build_50nt(self, design: Design, snv: SNVContext) -> str:
        """Construct BE-Hive's 50-nt input so the target base sits at gRNA position
        design.target_pos (i.e. seq index P0IDX + target_pos)."""
        # Use the design-placed 20-nt protospacer from the same context model used elsewhere,
        # then pad to 50 nt with the SNV's genomic flanks (or deterministic synthetic flanks).
        proto = snv.window_context(design)  # length PROTOSPACER_LEN, target base at target_pos
        # left flank 19 nt + protospacer(20) + right flank up to 50
        import hashlib
        h = hashlib.sha256(f"{snv.chrom}:{snv.pos}:flank".encode()).hexdigest()
        bases = "ACGT"
        filler = [bases[int(c, 16) % 4] for c in (h * 2)]
        left = "".join(filler[:PROTO_START])               # 20 nt, index 0..19
        right = "".join(filler[PROTO_START:PROTO_START + (SEQ_LEN - PROTO_START - PROTOSPACER_LEN)])
        seq = (left + proto + right)
        seq = (seq + "A" * SEQ_LEN)[:SEQ_LEN]
        return seq.upper()

    def _purity_from_pred(self, pred_df, stats, target_seq_index: int, target_base: str) -> tuple:
        """Compute (edited_mass, purity) from BE-Hive's genotype distribution.

        edited_mass = total predicted frequency of genotypes differing from the reference
                      anywhere in the substrate window (i.e. 1 - WT mass).
        purity      = (mass of genotypes where ONLY target_seq_index changed, and to the
                      intended product) / edited_mass.
        """
        pred_df = self._bystander.add_genotype_column(pred_df, stats)
        ref_seq = stats["50-nt target sequence"]
        freq_col = "Predicted frequency"

        wt_mass = 0.0
        edited_mass = 0.0
        pure_mass = 0.0
        for _, row in pred_df.iterrows():
            geno = row["Genotype"]
            f = float(row[freq_col])
            diffs = [i for i in range(len(geno)) if geno[i] != ref_seq[i]]
            if not diffs:
                wt_mass += f
                continue
            edited_mass += f
            # "pure" = the only changed position is the target, and it changed (any product)
            if diffs == [target_seq_index]:
                pure_mass += f
        purity = (pure_mass / edited_mass) if edited_mass > 0 else 0.0
        return edited_mass, purity

    # ----------------------------------------------------------------- #
    def predict(self, design: Design, snv: SNVContext) -> OracleReading:
        if self.offline:
            raise RuntimeError("BEHiveOracle(offline=True) is a structural placeholder; it "
                               "cannot predict. Use the surrogate oracle for offline runs.")
        editor = EDITORS[design.editor]
        behive_editor = self.resolve_editor(design.editor)
        if behive_editor is None:
            raise UnsupportedDesign(
                f"{design.editor} has no BE-Hive model in celltype '{self.celltype}'")
        self._ensure_editor(behive_editor)

        seq = self._build_50nt(design, snv)
        target_seq_index = P0IDX + design.target_pos  # where the target base lives in the 50-nt

        with _suppress_stdout():
            pred_df, stats = self._bystander.predict(seq)
        edited_mass, purity = self._purity_from_pred(pred_df, stats, target_seq_index,
                                                     editor.edit[1])

        # efficiency: prefer the dedicated model, else proxy by edited probability mass
        if self._efficiency is not None:
            try:
                with _suppress_stdout():
                    eff_out = self._efficiency.predict(seq)
                # be_predict_efficiency.predict returns a scalar (or (scalar, stats)); normalize.
                if isinstance(eff_out, tuple):
                    eff_val = eff_out[0]
                else:
                    eff_val = eff_out
                eff_val = float(eff_val)
                # Some BE-Hive efficiency models output a logit; squash if clearly out of [0,1].
                if eff_val < 0.0 or eff_val > 1.0:
                    from math import exp
                    eff_val = 1.0 / (1.0 + exp(-eff_val))
                efficiency = float(min(max(eff_val, 0.0), 1.0))
            except Exception:
                efficiency = float(min(max(edited_mass, 0.0), 1.0))
        else:
            efficiency = float(min(max(edited_mass, 0.0), 1.0))

        return OracleReading(
            efficiency=efficiency,
            bystander_purity=float(min(max(purity, 0.0), 1.0)),
            offtarget_risk=0.0,  # not modeled by BE-Hive; supply via CompositeOracle
            meta={"behive_editor": behive_editor, "edited_mass": edited_mass,
                  "celltype": self.celltype},
        )
