"""
B1 — Dedicated efficiency oracle (drop-in for CompositeOracle).

Wraps a trained sequence-based efficiency model (from experiments/train_efficiency_model.py) OR any
external callable model (DeepBE / CRISPRon-BE / BE-DICT efficiency head) behind the standard Oracle
interface, so it substitutes for the biophysical efficiency source without touching the optimizer.

Two ways to construct:
  1) From a trained pickle:
        eff = DeepEfficiencyOracle.from_pickle("trained_efficiency.pkl")
  2) From any callable f(protospacer:str, editor:str) -> float in [0,1]:
        eff = DeepEfficiencyOracle(predict_fn=my_model_fn)

Only the `efficiency` field of the returned OracleReading is meaningful; purity/off-target are left
at 0.0 because CompositeOracle takes efficiency from this source and the other objectives from their
own oracles.
"""
import pickle
import numpy as np

from oracles.base import Oracle, OracleReading


# Catalogue editor name -> the name used in the BE-dataHIVE training data.
# The trained model's vocabulary is ['ABE','ABE8e','ABEmax','CBE','CBE4max','Target-AID'];
# the design catalogue uses ['ABE8e','ABE7.10','BE4max','evoAPOBEC']. Only ABE8e matches, so
# without a mapping three of the four editors get an all-zero one-hot and the model cannot
# tell them apart.
#
# [[AUTHOR: this mapping is a scientific judgement, so it is left empty rather than guessed.
#  Fill it in and justify each pair in the Methods, e.g. whether ABE7.10 is best represented
#  by 'ABE' or 'ABEmax', and which training editor (if any) stands for evoAPOBEC. If no
#  defensible mapping exists for an editor, say so and treat its efficiency as unmodelled
#  rather than silently feeding the model a zero vector.]]
EDITOR_NAME_MAP = {
    # Confident: ABE7.10 is the original adenine base editor, which the generic "ABE"
    # label covers; "ABEmax" is an optimized derivative and a poorer match.
    "ABE7.10":   "ABE",
    # Confident: BE4max and CBE4max are the same editor under two naming conventions.
    "BE4max":    "CBE4max",
    # PROXY, and it must be disclosed. evoAPOBEC uses an evolved APOBEC1 deaminase;
    # the training vocabulary has no equivalent ("Target-AID" is CDA1-based, a different
    # deaminase family), so the generic cytosine-editor label is the nearest defensible
    # stand-in. This mirrors the purity-side proxying already recorded in
    # BEHIVE_EDITOR_IS_PROXY, and should be reported the same way.
    # [[AUTHOR: confirm or replace; if you would rather not proxy it, drop evoAPOBEC from
    #  the direct-model catalogue instead of feeding the model a zero vector.]]
    "evoAPOBEC": "CBE",
}


class DeepEfficiencyOracle(Oracle):
    def __init__(self, predict_fn=None, model_payload=None):
        """
        predict_fn   : callable(protospacer:str, editor:str) -> float in [0,1]
        model_payload: dict from train_efficiency_model.py (mutually exclusive with predict_fn)
        """
        if predict_fn is None and model_payload is None:
            raise ValueError("Provide predict_fn or model_payload")
        self._fn = predict_fn
        self._payload = model_payload
        self._warned_editors = set()
        if model_payload is not None:
            self._model = model_payload["model"]
            self._editor_levels = model_payload["editor_levels"]
            self._k = model_payload.get("k", 3)
        else:
            self._editor_levels = []

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(model_payload=payload)

    # --- featurization must match the trainer ---
    def _featurize_one(self, protospacer, editor):
        from experiments.train_efficiency_model import kmer_features, onehot_window
        km = kmer_features(protospacer, self._k)
        ow = onehot_window(protospacer, 20)
        ed = np.zeros(len(self._editor_levels), dtype=np.float32)
        if editor in self._editor_levels:
            ed[self._editor_levels.index(editor)] = 1.0
        # if editor unseen, leave all-zero (model sees an out-of-catalogue editor)
        return np.concatenate([km, ow, ed]).reshape(1, -1)

    def _predict_efficiency(self, protospacer, editor):
        if self._fn is not None:
            try:
                return float(np.clip(self._fn(protospacer, editor), 0, 1))
            except Exception:
                return 0.0
        X = self._featurize_one(protospacer, editor)
        return float(np.clip(self._model.predict(X)[0], 0, 1))

    def predict(self, design, snv) -> OracleReading:
        # The design-placed 20-nt protospacer, via the SAME accessor the surrogate and
        # BE-Hive oracles use (oracles/surrogate.py:52, oracles/behive.py:233). The previous
        # implementation looked for design.protospacer / .guide / .guide_seq -- none of which
        # exist on Design -- then called snv.context.protospacer_for(), which does not exist
        # either because snv.context is a plain string. Both failures were swallowed and the
        # model was handed an empty string, so its prediction depended only on the editor
        # one-hot and was constant within every locus. No exception is caught here: if the
        # accessor ever breaks again it must be loud.
        protospacer = snv.window_context(design)

        editor = getattr(design, "editor", None) or getattr(design, "editor_name", "")
        editor = EDITOR_NAME_MAP.get(str(editor), str(editor))

        if editor not in self._editor_levels and editor not in self._warned_editors:
            self._warned_editors.add(editor)
            import warnings
            warnings.warn(
                f"DeepEfficiencyOracle: editor {editor!r} is not in the trained model's "
                f"vocabulary {self._editor_levels}. Its one-hot is all-zero, so predictions "
                f"for it are not editor-specific. Add a mapping to EDITOR_NAME_MAP in "
                f"oracles/deep_efficiency.py, or retrain on data covering this editor.",
                RuntimeWarning, stacklevel=2)

        eff = self._predict_efficiency(protospacer, editor)
        return OracleReading(efficiency=eff, bystander_purity=0.0, offtarget_risk=0.0)
