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
        if model_payload is not None:
            self._model = model_payload["model"]
            self._editor_levels = model_payload["editor_levels"]
            self._k = model_payload.get("k", 3)

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
        # derive the protospacer for this design from the SNV context + placement,
        # mirroring how the surrogate/behive oracles read the guide.
        protospacer = None
        for attr in ("protospacer", "guide", "guide_seq"):
            protospacer = getattr(design, attr, None)
            if protospacer:
                break
        if protospacer is None and getattr(snv, "context", None) is not None:
            # fall back to window_context extraction if the design carries a position
            try:
                protospacer = snv.context.protospacer_for(design)
            except Exception:
                protospacer = None
        editor = getattr(design, "editor", None) or getattr(design, "editor_name", "")
        eff = self._predict_efficiency(protospacer or "", str(editor))
        return OracleReading(efficiency=eff, bystander_purity=0.0, offtarget_risk=0.0)
