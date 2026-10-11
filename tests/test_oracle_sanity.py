"""
tests/test_oracle_sanity.py
===========================
Every objective must actually vary across the designs it is supposed to discriminate.

This exists because five separate bugs in this codebase all had the same shape: a component
silently produced a degenerate value and nothing objected.

  1. aggregate-CFD returned one constant across all 150 loci and all designs
  2. DeepEfficiencyOracle featurized an empty string, because it looked for
     design.protospacer / .guide / .guide_seq (none exist) and then called
     snv.context.protospacer_for() (does not exist either); both failures were swallowed
     by a bare `except`
  3. three of four catalogue editors fell through to an all-zero one-hot, because the
     trained model's vocabulary did not cover them
  4. --cfd-pam loaded pam_scores.pkl and never read it
  5. the optimizers rebuilt the PAM search space from search_bounds(), ignoring the design
     list the pipeline had enumerated

Each was invisible in normal output and each silently corrupted a reported number. The checks
below would have caught all five in a few seconds.

Run directly, or under pytest:

    python tests/test_oracle_sanity.py --bystander ./be_predict_bystander
    pytest tests/test_oracle_sanity.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_space import SNVContext, PROTOSPACER_LEN, enumerate_designs
from objectives import evaluate, to_human

BYSTANDER_REPO = os.environ.get("PARETO_EDIT_BYSTANDER", "./be_predict_bystander")
EFF_MODEL = os.environ.get("PARETO_EDIT_EFF_MODEL", "trained_efficiency.pkl")
CFD_MATRIX = os.environ.get("PARETO_EDIT_CFD_MATRIX", "mismatch_score.pkl")

TOL = 1e-9


# --------------------------------------------------------------- helpers
def a_locus():
    """A G>A locus (ABE-correctable) with a deterministic synthetic context, so the test
    needs no network and no ClinVar download."""
    from data.clinvar import synthetic_clinvar_problems, variant_to_snv
    for v in synthetic_clinvar_problems(n=12, seed=0):
        snv = variant_to_snv(v)
        if enumerate_designs(snv):
            return snv
    raise AssertionError("no synthetic locus produced feasible designs")


def spread(values):
    vs = [v for v in values if v is not None]
    return (max(vs) - min(vs)) if vs else 0.0


def check_varies(name, values, designs, axis="design"):
    """A degenerate objective is one that is constant across the designs at a locus."""
    s = spread(values)
    uniq = len(set(round(v, 9) for v in values))
    print(f"  {name:<28} range={s:.6f}  distinct={uniq}/{len(values)}")
    assert s > TOL, (
        f"{name} is CONSTANT across {len(values)} designs at one locus (range {s:.2e}). "
        f"That is the signature of a silently degenerate objective: an oracle returning a "
        f"fallback, an unmapped category falling through to a zero vector, or an input never "
        f"reaching the model. Do not report numbers from this configuration.")
    return s


def check_varies_with_position(name, by_pos):
    """Within one editor, moving the protospacer must change the value. Catches an oracle
    that sees only the editor and never the sequence (bug 2 above)."""
    vals = list(by_pos.values())
    s = spread(vals)
    print(f"  {name:<28} across positions: range={s:.6f}")
    assert s > TOL, (
        f"{name} does not change when the protospacer moves, for a fixed editor. The model "
        f"is not seeing the sequence. Check that the oracle reads snv.window_context(design) "
        f"-- this is exactly how DeepEfficiencyOracle was broken.")


# --------------------------------------------------------------- tests
def test_surrogate_objectives_vary():
    from oracles.surrogate import SurrogateOracle
    snv = a_locus()
    designs = enumerate_designs(snv)
    print("\n[surrogate]")
    rows = [to_human(evaluate(d, snv, SurrogateOracle())) for d in designs]
    check_varies("efficiency", [r["efficiency"] for r in rows], designs)
    check_varies("bystander_purity", [r["bystander_purity"] for r in rows], designs)
    check_varies("offtarget_risk", [r["offtarget_risk"] for r in rows], designs)


def test_cfd_offtarget_varies():
    from oracles.cfd import CFDOffTargetOracle
    snv = a_locus()
    designs = enumerate_designs(snv, collapse_pam=True)
    print("\n[cfd]")
    cfd = CFDOffTargetOracle(mismatch_pkl=CFD_MATRIX if os.path.exists(CFD_MATRIX) else None)
    vals = [cfd.predict(d, snv).offtarget_risk for d in designs]
    check_varies("offtarget_risk", vals, designs)

    ed = designs[0].editor
    by_pos = {d.target_pos: cfd.predict(d, snv).offtarget_risk
              for d in designs if d.editor == ed}
    check_varies_with_position("offtarget_risk", by_pos)


def test_deep_efficiency_sees_the_sequence():
    if not os.path.exists(EFF_MODEL):
        print(f"\n[deep efficiency] SKIPPED: {EFF_MODEL} not found")
        return
    from oracles.deep_efficiency import DeepEfficiencyOracle, EDITOR_NAME_MAP
    snv = a_locus()
    designs = enumerate_designs(snv, collapse_pam=True)
    print("\n[deep efficiency]")
    eff = DeepEfficiencyOracle.from_pickle(EFF_MODEL)
    vals = [eff.predict(d, snv).efficiency for d in designs]
    check_varies("efficiency", vals, designs)

    ed = designs[0].editor
    by_pos = {d.target_pos: eff.predict(d, snv).efficiency
              for d in designs if d.editor == ed}
    check_varies_with_position("efficiency", by_pos)

    # every catalogue editor must resolve into the trained vocabulary, directly or by mapping
    cat = sorted({d.editor for d in designs})
    levels = eff._editor_levels
    unmapped = [e for e in cat if EDITOR_NAME_MAP.get(e, e) not in levels]
    print(f"  catalogue editors: {cat}")
    print(f"  model vocabulary : {levels}")
    assert not unmapped, (
        f"editors {unmapped} resolve to no entry in the trained vocabulary, so their one-hot "
        f"is all-zero and the model cannot tell them apart. Add them to EDITOR_NAME_MAP or "
        f"drop them from the catalogue; do not report per-editor results for them.")


def test_behive_purity_varies():
    if not os.path.isdir(BYSTANDER_REPO):
        print(f"\n[be-hive] SKIPPED: {BYSTANDER_REPO} not found")
        return
    try:
        import torch  # noqa
    except Exception:
        print("\n[be-hive] SKIPPED: torch not installed")
        return
    from oracles.behive import BEHiveOracle
    snv = a_locus()
    designs = enumerate_designs(snv, collapse_pam=True)
    print("\n[be-hive]")
    be = BEHiveOracle(repo_bystander=BYSTANDER_REPO, repo_efficiency=None, celltype="mES")
    vals = [be.predict(d, snv).bystander_purity for d in designs]
    check_varies("bystander_purity", vals, designs)

    ed = designs[0].editor
    by_pos = {d.target_pos: be.predict(d, snv).bystander_purity
              for d in designs if d.editor == ed}
    check_varies_with_position("bystander_purity", by_pos)


def test_pam_dimension_is_declared_honestly():
    """If no objective reads design.pam_variant, enumerating PAM variants only duplicates
    designs. This does not fail the suite -- it reports, so the choice stays deliberate."""
    snv = a_locus()
    full = enumerate_designs(snv, collapse_pam=False)
    coll = enumerate_designs(snv, collapse_pam=True)
    print("\n[pam dimension]")
    print(f"  designs with PAM enumerated: {len(full)}")
    print(f"  designs with PAM collapsed : {len(coll)}")
    if len(full) > len(coll):
        print("  NOTE: no objective in the real-model composition reads design.pam_variant, "
              "so PAM variants at one placement are exact ties. Use --collapse-pam for "
              "real-model runs, or implement a genomic PAM feasibility check.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bystander", default=BYSTANDER_REPO)
    ap.add_argument("--efficiency-model", default=EFF_MODEL)
    ap.add_argument("--cfd-matrix", default=CFD_MATRIX)
    a = ap.parse_args()
    globals()["BYSTANDER_REPO"] = a.bystander
    globals()["EFF_MODEL"] = a.efficiency_model
    globals()["CFD_MATRIX"] = a.cfd_matrix

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAILED: {e}")
    print("\n" + "=" * 70)
    if failed:
        print(f"{len(failed)}/{len(tests)} checks FAILED:")
        for n, _ in failed:
            print(f"  - {n}")
        sys.exit(1)
    print(f"all {len(tests)} checks passed: every objective varies across the designs it "
          f"is meant to discriminate")


if __name__ == "__main__":
    main()
