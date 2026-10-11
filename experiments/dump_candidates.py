"""
experiments/dump_candidates.py
==============================
Dump the FULL enumerated candidate set per locus, with all three objectives scored,
before any dominance filtering. Consumed by d6_compare.py, and it also supplies the
efficiency-purity correlation the manuscript currently lacks.

Mirrors layer2_application.py's oracle construction exactly, so the numbers are
comparable with layer2_full_direct.json. One deliberate difference:

    --efficiency-model PATH   uses DeepEfficiencyOracle (the sequence-trained model)

layer2_application.py builds the Layer-2 oracle with efficiency_source=surrogate;
DeepEfficiencyOracle appears only in efficiency_recovery_run.py. So the 150-locus
results were produced with the surrogate, not the sequence-trained model the
manuscript names. Run both ways and compare.

CRASH SAFETY
------------
This run can take hours, so it never holds results only in memory. Each locus is
appended to a JSON Lines checkpoint and flushed to disk before the next locus starts.
A power cut loses at most the locus in flight.

  * Re-run the same command to resume. Completed loci are skipped; the oracle is not
    called again for them.
  * A half-written final line from a hard cut is detected and discarded on resume.
  * The consolidated .json is written through a temporary file and renamed, so it is
    never left truncated.
  * --no-resume starts over, --consolidate-only rebuilds the .json from the
    checkpoint without touching the oracle.

d6_compare.py reads the .jsonl directly, so you can analyse a partial run while the
rest is still going.

    python experiments/dump_candidates.py \
        --bystander ./be_predict_bystander --celltype mES \
        --clinvar variant_summary.txt.gz --n 150 --direct-only \
        --cfd-matrix mismatch_score.pkl \
        --out candidates.json

Start with --n 5: this scores EVERY candidate, not just the ~85 the optimizer visits,
so expect roughly an order of magnitude more BE-Hive calls than layer2_application.py.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_space import SNVContext, PROTOSPACER_LEN, enumerate_designs
from objectives import evaluate, to_human
from oracles.base import CompositeOracle
from oracles.surrogate import SurrogateOracle
from data.genome import GenomeContext


# --------------------------------------------------------------- crash safety
_STOP = {"flag": False}


def _request_stop(signum, frame):
    """Ctrl-C finishes the locus in flight, then exits cleanly."""
    if _STOP["flag"]:
        print("\nSecond interrupt - exiting now.")
        sys.exit(130)
    _STOP["flag"] = True
    print("\nInterrupt received. Finishing the current locus, then stopping. "
          "Re-run the same command to resume.")


def append_record(fh, rec):
    """Append one locus and force it to disk before returning."""
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass          # some filesystems/network drives do not support fsync


def read_checkpoint(path):
    """Return (records_by_locus, n_dropped). Tolerates a truncated last line."""
    recs, dropped = {}, 0
    if not os.path.exists(path):
        return recs, dropped
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1          # the cut landed mid-write; that locus is redone
                continue
            if isinstance(r, dict) and r.get("locus"):
                recs[r["locus"]] = r
    return recs, dropped


def write_atomic(path, obj):
    """Write JSON through a temp file and rename, so `path` is never half-written."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, path)           # atomic on Windows and POSIX


def consolidate(ckpt_path, out_path):
    recs, dropped = read_checkpoint(ckpt_path)
    out = {}
    for lab, r in recs.items():
        d = {k: v for k, v in r.items() if k != "locus"}
        out[lab] = d
    write_atomic(out_path, out)
    sizes = [len(v.get("candidates", [])) for v in out.values()]
    print(f"Consolidated {len(out)} loci, {sum(sizes)} candidates -> {out_path}"
          + (f"  ({dropped} incomplete line(s) ignored)" if dropped else ""))
    return out


# --------------------------------------------------------------- main
def attach_real_context(snv: SNVContext, genome: GenomeContext) -> bool:
    """Same as layer2_application.attach_real_context."""
    window = genome.fetch_window(snv.chrom, snv.pos,
                                 ref=snv.ref if snv.ref in "ACGT" else None)
    if window is None:
        return False
    snv.context = window
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bystander", required=True)
    ap.add_argument("--celltype", default="mES")
    ap.add_argument("--clinvar", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--offline-genome", action="store_true")
    ap.add_argument("--cfd-matrix", default=None)
    ap.add_argument("--cfd-pam", default=None)
    ap.add_argument("--direct-only", action="store_true")
    ap.add_argument("--aggregate-offtarget", action="store_true")
    ap.add_argument("--ot-max-mismatches", type=int, default=2)
    ap.add_argument("--collapse-pam", action="store_true",
                    help="one design per (editor, position); see "
                         "design_space.enumerate_designs")
    ap.add_argument("--efficiency-calibration", default=None)
    ap.add_argument("--efficiency-model", default=None,
                    help="trained sequence model pickle; efficiency then comes from "
                         "DeepEfficiencyOracle instead of the surrogate")
    ap.add_argument("--out", default="candidates.json")
    ap.add_argument("--checkpoint", default=None,
                    help="JSON Lines checkpoint (default: <out> with .jsonl)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore and overwrite any existing checkpoint")
    ap.add_argument("--consolidate-only", action="store_true",
                    help="rebuild the .json from the checkpoint and exit")
    args = ap.parse_args()

    ckpt = args.checkpoint or (os.path.splitext(args.out)[0] + ".jsonl")

    if args.consolidate_only:
        consolidate(ckpt, args.out)
        return

    done, dropped = ({}, 0) if args.no_resume else read_checkpoint(ckpt)
    if args.no_resume and os.path.exists(ckpt):
        os.replace(ckpt, ckpt + ".bak")
        print(f"--no-resume: previous checkpoint moved to {ckpt}.bak")
    if done:
        print(f"Resuming: {len(done)} loci already in {ckpt}"
              + (f"; {dropped} incomplete line(s) will be redone" if dropped else ""))

    try:
        import torch  # noqa
    except Exception:
        print("PyTorch required for BE-Hive. pip install torch")
        sys.exit(2)

    from oracles.behive import BEHiveOracle, editor_is_proxy, BEHIVE_EDITOR_IS_PROXY
    from oracles.cfd import CFDOffTargetOracle

    behive = BEHiveOracle(repo_bystander=args.bystander, repo_efficiency=None,
                          celltype=args.celltype)
    cfd = CFDOffTargetOracle(mismatch_pkl=args.cfd_matrix, pam_pkl=args.cfd_pam,
                             aggregate=args.aggregate_offtarget,
                             max_mismatches=args.ot_max_mismatches)

    eff_cal = None
    if args.efficiency_model:
        from oracles.deep_efficiency import DeepEfficiencyOracle
        eff_source = DeepEfficiencyOracle.from_pickle(args.efficiency_model)
        eff_label = f"sequence-trained model ({args.efficiency_model})"
    else:
        eff_source = SurrogateOracle()
        eff_label = "surrogate"
        if args.efficiency_calibration:
            from oracles.efficiency_calibration import CalibratedEfficiency
            eff_cal = CalibratedEfficiency.load(args.efficiency_calibration)
            eff_label = f"surrogate calibrated to BE-dataHIVE ({args.efficiency_calibration})"

    oracle = CompositeOracle(efficiency_source=eff_source,
                             purity_source=behive,
                             offtarget_source=cfd,
                             efficiency_calibrator=eff_cal)

    print(f"Efficiency : {eff_label}")
    print(f"Purity     : BE-Hive ({args.celltype})")
    print(f"Off-target : CFD ({'aggregate' if args.aggregate_offtarget else 'single-mismatch'}, "
          f"{'REAL matrix' if cfd.using_real_matrix else 'fallback'})")
    print(f"Checkpoint : {ckpt}  (safe to kill and re-run; completed loci are skipped)\n")

    genome = GenomeContext(half_width=PROTOSPACER_LEN, offline=args.offline_genome)

    if args.clinvar:
        from data.clinvar import load_problems
        snvs = load_problems(args.clinvar, max_variants=args.n)
        labels = [f"{s.chrom}:{s.pos}{s.ref}>{s.alt}[{s.target_base}]" for s in snvs]
    else:
        from data.clinvar import synthetic_clinvar_problems, variant_to_snv
        variants = synthetic_clinvar_problems(n=args.n, seed=0)
        snvs = [variant_to_snv(v) for v in variants]
        labels = [f"{v.gene} {v.chrom}:{v.pos}{v.ref}>{v.alt}[{v.chemistry}]" for v in variants]

    proxy_editors = tuple(e for e, p in BEHIVE_EDITOR_IS_PROXY.items() if p)
    exclude = proxy_editors if args.direct_only else ()

    signal.signal(signal.SIGINT, _request_stop)
    try:
        signal.signal(signal.SIGTERM, _request_stop)
    except (AttributeError, ValueError):
        pass

    todo = [(s, l) for s, l in zip(snvs, labels) if l not in done]
    print(f"{len(todo)} loci to score ({len(done)} already done)\n")

    n_calls = 0
    t0 = time.time()
    with open(ckpt, "a", encoding="utf-8") as fh:
        for i, (snv, label) in enumerate(todo, 1):
            if _STOP["flag"]:
                break

            ok = attach_real_context(snv, genome)
            designs = enumerate_designs(snv, exclude_editors=exclude,
                                        collapse_pam=args.collapse_pam)
            if not designs:
                print(f"{label}: no feasible design, skipped")
                append_record(fh, {"locus": label, "candidates": [],
                                   "context": "real GRCh38" if ok else "SYNTHETIC",
                                   "note": "no feasible design"})
                continue

            rows = []
            for d in designs:
                try:
                    f = evaluate(d, snv, oracle)
                except Exception as e:
                    if type(e).__name__ == "UnsupportedDesign":
                        continue          # same rule the pipeline applies
                    raise
                h = to_human(f)
                rows.append({
                    "editor": d.editor,
                    "pam": d.pam_variant,
                    "position": d.target_pos,
                    "efficiency": h["efficiency"],
                    "purity": h["bystander_purity"],
                    "offtarget": h["offtarget_risk"],
                    "proxy_editor": editor_is_proxy(d.editor),
                })
                n_calls += 1

            gene = (getattr(snv, "gene", None)
                    or getattr(snv, "symbol", None)
                    or (label.split()[0] if args.clinvar is None else None))

            rec = {"locus": label, "candidates": rows,
                   "context": "real GRCh38" if ok else "SYNTHETIC"}
            if gene:
                rec["gene"] = gene
            append_record(fh, rec)          # on disk before the next locus starts

            if i % 5 == 0 or i == len(todo):
                el = time.time() - t0
                rate = el / i
                left = rate * (len(todo) - i)
                print(f"  {i}/{len(todo)} | {n_calls} oracle calls | {el/60:.1f} min "
                      f"| ~{left/60:.0f} min left")

    out = consolidate(ckpt, args.out)
    if _STOP["flag"]:
        print(f"\nStopped early. {len(out)} loci are safely on disk. "
              "Re-run the same command to continue.")
        sys.exit(130)

    sizes = [len(v.get("candidates", [])) for v in out.values()]
    print(f"\nDone: {len(out)} loci, {sum(sizes)} candidates, "
          f"mean {sum(sizes)/max(1,len(sizes)):.1f} per locus, "
          f"{n_calls} oracle calls this session, {(time.time()-t0)/60:.1f} min")
    if not any("gene" in v for v in out.values()):
        print("\nNo gene symbols from the loader. d6_compare.py will report locus-level "
              "figures only; supply a locus,gene CSV with --genes, or add a gene field in "
              "data/clinvar.load_problems.")


if __name__ == "__main__":
    main()
