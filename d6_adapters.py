#!/usr/bin/env python3
r"""
Adapters: run each tool over the 150 loci and emit the normalised CSV that
d6_compare.py consumes.

    locus,editor,pam,position,rank,tool_score

HONESTY NOTE. I have not executed these tools, so I have not invented their call
signatures. Each adapter below is a skeleton with the logic and the normalisation
filled in, and a [[VERIFY]] marker wherever you must confirm the actual API against
the package you install. Fabricated function names would cost you more time than
they save.

Run order, by cost:
  1. behive        - already installed; no new dependency
  2. crisprdesign  - R/Bioconductor; a few hours to set up
  3. bedict        - Python; independent of BE-Hive, which is the point

ONE RULE FOR ALL THREE. The adapter records only WHICH DESIGN the tool picked --
editor, PAM, protospacer position. Never copy the tool's own efficiency or purity
numbers into the comparison. d6_compare.py re-scores every pick with your oracle,
which is what makes the comparison about selection rather than about whose model
is better.
"""

import csv
import sys

OUT_FIELDS = ["locus", "editor", "pam", "position", "rank", "tool_score"]


def write_picks(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUT_FIELDS})
    print(f"wrote {len(rows)} picks to {path}")


def load_loci(path):
    """Your 150 loci: locus id, 20nt protospacer context, lesion, strand."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ====================================================================== 1
def run_behive(loci, out="behive_picks.csv"):
    r"""
    BE-Hive's own correction recommender (Arbab et al. 2020).

    The most valuable comparator and the cheapest: BE-Hive already supplies your
    purity objective, so there is no model confound at all. Any difference is
    purely the difference between returning one design and returning the set.

    [[VERIFY]] You already call BE-Hive for purity, so use the same import you
    use there. In the public release the two models are separate packages
    (bystander and efficiency); the recommender in the paper combines them to
    choose an editor and protospacer placement for a target. Confirm whether your
    installed copy exposes that recommender directly. If it does not, implement
    the paper's documented rule explicitly and SAY SO in the Methods -- an
    explicit reimplementation is defensible, a silent one is not.
    """
    rows = []
    for L in loci:
        # [[VERIFY]] replace with the recommender call, or the documented rule.
        #   best = behive.recommend(context=L["context"], lesion=L["lesion"])
        #   editor, position, pam = best.editor, best.position, best.pam
        raise NotImplementedError(
            "Wire BE-Hive's recommender here, or implement the paper's rule "
            "explicitly and document it in the Methods.")
    write_picks(out, rows)


# ====================================================================== 2
def run_crisprdesign(loci, out="crisprdesign_picks.csv"):
    r"""
    crisprDesign (Bioconductor). Has a dedicated base-editing vignette and runs
    in batch, so it is the most practical independent comparator.

    Setup, from R:
        BiocManager::install(c("crisprDesign","crisprBase","crisprBowtie"))
        BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")

    [[VERIFY]] the exact function names in the version you install -- the
    base-editing vignette is the reference. The shape of the workflow is:
    build a GuideSet over the locus window, attach a base-editor object, add
    editing scores, then take the top-ranked guide.

    Write a small R script that loops the loci and writes the CSV directly; it is
    less fragile than calling R from Python. Emit the SELECTION only.
    """
    print(run_crisprdesign.__doc__)
    print("See d6_crisprdesign.R for the script skeleton.")


# ====================================================================== 3
def run_bedict(loci, out="bedict_picks.csv"):
    r"""
    BE-DICT (Marquart et al. 2021). Independent of BE-Hive, which is exactly why
    it is worth the effort: it tests whether your finding is an artefact of one
    outcome model.

    BE-DICT returns per-base editing probabilities for a protospacer. It is a
    predictor, not a designer, so you must state the selection rule you apply on
    top of it and justify it. The defensible choice is the rule a user following
    the paper would apply: rank candidate placements by predicted efficiency at
    the target base. Do NOT silently pick the highest-purity option -- that would
    hand your own thesis to the comparator and invert the comparison.

    [[VERIFY]] the inference entry point in the repository you clone.
    """
    rows = []
    for L in loci:
        # [[VERIFY]] replace with BE-DICT inference over each candidate placement
        #   probs = bedict.predict(protospacer=seq, editor=ed)
        #   score = probs[target_index]
        raise NotImplementedError(
            "Wire BE-DICT inference here and state the selection rule in Methods.")
    write_picks(out, rows)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python d6_adapters.py {behive|crisprdesign|bedict} loci.csv")
    which, loci_path = sys.argv[1], sys.argv[2]
    loci = load_loci(loci_path)
    {"behive": run_behive,
     "crisprdesign": run_crisprdesign,
     "bedict": run_bedict}[which](loci)
