"""
data/clinvar.py
===============
ClinVar loader for PARETO-EDIT (Phase 2).

Loads pathogenic / likely-pathogenic single-nucleotide variants and filters them to the subset
that base editors can in principle CORRECT, then converts each into an SNVContext problem
instance for the optimizer.

Data source (public, no auth):
  Tab-delimited summary (recommended, simplest to parse):
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
  VCF (alternative):
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz

IMPORTANT coordinate note (from ClinVar FTP primer): tab-delimited/XML positions follow HGVS
(right-shifted) convention while the VCF follows the VCF standard (left-shifted). For SNVs
outside repeat regions these coincide, which is the only class we keep here, so the distinction
does not affect us. We additionally restrict to GRCh38 ("Assembly == GRCh38").

Base-editing correction logic
------------------------------
A base editor edits a base on the protospacer (sense) strand: CBE does C->T, ABE does A->G
(CGBE does C->G). To *correct* a pathogenic variant back to the reference allele, the editor
must perform the conversion that turns the ALT base into the REF base on one of the two strands.
Concretely, on the genomic +/- strands the correctable pathogenic ref>alt classes are:

    CBE  (C->T): corrects  G>A (+ strand: A->G is ABE; but C->T on - strand fixes G>A),
                 and       C>T  read on the complementary strand.
    ABE  (A->G): corrects  T>C and A>G-type lesions via A->G on the appropriate strand.

To keep the Phase-2 loader transparent and auditable we encode the canonical mapping used in
the base-editing literature (e.g. the BE patent ClinVar survey): the four transition classes
G>A, C>T (CBE-correctable) and T>C, A>G (ABE-correctable) are the base-editable transitions.
Transversions (e.g. C>G) are only CGBE-relevant and are flagged separately.

The loader emits, for each kept variant, the editor *chemistry* (target_base) needed and the
flanking genomic context (if a reference FASTA is supplied; otherwise context is left empty and
the surrogate synthesises it, exactly as in Phase 0).
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional

from design_space import SNVContext

CLINVAR_TSV_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

# Transition classes that base editors can address, mapped to the editor chemistry + the
# sense-strand target base the editor must convert. (Complement handled by strand choice.)
#   key = (ref, alt) on the reported strand
#   value = (chemistry, target_base_on_sense_strand)
BASE_EDITABLE = {
    ("G", "A"): ("CBE", "C"),   # correct A->G on (-) strand == C->T sense; CBE
    ("C", "T"): ("CBE", "C"),   # CBE corrects the T back to C (read on complementary strand)
    ("T", "C"): ("ABE", "A"),   # ABE
    ("A", "G"): ("ABE", "A"),   # ABE
}
# transversions correctable only by CGBE (flagged, not used by default)
CGBE_EDITABLE = {("C", "G"): ("CGBE", "C"), ("G", "C"): ("CGBE", "C")}


@dataclass
class ClinVarVariant:
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str
    clinical_significance: str
    chemistry: str          # CBE / ABE / CGBE
    target_base: str        # base the editor converts on the sense strand
    name: str               # ClinVar variant name / HGVS


def _open_maybe_gzip(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def stream_variant_summary(
    path: str,
    assembly: str = "GRCh38",
    significances=("Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"),
    include_cgbe: bool = False,
    max_variants: Optional[int] = None,
) -> Iterator[ClinVarVariant]:
    """Stream base-editable pathogenic SNVs from a local variant_summary[.txt|.gz].

    Download once with:
        wget {CLINVAR_TSV_URL}

    The file is large (~hundreds of MB); we stream it line by line and never hold it in memory.
    """
    sig_set = set(significances)
    editable = dict(BASE_EDITABLE)
    if include_cgbe:
        editable.update(CGBE_EDITABLE)

    kept = 0
    with _open_maybe_gzip(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        # required columns in variant_summary
        needed = ["Type", "ClinicalSignificance", "Assembly", "Chromosome",
                  "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF", "GeneSymbol", "Name"]
        # Some releases name the allele columns differently; fall back gracefully.
        ref_key = "ReferenceAlleleVCF" if "ReferenceAlleleVCF" in col else "ReferenceAllele"
        alt_key = "AlternateAlleleVCF" if "AlternateAlleleVCF" in col else "AlternateAllele"

        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            if parts[col["Type"]] != "single nucleotide variant":
                continue
            if parts[col["Assembly"]] != assembly:
                continue
            if parts[col["ClinicalSignificance"]] not in sig_set:
                continue
            ref = parts[col[ref_key]].strip().upper()
            alt = parts[col[alt_key]].strip().upper()
            if ref not in "ACGT" or alt not in "ACGT" or ref == alt:
                continue
            mapping = editable.get((ref, alt))
            if mapping is None:
                continue
            chem, tb = mapping
            try:
                pos = int(parts[col["Start"]])
            except ValueError:
                continue
            yield ClinVarVariant(
                chrom=str(parts[col["Chromosome"]]),
                pos=pos, ref=ref, alt=alt,
                gene=parts[col["GeneSymbol"]],
                clinical_significance=parts[col["ClinicalSignificance"]],
                chemistry=chem, target_base=tb,
                name=parts[col["Name"]],
            )
            kept += 1
            if max_variants and kept >= max_variants:
                return


def variant_to_snv(v: ClinVarVariant, flank: str = "") -> SNVContext:
    """Convert a ClinVarVariant into an optimizer problem instance. If `flank` (the genomic
    neighbourhood centred on the variant) is supplied it is used directly; otherwise the
    surrogate synthesises context (Phase-0 behaviour)."""
    return SNVContext(
        chrom=v.chrom, pos=v.pos, ref=v.ref, alt=v.alt,
        target_base=v.target_base, context=flank,
    )


def load_problems(
    path: str,
    chemistry: Optional[str] = None,
    max_variants: Optional[int] = None,
    **kwargs,
) -> List[SNVContext]:
    """Convenience: load a list of SNVContext problems, optionally restricted to one chemistry
    (e.g. 'CBE' or 'ABE')."""
    out = []
    for v in stream_variant_summary(path, max_variants=None, **kwargs):
        if chemistry and v.chemistry != chemistry:
            continue
        out.append(variant_to_snv(v))
        if max_variants and len(out) >= max_variants:
            break
    return out


# --------------------------------------------------------------------------- #
# Offline synthetic fallback (so the pipeline is testable without the download)
# --------------------------------------------------------------------------- #

def synthetic_clinvar_problems(n: int = 10, seed: int = 0) -> List[ClinVarVariant]:
    """Generate plausible base-editable pathogenic SNV records for testing the loader logic
    WITHOUT the multi-hundred-MB download. Real runs use stream_variant_summary()."""
    import numpy as np
    rng = np.random.default_rng(seed)
    genes = ["BRCA1", "TP53", "CFTR", "MLH1", "APC", "PTEN", "HBB", "LDLR"]
    classes = list(BASE_EDITABLE.items())
    out = []
    for i in range(n):
        (ref, alt), (chem, tb) = classes[i % len(classes)]
        out.append(ClinVarVariant(
            chrom=str(int(rng.integers(1, 23))),
            pos=int(rng.integers(1_000_000, 200_000_000)),
            ref=ref, alt=alt,
            gene=genes[i % len(genes)],
            clinical_significance="Pathogenic",
            chemistry=chem, target_base=tb,
            name=f"NM_000000.0(GENE):c.{i}{ref}>{alt}",
        ))
    return out
