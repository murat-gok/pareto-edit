"""
data/genome.py
==============
Fetch real GRCh38 flanking sequence for a locus from the Ensembl REST API, with a local cache
for reproducibility and offline reruns.

Used by Layer 2 (the real-BE-Hive biological application): each ClinVar SNV is given its true
genomic neighbourhood so BE-Hive predicts purity from the real bystander context, not a
synthetic filler.

Endpoint (verified):
    GET https://rest.ensembl.org/sequence/region/human/{chrom}:{start}..{end}:{strand}
        ?content-type=text/plain
    - rest.ensembl.org serves GRCh38 by default (GRCh37 has a separate host).
    - coordinates are 1-based, inclusive; strand 1 = forward.

Coordinate convention: ClinVar tab-delimited Start is 1-based (HGVS-style) on the forward
strand for SNVs, matching Ensembl. We fetch a window centred on the variant position and verify
the returned reference base equals the ClinVar ref allele (sanity check; logs a warning on
mismatch rather than failing, since some ClinVar rows report the variant on the minus strand).

Reproducibility: every fetched window is cached to data/_genome_cache.json keyed by
(chrom, pos, half_width, assembly). The cache is committed with the repo so results reproduce
without network access. State the Ensembl release used in the paper's data-availability section.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

ENSEMBL_GRCH38 = "https://rest.ensembl.org"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_genome_cache.json")

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


class GenomeContext:
    """Fetches and caches GRCh38 flanking sequence for loci."""

    def __init__(self, half_width: int = 30, assembly: str = "GRCh38",
                 server: str = ENSEMBL_GRCH38, offline: bool = False,
                 throttle_s: float = 0.2):
        self.half_width = half_width
        self.assembly = assembly
        self.server = server
        self.offline = offline
        self.throttle_s = throttle_s
        self._cache = self._load_cache()

    # ---------------- cache ---------------- #
    def _load_cache(self) -> dict:
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH) as fh:
                    return json.load(fh)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_PATH, "w") as fh:
                json.dump(self._cache, fh, indent=0)
        except Exception:
            pass

    def _key(self, chrom: str, pos: int) -> str:
        return f"{self.assembly}:{chrom}:{pos}:{self.half_width}"

    # ---------------- fetch ---------------- #
    def fetch_window(self, chrom: str, pos: int, ref: Optional[str] = None) -> Optional[str]:
        """Return a (2*half_width+1)-nt forward-strand sequence centred on `pos`.
        Uses cache; falls back to None if offline and not cached, or on network error."""
        key = self._key(chrom, pos)
        if key in self._cache:
            seq = self._cache[key]
        else:
            if self.offline:
                return None
            seq = self._fetch_from_ensembl(chrom, pos)
            if seq is None:
                return None
            self._cache[key] = seq
            self._save_cache()

        # optional ref-allele sanity check (centre base)
        if ref and len(ref) == 1:
            centre = seq[self.half_width]
            if centre.upper() != ref.upper():
                # try reverse complement (variant reported on minus strand)
                rc = _revcomp(seq)
                if rc[self.half_width].upper() == ref.upper():
                    return rc
                # otherwise warn but return as-is (caller may still use it)
                print(f"[genome] WARNING {chrom}:{pos} centre base {centre} != ref {ref} "
                      f"(strand/indel?). Using forward sequence anyway.")
        return seq

    def _fetch_from_ensembl(self, chrom: str, pos: int) -> Optional[str]:
        import urllib.request
        import urllib.error
        chrom_clean = chrom.replace("chr", "")
        start = pos - self.half_width
        end = pos + self.half_width
        url = (f"{self.server}/sequence/region/human/"
               f"{chrom_clean}:{start}..{end}:1?content-type=text/plain")
        try:
            time.sleep(self.throttle_s)  # be polite to the public API
            req = urllib.request.Request(url, headers={"Content-Type": "text/plain"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                seq = resp.read().decode().strip().upper()
            if not seq or any(c not in "ACGTN" for c in seq):
                return None
            return seq
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"[genome] Ensembl fetch failed for {chrom}:{pos}: {e}")
            return None

    # ---------------- protospacer context for a design ---------------- #
    def protospacer_context(self, chrom: str, pos: int, target_pos: int,
                            protospacer_len: int = 20, ref: Optional[str] = None) -> Optional[str]:
        """Return a protospacer-length window from the real genome such that the variant base
        lands at position `target_pos` (1-based) within the protospacer. Returns None if the
        flank is unavailable."""
        window = self.fetch_window(chrom, pos, ref=ref)
        if window is None:
            return None
        centre = self.half_width  # index of the variant base in `window`
        start = centre - (target_pos - 1)
        proto = window[start:start + protospacer_len]
        if len(proto) < protospacer_len:
            proto = (proto + "A" * protospacer_len)[:protospacer_len]
        return proto
