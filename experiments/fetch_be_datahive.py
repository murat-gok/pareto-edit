"""
experiments/fetch_be_datahive.py
================================
Resilient, resumable downloader for the BE-dataHIVE efficiency table.

The packaged client (`be_datahive.get_efficiency()`) downloads all ~460k rows in one pass and
aborts the entire download on the first non-200 response. BE-dataHIVE's server (Heroku) returns
intermittent 503s on long pulls, so that single-pass approach rarely completes.

This script instead pages through the API in small batches, writes each batch to a local cache as
it arrives (checkpointing), retries individual batches with exponential backoff on 503/timeout,
and can resume from where a previous run stopped. The result is a single local file
(`be_datahive_efficiency.parquet`, or .pkl fallback) that the empirical-recovery analysis reads
directly — so the slow, flaky download happens once.

Run (resumable — just re-run if it stops):
    python experiments/fetch_be_datahive.py --out be_datahive_efficiency.parquet

Options:
    --limit 250         rows per batch (server default)
    --max-retries 8     per-batch retry attempts on 503/timeout
    --max-rows N        stop after N rows (for a quick partial pull / testing)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

BASE_URL = "https://be-server.herokuapp.com"
ENDPOINT = "efficiency"
DATASIZE = 460000
DEFAULT_LIMIT = 250


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa
        return True
    except Exception:
        try:
            import fastparquet  # noqa
            return True
        except Exception:
            return False


def _save_df(df: pd.DataFrame, path: str) -> str:
    """Save to parquet if an engine is available, else pickle. Returns the actual path used."""
    if path.endswith(".parquet") and not _has_parquet():
        path = path.rsplit(".", 1)[0] + ".pkl"
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)
    return path


def _read_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def _checkpoint_path(out: str) -> str:
    # checkpoint always as pickle (no engine dependency, always writable)
    base = out.rsplit(".", 1)[0]
    return base + ".partial.pkl"


def _load_progress(ckpt: str):
    """Return (cached_dataframe, next_offset) from a checkpoint if present."""
    if os.path.exists(ckpt):
        try:
            df = pd.read_pickle(ckpt)
            return df, len(df)
        except Exception:
            pass
    return None, 0


def _save_checkpoint(df: pd.DataFrame, ckpt: str):
    df.to_pickle(ckpt)


def _shard_dir(out: str) -> str:
    base = out.rsplit(".", 1)[0]
    return base + "_shards"


# columns we actually need for empirical recovery (drop the 28 energy/feature cols to save
# memory and disk; keep guide, editor, cell, efficiency, and a couple of useful context fields)
KEEP_COLS = ["grna", "base_editor", "cell", "efficiency_full_grna_calculated",
             "pam_sequence", "sequence", "study", "source"]


def _slim(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEEP_COLS if c in df.columns]
    return df[cols] if cols else df


def fetch(out: str, limit: int = DEFAULT_LIMIT, max_retries: int = 8,
          max_rows: int | None = None):
    import glob
    import requests

    shard_dir = _shard_dir(out)
    os.makedirs(shard_dir, exist_ok=True)

    # migrate an old single-file checkpoint (from the previous downloader version) into a shard
    # so we resume instead of restarting.
    old_ckpt = out.rsplit(".", 1)[0] + ".partial.pkl"
    if os.path.exists(old_ckpt) and not glob.glob(os.path.join(shard_dir, "shard_*.pkl")):
        try:
            old = pd.read_pickle(old_ckpt)
            n = len(old)
            _slim(old).to_pickle(os.path.join(shard_dir, f"shard_{n:010d}.pkl"))
            print(f"Migrated old checkpoint ({n} rows) into shard format.")
            os.remove(old_ckpt)
        except Exception as e:
            print(f"(could not migrate old checkpoint: {e}; starting fresh)")

    # resume is handled per-batch below (batch_*.pkl files already on disk are skipped)

    target = max_rows or DATASIZE
    session = requests.Session()
    SHARD_EVERY = limit  # one shard per batch now, so skipped offsets don't block others

    # which offsets are already on disk?
    def done_offsets():
        out = set()
        for s in glob.glob(os.path.join(shard_dir, "batch_*.pkl")):
            try:
                out.add(int(os.path.basename(s).split("_")[1].split(".")[0]))
            except Exception:
                pass
        return out

    # migrate legacy big shards (shard_*.pkl, ~5000 rows each) — keep them, they still assemble;
    # we just track new per-batch files separately as batch_*.pkl
    all_offsets = list(range(0, target, limit))
    legacy = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pkl")))
    legacy_covered = 0
    if legacy:
        try:
            legacy_covered = max(int(os.path.basename(s).split("_")[1].split(".")[0])
                                 for s in legacy)
        except Exception:
            legacy_covered = 0
        print(f"{len(legacy)} legacy shard(s) cover up to offset {legacy_covered}.")

    def fetch_one(offset):
        """Try to fetch a single batch; return list (possibly empty) or None on failure."""
        params = {"limit": limit, "offset": offset}
        for attempt in range(max_retries):
            try:
                r = session.get(f"{BASE_URL}/{ENDPOINT}", params=params, timeout=120)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = min(2 ** attempt, 60)
                    print(f"  offset {offset}: HTTP {r.status_code}, retry {attempt+1}/"
                          f"{max_retries} in {wait}s")
                    time.sleep(wait)
                    continue
                print(f"  offset {offset}: HTTP {r.status_code} (non-retryable)")
                return None
            except (requests.Timeout, requests.ConnectionError) as e:
                wait = min(2 ** attempt, 60)
                print(f"  offset {offset}: {type(e).__name__}, retry {attempt+1}/{max_retries}")
                time.sleep(wait)
            except (MemoryError, ValueError):
                time.sleep(min(2 ** attempt, 60))
        return None

    def run_pass(offsets, label):
        """Attempt a list of offsets once; save each success as its own batch file; return the
        list of offsets that still failed (to retry in a later round)."""
        failed = []
        for i, off in enumerate(offsets):
            if off >= legacy_covered and off not in done:  # skip what legacy already covers
                batch = fetch_one(off)
                if batch is None:
                    failed.append(off)
                    continue
                if len(batch) == 0:
                    # past the end; treat as covered, stop early on a run of empties
                    _slim(pd.DataFrame(columns=list(KEEP_COLS))).to_pickle(
                        os.path.join(shard_dir, f"batch_{off:010d}.pkl"))
                else:
                    _slim(pd.DataFrame(batch)).to_pickle(
                        os.path.join(shard_dir, f"batch_{off:010d}.pkl"))
                done.add(off)
            if (i + 1) % 20 == 0:
                got = len(done) + (legacy_covered // limit)
                pct = 100 * min(got * limit, target) / target
                print(f"[{label}] ~{pct:.1f}%  ({len(done)} new batches, "
                      f"{len(failed)} failed so far)")
        return failed

    done = done_offsets()
    todo = [o for o in all_offsets if o >= legacy_covered and o not in done]
    print(f"Need {len(todo)} batches (each {limit} rows); "
          f"{len(done)} already saved beyond legacy.")

    # round 1, then retry the failures a few more times (server may recover between rounds)
    failed = run_pass(todo, "pass1")
    rounds = 1
    while failed and rounds < 4:
        rounds += 1
        print(f"\nRetrying {len(failed)} failed batches (round {rounds})...")
        time.sleep(10)
        failed = run_pass(failed, f"pass{rounds}")

    if failed:
        print(f"\n{len(failed)} batches still failed after {rounds} rounds: "
              f"{failed[:10]}{'...' if len(failed) > 10 else ''}")
        print("Re-run later to attempt these again (already-saved batches are skipped).")

    # ---- assemble everything (legacy shards + per-batch files) ----
    _assemble(out, shard_dir, glob)


def _assemble(out, shard_dir, glob):
    parts = []
    for s in sorted(glob.glob(os.path.join(shard_dir, "shard_*.pkl"))):
        parts.append(pd.read_pickle(s))
    for s in sorted(glob.glob(os.path.join(shard_dir, "batch_*.pkl"))):
        df = pd.read_pickle(s)
        if len(df):
            parts.append(df)
    if not parts:
        print("No data downloaded.")
        return
    final = pd.concat(parts, ignore_index=True)
    del parts
    keys = [c for c in ["grna", "base_editor", "cell"] if c in final.columns]
    if keys:
        final = final.drop_duplicates(subset=keys)
    out = _save_df(final, out)
    print(f"\nDone. Wrote {len(final)} rows to {out}")
    if "grna" in final.columns:
        uniq = final["grna"].nunique()
        print(f"Unique gRNAs: {uniq}")
        _write_provenance(out, final, uniq, shard_dir)
    print(f"(Per-batch files kept in {shard_dir}/ — re-run to fetch any missing batches.)")
    print("Use this file with empirical_recovery_run.py via --be-datahive-file.")


def _write_provenance(out, final, uniq, shard_dir):
    """Record what was downloaded so future studies can reuse this dataset with a clear lineage,
    and save the unique gRNA list as a standalone CSV."""
    import datetime
    base = out.rsplit(".", 1)[0]
    # unique gRNA list (handy for downstream work)
    if "grna" in final.columns:
        try:
            final[["grna"]].drop_duplicates().to_csv(base + "_unique_grnas.csv", index=False)
        except Exception:
            pass
    # full table also as CSV for portability (optional; can be large)
    try:
        final.to_csv(base + ".csv", index=False)
    except Exception:
        pass
    prov = [
        "BE-dataHIVE efficiency download — provenance",
        f"timestamp_utc: {datetime.datetime.utcnow().isoformat()}Z",
        f"source_api: {BASE_URL}/{ENDPOINT}",
        f"rows: {len(final)}",
        f"unique_grnas: {uniq}",
        f"columns: {list(final.columns)}",
        f"note: partial download is fine; server returns intermittent 503s.",
        f"files: {os.path.basename(out)} (table), "
        f"{os.path.basename(base)}_unique_grnas.csv (unique guides), "
        f"{os.path.basename(base)}.csv (CSV copy)",
    ]
    try:
        with open(base + "_PROVENANCE.txt", "w") as fh:
            fh.write("\n".join(prov) + "\n")
        print(f"Wrote provenance + unique-gRNA CSV next to {os.path.basename(out)}.")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="be_datahive_efficiency.parquet")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    fetch(args.out, limit=args.limit, max_retries=args.max_retries, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
