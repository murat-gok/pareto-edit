"""
Resilient downloader for BE-dataHIVE BYSTANDER (outcome) data.

The wrapper's get_bystander() pulls the whole DB in large chunks and times out against the
(slow, free-tier) Heroku API. This version talks to the same API directly but pages in SMALL
batches with retries and RESUME, so a single slow response can't kill the whole download:

  - small --batch (default 500 rows) so each request finishes well under the timeout
  - per-batch retries with backoff on timeout/connection errors
  - long per-request timeout
  - RESUME: appends each batch to a parts folder; re-running continues from the last saved offset
  - at the end, concatenates parts, filters (cell/editor), drops heavy encodings, saves one pkl

Endpoint/params inferred from the wrapper: GET {BASE}/bystander?limit=..&offset=..

Usage (full HEK293T, resilient):
  python -m experiments.fetch_be_datahive_bystander --cell HEK293T --out be_datahive_bystander.pkl
If it stops, just run the SAME command again — it resumes from where it left off.

Quick smaller pull (one editor, cap rows):
  python -m experiments.fetch_be_datahive_bystander --cell HEK293T --base-editor CBE --max-rows 8000 --out bystander_cbe.pkl
"""
import argparse
import os
import time
import glob
import json
import pandas as pd
import requests

BASE = "https://be-server.herokuapp.com"
ENDPOINT = "bystander"


def fetch_batch(session, limit, offset, timeout, retries, backoff):
    url = f"{BASE}/{ENDPOINT}"
    params = {"limit": limit, "offset": offset}
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            wait = backoff * (attempt + 1)
            print(f"    batch@{offset} attempt {attempt+1}/{retries} failed ({type(e).__name__}); "
                  f"retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"batch@{offset} failed after {retries} retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None)
    ap.add_argument("--base-editor", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="stop after this many rows (across batches)")
    ap.add_argument("--batch", type=int, default=500, help="rows per request (small = robust)")
    ap.add_argument("--timeout", type=int, default=120, help="per-request timeout (s)")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=int, default=5, help="seconds * attempt between retries")
    ap.add_argument("--parts-dir", default="_bystander_parts")
    ap.add_argument("--out", default="be_datahive_bystander.pkl")
    ap.add_argument("--also-csv", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.parts_dir, exist_ok=True)
    session = requests.Session()

    # Clean any stale .tmp files from a power cut mid-write (they are incomplete by definition).
    for tmp in glob.glob(os.path.join(args.parts_dir, "part_*.json.tmp")):
        try:
            os.remove(tmp)
            print(f"Removed stale temp file: {os.path.basename(tmp)}")
        except OSError:
            pass

    # RESUME: find the highest VALID offset already saved (skip any unreadable part).
    def valid_parts():
        good = []
        for p in glob.glob(os.path.join(args.parts_dir, "part_*.json")):
            try:
                with open(p) as f:
                    json.load(f)
                good.append(p)
            except Exception:
                print(f"Discarding corrupt part (will refetch): {os.path.basename(p)}")
                try:
                    os.remove(p)
                except OSError:
                    pass
        return sorted(good, key=lambda p: int(p.split("_")[-1].split(".")[0]))

    existing = valid_parts()
    offset = 0
    if existing:
        last = existing[-1]
        offset = int(last.split("_")[-1].split(".")[0]) + args.batch
        print(f"Resuming: {len(existing)} valid parts saved, continuing from offset {offset}")

    total = sum(len(json.load(open(p))) for p in existing) if existing else 0
    empty_streak = 0
    t0 = time.time()
    while True:
        if args.max_rows and total >= args.max_rows:
            print(f"Reached max-rows {args.max_rows}; stopping.")
            break
        try:
            rows = fetch_batch(session, args.batch, offset, args.timeout, args.retries, args.backoff)
        except RuntimeError as e:
            print(f"\nStopping on persistent error: {e}")
            print("Re-run the SAME command to resume from this offset.")
            break
        if not rows:
            empty_streak += 1
            if empty_streak >= 2:
                print("Empty batches — reached end of database.")
                break
            offset += args.batch
            continue
        empty_streak = 0
        part_path = os.path.join(args.parts_dir, f"part_{offset:09d}.json")
        # ATOMIC WRITE: write to a temp file, fsync, then rename. A power cut mid-write leaves the
        # .tmp (ignored on resume), never a half-written part_*.json.
        tmp_path = part_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(rows, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, part_path)  # atomic on the same filesystem
        total += len(rows)
        offset += args.batch
        if (total // args.batch) % 10 == 0:
            rate = total / max(1, time.time() - t0)
            print(f"  {total} rows saved (offset {offset}, ~{rate:.0f} rows/s)")

    # assemble (tolerant of any unreadable part)
    parts = sorted(glob.glob(os.path.join(args.parts_dir, "part_*.json")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not parts:
        print("No data downloaded."); return
    frames = []
    bad = 0
    for p in parts:
        try:
            frames.append(pd.DataFrame(json.load(open(p))))
        except Exception:
            bad += 1
            print(f"  skipping unreadable part {os.path.basename(p)} (re-run to refetch it)")
    if bad:
        print(f"  ({bad} parts were unreadable and skipped)")
    df = pd.concat(frames, ignore_index=True)
    print(f"\nAssembled {len(df)} rows from {len(frames)} parts")

    if args.cell and "cell" in df.columns:
        df = df[df["cell"] == args.cell].copy()
    if args.base_editor and "base_editor" in df.columns:
        df = df[df["base_editor"].astype(str).str.contains(args.base_editor, case=False)].copy()
    drop_like = [c for c in df.columns if c.startswith(("one_hot_", "hilbert_curve_"))]
    if drop_like:
        df = df.drop(columns=drop_like)

    print(f"After filtering (cell={args.cell}, editor={args.base_editor}): {len(df)} rows")
    if "base_editor" in df.columns:
        print("editors:", sorted(df["base_editor"].astype(str).unique()))
    if "cell" in df.columns:
        print("cells:", sorted(df["cell"].astype(str).unique()))
    pos_cols = [c for c in df.columns if c.startswith("Position_") and " " not in c]
    print(f"position columns: {len(pos_cols)} (e.g. {pos_cols[:6]})")

    df.to_pickle(args.out)
    print(f"Wrote {args.out}")
    if args.also_csv:
        df.to_csv(args.out.rsplit('.', 1)[0] + ".csv", index=False)
        print("Wrote CSV too")
    with open(args.out.rsplit('.', 1)[0] + "_PROVENANCE.txt", "w") as f:
        f.write(f"BE-dataHIVE bystander endpoint (resilient batched fetch)\n"
                f"fetched: {time.strftime('%Y-%m-%d %H:%M')}\nrows: {len(df)}\n"
                f"cell: {args.cell}\nbase_editor: {args.base_editor}\n")
    print("Done. (You can delete the _bystander_parts folder once the pkl looks correct.)")


if __name__ == "__main__":
    main()
