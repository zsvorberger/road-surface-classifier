#!/usr/bin/env python3
"""
007_naip_indexer.py

Build NAIP index CSVs for all trimmed state Parquets.

Defaults:
- Input:  ../osmtrimmed/*.parquet
- Output: ../NAIP_indexs/naip_index_<state>.csv
- Uses Planetary Computer STAC NAIP (0.6m)
- Resumes if CSV exists (skips already processed IDs)
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
from planetary_computer import sign
from pystac_client import Client


DEFAULT_GSD_MIN = 0.59
DEFAULT_GSD_MAX = 0.61
DEFAULT_THREADS = 1
DEFAULT_STATE_WORKERS = 4


def state_from_filename(path: Path) -> str:
    # texas_roads_only.parquet -> texas
    base = path.stem
    if base.endswith("_roads_only"):
        return base[:-11]
    return base


def connect_catalog(retries: int = 3, sleep_s: int = 5) -> Client:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            catalog = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                headers={"Accept": "application/json"},
            )
            print("✅ Connected to Planetary Computer")
            return catalog
        except Exception as exc:
            last_exc = exc
            print(f"⚠️ Connection failed (attempt {attempt}/{retries}): {exc}")
            time.sleep(sleep_s)
    raise SystemExit(f"❌ Could not connect after {retries} attempts: {last_exc}")


def build_index_for_state(
    parquet_path: Path,
    out_dir: Path,
    threads: int,
    gsd_min: float,
    gsd_max: float,
    signed_preview_count: int = 3,
) -> None:
    state = state_from_filename(parquet_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"naip_index_{state}.csv"
    log_file = out_dir / f"naip_index_{state}.log"

    print(f"\n=== {state.upper()} ===")
    print(f"📂 Input:  {parquet_path}")
    print(f"💾 Output: {out_csv}")
    print(f"🪵 Log:    {log_file}")

    if not parquet_path.exists():
        print(f"❌ Missing parquet: {parquet_path}")
        return

    gdf = gpd.read_parquet(parquet_path)
    if "osm_id" in gdf.columns and "id" not in gdf.columns:
        gdf = gdf.rename(columns={"osm_id": "id"})
    if "id" not in gdf.columns or "geometry" not in gdf.columns:
        raise ValueError(f"{parquet_path.name} must include 'id' and 'geometry'.")

    gdf["id"] = gdf["id"].astype(str)
    points = gdf.geometry.centroid
    total = len(points)

    done_ids = set()
    if out_csv.exists():
        try:
            existing = pd.read_csv(out_csv, usecols=["id"])
            done_ids = set(existing["id"].astype(str))
            print(f"🔁 Resuming: {len(done_ids)} already processed.")
        except Exception:
            print("⚠️ Could not read existing CSV for resume; starting fresh.")

    to_process = [
        (i, pt.x, pt.y, gdf.iloc[i].get("id", i))
        for i, pt in enumerate(points)
        if gdf.iloc[i].get("id", i) not in done_ids
    ]
    print(f"🚀 Starting run for {len(to_process)} remaining roads.")

    catalog = connect_catalog()
    lock = threading.Lock()
    results_written = 0
    start = time.time()

    def process_point(i: int, lon: float, lat: float, id_: str):
        try:
            search = catalog.search(
                collections=["naip"],
                intersects={"type": "Point", "coordinates": [lon, lat]},
                query={"gsd": {"gte": gsd_min, "lte": gsd_max}},
                sortby=[{"field": "datetime", "direction": "desc"}],
                limit=1,
            )
            items = list(search.get_all_items())
            if not items:
                return None
            item = items[0]
            asset = item.assets.get("image")
            href = asset.href if asset else None
            signed = sign(asset.href) if (asset and i < signed_preview_count) else ""
            return {
                "id": id_,
                "lon": lon,
                "lat": lat,
                "tile_id": item.id,
                "datetime": item.properties.get("datetime"),
                "gsd": item.properties.get("gsd"),
                "tile_url": href,
                "signed_url": signed,
            }
        except Exception as exc:
            return {"id": id_, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_point, *args) for args in to_process]
        for j, f in enumerate(as_completed(futures), start=1):
            res = f.result()
            if res:
                with lock:
                    pd.DataFrame([res]).to_csv(
                        out_csv,
                        mode="a",
                        index=False,
                        header=not out_csv.exists(),
                    )
                    results_written += 1

            if j % 500 == 0:
                elapsed = (time.time() - start) / 60
                msg = f"🧾 Progress: {j}/{len(to_process)} processed in {elapsed:.1f} min"
                print(msg)
                with open(log_file, "a") as log:
                    log.write(msg + "\n")

    elapsed = (time.time() - start) / 60
    msg = f"✅ Finished {state}. new_records={results_written} elapsed={elapsed:.2f} min"
    print(msg)
    with open(log_file, "a") as log:
        log.write(msg + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NAIP index CSVs for trimmed state Parquets")
    parser.add_argument(
        "--in-dir",
        default="../osmtrimmed",
        help="Folder containing trimmed Parquet files (default: ../osmtrimmed)",
    )
    parser.add_argument(
        "--out-dir",
        default="../NAIP_indexs",
        help="Output folder for NAIP index CSVs (default: ../NAIP_indexs)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Threads per state (default: {DEFAULT_THREADS})",
    )
    parser.add_argument(
        "--state-workers",
        type=int,
        default=DEFAULT_STATE_WORKERS,
        help=f"Number of states to process in parallel (default: {DEFAULT_STATE_WORKERS})",
    )
    parser.add_argument(
        "--gsd-min",
        type=float,
        default=DEFAULT_GSD_MIN,
        help=f"Min NAIP GSD (default: {DEFAULT_GSD_MIN})",
    )
    parser.add_argument(
        "--gsd-max",
        type=float,
        default=DEFAULT_GSD_MAX,
        help=f"Max NAIP GSD (default: {DEFAULT_GSD_MAX})",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Run only one state file (provide filename or state prefix)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}")
        return 2

    parquets = sorted(in_dir.glob("*.parquet"))
    if not parquets:
        print(f"No parquet files found in: {in_dir}")
        return 2

    if args.test:
        test = args.test
        match = None
        for p in parquets:
            if p.name == test or p.stem.startswith(test):
                match = p
                break
        if not match:
            print(f"Test file not found for: {test}")
            return 2
        build_index_for_state(match, out_dir, args.threads, args.gsd_min, args.gsd_max)
        return 0

    with ThreadPoolExecutor(max_workers=args.state_workers) as executor:
        futures = [
            executor.submit(
                build_index_for_state,
                p,
                out_dir,
                args.threads,
                args.gsd_min,
                args.gsd_max,
            )
            for p in parquets
        ]
        for f in as_completed(futures):
            f.result()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
