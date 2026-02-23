#!/usr/bin/env python3
"""
35_downloadtiles_locally.py
Download a small random sample of NAIP tiles locally for quick testing.
Saves GeoTIFFs into ClassifierTest/input/ so you can test script 50 locally.
"""

import os
import random
import math
from pathlib import Path

import pandas as pd
import requests

SAMPLE_COUNT = 15
SEED = 42
TIMEOUT = 60
MIN_DISTANCE_KM = 60

BASE_DIR = Path(__file__).resolve().parents[1]
NAIP_INDEX = BASE_DIR / "output" / "naip_index.csv"
OUT_DIR = BASE_DIR / "input"


def download_tile(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"🟡 Cached: {out_path.name}")
        return True
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)
        print(f"⬇️  Downloaded: {out_path.name}")
        return True
    except Exception as exc:
        print(f"❌ Failed: {out_path.name} ({exc})")
        return False


def main():
    if not NAIP_INDEX.exists():
        raise FileNotFoundError(f"Missing {NAIP_INDEX}")

    existing_files = {p.name for p in OUT_DIR.glob("*.tif")}
    existing_tiles = sorted(existing_files)
    print(f"Existing tiles: {len(existing_tiles)}")

    tiles = pd.read_csv(NAIP_INDEX, low_memory=False)
    if "tile_url" not in tiles.columns:
        raise SystemExit("naip_index.csv is missing tile_url column.")

    tiles = tiles[tiles["tile_url"].notna()]
    tile_urls = tiles["tile_url"].unique().tolist()
    if not tile_urls:
        raise SystemExit("No tile URLs found in naip_index.csv")

    random.seed(SEED)
    picks = []
    if "lon" in tiles.columns and "lat" in tiles.columns:
        tiles = tiles[tiles["lon"].notna() & tiles["lat"].notna()].copy()
        tiles["lon"] = tiles["lon"].astype(float)
        tiles["lat"] = tiles["lat"].astype(float)
        rows = tiles[["tile_url", "lon", "lat"]].drop_duplicates().to_dict("records")
        random.shuffle(rows)

        def haversine_km(lon1, lat1, lon2, lat2):
            r = 6371.0
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
            return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        existing_points = []
        if "lon" in tiles.columns and "lat" in tiles.columns and existing_files:
            by_name = {}
            for row in rows:
                filename = os.path.basename(row["tile_url"].split("?")[0])
                if filename in existing_files:
                    by_name[filename] = (row["lon"], row["lat"])
            existing_points = list(by_name.values())

        for row in rows:
            if len(picks) >= SAMPLE_COUNT:
                break
            filename = os.path.basename(row["tile_url"].split("?")[0])
            if filename in existing_files:
                continue
            if not picks:
                picks.append(row)
                continue
            ok = True
            for chosen in picks:
                dist = haversine_km(row["lon"], row["lat"], chosen["lon"], chosen["lat"])
                if dist < MIN_DISTANCE_KM:
                    ok = False
                    break
            if ok:
                for lon, lat in existing_points:
                    dist = haversine_km(row["lon"], row["lat"], lon, lat)
                    if dist < MIN_DISTANCE_KM:
                        ok = False
                        break
            if ok:
                picks.append(row)

        if len(picks) < SAMPLE_COUNT:
            remaining = [r for r in rows if r["tile_url"] not in {p["tile_url"] for p in picks}]
            fill = []
            for row in remaining:
                if len(picks) + len(existing_files) + len(fill) >= SAMPLE_COUNT:
                    break
                filename = os.path.basename(row["tile_url"].split("?")[0])
                if filename in existing_files:
                    continue
                ok = True
                for chosen in picks:
                    dist = haversine_km(row["lon"], row["lat"], chosen["lon"], chosen["lat"])
                    if dist < MIN_DISTANCE_KM:
                        ok = False
                        break
                if ok:
                    for lon, lat in existing_points:
                        dist = haversine_km(row["lon"], row["lat"], lon, lat)
                        if dist < MIN_DISTANCE_KM:
                            ok = False
                            break
                if ok:
                    fill.append(row)
            picks.extend(fill)

        picks = [p["tile_url"] for p in picks]
        print(f"Downloading {len(picks)} spread-out tiles into {OUT_DIR}...")
    else:
        picks = random.sample(tile_urls, k=min(SAMPLE_COUNT, len(tile_urls)))
        print(f"Downloading {len(picks)} random tiles into {OUT_DIR}...")

    to_download = []
    for url in picks:
        filename = os.path.basename(url.split("?")[0])
        if filename in existing_files:
            continue
        to_download.append((url, filename))

    for url, filename in to_download:
        download_tile(url, OUT_DIR / filename)

    final_files = sorted({p.name for p in OUT_DIR.glob("*.tif")})
    print(f"New tiles downloaded: {len(to_download)}")
    print(f"Final tile count: {len(final_files)}")
    print("Tile IDs:")
    for name in final_files:
        print(f"- {name}")

    if len(final_files) < SAMPLE_COUNT:
        print("⚠️ Warning: candidate pool exhausted before reaching target tile count.")
    print("✅ Done.")


if __name__ == "__main__":
    main()
