#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 (test): Extract NAIP patches around labeled roads.
- Merges: merged_labels.parquet + pa_roads_only2.parquet + naip_index.csv
- Works with gold/silver subsets
- Uses true midpoint along road, not centroid
- Handles edge/nudge correction
- Cleans /output/patches before running
"""

from pathlib import Path
import os
import sys
import shutil
import random

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
from PIL import Image
import planetary_computer as pc  # for URL signing

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
IN_MERGED = BASE / "output" / "merged_labels.parquet"
IN_ROADS = BASE / "input" / "pa_roads_only2.parquet"
IN_INDEX = BASE / "output" / "naip_index.csv"
OUT_DIR = BASE / "output" / "patches"
LOG = BASE / "output" / "extraction_log.txt"

SIZES = [96, 64, 32]
N_GOLD_PER_SIZE = 3
N_SILVER_PER_SIZE = 2
NUDGE_FRACTIONS = (+0.10, -0.10, +0.25, -0.25, +0.40, -0.40)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def longest_linestring(geom):
    """Return the longest LineString from geometry."""
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        return max(geom.geoms, key=lambda g: g.length)
    try:
        uni = unary_union(geom)
        if isinstance(uni, (LineString, MultiLineString)):
            return longest_linestring(uni)
    except Exception:
        pass
    return None

def midpoint_along(line: LineString):
    """Return the midpoint Point along the true geometry."""
    return line.interpolate(line.length / 2.0)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_output():
    """Remove and recreate output folders."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for subset in ("gold", "silver"):
        for s in SIZES:
            ensure_dir(OUT_DIR / subset / f"{s}px")

def to_png(arr: np.ndarray, out_path: Path):
    """Save raster array as RGB PNG."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.transpose(np.repeat(arr[:1], 3, axis=0), (1, 2, 0))
    Image.fromarray(rgb).save(out_path)

def reproject_point(lon: float, lat: float, dst_crs):
    """Reproject lon/lat to given CRS."""
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def get_center_window(src, x, y, size):
    """Return a window centered at (x,y)."""
    half = size // 2
    col, row = src.index(x, y)
    w = Window(col - half, row - half, size, size)
    if w.col_off < 0 or w.row_off < 0 or \
       w.col_off + w.width > src.width or \
       w.row_off + w.height > src.height:
        return None
    return w

def extract_patch(signed_url, lon, lat, size, out_path):
    """Extract and save one patch."""
    try:
        with rasterio.Env():
            with rasterio.open(signed_url) as src:
                x, y = reproject_point(lon, lat, src.crs)
                win = get_center_window(src, x, y, size)
                if win is None:
                    return False
                data = src.read(window=win)
        ensure_dir(out_path.parent)
        to_png(data, out_path)
        return True
    except Exception:
        return False

def try_nudged(signed_url, line, size, out_path):
    """Try midpoint, then nudge if needed."""
    midpt = midpoint_along(line)
    if extract_patch(signed_url, midpt.x, midpt.y, size, out_path):
        return True
    for frac in NUDGE_FRACTIONS:
        t = 0.5 + frac
        t = max(0.0, min(1.0, t))
        pt = line.interpolate(t * line.length)
        if extract_patch(signed_url, pt.x, pt.y, size, out_path):
            return True
    return False

def pick_samples(df, quality, n):
    sub = df[df["quality"] == quality]
    return sub.sample(n=min(n, len(sub)), random_state=SEED)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("\n================= Stage 2: EXTRACT PATCHES (TEST) =================\n")

    # --- Load inputs ---
    if not IN_MERGED.exists():
        sys.exit(f"Missing {IN_MERGED}")
    if not IN_ROADS.exists():
        sys.exit(f"Missing {IN_ROADS}")
    if not IN_INDEX.exists():
        sys.exit(f"Missing {IN_INDEX}")

    clean_output()
    if LOG.exists():
        LOG.unlink()

    # --- Load merged_labels (id, label, quality) ---
    labels_df = pd.read_parquet(IN_MERGED)[["id", "label", "quality"]]
    labels_df["id"] = labels_df["id"].astype(str)

    # --- Load roads parquet ---
    roads_gdf = gpd.read_parquet(IN_ROADS)
    if "geometry" not in roads_gdf.columns:
        sys.exit("ERROR: geometry column missing in pa_roads_only2.parquet")
    roads_gdf = roads_gdf.rename(columns={"osm_id": "id"})
    roads_gdf["id"] = roads_gdf["id"].astype(str)
    roads_gdf = roads_gdf[["id", "geometry"]]

    # --- Merge labels + geometry ---
    df = labels_df.merge(roads_gdf, on="id", how="inner")

    # --- Load index csv ---
    index_df = pd.read_csv(IN_INDEX, low_memory=False)
    if "tile_url" not in index_df.columns:
        if "signed_url" in index_df.columns:
            index_df = index_df.rename(columns={"signed_url": "tile_url"})
        else:
            sys.exit("naip_index.csv missing tile_url or signed_url column.")
    index_df["id"] = index_df["id"].astype(str)
    index_df = index_df[["id", "tile_id", "datetime", "tile_url"]]

    # --- Merge NAIP index ---
    df = df.merge(index_df, on="id", how="inner")

    print(f"Loaded {len(df)} labeled roads with geometry and NAIP tiles.\n")

    # --- Sample ---
    samples = []
    for size in SIZES:
        g = pick_samples(df, "gold", N_GOLD_PER_SIZE)
        s = pick_samples(df, "silver", N_SILVER_PER_SIZE)
        g = g.assign(_subset="gold", _size=size)
        s = s.assign(_subset="silver", _size=size)
        samples.append(pd.concat([g, s], ignore_index=True))
    test_df = pd.concat(samples, ignore_index=True)

    print(f"Extracting {len(test_df)} patches total.\n")

    # --- Extract patches ---
    ok, fail = 0, 0
    for i, row in test_df.iterrows():
        road_id = row["id"]
        subset = row["_subset"]
        size = int(row["_size"])
        tile_id = row["tile_id"]
        dt = str(row.get("datetime", "")).replace(":", "").replace(" ", "T")
        tile_url = row["tile_url"]

        geom = longest_linestring(row["geometry"])
        if geom is None or geom.length == 0:
            print(f"[{i+1}/{len(test_df)}] {subset} id={road_id}: invalid geom")
            fail += 1
            continue

        try:
            signed = pc.sign(tile_url)
        except Exception as e:
            print(f"[{i+1}/{len(test_df)}] {subset} id={road_id}: sign fail ({e})")
            fail += 1
            continue

        out_name = f"{tile_id}_{road_id}_{dt}_{size}px.png"
        out_path = OUT_DIR / subset / f"{size}px" / out_name

        success = try_nudged(signed, geom, size, out_path)
        if success:
            print(f"[{i+1}/{len(test_df)}] {subset} id={road_id}: saved {out_path.relative_to(BASE)}")
            ok += 1
        else:
            print(f"[{i+1}/{len(test_df)}] {subset} id={road_id}: failed (edge/off-tile)")
            fail += 1

    # --- Summary ---
    print("\n================= SUMMARY =================")
    print(f"Saved:   {ok}")
    print(f"Failed:  {fail}")
    print(f"Output:  {OUT_DIR}")
    print("==========================================\n")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
