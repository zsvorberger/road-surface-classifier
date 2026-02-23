#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
40_extract_road_patches.py
Draws colored road overlays on NAIP tiles from Azure Blob Storage and uploads PNGs.
"""

import os
import io
import gc
import math
import random
import tempfile
import numpy as np
import geopandas as gpd
from shapely.ops import transform as shp_transform
from shapely.geometry import box
from pyproj import Transformer
import rasterio
from PIL import Image, ImageDraw
from azure.storage.blob import BlobServiceClient, ContentSettings

# ============================================
# USER SETTINGS
# ============================================

AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=mapstoragepa;AccountKey=5Vn4LGabS+qBqyKqgtPOSooAr3MNzRUsGbiC792YADRdScC17178H/Ogv4ShoXIJdQF3eDQE1cND+AStC0ZWKg==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "naip-tiles"
INPUT_PREFIX = "pa/"
OUTPUT_PREFIX = "test/overlays/"
ROADS_PATH = "input/pa_roads_only2.parquet"

ROAD_WIDTH_METERS = 4.5
TILE_LIMIT = 1  # run just one tile for testing
TILE_FILTER = ""  # e.g. "m_39075" to test a specific tile

# ============================================
# UTILITY FUNCTIONS
# ============================================

def connect_blob():
    svc = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    print("✅ Connected to Azure Blob Storage")
    return svc

def list_tiles(svc):
    cc = svc.get_container_client(CONTAINER_NAME)
    for b in cc.list_blobs(name_starts_with=INPUT_PREFIX):
        name = b.name
        if name.lower().endswith(".tif"):
            if TILE_FILTER and not os.path.basename(name).startswith(TILE_FILTER):
                continue
            yield name

def blob_exists(svc, name):
    try:
        svc.get_container_client(CONTAINER_NAME).get_blob_client(name).get_blob_properties()
        return True
    except Exception:
        return False

def save_blob(svc, name, data, mime):
    cc = svc.get_container_client(CONTAINER_NAME)
    bc = cc.get_blob_client(name)
    settings = ContentSettings(content_type=mime)
    bc.upload_blob(data, overwrite=True, content_settings=settings)

def download_blob_temp(svc, name):
    cc = svc.get_container_client(CONTAINER_NAME)
    bc = cc.get_blob_client(name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f:
        bc.download_blob().readinto(f)
        return f.name

def random_color(seed):
    rnd = random.Random(seed)
    return tuple(rnd.randint(50, 255) for _ in range(3))

# ============================================
# MAIN LOGIC
# ============================================

def main():
    svc = connect_blob()
    cc = svc.get_container_client(CONTAINER_NAME)

    # Load roads file
    print("📂 Loading roads...")
    roads = gpd.read_parquet(ROADS_PATH)
    if roads.crs is None:
        roads.set_crs(4326, inplace=True)
    if "road_id" not in roads.columns:
        roads["road_id"] = np.arange(len(roads))
    print(f"   → {len(roads):,} roads loaded")

    done = 0
    for blob_name in list_tiles(svc):
        base = os.path.basename(blob_name)
        stem = os.path.splitext(base)[0]
        png_blob = f"{OUTPUT_PREFIX}{stem}.png"
        csv_blob = f"{OUTPUT_PREFIX}{stem}.csv"

        if blob_exists(svc, png_blob):
            print(f"⚠️  Skipping {stem} (already exists)")
            continue

        print(f"🟢 Processing tile: {base}")
        tmp = download_blob_temp(svc, blob_name)

        try:
            with rasterio.open(tmp) as ds:
                tile_bounds = ds.bounds
                tile_crs = ds.crs
                arr = ds.read([1, 2, 3], out_dtype=np.uint8)
                img = np.transpose(arr, (1, 2, 0))
                pil = Image.fromarray(img, "RGB")
                draw = ImageDraw.Draw(pil)

                # Transformers
                to_tile = Transformer.from_crs(roads.crs, tile_crs, always_xy=True).transform
                to_roads = Transformer.from_crs(tile_crs, roads.crs, always_xy=True).transform

                # Filter roads in this tile
                tile_poly_roads = shp_transform(to_roads, box(*tile_bounds))
                idx = list(roads.sindex.intersection(tile_poly_roads.bounds))
                subset = roads.iloc[idx]
                subset = subset[subset.intersects(tile_poly_roads)]
                print(f"   → {len(subset)} roads intersect this tile")

                # Meters to pixels
                px_size = abs(ds.transform.a)
                px_per_m = 1.0 / px_size
                width_px = max(1, int(ROAD_WIDTH_METERS * px_per_m))

                rows = []
                for _, r in subset.iterrows():
                    geom = shp_transform(to_tile, r.geometry)
                    color = random_color(int(r.road_id))
                    for line in geom.geoms if geom.geom_type == "MultiLineString" else [geom]:
                        pts = [(~ds.transform)[x, y] for x, y in line.coords]
                        draw.line(pts, fill=color, width=width_px)
                    rows.append((r.road_id, color))

                # Save image
                buf = io.BytesIO()
                pil.save(buf, format="PNG", optimize=True)
                save_blob(svc, png_blob, buf.getvalue(), "image/png")

                # Save CSV
                csv_buf = io.StringIO()
                csv_buf.write("road_id,color\n")
                for rid, c in rows:
                    csv_buf.write(f"{rid},{c}\n")
                save_blob(svc, csv_blob, csv_buf.getvalue().encode("utf-8"), "text/csv")

                print(f"✅ Saved {png_blob} with {len(rows)} roads")

        finally:
            os.unlink(tmp)
            done += 1
            gc.collect()
            if TILE_LIMIT and done >= TILE_LIMIT:
                print("🛑 TILE_LIMIT reached, stopping.")
                break

    print("🎯 Done.")

# ============================================

if __name__ == "__main__":
    main()
