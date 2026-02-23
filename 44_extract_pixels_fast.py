# scripts/43_extract_final.py
import os, io, sys, json, math, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import box
from shapely.errors import TopologicalError
from azure.storage.blob import BlobServiceClient, ContentSettings

# -----------------------------
# CONFIG — edit if needed
# -----------------------------
AZURE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;"
    "AccountName=tilesandpixels;"
    "AccountKey=1T7HQb4GcaPRiKgBGpiLRt0aTdohVn+1ULgigm9SEX/hVFAW2b/JgOysjfvm++6YDfC+MKrNIWPe+AStbj7GJw==;"
    "EndpointSuffix=core.windows.net"
)

CONTAINER        = "naip-tiles"
ROADS_PARQUET    = "/home/azureuser/ClassifierTest/output/merged_labels.parquet"
STAGE4_PREFIX    = "stage4/roads_test"

# Speed/quality knobs
LINE_WIDTH_METERS  = 1.8       # was 2.4 m
MIN_ROAD_LENGTH_M  = 20.0      # skip short OSM segments
MAX_WORKERS        = 2         # tiles processed in parallel (safe on F4s_v2)
MAX_TILES          = 2    # set large to do all

# Local temp (fast NVMe on Azure Linux VMs)
LOCAL_TMP_ROOT = "/mnt/fast/stage4_tmp"   # falls back to /tmp if missing

STATE_CODE         = "PA"
TIMESTAMP          = "20220517"


# -----------------------------
# Helpers
# -----------------------------
def get_container():
    svc = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    return svc.get_container_client(CONTAINER)

def blob_exists(container, blob_path: str) -> bool:
    try:
        return container.get_blob_client(blob_path).exists()
    except Exception:
        return False

def upload_bytes(container, blob_path: str, data: bytes, content_type: Optional[str] = None):
    kwargs = {}
    if content_type:
        kwargs["content_settings"] = ContentSettings(content_type=content_type)
    container.upload_blob(name=blob_path, data=data, overwrite=True, **kwargs)

def upload_file(container, local_path: str, blob_path: str, content_type: Optional[str] = None):
    with open(local_path, "rb") as f:
        upload_bytes(container, blob_path, f.read(), content_type=content_type)

def safe_tmp_dir(tile_name: str) -> str:
    root = LOCAL_TMP_ROOT if os.path.isdir(LOCAL_TMP_ROOT) else "/tmp"
    path = os.path.join(root, f"tile_{tile_name}")
    os.makedirs(path, exist_ok=True)
    return path

def list_all_tiffs(container) -> List[str]:
    return [b.name for b in container.list_blobs(name_starts_with="") if b.name.lower().endswith(".tif")]


# -----------------------------
# Tile worker
# -----------------------------
def process_tile(tiff_blob: str) -> Tuple[str, int, int, Optional[str]]:
    """
    Returns: (tile_name, roads_written, roads_skipped, error_message_or_None)
    """
    container = get_container()
    tile_name = os.path.basename(tiff_blob).replace(".tif", "")
    out_prefix = f"{STAGE4_PREFIX}/{tile_name}"
    csv_blob   = f"{out_prefix}/{tile_name}_roads.csv"
    zip_blob   = f"{out_prefix}/{tile_name}_roads.zip"

    # Skip if already done
    if blob_exists(container, csv_blob) and blob_exists(container, zip_blob):
        return (tile_name, 0, 0, None)

    # Download TIFF into memory
    try:
        tile_bytes = io.BytesIO(container.download_blob(tiff_blob).readall())
    except Exception as e:
        return (tile_name, 0, 0, f"download failed: {e}")

    # Read roads parquet inside worker (simple & robust for multiprocessing)
    try:
        roads = gpd.read_parquet(ROADS_PARQUET)
    except Exception as e:
        return (tile_name, 0, 0, f"roads parquet read failed: {e}")

    roads_written = 0
    roads_skipped = 0
    records = []

    # Temp workspace
    tmp_dir = safe_tmp_dir(tile_name)
    roads_dir = os.path.join(tmp_dir, "roads")
    os.makedirs(roads_dir, exist_ok=True)

    try:
        with rasterio.open(tile_bytes) as src:
            tile_crs = src.crs
            res_mpp  = float(src.res[0])
            width_px = max(1, int(round(LINE_WIDTH_METERS / res_mpp)))

            # Clip roads to tile bounds in same CRS
            tile_poly = box(*src.bounds)
            roads_proj = roads.to_crs(tile_crs)
            roads_clip = roads_proj[roads_proj.intersects(tile_poly)]

            if roads_clip.empty:
                # nothing to upload; clean local and exit quietly
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return (tile_name, 0, 0, None)

            # Iterate roads
            for idx, row in roads_clip.iterrows():
                geom = row.geometry
                # Keep only the portion inside the tile
                geom = geom.intersection(tile_poly)
                if geom.is_empty:
                    roads_skipped += 1
                    continue

                # Skip short segments
                if geom.length < MIN_ROAD_LENGTH_M:
                    roads_skipped += 1
                    continue

                # Buffer around centerline (half-width)
                try:
                    geom_buf = geom.buffer(LINE_WIDTH_METERS / 2.0)
                except Exception as e:
                    roads_skipped += 1
                    continue

                # Rasterize mask
                try:
                    mask = geometry_mask(
                        [geom_buf],
                        transform=src.transform,
                        invert=True,
                        out_shape=(src.height, src.width),
                    )
                except (TopologicalError, ValueError, MemoryError):
                    roads_skipped += 1
                    continue

                rows, cols = np.where(mask)
                if rows.size == 0 or cols.size == 0:
                    roads_skipped += 1
                    continue

                # tight crop window
                r0, r1 = rows.min(), rows.max() + 1
                c0, c1 = cols.min(), cols.max() + 1

                # Read only needed window (faster than reading full bands)
                window = Window.from_slices((r0, r1), (c0, c1))
                bands = src.read([1, 2, 3, 4], window=window)  # (4, H, W)
                crop_mask = mask[r0:r1, c0:c1]

                # apply mask
                out = np.zeros_like(bands)
                out[:, crop_mask] = bands[:, crop_mask]
                out = np.moveaxis(out, 0, -1)  # (H, W, 4)

                road_id = row.get("id", idx)
                if isinstance(road_id, (int, np.integer, float)):
                    try:
                        road_id = str(int(road_id))
                    except Exception:
                        road_id = str(road_id)
                else:
                    road_id = str(road_id)

                # Save to local tmp (per-road npy)
                npy_local = os.path.join(roads_dir, f"road_{road_id}.npy")
                with open(npy_local, "wb") as f:
                    np.save(f, out)

                center = geom.centroid
                road_length_m = float(geom.length)
                pixel_count   = int(np.count_nonzero(out))

                # For CSV we will point to the path inside the ZIP (logical path)
                npy_in_zip = f"roads/road_{road_id}.npy"

                records.append({
                    "patch_id": f"{tile_name}_road_{road_id}",
                    "tile_name": tile_name,
                    "road_id": road_id,
                    "lat": float(center.y),
                    "lon": float(center.x),
                    "width_m": LINE_WIDTH_METERS,
                    "resolution_mpp": res_mpp,
                    "road_length_m": road_length_m,
                    "pixel_count": pixel_count,
                    "path": npy_in_zip,
                    "tile_crs": str(tile_crs),
                    "timestamp": TIMESTAMP,
                    "state": STATE_CODE,
                    "note": "masked single road",
                    "true_label": ""
                })

                roads_written += 1

        if roads_written == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return (tile_name, 0, roads_skipped, None)

        # Build CSV locally
        df = pd.DataFrame.from_records(records)
        csv_local = os.path.join(tmp_dir, f"{tile_name}_roads.csv")
        df.to_csv(csv_local, index=False)

        # ZIP all npys (roads/ folder) → {tile}_roads.zip
        zip_base = os.path.join(tmp_dir, f"{tile_name}_roads")
        shutil.make_archive(zip_base, "zip", root_dir=tmp_dir, base_dir="roads")
        zip_local = f"{zip_base}.zip"

        # Upload CSV + ZIP
        upload_file(container, csv_local, csv_blob, content_type="text/csv")
        upload_file(container, zip_local, zip_blob, content_type="application/zip")

        # Cleanup local
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return (tile_name, roads_written, roads_skipped, None)

    except Exception as e:
        # Best-effort cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return (tile_name, roads_written, roads_skipped, f"processing failed: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    print(f"🚀 Stage 4 — parallel per-tile extract (width={LINE_WIDTH_METERS} m, min_len={MIN_ROAD_LENGTH_M} m)")
    print(f"   Output: per-tile CSV + per-tile roads.zip (per-road .npy inside)")
    print(f"   Parallel tiles: {MAX_WORKERS}")

    container = get_container()
    tiffs = list_all_tiffs(container)
    total = len(tiffs)
    print(f"🗺️  Found {total} tiles in container")
    if total == 0:
        print("No TIFFs found; exiting.")
        return

    # Respect MAX_TILES for test runs
    if total > MAX_TILES:
        tiffs = tiffs[:MAX_TILES]
        print(f"🔹 Limiting to first {MAX_TILES} tiles")

    # Pre-skip tiles that already have both CSV and ZIP
    pending = []
    for t in tiffs:
        tile_name = os.path.basename(t).replace(".tif", "")
        out_prefix = f"{STAGE4_PREFIX}/{tile_name}"
        if blob_exists(container, f"{out_prefix}/{tile_name}_roads.csv") and \
           blob_exists(container, f"{out_prefix}/{tile_name}_roads.zip"):
            continue
        pending.append(t)

    print(f"🧮 Tiles to process now: {len(pending)}")

    # Parallel pool
    completed = 0
    errors = 0
    total_written = 0
    total_skipped = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        future_map = {exe.submit(process_tile, t): t for t in pending}
        for fut in as_completed(future_map):
            tiff_blob = future_map[fut]
            try:
                tile_name, wrote, skipped, err = fut.result()
            except Exception as e:
                errors += 1
                print(f"❌ Worker crashed on {os.path.basename(tiff_blob)}: {e}")
                continue

            completed += 1
            total_written += wrote
            total_skipped += skipped
            if err:
                errors += 1
                print(f"⚠️  {tile_name}: {err}")
            else:
                print(f"✅ {tile_name}: wrote {wrote} roads, skipped {skipped}  "
                      f"({completed}/{len(pending)})")

    print("\n🎉 Done.")
    print(f"   Tiles completed: {completed}/{len(pending)}  | Errors: {errors}")
    print(f"   Roads written: {total_written:,}  | Roads skipped: {total_skipped:,}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
