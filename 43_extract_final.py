# scripts/43_extract_all_tiles_final.py
import os, io, sys, json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
from shapely.errors import TopologicalError
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

# -----------------------------
# CONFIG — edit if needed
# -----------------------------
AZURE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;"
    "AccountName=tilesandpixels;"
    "AccountKey=1T7HQb4GcaPRiKgBGpiLRt0aTdohVn+1ULgigm9SEX/hVFAW2b/JgOysjfvm++6YDfC+MKrNIWPe+AStbj7GJw==;"
    "EndpointSuffix=core.windows.net"
)

CONTAINER = "naip-tiles"
ROADS_PARQUET = "output/merged_labels.parquet"

LINE_WIDTH_METERS = 2.4   # ~4 px at 0.6 m/px
STATE_CODE = "PA"
TIMESTAMP = "20220517"
STAGE4_PREFIX = "stage4/roads_test"
MAX_TILES = 100000  # change to 100000 for full run


def upload_bytes(container_client, blob_name, data_bytes, content_type=None):
    """Upload bytes to Azure Blob."""
    container_client.upload_blob(
        name=blob_name,
        data=data_bytes,
        overwrite=True,
        content_settings=None if content_type is None else
        __import__("azure.storage.blob").storage.blob.ContentSettings(content_type=content_type)
    )


def tile_already_processed(container, tile_name):
    """Check if the tile's CSV already exists."""
    csv_path = f"{STAGE4_PREFIX}/{tile_name}/{tile_name}_roads.csv"
    blob_client = container.get_blob_client(csv_path)
    return blob_client.exists()



def main():
    print(f"🚀 Stage 4 — per-road NPY + master CSV (no PNG, limit={MAX_TILES})")

    # Connect to Azure
    svc = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container = svc.get_container_client(CONTAINER)
    print("✅ Connected to Azure Blob Storage")

    # Load road geometries
    roads = gpd.read_parquet(ROADS_PARQUET)
    print(f"✅ Loaded {len(roads):,} total road features")

    # List all TIFF tiles
    all_blobs = [b.name for b in container.list_blobs() if b.name.lower().endswith(".tif")]
    print(f"🗺️  Found {len(all_blobs)} total tiles in container")

    if len(all_blobs) > MAX_TILES:
        all_blobs = all_blobs[:MAX_TILES]
        print(f"🔹 Limiting run to first {MAX_TILES} tiles")

    for tiff_blob in tqdm(all_blobs, desc="Processing tiles", ncols=100):
        tile_name = os.path.basename(tiff_blob).replace(".tif", "")

        # Skip existing tiles
        if tile_already_processed(container, tile_name):
            print(f"⏩ Skipping {tile_name} (already processed)")
            continue

        print(f"\n⬇️  Downloading tile: {tiff_blob}")
        try:
            tile_bytes = io.BytesIO(container.download_blob(tiff_blob).readall())
        except Exception as e:
            print(f"  ⚠️  Failed to download {tile_name}: {e}")
            continue

        try:
            with rasterio.open(tile_bytes) as src:
                tile_crs = src.crs
                res_mpp = float(src.res[0])
                width_px = max(1, int(round(LINE_WIDTH_METERS / res_mpp)))

                tile_bounds = src.bounds
                tile_poly = box(*tile_bounds)

                roads_proj = roads.to_crs(tile_crs)
                roads_clip = roads_proj[roads_proj.intersects(tile_poly)]

                if roads_clip.empty:
                    print(f"⚠️  No roads intersect {tile_name}, skipping.")
                    continue

                out_prefix = f"{STAGE4_PREFIX}/{tile_name}"
                roads_prefix = f"{out_prefix}/roads"
                records = []

                for idx, row in roads_clip.iterrows():
                    geom = row.geometry
                    geom = geom.intersection(tile_poly)
                    if geom.is_empty:
                        continue

                    try:
                        geom_buf = geom.buffer(LINE_WIDTH_METERS / 2.0)
                    except Exception as e:
                        print(f"  ⚠️  Buffer failed on feature {idx}: {e}")
                        continue

                    try:
                        mask = geometry_mask(
                            [geom_buf],
                            transform=src.transform,
                            invert=True,
                            out_shape=(src.height, src.width),
                        )
                    except (TopologicalError, ValueError, MemoryError) as e:
                        print(f"  ⚠️  Skipping road (mask error): {e}")
                        continue

                    rows, cols = np.where(mask)
                    if rows.size == 0 or cols.size == 0:
                        continue
                    r0, r1 = rows.min(), rows.max() + 1
                    c0, c1 = cols.min(), cols.max() + 1

                    bands = src.read([1, 2, 3, 4])
                    crop = bands[:, r0:r1, c0:c1]
                    crop_mask = mask[r0:r1, c0:c1]

                    out = np.zeros_like(crop)
                    out[:, crop_mask] = crop[:, crop_mask]
                    out = np.moveaxis(out, 0, -1)

                    road_id = row.get("id", idx)
                    road_id = str(int(road_id)) if isinstance(road_id, (int, np.integer, float)) else str(road_id)

                    npy_buf = io.BytesIO()
                    np.save(npy_buf, out)
                    npy_buf.seek(0)

                    npy_blob = f"{roads_prefix}/road_{road_id}.npy"
                    upload_bytes(container, npy_blob, npy_buf.getvalue(), content_type="application/octet-stream")

                    center = geom.centroid
                    road_length_m = geom.length
                    pixel_count = int(np.count_nonzero(out))

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
                        "path": npy_blob,
                        "tile_crs": str(tile_crs),
                        "timestamp": TIMESTAMP,
                        "state": STATE_CODE,
                        "note": "masked single road",
                        "true_label": ""
                    })

                if not records:
                    print(f"⚠️  No valid roads extracted for {tile_name}")
                    continue

                csv_name = f"{out_prefix}/{tile_name}_roads.csv"
                df = pd.DataFrame.from_records(records)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                upload_bytes(container, csv_name, csv_bytes, content_type="text/csv")

                print(f"✅ Finished {tile_name}: wrote {len(records)} roads → {csv_name}")

        except Exception as e:
            print(f"❌ Error processing {tile_name}: {e}")
            continue

    print("\n🎉 All tiles complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
