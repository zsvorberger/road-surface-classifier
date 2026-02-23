# scripts/41_extract_pixels.py
import os, io, sys, json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.plot import reshape_as_image
from shapely.geometry import box
from shapely.errors import TopologicalError
from azure.storage.blob import BlobServiceClient
from PIL import Image, ImageDraw
from tqdm import tqdm

# -----------------------------
# CONFIG — edit here if needed
# -----------------------------
AZURE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;"
    "AccountName=tilesandpixels;"
    "AccountKey=1T7HQb4GcaPRiKgBGpiLRt0aTdohVn+1ULgigm9SEX/hVFAW2b/JgOysjfvm++6YDfC+MKrNIWPe+AStbj7GJw==;"
    "EndpointSuffix=core.windows.net"
)

# Container that has the NAIP TIFF and where we’ll write outputs
CONTAINER = "naip-tiles"

# Input data already in your VM
ROADS_PARQUET = "output/merged_labels.parquet"

# Test a single tile for QA (blob path inside the container)
TEST_TILE_BLOB = "pa/m_3907501_ne_18_060_20220517.tif"

# Desired physical line width (meters) = what you want “behind the road”
# 2.4 m ≈ 4 px at 0.6 m/px
LINE_WIDTH_METERS = 2.4

# Metadata fields
STATE_CODE = "PA"
TIMESTAMP = "20220517"

# Root prefix inside the container for outputs
STAGE4_PREFIX = "stage4/roads_test"


def upload_bytes(container_client, blob_name, data_bytes, content_type=None):
    container_client.upload_blob(
        name=blob_name,
        data=data_bytes,
        overwrite=True,
        content_settings=None if content_type is None else
        __import__("azure.storage.blob").storage.blob.ContentSettings(content_type=content_type)
    )


def main():
    print("🚀 Stage 4 — per-road NPY + master CSV + single PNG overlay (2.4 m width)")
    # Connect storage
    svc = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container = svc.get_container_client(CONTAINER)
    print("✅ Connected to Azure Blob Storage")

    # Load roads
    roads = gpd.read_parquet(ROADS_PARQUET)
    print(f"✅ Loaded {len(roads):,} total road features")

    # Download TIFF into memory
    print(f"⬇️  Downloading tile: {TEST_TILE_BLOB}")
    tile_bytes = io.BytesIO(container.download_blob(TEST_TILE_BLOB).readall())

    # Open raster
    with rasterio.open(tile_bytes) as src:
        tile_name = os.path.basename(TEST_TILE_BLOB).replace(".tif", "")
        tile_crs = src.crs
        res_mpp = float(src.res[0])  # meters per pixel
        width_px = max(1, int(round(LINE_WIDTH_METERS / res_mpp)))

        tile_bounds = src.bounds
        tile_poly = box(*tile_bounds)
        print(f"🧭 CRS: {tile_crs}, Resolution: {res_mpp:.2f} m/px, Overlay line width: {width_px} px (~{LINE_WIDTH_METERS} m)")

        # Reproject & clip roads
        roads_proj = roads.to_crs(tile_crs)
        roads_clip = roads_proj[roads_proj.intersects(tile_poly)]
        if roads_clip.empty:
            print("⚠️  No roads intersect this tile. Exiting.")
            return
        print(f"🧩 Roads in tile: {len(roads_clip):,}")

        # Read tile bands for overlay
        rgb = reshape_as_image(src.read([1, 2, 3])).astype(np.uint8)
        overlay_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(overlay_img, "RGBA")

        out_prefix = f"{STAGE4_PREFIX}/{tile_name}"
        roads_prefix = f"{out_prefix}/roads"
        records = []

        # Loop roads
        for idx, row in tqdm(list(roads_clip.iterrows()), desc="Processing roads", ncols=80):
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

            def _draw_line_coords(g):
                if g.geom_type == "LineString":
                    pts = [src.index(x, y)[::-1] for x, y in g.coords]
                    if len(pts) >= 2:
                        draw.line(pts, fill=(255, 0, 0, 255), width=width_px)
                elif g.geom_type == "MultiLineString":
                    for ln in g.geoms:
                        _draw_line_coords(ln)
            _draw_line_coords(geom)

            # Metadata row
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

        # Upload CSV
        csv_name = f"{out_prefix}/{tile_name}_roads.csv"
        df = pd.DataFrame.from_records(records)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        upload_bytes(container, csv_name, csv_bytes, content_type="text/csv")

        # Upload overlay PNG
        png_name = f"{out_prefix}/{tile_name}_roads_overlay.png"
        png_buf = io.BytesIO()
        overlay_img.save(png_buf, format="PNG")
        png_buf.seek(0)
        upload_bytes(container, png_name, png_buf.getvalue(), content_type="image/png")

        print("\n✅ Uploads complete:")
        print(f"   • Per-road NPYs:  {roads_prefix}/road_<id>.npy")
        print(f"   • CSV:            {csv_name}")
        print(f"   • Overlay PNG:    {png_name}")
        print("🎉 Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
