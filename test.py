import os
import io
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import rasterize
from shapely.geometry import box
from azure.storage.blob import BlobServiceClient
from matplotlib import pyplot as plt
import tempfile

# ==================================================
# USER CONFIGURATION SECTION
# ==================================================

# Azure connection
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=mapstoragepa;AccountKey=5Vn4LGabS+qBqyKqgtPOSooAr3MNzRUsGbiC792YADRdScC17178H/Ogv4ShoXIJdQF3eDQE1cND+AStC0ZWKg==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "naip-tiles"
INPUT_PREFIX = "pa/"                   # where your NAIP tiles live
OVERLAY_PREFIX = "overlays/"           # where to save drawn overlays
PIXEL_PREFIX = "pixels/"               # where to save extracted pixel arrays

# Roads input
ROADS_PATH = "input/pa_roads_only2.parquet"  # your parquet file with all roads (WGS84 lon/lat)

# Drawing & extraction
ROAD_BUFFER_METERS = 3.5   # half-width of red overlay buffer (meters)
TILE_LIMIT = 1             # set to 1 for test; remove or set None for full run
TILE_FILTER = ""           # optionally restrict by partial tile name

# ==================================================
# ENVIRONMENT + SETUP
# ==================================================
os.environ["TMPDIR"] = "/mnt/tmp"

def connect_blob():
    svc = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    print("✅ Connected to Azure Blob Storage")
    return svc

def list_tiles(svc):
    container = svc.get_container_client(CONTAINER_NAME)
    blobs = [b.name for b in container.list_blobs(name_starts_with=INPUT_PREFIX) if b.name.endswith(".tif")]
    if TILE_FILTER:
        blobs = [b for b in blobs if TILE_FILTER in b]
    if TILE_LIMIT:
        blobs = blobs[:TILE_LIMIT]
    print(f"🗺️  Found {len(blobs)} tile(s) to process")
    return blobs

# ==================================================
# MAIN FUNCTION
# ==================================================
def main():
    svc = connect_blob()
    container = svc.get_container_client(CONTAINER_NAME)
    roads = gpd.read_parquet(ROADS_PATH)
    print(f"🛣️  Loaded {len(roads)} roads total")

    tiles = list_tiles(svc)

    for blob_name in tiles:
        stem = os.path.splitext(os.path.basename(blob_name))[0]
        overlay_name = f"{OVERLAY_PREFIX}{stem}_roads_overlay.png"
        pixel_name = f"{PIXEL_PREFIX}{stem}_road_pixels.npy"

        # Skip existing
        existing = [b.name for b in container.list_blobs(name_starts_with=OVERLAY_PREFIX)]
        if overlay_name in existing:
            print(f"⚠️ Skipping {stem} (already processed)")
            continue

        print(f"📥 Processing {stem}...")

        # Stream the tile directly from Blob
        blob_client = container.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()

        with MemoryFile(blob_data) as memfile:
            with memfile.open() as ds:
                bounds = ds.bounds
                bbox = box(*bounds)

                # Reproject & clip roads
                roads_proj = roads.to_crs(ds.crs)
                clipped = roads_proj[roads_proj.intersects(bbox)]
                print(f"   ↳ {len(clipped)} roads intersect this tile")

                # Read RGB image
                img = ds.read([1, 2, 3])
                img = np.moveaxis(img, 0, -1).astype(np.uint8)

                transform = ds.transform

                # Create road mask (binary raster where roads are 1)
                mask = rasterize(
                    [(geom.buffer(ROAD_BUFFER_METERS), 1) for geom in clipped.geometry if geom.is_valid],
                    out_shape=(ds.height, ds.width),
                    transform=transform,
                    fill=0,
                    dtype="uint8"
                )

                # === Create overlay ===
                overlay = img.copy()
                overlay[mask == 1] = [255, 0, 0]

                # Save overlay to buffer
                buf = io.BytesIO()
                plt.imsave(buf, overlay)
                buf.seek(0)
                container.upload_blob(name=overlay_name, data=buf, overwrite=True)
                print(f"✅ Uploaded overlay for {stem}")

                # === Extract pixels under road mask ===
                road_pixels = img[mask == 1]  # (N, 3) array of RGB values

                # Save road pixels to npy buffer
                np_buf = io.BytesIO()
                np.save(np_buf, road_pixels)
                np_buf.seek(0)

                # Upload .npy to Blob
                container.upload_blob(name=pixel_name, data=np_buf, overwrite=True)
                print(f"✅ Uploaded pixel data for {stem} ({road_pixels.shape[0]} pixels)")

    print("🎉 All done! Check your Azure Blob under 'overlays/' and 'pixels/'")

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    main()
  