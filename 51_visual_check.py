import os, io
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient

# ---------- CONFIG ----------
CONNECTION_STRING = "YOUR_CONNECTION_STRING_HERE"
CONTAINER_NAME = "naiptiles"
OUTPUT_DIR = os.path.expanduser("~/ClassifierTest/output")
MERGED_LABELS = os.path.expanduser("~/ClassifierTest/output/merged_labels.parquet")
TEST_TILE = "m_3907501_ne_18_060_20220517.tif"
ROAD_WIDTH_M = 2.4
# ----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load roads ----------
labels = pd.read_parquet(MERGED_LABELS)
# convert to GeoDataFrame safely
if "geometry" not in labels.columns:
    raise RuntimeError("No geometry column found in merged_labels.")
if isinstance(labels["geometry"].iloc[0], str):
    labels["geometry"] = labels["geometry"].apply(wkt.loads)

gdf = gpd.GeoDataFrame(labels, geometry="geometry", crs="EPSG:4326")
print(f"Loaded {len(gdf)} roads.")

# ---------- Azure NAIP tile ----------
blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service.get_container_client(CONTAINER_NAME)
blob_name = f"Stage3/{TEST_TILE}"
data = container_client.download_blob(blob_name).readall()
src = rasterio.open(io.BytesIO(data))

# ---------- Convert width to pixels ----------
width_px = int(ROAD_WIDTH_M / src.res[0])

# ---------- Draw helper ----------
def _draw_line_coords(g, src, draw, width_px):
    if g.geom_type == "LineString":
        pts = [src.index(x, y)[::-1] for x, y in g.coords]
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 0, 0, 255), width=width_px)
    elif g.geom_type == "MultiLineString":
        for ln in g.geoms:
            _draw_line_coords(ln, src, draw, width_px)

# ---------- Filter roads that intersect the tile bbox ----------
bbox = gpd.GeoDataFrame(
    [{"geometry": gpd.GeoSeries.box(*src.bounds)[0]}], crs=src.crs
).to_crs("EPSG:4326")
roads_in_tile = gpd.sjoin(gdf, bbox, how="inner", predicate="intersects")
print(f"Drawing overlay for {TEST_TILE} with {len(roads_in_tile)} roads.")

# ---------- Read and normalize image ----------
img = src.read()
if img.shape[0] >= 3:
    img = img[:3]          # take RGB
img = np.transpose(img, (1, 2, 0))  # (H, W, C)
img = np.clip(img / np.nanpercentile(img, 99), 0, 1)
base = Image.fromarray((img * 255).astype(np.uint8))

overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay, "RGBA")

for _, row in roads_in_tile.iterrows():
    _draw_line_coords(row.geometry, src, draw, width_px)

combined = Image.alpha_composite(base.convert("RGBA"), overlay)
out_path = os.path.join(OUTPUT_DIR, TEST_TILE.replace(".tif", "_roads_overlay.png"))
combined.save(out_path)

plt.imshow(combined)
plt.title("Road Overlay")
plt.axis("off")
plt.show()

print(f"✅ Saved overlay: {out_path}")
