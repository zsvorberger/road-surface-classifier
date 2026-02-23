"""
10_make_index.py
Build an index linking each road centroid to its NAIP imagery tile.
- Works on merged_labels_subset.parquet for quick tests.
- Writes permanent (unsigned) NAIP tile URLs for every record.
- Adds signed preview URLs for the first 3 records.
"""

from pystac_client import Client
from planetary_computer import sign
import geopandas as gpd
import pandas as pd
from pathlib import Path
import time

# ---------- Paths ----------
BASE = Path(__file__).resolve().parents[1]
IN_FILE = BASE / "output" / "merged_labels_subset.parquet"
OUT_FILE = BASE / "output" / "naip_index_subset.csv"

print(f"📂 Input:  {IN_FILE}")
print(f"💾 Output: {OUT_FILE}")

# ---------- Load input ----------
if not IN_FILE.exists():
    raise FileNotFoundError(f"Missing {IN_FILE}")
gdf = gpd.read_parquet(IN_FILE)
points = gdf.geometry.centroid
print(f"✅ Loaded {len(points)} road centroids.\n")

# ---------- Connect to Planetary Computer ----------
for attempt in range(3):
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            headers={"Accept": "application/json"}
        )
        print("✅ Connected to Planetary Computer\n")
        break
    except Exception as e:
        print(f"⚠️ Connection failed (attempt {attempt+1}/3): {e}")
        time.sleep(5)
else:
    raise SystemExit("❌ Could not connect after 3 attempts.")

# ---------- Query imagery ----------
results = []
start = time.time()

for i, pt in enumerate(points):
    lon, lat = pt.x, pt.y
    print(f"({i+1}/{len(points)}) → ({lon:.5f}, {lat:.5f})")
    try:
        search = catalog.search(
            collections=["naip"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            query={"gsd": {"gte": 0.59, "lte": 0.61}},  # only 0.6 m tiles
            sortby=[{"field": "datetime", "direction": "desc"}],
            limit=1,
        )
        items = list(search.get_all_items())
        if not items:
            print("   ⚠️ No imagery found.")
            continue

        item = items[0]
        asset = item.assets.get("image")
        href = asset.href if asset else None
        signed = sign(asset.href) if (asset and i < 3) else ""

        results.append({
            "id": gdf.iloc[i].get("id", i),
            "lon": lon,
            "lat": lat,
            "tile_id": item.id,
            "datetime": item.properties.get("datetime"),
            "gsd": item.properties.get("gsd"),
            "tile_url": href,
            "signed_url": signed,
        })

        tag = "🧾 (signed preview)" if signed else ""
        print(f"   ✅ {item.id}  {tag}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

# ---------- Save output ----------
df = pd.DataFrame(results)
df.to_csv(OUT_FILE, index=False)
print(f"\n✅ Finished.  {len(df)} records written to {OUT_FILE}")
print(f"⏱️  Elapsed: {(time.time()-start)/60:.2f} min")
