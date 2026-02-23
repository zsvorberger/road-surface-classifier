from pystac_client import Client
from planetary_computer import sign
import geopandas as gpd
import time

# Load your subset file
gdf = gpd.read_parquet("output/merged_labels_subset.parquet")
sample_points = gdf.geometry.centroid[:3]

# Connect safely to Planetary Computer
for attempt in range(3):
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            headers={"Accept": "application/json"}
        )
        print("✅ Connected to Planetary Computer")
        break
    except Exception as e:
        print(f"⚠️ Connection failed (attempt {attempt+1}/3): {e}")
        time.sleep(5)
else:
    raise SystemExit("❌ Could not connect after 3 attempts.")

# Query three points
for i, pt in enumerate(sample_points):
    lon, lat = pt.x, pt.y
    print(f"\n=== Checking Point {i+1} @ ({lon:.5f}, {lat:.5f}) ===")
    try:
        search = catalog.search(
            collections=["naip"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            query={"gsd": {"lt": 0.7}},  # 0.6 m or better
            sortby=[{"field": "datetime", "direction": "desc"}],
            limit=1
        )
        items = list(search.get_all_items())
        if not items:
            print("No imagery found.")
            continue

        item = items[0]
        print(f"🛰️ Tile ID: {item.id}")
        print(f"📅 Acquired: {item.properties.get('datetime')}")
        print(f"📏 GSD: {item.properties.get('gsd')} m")
        for k, v in item.assets.items():
            if k == "image":
                print(f"🔗 Signed URL: {sign(v.href)}")
    except Exception as e:
        print(f"❌ Error: {e}")
