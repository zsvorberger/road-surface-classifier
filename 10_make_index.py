"""
11_make_index_full.py
Build an index linking each road centroid to its NAIP imagery tile.
- Works on merged_labels.parquet (full file).
- Writes unsigned NAIP tile URLs for all records.
- Adds signed preview URLs for the first 3 records.
- Parallelized with threads for speed.
- Supports stop/resume (appends to CSV and skips completed IDs).
"""

from pystac_client import Client
from planetary_computer import sign
import geopandas as gpd
import pandas as pd
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---------- Paths ----------
BASE = Path(__file__).resolve().parents[1]
IN_FILE = BASE / "output" / "merged_labels.parquet"
OUT_FILE = BASE / "output" / "naip_index.csv"
LOG_FILE = BASE / "output" / "make_index_log.txt"

print(f"📂 Input:  {IN_FILE}")
print(f"💾 Output: {OUT_FILE}")
print(f"🪵 Log:    {LOG_FILE}\n")

# ---------- Load input ----------
if not IN_FILE.exists():
    raise FileNotFoundError(f"Missing {IN_FILE}")
gdf = gpd.read_parquet(IN_FILE)
points = gdf.geometry.centroid
total = len(points)
print(f"✅ Loaded {total} road centroids.\n")

# ---------- Resume logic ----------
done_ids = set()
if OUT_FILE.exists():
    existing = pd.read_csv(OUT_FILE)
    done_ids = set(existing["id"])
    print(f"🔁 Resuming: {len(done_ids)} already processed.\n")

lock = threading.Lock()
results = []

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

# ---------- Per-point processing ----------
def process_point(i, lon, lat, id_):
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
            return None

        item = items[0]
        asset = item.assets.get("image")
        href = asset.href if asset else None
        signed = sign(asset.href) if (asset and i < 3) else ""

        return {
            "id": id_,
            "lon": lon,
            "lat": lat,
            "tile_id": item.id,
            "datetime": item.properties.get("datetime"),
            "gsd": item.properties.get("gsd"),
            "tile_url": href,
            "signed_url": signed,
        }
    except Exception as e:
        return {"id": id_, "error": str(e)}

# ---------- Run parallel processing ----------
start = time.time()
to_process = [(i, pt.x, pt.y, gdf.iloc[i].get("id", i))
              for i, pt in enumerate(points)
              if gdf.iloc[i].get("id", i) not in done_ids]

print(f"🚀 Starting run for {len(to_process)} remaining roads.\n")

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_point, *args) for args in to_process]
    for j, f in enumerate(as_completed(futures), start=1):
        res = f.result()
        if res:
            with lock:
                pd.DataFrame([res]).to_csv(OUT_FILE, mode='a', index=False,
                                           header=not OUT_FILE.exists())
            results.append(res)

        if j % 500 == 0:
            elapsed = (time.time() - start) / 60
            msg = f"🧾 Progress: {j}/{len(to_process)} processed in {elapsed:.1f} min"
            print(msg)
            with open(LOG_FILE, "a") as log:
                log.write(msg + "\n")

# ---------- Finish ----------
elapsed = (time.time() - start) / 60
msg = f"\n✅ Finished. {len(results)} new records written to {OUT_FILE}\n⏱️ Elapsed: {elapsed:.2f} min"
print(msg)
with open(LOG_FILE, "a") as log:
    log.write(msg + "\n")
