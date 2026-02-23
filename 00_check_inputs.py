#!/usr/bin/env python3
"""
00_check_inputs.py  —  Stage 0: Validate, align, and merge road + label data
---------------------------------------------------------------------------
Purpose:
Prepares your labeled road data so each ID lines up with its exact geometry
( EPSG:4326 lat/lon ) and is ready for NAIP tile lookup + training.

What this does:
1) Loads roads (prefers Parquet cache; falls back to GeoJSON once).
2) Standardizes road ID to 'id' (handles osm_id/FID/FID_1/osmid).
3) Loads gold & silver CSVs, adds quality column, and standardizes their ID to 'id'.
4) Validates required columns & CRS (EPSG:4326); filters out null geometry.
5) Saves cleaned outputs:
   • output/merged_labels.parquet  (GeoDataFrame for training)
   • output/merged_labels.csv      (flat preview)
   • output/check_inputs_log.txt   (full run log)

Notes:
- Keep your CSV headers as-is (e.g., 'osm_id'); this script normalizes them.
- Install 'pyarrow' to enable Parquet read/write caching.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import time, sys, warnings, traceback

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- paths & logging ----------------
BASE = Path(__file__).resolve().parents[1]
IN   = BASE / "input"
OUT  = BASE / "output"
OUT.mkdir(exist_ok=True)
LOG  = OUT / "check_inputs_log.txt"

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, x): 
        for f in self.files: f.write(x); f.flush()
    def flush(self): 
        for f in self.files: f.flush()

_logf = open(LOG, "a")
sys.stdout = Tee(sys.stdout, _logf)
sys.stderr = sys.stdout  # capture tracebacks too

print("\n==================== Stage 0: CHECK INPUTS ====================")
t0 = time.time()

try:
    # ---------------- STEP 1: roads ----------------
    print("\n[STEP 1] Loading road geometries...")
    roads_gj = IN / "pa_roads_only2.geojson"
    roads_pq = IN / "pa_roads_only2.parquet"

    if roads_pq.exists():
        t = time.time()
        roads = gpd.read_parquet(roads_pq)
        print(f"⚡ Loaded Parquet roads ({len(roads):,}) in {time.time()-t:.1f}s.")
    else:
        t = time.time()
        roads = gpd.read_file(roads_gj)
        print(f"✅ Loaded {len(roads):,} road segments from GeoJSON in {time.time()-t:.1f}s.")
        # cache for next time (skip gracefully if pyarrow missing)
        try:
            roads.to_parquet(roads_pq)
            print("💾 Saved Parquet copy for faster future loads.")
        except Exception as e:
            print(f"⚠️ Skipping Parquet cache ({type(e).__name__}: {e}). Tip: pip install pyarrow")

    # standardize road ID column to 'id'
    road_id_candidates = ["id", "osm_id", "FID_1", "FID", "osmid", "way_id", "osm_way_id"]
    found = next((c for c in road_id_candidates if c in roads.columns), None)
    if found and found != "id":
        roads = roads.rename(columns={found: "id"})
        print(f"🔧 Roads: renamed '{found}' → 'id'")
    if "id" not in roads.columns or "geometry" not in roads.columns:
        raise ValueError("❌ Roads file must contain 'id' and 'geometry' columns.")
    roads["id"] = roads["id"].astype(str)
    print("✅ Road file structure verified.")

    # ---------------- STEP 2: labels ----------------
    print("\n[STEP 2] Loading label CSVs...")
    gold   = pd.read_csv(IN / "gold.csv")
    silver = pd.read_csv(IN / "silver.csv")

    def normalize_label_ids(df, name):
        df.columns = [c.strip() for c in df.columns]
        candidates = [
            "id","ID","Id",
            "osm_id","osmid",
            "road_id","segment_id","segmentID","segment",
            "way_id","osm_way_id","FID","FID_1"
        ]
        for c in candidates:
            if c in df.columns:
                if c != "id":
                    df = df.rename(columns={c: "id"})
                    print(f"🔧 {name}: using '{c}' as id → renamed to 'id'")
                return df
        raise KeyError(f"{name}: no recognizable ID column. Columns: {list(df.columns)[:15]}")

    gold   = normalize_label_ids(gold,   "gold.csv")
    silver = normalize_label_ids(silver, "silver.csv")
    gold["quality"]   = "gold"
    silver["quality"] = "silver"

    labels = pd.concat([gold, silver], ignore_index=True)
    if "id" not in labels.columns:
        raise KeyError("❌ Label CSVs must contain an 'id' column after normalization.")
    labels["id"] = labels["id"].astype(str)

    print(f"✅ Loaded {len(labels):,} total labels ({len(gold)} gold + {len(silver)} silver).")

    # ---------------- STEP 3: merge ----------------
    print("\n[STEP 3] Merging label IDs with road geometries...")
    t = time.time()
    merged = roads.merge(labels, on="id", how="inner")
    before = len(merged)
    merged = merged[merged.geometry.notnull()]
    dropped_null = before - len(merged)
    print(f"🔗 Merged {len(merged):,} labeled geometries in {time.time()-t:.1f}s."
          + (f" (dropped {dropped_null:,} null-geom rows)" if dropped_null else ""))

    # ---------------- STEP 4: CRS ----------------
    print("\n[STEP 4] Checking CRS...")
    if merged.crs is None or (merged.crs.to_epsg() if hasattr(merged.crs, "to_epsg") else None) != 4326:
        print(f"⚠️ Warning: CRS is {merged.crs}, expected EPSG:4326 (lat/lon).")
    else:
        print("✅ CRS is EPSG:4326 (good).")

    # ---------------- STEP 5: save ----------------
    print("\n[STEP 5] Saving cleaned outputs...")
    pq_out  = OUT / "merged_labels.parquet"
    csv_out = OUT / "merged_labels.csv"
    try:
        merged.to_parquet(pq_out, index=False)
        print(f"💾 {pq_out.name}")
    except Exception as e:
        print(f"⚠️ Could not write Parquet ({type(e).__name__}: {e}).")
    merged.drop(columns="geometry").to_csv(csv_out, index=False)
    print(f"💾 {csv_out.name}")

    # ---------------- SUMMARY ----------------
    print("\n===== SUMMARY =====")
    print(f"Roads loaded:   {len(roads):,}")
    print(f"Labels merged:  {len(merged):,}")
    if "quality" in merged.columns:
        q = merged["quality"].value_counts().to_dict()
        print(f"Quality mix:    {q}")
    if "label" in merged.columns:
        print(f"Classes:        {sorted(map(str, merged['label'].unique()))}")
    print(f"Elapsed:        {time.time()-t0:.1f}s")
    print(f"Log:            {LOG}")
    print("==================== DONE ====================\n")

except Exception:
    print("\n❌ FATAL ERROR\n" + "-"*60)
    traceback.print_exc()
    print("-"*60 + f"\nSee full log at: {LOG}\n")
    sys.exit(1)
finally:
    _logf.flush(); _logf.close()
