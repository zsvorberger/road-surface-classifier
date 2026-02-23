
#!/usr/bin/env python3
"""
MakeCleanLabels.py  — roads-only labels CSV + lightweight QGIS visualizer (GPKG)

Reads (inside ClassifierTest/ only):
  - input/pa_roads_only2.geojson       (your PA roads layer)
  - input/labels2.csv                  (your labels; uses road_id; gravel trusted)

Writes to output/ (timestamped):
  - labels_clean_*.csv                 (osm_id,label)  <-- for your pipeline
  - roads_labeled_visual_*.gpkg        (ALL kept roads; fields: osm_id,highway,label)  <-- for QGIS
  - gravel_not_in_clean_*.csv          (FYI report: gravel IDs not in kept roads)
  - summary_*.txt                      (counts, kept classes, etc.)

Whitelist (SAFE): motorway, trunk, primary, secondary, tertiary, residential
Excluded: unclassified, living_street (to avoid risk)
"""

import sys, time
from pathlib import Path
import pandas as pd

# Geo deps
try:
    import geopandas as gpd
    HAS_GPD = True
except Exception:
    HAS_GPD = False

# --------------- CONFIG (matches your filenames) ---------------
BASE_DIR       = Path("ClassifierTest")
IN_DIR         = BASE_DIR / "input"
OUT_DIR        = BASE_DIR / "output"

INPUT_ROADS    = IN_DIR / "pa_roads_only2.geojson"
INPUT_LABELS   = IN_DIR / "labels2.csv"

# SAFE whitelist (no unclassified, no living_street)
HIGHWAY_KEEP = {"motorway","trunk","primary","secondary","tertiary","residential"}

# Visualizer options
VIS_LAYER_NAME = "roads_labeled_visual"
SIMPLIFY_TOLERANCE_METERS = 1.5   # ~1–2 m keeps lines visually fine in QGIS
KEEP_FIELDS = ["osm_id","highway","label"]  # only what you want to see

# Do NOT append gravel IDs that aren't in clean roads (prevents trails sneaking back)
FORCE_APPEND_MISSING_GRAVEL = False
# ---------------------------------------------------------------

def ts(): return time.strftime("%Y%m%d_%H%M")

def norm_id(x):
    s = str(x).strip()
    if "/" in s: s = s.split("/")[-1]  # handle 'way/123'
    return s

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def simplify_in_meters(gdf: "gpd.GeoDataFrame", tol_m: float) -> "gpd.GeoDataFrame":
    """Project to EPSG:3857, simplify, return to EPSG:4326."""
    if gdf.empty:
        return gdf
    src = gdf.crs or "EPSG:4326"
    gdf_proj = gdf.to_crs(3857)
    gdf_proj["geometry"] = gdf_proj.geometry.simplify(tol_m, preserve_topology=True)
    return gdf_proj.to_crs(4326)

def atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)

def main():
    if not HAS_GPD:
        raise RuntimeError("GeoPandas required. Install with: pip install geopandas")

    cwd = Path.cwd().resolve()
    out_dir = (cwd / OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = ts()
    OUT_LABELS  = out_dir / f"labels_clean_{T}.csv"
    OUT_MISS    = out_dir / f"gravel_not_in_clean_{T}.csv"
    OUT_SUMMARY = out_dir / f"summary_{T}.txt"
    OUT_VIZ_GPKG= out_dir / f"roads_labeled_visual_{T}.gpkg"

    # ----- Load roads -----
    roads_path = (cwd / INPUT_ROADS)
    print(f"[1/7] Reading roads: {roads_path}")
    gdf = gpd.read_file(roads_path)

    id_col = find_col(gdf, ["OSMID","osm_id","@id","id","osm_way_id","way_id","ROADID","road_id","roadid"])
    if not id_col:
        raise ValueError("No ID column found in roads (OSMID/osm_id/@id/id/osm_way_id/way_id/ROADID/road_id).")
    if "highway" not in gdf.columns:
        raise ValueError("Roads file has no 'highway' column.")

    gdf[id_col] = gdf[id_col].map(norm_id)
    gdf["highway"] = gdf["highway"].astype(str).str.strip()

    print(f"[2/7] Total features loaded: {len(gdf)} | Keeping classes: {sorted(HIGHWAY_KEEP)}")
    clean = gdf[gdf["highway"].isin(HIGHWAY_KEEP)].copy()
    clean[id_col] = clean[id_col].astype(str)
    clean_ids = set(clean[id_col].tolist())
    print(f"[3/7] Kept (roads-only): {len(clean)} | Unique IDs: {len(clean_ids)}")

    # ----- Load labels -----
    labels_path = (cwd / INPUT_LABELS)
    print(f"[4/7] Reading labels: {labels_path}")
    lab = pd.read_csv(labels_path)

    # Your CSV uses road_id
    lab_id_col = find_col(lab, ["road_id","ROADID","roadid","OSMID","osm_id","id","osm_way_id","way_id"])
    if not lab_id_col:
        raise ValueError("labels2.csv must have road_id (or ROADID/OSMID/osm_id/id/osm_way_id/way_id).")
    lab[lab_id_col] = lab[lab_id_col].map(norm_id).astype(str)

    # Accept label column values like 'gravel', '1', 'true', 'yes'
    lab_label_col = find_col(lab, ["label","class","is_gravel","surface"])
    if lab_label_col:
        vals = lab[lab_label_col].astype(str).str.lower().str.strip()
        gravel_df = lab[vals.isin(["1","true","gravel","yes"])]
    else:
        gravel_df = lab  # treat all as gravel list
    gravel_ids = set(gravel_df[lab_id_col].tolist())
    print(f"[5/7] Gravel IDs from labels file: {len(gravel_ids)} (column '{lab_id_col}')")

    # ----- Build labels CSV (osm_id,label) -----
    print("[6/7] Building labels CSV…")
    labels = pd.DataFrame({"osm_id": sorted(clean_ids)})
    labels["label"] = 0
    labels.loc[labels["osm_id"].isin(gravel_ids), "label"] = 1

    missing_gravel = sorted(list(gravel_ids - clean_ids))
    pd.DataFrame({"osm_id": missing_gravel}).to_csv(OUT_MISS, index=False)

    if FORCE_APPEND_MISSING_GRAVEL and missing_gravel:
        extra = pd.DataFrame({"osm_id": missing_gravel, "label": 1})
        labels = pd.concat([labels, extra], ignore_index=True).drop_duplicates("osm_id")

    labels.to_csv(OUT_LABELS, index=False)

    # ----- Visualizer layer (GPKG, simplified, minimal fields) -----
    print("[7/7] Building visualizer for QGIS…")
    # Join labels back
    clean = clean.rename(columns={id_col: "osm_id"}).copy()
    clean["osm_id"] = clean["osm_id"].astype(str)
    viz = clean.merge(labels, on="osm_id", how="left")
    viz["label"] = viz["label"].fillna(0).astype(int)

    # Keep only the fields you care about
    keep_cols = [c for c in KEEP_FIELDS if c in viz.columns]
    viz = viz[keep_cols + ["geometry"]].copy()

    # Simplify geometry lightly to shrink size, keep WGS84
    viz = viz.set_crs(gdf.crs or "EPSG:4326")
    if SIMPLIFY_TOLERANCE_METERS and SIMPLIFY_TOLERANCE_METERS > 0:
        print(f"      Simplifying geometry (~{SIMPLIFY_TOLERANCE_METERS} m)…")
        viz = simplify_in_meters(viz, SIMPLIFY_TOLERANCE_METERS)
    viz = viz.to_crs(4326)

    print("      Writing GeoPackage visual layer…")
    # Write to GPKG with only one layer and minimal columns
    if OUT_VIZ_GPKG.exists():
        OUT_VIZ_GPKG.unlink()  # ensure fresh file
    viz.to_file(OUT_VIZ_GPKG, driver="GPKG", layer=VIS_LAYER_NAME)

    # ----- Summary -----
    kept_counts = viz["highway"].value_counts().to_dict()
    summary = (
        "Clean labels build (SAFE whitelist + lightweight visualizer)\n"
        f"- Input roads:  {INPUT_ROADS}\n"
        f"- Input labels: {INPUT_LABELS}\n"
        f"- Kept highways: {sorted(HIGHWAY_KEEP)}\n\n"
        f"- Clean features: {len(viz)}\n"
        f"- Unique clean IDs: {viz['osm_id'].nunique()}\n"
        f"- Gravel IDs (labels file): {len(gravel_ids)}\n"
        f"- Final labels rows: {len(labels)}\n"
        f"- Gravel missing in clean (not included): {len(missing_gravel)}\n"
        f"- Labels CSV: {OUT_LABELS}\n"
        f"- Visualizer GPKG: {OUT_VIZ_GPKG} (layer='{VIS_LAYER_NAME}')\n"
        f"- Per-highway kept counts: {kept_counts}\n"
    )
    atomic_write_text(OUT_SUMMARY, summary)
    print("\nDONE ✅")
    print(summary)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
