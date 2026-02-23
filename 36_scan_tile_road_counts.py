import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
ROADS_PARQUET = OUTPUT_DIR / "merged_labels.parquet"

OUTPUT_CSV = OUTPUT_DIR / "tile_road_counts.csv"

# Column in merged_labels.parquet that encodes gravel vs paved
# Assumes: 1 = gravel, 0 = paved
LABEL_COL = "label"

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading roads data...")
    roads = gpd.read_parquet(ROADS_PARQUET)

    if LABEL_COL not in roads.columns:
        raise ValueError(f"Expected column '{LABEL_COL}' in roads parquet")

    results = []

    tif_files = sorted(INPUT_DIR.glob("*.tif"))

    print(f"Found {len(tif_files)} tif tiles\n")

    for tif_path in tif_files:
        tile_id = tif_path.stem

        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            crs = src.crs

        tile_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        tile_gdf = gpd.GeoDataFrame(
            {"tile": [tile_id]},
            geometry=[tile_geom],
            crs=crs,
        )

        # Reproject roads if needed
        if roads.crs != tile_gdf.crs:
            roads_proj = roads.to_crs(tile_gdf.crs)
        else:
            roads_proj = roads

        # Intersect roads with tile
        intersecting = roads_proj[roads_proj.intersects(tile_geom)]

        paved_count = int((intersecting[LABEL_COL] == 0).sum())
        gravel_count = int((intersecting[LABEL_COL] == 1).sum())
        total_count = len(intersecting)

        results.append({
            "tile_id": tile_id,
            "paved_count": paved_count,
            "gravel_count": gravel_count,
            "total_roads": total_count,
            "gravel_ratio": gravel_count / total_count if total_count > 0 else 0.0,
        })

        print(
            f"{tile_id}: "
            f"paved={paved_count}, "
            f"gravel={gravel_count}, "
            f"total={total_count}, "
            f"gravel_ratio={gravel_count / total_count if total_count > 0 else 0.0:.3f}"
        )

    # Write CSV
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(results).sort_values(by="gravel_count", ascending=False)

    df.to_csv(OUTPUT_CSV, index=False)

    print("\n-----------------------------------")
    print(f"Wrote tile summary to: {OUTPUT_CSV}")
    print("-----------------------------------")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
