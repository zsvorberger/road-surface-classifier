import geopandas as gpd
import mercantile
import os

# Paths (relative to ClassifierTest folder)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, "input", "pa_roads_only2.geojson")
output_file = os.path.join(base_dir, "output", "pa_roads_only2_tiles.txt")

# Load the GeoJSON
print(f"Loading GeoJSON from: {input_file}")
gdf = gpd.read_file(input_file)

# Collect unique tiles
tiles = set()

print("Computing tiles at zoom 16...")
for geom in gdf.geometry:
    if geom is None:
        continue
    bounds = geom.bounds  # (minx, miny, maxx, maxy)
    for tile in mercantile.tiles(bounds[0], bounds[1], bounds[2], bounds[3], zooms=[16]):
        tiles.add((tile.z, tile.x, tile.y))

print(f"Found {len(tiles)} unique tiles.")

# Save to output file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    for z, x, y in sorted(tiles):
        f.write(f"{z}/{x}/{y}\n")

print(f"Tile list written to: {output_file}")
