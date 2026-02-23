### Made9/30 to convert my notes app of gold labels of roads to a clean csv file

import csv, os, re

# Resolve paths relative to this script location
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir   = os.path.dirname(script_dir)  # -> ClassifierTest
input_file = os.path.join(base_dir, "input", "raw_gold.txt")
output_file= os.path.join(base_dir, "output", "gold.csv")

print("Input:", input_file)
print("Output:", output_file)

with open(input_file, "r", errors="ignore") as f:
    lines = f.readlines()

current_label = None  # 1=gravel, 0=paved
rows = []

for raw in lines:
    line = raw.strip()
    if not line:
        continue

    up = line.upper()
    if "GRAVEL" in up:
        current_label = 1
        continue
    if "PAVED"  in up:
        current_label = 0
        continue

    if set(line) <= set("#- "):
        continue

    # last number on the line is the OSM ID
    nums = re.findall(r"\d+", line)
    if not nums or current_label is None:
        continue
    osm_id = nums[-1]
    rows.append((osm_id, current_label))

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["osm_id","label"])
    w.writerows(rows)

print(f"Done. Wrote {len(rows)} rows.")
