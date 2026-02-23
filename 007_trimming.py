#!/usr/bin/env python3
"""
007_trimming.py

Batch-convert state OSM PBF files to trimmed roads-only Parquet.

Default behavior:
- Uses osmium to filter highway=* ways into a temp PBF
- Uses ogr2ogr to export to Parquet with a minimal column set

Notes:
- Requires external tools: osmium, ogr2ogr (GDAL)
- Output Parquet is EPSG:4326 with geometry
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import sys


DEFAULT_KEEP_FIELDS = ["osm_id", "highway", "name"]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Missing required tool: {name}")
    return path


def state_from_filename(pbf_path: Path) -> str:
    # e.g., pennsylvania-260202.osm.pbf -> pennsylvania
    base = pbf_path.name
    if base.endswith(".osm.pbf"):
        base = base[:-8]
    if "-" in base:
        return base.split("-")[0]
    return base


def build_output_path(out_dir: Path, pbf_path: Path) -> Path:
    state = state_from_filename(pbf_path)
    return out_dir / f"{state}_roads_only.parquet"


def detect_keep_fields(ogrinfo: str, pbf_path: Path, requested: list[str]) -> list[str]:
    """Inspect available fields and keep only those present."""
    cmd = [
        ogrinfo,
        "-so",
        "-al",
        str(pbf_path),
        "lines",
    ]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    available = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if ":" in line and "(" in line and ")" in line:
            # Example: "osm_id: Integer64 (0.0)"
            field = line.split(":", 1)[0].strip()
            if field:
                available.add(field)
    # geometry is implicit; always keep osm_id if present
    keep = [f for f in requested if f in available]
    if "osm_id" in available and "osm_id" not in keep:
        keep = ["osm_id"] + keep
    return keep


def convert_one(
    osmium: str,
    ogr2ogr: str,
    ogrinfo: str,
    pbf_path: Path,
    out_path: Path,
    keep_fields: list[str],
    overwrite: bool,
    tmp_dir: Path,
) -> None:
    if out_path.exists() and not overwrite:
        print(f"Skipping (exists): {out_path}")
        return

    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_pbf = tmp_dir / f"{pbf_path.stem}.highway.pbf"

    # 1) Filter to highway ways only
    run([
        osmium,
        "tags-filter",
        "-o",
        str(tmp_pbf),
        str(pbf_path),
        "w/highway",
    ])

    # 2) Export to Parquet with selected fields (only those present)
    keep_list = detect_keep_fields(ogrinfo, tmp_pbf, keep_fields)
    if not keep_list:
        raise RuntimeError(f"No requested fields found in {tmp_pbf.name}")
    keep = ",".join(keep_list)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        out_path.unlink()

    run([
        ogr2ogr,
        "-f",
        "Parquet",
        str(out_path),
        str(tmp_pbf),
        "lines",
        "-select",
        keep,
        "-where",
        "highway IS NOT NULL",
        "-t_srs",
        "EPSG:4326",
    ])

    # Clean temp
    try:
        tmp_pbf.unlink()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Trim OSM PBFs to roads-only Parquet")
    parser.add_argument(
        "--osm-dir",
        default="../osm",
        help="Folder containing .osm.pbf files (default: ../osm)",
    )
    parser.add_argument(
        "--out-dir",
        default="../input/roads_trimmed",
        help="Output folder for trimmed Parquet files (default: ../input/roads_trimmed)",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Run only one file (provide filename or state prefix)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all .osm.pbf files in osm-dir",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Keep only osm_id + geometry (drops highway/surface/name)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    args = parser.parse_args()

    osm_dir = Path(args.osm_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    tmp_dir = out_dir / "_tmp"

    if not osm_dir.exists():
        print(f"OSM dir not found: {osm_dir}")
        return 2

    osmium = find_tool("osmium")
    ogr2ogr = find_tool("ogr2ogr")
    ogrinfo = find_tool("ogrinfo")

    keep_fields = ["osm_id"] if args.minimal else DEFAULT_KEEP_FIELDS
    if args.minimal:
        print("Using minimal schema: osm_id + geometry")
    else:
        print(f"Using fields: {', '.join(keep_fields)}")

    pbfs = sorted(osm_dir.glob("*.osm.pbf"))
    if not pbfs:
        print(f"No .osm.pbf files found in: {osm_dir}")
        return 2

    if args.test:
        # Allow passing either exact filename or state prefix
        test = args.test
        match = None
        for pbf in pbfs:
            if pbf.name == test or pbf.stem.startswith(test):
                match = pbf
                break
        if not match:
            print(f"Test file not found for: {test}")
            return 2
        out_path = build_output_path(out_dir, match)
        convert_one(osmium, ogr2ogr, ogrinfo, match, out_path, keep_fields, args.overwrite, tmp_dir)
        return 0

    if not args.all:
        print("Nothing to do. Use --test <state> or --all.")
        return 2

    for pbf in pbfs:
        out_path = build_output_path(out_dir, pbf)
        convert_one(osmium, ogr2ogr, ogrinfo, pbf, out_path, keep_fields, args.overwrite, tmp_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
