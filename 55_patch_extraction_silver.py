import os
import io
import sys
import math
import random
import logging
import time
import csv
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import cv2
from rasterio.features import geometry_mask
from shapely.geometry import box, LineString
from azure.storage.blob import BlobServiceClient, ContentSettings
from PIL import Image

LOG_PATH = Path(__file__).resolve().parents[1] / "patch_extraction.log"
logger = logging.getLogger("patch_extraction")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "input"
ROADS_PARQUET = BASE_DIR / "output" / "merged_labels.parquet"
SILVER_DATASET = INPUT_DIR / "silver.csv"
NAIP_INDEX = BASE_DIR / "output" / "naip_index.csv"

BUFFER_METERS = 10.0
SEGMENT_LENGTH_M = 5.0
MIN_ROAD_LENGTH_M = 20.0
PATCH_SIZE = 96
MIN_MASK_COVERAGE = 0.0
SEED = 42
DEBUG_MODE = True
DEBUG_ROADS_PER_CLASS = 10
SAMPLE_SPACING_M = 2.5
ROAD_BUFFER_M = 6.0
PATCH_SPACING_M = 6.0
PATCH_SIZE_M = 8.0
MIN_ROAD_PIXEL_RATIO = 0.60
USE_ROAD_ONLY_PIXELS = True
ROAD_EDGE_DILATION_PX = 2
TILE_FETCH_LOG_EVERY = 25
MAX_PATCHES_PER_ROAD = 120
MIN_PATCH_VARIANCE = 50.0
MAX_PATCH_ATTEMPTS_PER_ROAD = 120
PREFLIGHT_BUFFER_M = 5.0
PREFLIGHT_INTERSECTION_AREA_M2 = 1.0

SILVER_PAVED_TARGET = 200
SILVER_GRAVEL_TARGET = 200
ROAD_BATCH_SIZE = 25

INDEX_OUTPUT = BASE_DIR / "output" / "patch_index_silver.csv"
INDEX_UPLOAD_EVERY = 100

AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT", "tilestorage01")
AZURE_STORAGE_CONTAINER = "patiles"
AZURE_STORAGE_KEY = os.environ.get("AZURE_STORAGE_KEY")

AZURE_PATCH_ACCOUNT = os.environ.get("AZURE_PATCH_ACCOUNT", "maskedpatches")
AZURE_PATCH_KEY = os.environ.get("AZURE_PATCH_KEY")
PATCHES_CONTAINER = "patches"
MASKS_CONTAINER = "masks"
INDEXES_CONTAINER = "index"

if not AZURE_STORAGE_ACCOUNT:
    raise EnvironmentError("Missing required environment variable: AZURE_STORAGE_ACCOUNT")
if not AZURE_STORAGE_CONTAINER:
    raise EnvironmentError("Missing required environment variable: AZURE_STORAGE_CONTAINER")
if not AZURE_STORAGE_KEY:
    raise EnvironmentError("Missing required environment variable: AZURE_STORAGE_KEY")
if not AZURE_PATCH_ACCOUNT:
    raise EnvironmentError("Missing required environment variable: AZURE_PATCH_ACCOUNT")
if not AZURE_PATCH_KEY:
    raise EnvironmentError("Missing required environment variable: AZURE_PATCH_KEY")

logger.info(
    "Azure tile credential loading: account=%s container=%s key_present=%s",
    AZURE_STORAGE_ACCOUNT,
    AZURE_STORAGE_CONTAINER,
    bool(AZURE_STORAGE_KEY),
)
logger.info(
    "Azure patch credential loading: account=%s key_present=%s",
    AZURE_PATCH_ACCOUNT,
    bool(AZURE_PATCH_KEY),
)

# -----------------------------
# Helpers (from 50_train_cnn_test.py)
# -----------------------------

def open_remote_tiff(tile_url: str):
    return rasterio.open(tile_url)


def resize_or_pad(arr, size):
    h, w, _ = arr.shape
    out = np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
    out[: min(h, size), : min(w, size), :] = arr[: min(h, size), : min(w, size), :]
    return out


def longest_linestring(geom):
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        return max(geom.geoms, key=lambda g: g.length)
    return None


def _distance_channel(window_transform, shape, segment, size_m):
    rows, cols = np.indices(shape)
    xs = window_transform.c + (cols + 0.5) * window_transform.a + (rows + 0.5) * window_transform.b
    ys = window_transform.f + (cols + 0.5) * window_transform.d + (rows + 0.5) * window_transform.e
    coords = list(segment.coords)
    if len(coords) < 2:
        return None
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy
    if denom == 0:
        return None
    wx = xs - x0
    wy = ys - y0
    t = (vx * wx + vy * wy) / denom
    t = np.clip(t, 0.0, 1.0)
    proj_x = x0 + t * vx
    proj_y = y0 + t * vy
    dist = np.hypot(xs - proj_x, ys - proj_y)
    max_dist = math.sqrt(2) * (size_m / 2.0)
    if max_dist <= 0:
        return None
    return np.clip(dist / max_dist, 0.0, 1.0).astype(np.float32)


def extract_patch_from_point(src, point, size_m, segment):
    half = size_m / 2.0
    minx = point.x - half
    maxx = point.x + half
    miny = point.y - half
    maxy = point.y + half
    window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    window = window.round_offsets().round_lengths()
    if window.width <= 0 or window.height <= 0:
        return None, None, None
    try:
        data = src.read(window=window, boundless=False)
    except Exception:
        return None, None, None
    if data.size == 0:
        return None, None, None
    if not np.any(data):
        return None, None, None
    if data.shape[0] < 3:
        return None, None, None
    rgb_patch = np.moveaxis(data[:3], 0, -1)
    window_transform = rasterio.windows.transform(window, src.transform)
    road_mask = geometry_mask(
        [segment.buffer(ROAD_BUFFER_M / 2.0)],
        transform=window_transform,
        invert=True,
        out_shape=(rgb_patch.shape[0], rgb_patch.shape[1]),
    ).astype(np.float32)
    if USE_ROAD_ONLY_PIXELS:
        road_mask_bin = (road_mask > 0).astype(np.uint8)
        if ROAD_EDGE_DILATION_PX > 0:
            kernel = np.ones(
                (2 * ROAD_EDGE_DILATION_PX + 1, 2 * ROAD_EDGE_DILATION_PX + 1),
                dtype=np.uint8,
            )
            road_mask_bin = cv2.dilate(road_mask_bin, kernel, iterations=1)
        rgb_patch = rgb_patch * road_mask_bin[..., None]
    distance_channel = _distance_channel(window_transform, road_mask.shape, segment, size_m)
    if distance_channel is None:
        return None, None, None
    return rgb_patch, road_mask, distance_channel


# -----------------------------
# Azure helpers
# -----------------------------

def connect_tile_storage():
    account_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=AZURE_STORAGE_KEY)


def connect_patch_storage():
    account_url = f"https://{AZURE_PATCH_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=AZURE_PATCH_KEY)


def upload_bytes(container_client, blob_name, data_bytes, content_type):
    settings = ContentSettings(content_type=content_type)
    container_client.upload_blob(
        name=blob_name,
        data=data_bytes,
        overwrite=True,
        content_settings=settings,
    )


def upload_file(container_client, blob_name, file_path, content_type):
    settings = ContentSettings(content_type=content_type)
    with open(file_path, "rb") as handle:
        container_client.upload_blob(
            name=blob_name,
            data=handle,
            overwrite=True,
            content_settings=settings,
        )


def _label_name(label):
    return "gravel" if int(label) == 1 else "paved"


def _format_patch_id(dataset, road_id, patch_index):
    return f"{dataset}_{road_id}_{patch_index:04d}"


def _png_bytes_from_array(arr):
    img = Image.fromarray(arr)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def _load_existing_index(blob_client):
    road_ids = set()
    try:
        if blob_client.exists():
            data = blob_client.download_blob().readall().decode("utf-8")
            reader = csv.DictReader(io.StringIO(data))
            for row in reader:
                if row.get("dataset") == "silver":
                    road_ids.add(str(row.get("road_id", "")).strip())
            logger.info("Loaded existing index from blob: roads=%d", len(road_ids))
            return road_ids
        logger.info("Index blob not found; falling back to local index if present.")
    except Exception as exc:
        logger.warning("Failed reading index blob %s: %s", blob_client.blob_name, exc)
    if INDEX_OUTPUT.exists():
        try:
            with open(INDEX_OUTPUT, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row.get("dataset") == "silver":
                        road_ids.add(str(row.get("road_id", "")).strip())
            logger.info("Loaded existing index from local file: roads=%d", len(road_ids))
        except Exception as exc:
            logger.warning("Failed reading local index %s: %s", INDEX_OUTPUT, exc)
    return road_ids


def _iter_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    logger.info("Script startup. python=%s cwd=%s", sys.version.split()[0], Path.cwd())
    random.seed(SEED)
    np.random.seed(SEED)

    tile_service = connect_tile_storage()
    tile_container = tile_service.get_container_client(AZURE_STORAGE_CONTAINER)

    patch_service = connect_patch_storage()
    patches_container = patch_service.get_container_client(PATCHES_CONTAINER)
    masks_container = patch_service.get_container_client(MASKS_CONTAINER)
    indexes_container = patch_service.get_container_client(INDEXES_CONTAINER)
    index_blob_client = indexes_container.get_blob_client(INDEX_OUTPUT.name)

    existing_road_ids = _load_existing_index(index_blob_client)

    if not ROADS_PARQUET.exists():
        sys.exit(f"Missing labels file: {ROADS_PARQUET}")

    if not SILVER_DATASET.exists():
        sys.exit(f"Missing silver dataset: {SILVER_DATASET}")

    silver_candidates = {0: [], 1: []}
    with open(SILVER_DATASET, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "osm_id" not in reader.fieldnames or "label" not in reader.fieldnames:
            sys.exit("silver.csv must include 'osm_id' and 'label' columns.")
        for row in reader:
            road_id = str(row.get("osm_id", "")).strip()
            if not road_id or road_id in existing_road_ids:
                continue
            try:
                label = int(row.get("label", ""))
            except (TypeError, ValueError):
                continue
            if label not in (0, 1):
                continue
            silver_candidates[label].append(road_id)

    rng = random.Random(SEED)
    for label in (0, 1):
        rng.shuffle(silver_candidates[label])

    paved_take = min(SILVER_PAVED_TARGET, len(silver_candidates[0]))
    gravel_take = min(SILVER_GRAVEL_TARGET, len(silver_candidates[1]))
    selected = (
        [(rid, 0) for rid in silver_candidates[0][:paved_take]]
        + [(rid, 1) for rid in silver_candidates[1][:gravel_take]]
    )
    rng.shuffle(selected)

    logger.info(
        "Selected silver roads. paved=%d gravel=%d skipped_existing=%d",
        paved_take,
        gravel_take,
        len(existing_road_ids),
    )

    if not selected:
        sys.exit("No silver roads selected for extraction.")

    if not NAIP_INDEX.exists():
        sys.exit(f"Missing NAIP index: {NAIP_INDEX}")
    naip_cols = ["id", "tile_id", "tile_url", "signed_url"]
    naip_df = pd.read_csv(NAIP_INDEX, usecols=lambda c: c in naip_cols)
    if "id" not in naip_df.columns or "tile_id" not in naip_df.columns:
        sys.exit("naip_index.csv must include 'id' and 'tile_id' columns.")
    if "tile_url" not in naip_df.columns and "signed_url" not in naip_df.columns:
        sys.exit("naip_index.csv must include 'tile_url' or 'signed_url' columns.")
    naip_df["id"] = naip_df["id"].astype(str)
    naip_df = naip_df[naip_df["id"].isin({rid for rid, _ in selected})].copy()
    if "signed_url" in naip_df.columns:
        naip_df["tile_url"] = naip_df["signed_url"].where(
            naip_df["signed_url"].notna(),
            naip_df.get("tile_url"),
        )
    if naip_df["tile_url"].isna().any():
        naip_df = naip_df[naip_df["tile_url"].notna()].copy()
    road_to_tiles = naip_df.groupby("id")["tile_url"].apply(list).to_dict()

    logger.info(
        "Matched NAIP tiles for silver. roads=%d tiles=%d",
        len(road_to_tiles),
        naip_df["tile_id"].nunique(),
    )

    roads = gpd.read_parquet(ROADS_PARQUET)
    if "id" not in roads.columns:
        sys.exit("Roads parquet missing required 'id' column.")
    roads = roads.copy()
    roads["id"] = roads["id"].astype(str)
    roads = roads[roads["id"].isin({rid for rid, _ in selected})].copy()
    if roads.crs is None:
        logger.warning("Roads CRS is missing; assuming EPSG:4326.")
        roads = roads.set_crs("EPSG:4326")
    logger.info("Road geometries CRS: %s", roads.crs)

    INDEX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_OUTPUT, "a", newline="", encoding="utf-8") as index_handle:
        writer = csv.writer(index_handle)
        if INDEX_OUTPUT.stat().st_size == 0:
            writer.writerow(["patch_id", "patch_path", "mask_path", "label", "road_id", "dataset"])
            index_handle.flush()

        total_index_rows = 0
        roads = roads.set_index("id")

        def _upload_index_if_needed(total_index_rows, force=False):
            if not force and total_index_rows % INDEX_UPLOAD_EVERY != 0:
                return
            index_handle.flush()
            upload_file(indexes_container, INDEX_OUTPUT.name, INDEX_OUTPUT, "text/csv")
            logger.info("Index uploaded. rows=%d", total_index_rows)

        def _handle_patch_upload(
            dataset,
            road_id,
            patch_idx,
            label,
            rgb_weighted,
            mask_resized,
            total_index_rows,
        ):
            patch_id = _format_patch_id(dataset, road_id, patch_idx)
            patch_name = f"{patch_id}.png"
            patch_path = f"{PATCHES_CONTAINER}/{patch_name}"
            mask_path = f"{MASKS_CONTAINER}/{patch_name}"

            patch_img = np.clip(rgb_weighted, 0, 255).astype(np.uint8)
            mask_img = np.clip(mask_resized * 255.0, 0, 255).astype(np.uint8)

            try:
                patch_bytes = _png_bytes_from_array(patch_img)
                mask_bytes = _png_bytes_from_array(mask_img)
                upload_bytes(patches_container, patch_name, patch_bytes, "image/png")
                upload_bytes(masks_container, patch_name, mask_bytes, "image/png")
                logger.info("Upload success for %s", patch_id)
            except Exception as exc:
                logger.error("Upload failed for %s: %s", patch_id, exc)
                return None

            writer.writerow([
                patch_id,
                f"{PATCHES_CONTAINER}/{patch_name}",
                f"{MASKS_CONTAINER}/{patch_name}",
                _label_name(label),
                road_id,
                dataset,
            ])
            total_index_rows += 1
            if total_index_rows % INDEX_UPLOAD_EVERY == 0:
                _upload_index_if_needed(total_index_rows)
            return patch_id, total_index_rows

        def _process_road(road_id, label, tiles, dataset, total_index_rows):
            if not tiles:
                logger.warning("No tiles found for road %s; skipping.", road_id)
                return 0, total_index_rows
            road_rows = roads.loc[road_id]
            if isinstance(road_rows, pd.Series):
                road_rows = [road_rows]
            else:
                road_rows = [row for _, row in road_rows.iterrows()]

            patches_for_road = 0
            reject_notes = []
            rejected_for_road = 0
            attempts = 0
            preflight_failed = False

            logger.info("Road start: osm_id=%s dataset=%s", road_id, dataset)

            for row in road_rows:
                line = longest_linestring(row.geometry)
                if not line:
                    logger.warning(
                        "Rejected road %s geometry_invalid_or_short length=NA crs=%s",
                        road_id,
                        roads.crs,
                    )
                    if len(reject_notes) < 5:
                        reject_notes.append("geometry_invalid_or_short")
                    continue
                preflight_done = False
                for tile_idx, tile_url in enumerate(tiles, start=1):
                    if tile_idx == 1:
                        logger.info("Tile fetch start for road %s (tiles=%d)", road_id, len(tiles))
                    if tile_idx % TILE_FETCH_LOG_EVERY == 0:
                        logger.info("Fetched %d/%d tiles for road %s", tile_idx, len(tiles), road_id)
                    with open_remote_tiff(tile_url) as src:
                        tile_crs = src.crs
                        tile_poly = box(*src.bounds)
                        line_proj = gpd.GeoSeries([line], crs=roads.crs).to_crs(tile_crs).iloc[0]
                        if not preflight_done:
                            preflight_done = True
                            buffered = line_proj.buffer(PREFLIGHT_BUFFER_M)
                            bbox = buffered.envelope
                            intersection_area = bbox.intersection(tile_poly).area
                            if intersection_area < PREFLIGHT_INTERSECTION_AREA_M2:
                                logger.info(
                                    "Preflight skip road %s before tile download: intersection_area=%.2f",
                                    road_id,
                                    float(intersection_area),
                                )
                                preflight_failed = True
                                break
                        if preflight_failed:
                            break
                        if line_proj.length < MIN_ROAD_LENGTH_M:
                            logger.warning(
                                "Rejected road %s geometry_invalid_or_short length=%.3f crs=%s",
                                road_id,
                                float(line_proj.length),
                                tile_crs,
                            )
                            if len(reject_notes) < 5:
                                reject_notes.append("geometry_invalid_or_short")
                            rejected_for_road += 1
                            continue
                        buffer_dist = max(float(src.res[0]) * 2.0, 1.0)
                        if not line_proj.intersects(tile_poly.buffer(buffer_dist)):
                            if len(reject_notes) < 5:
                                reject_notes.append("no_tile_intersection")
                            continue
                        geom = line_proj.intersection(tile_poly)
                        if geom.is_empty or geom.length < MIN_ROAD_LENGTH_M:
                            if len(reject_notes) < 5:
                                reject_notes.append("intersection_empty_or_short")
                            rejected_for_road += 1
                            continue
                        distances = np.arange(0.0, geom.length, PATCH_SPACING_M)
                        for dist in distances:
                            if patches_for_road >= MAX_PATCHES_PER_ROAD:
                                break
                            if attempts >= MAX_PATCH_ATTEMPTS_PER_ROAD:
                                logger.info("Max patch attempts hit for road %s", road_id)
                                break
                            attempts += 1
                            point = geom.interpolate(dist)
                            seg_start = max(0.0, dist - PATCH_SPACING_M)
                            seg_end = min(geom.length, dist + PATCH_SPACING_M)
                            segment = LineString([geom.interpolate(seg_start), geom.interpolate(seg_end)])
                            rgb_patch, road_mask, distance_channel = extract_patch_from_point(
                                src, point, PATCH_SIZE_M, segment
                            )
                            if rgb_patch is None or road_mask is None or distance_channel is None:
                                rejected_for_road += 1
                                if len(reject_notes) < 5:
                                    reject_notes.append("patch_none")
                                continue
                            patch_resized = resize_or_pad(rgb_patch, PATCH_SIZE)
                            if float(np.var(patch_resized[:, :, :3])) < MIN_PATCH_VARIANCE:
                                rejected_for_road += 1
                                if len(reject_notes) < 5:
                                    reject_notes.append("low_variance")
                                continue
                            mask_resized = resize_or_pad(road_mask[..., np.newaxis], PATCH_SIZE)[..., 0]
                            dist_resized = resize_or_pad(distance_channel[..., np.newaxis], PATCH_SIZE)[..., 0]
                            rgb_weight = 0.5 + 0.5 * mask_resized
                            rgb_weighted = patch_resized[:, :, :3] * rgb_weight[..., np.newaxis]
                            patch_idx = patches_for_road + 1

                            upload_id, total_index_rows = _handle_patch_upload(
                                dataset,
                                road_id,
                                patch_idx,
                                label,
                                rgb_weighted,
                                mask_resized,
                                total_index_rows,
                            )
                            if upload_id is None:
                                logger.error("Upload failed; skipping index row for %s", road_id)
                                continue

                            patches_for_road += 1
                            logger.info(
                                "Road %s patch count=%d",
                                road_id,
                                patches_for_road,
                            )
                            if patches_for_road % TILE_FETCH_LOG_EVERY == 0:
                                logger.info(
                                    "Road %s patch progress: %d",
                                    road_id,
                                    patches_for_road,
                                )
                    if tile_idx == len(tiles):
                        logger.info("Tile fetch end for road %s", road_id)
                    if patches_for_road >= MAX_PATCHES_PER_ROAD:
                        break
                    if preflight_failed:
                        break
                if patches_for_road >= MAX_PATCHES_PER_ROAD:
                    break
                if preflight_failed:
                    break

            if preflight_failed:
                logger.info("Road %s skipped by preflight check.", road_id)
            if patches_for_road == 0 and not preflight_failed:
                logger.warning("No valid patches for road %s; skipping.", road_id)
                if reject_notes:
                    logger.warning("Patch rejection sample for road %s: %s", road_id, ", ".join(reject_notes))
            else:
                logger.info("Extracted %d patches for road %s", patches_for_road, road_id)
            return patches_for_road, total_index_rows

        batches = list(_iter_batches(selected, ROAD_BATCH_SIZE))
        for batch_idx, batch in enumerate(batches, start=1):
            logger.info(
                "Processing batch %d/%d (roads=%d)",
                batch_idx,
                len(batches),
                len(batch),
            )
            for road_id, label in batch:
                if road_id not in roads.index:
                    logger.warning("Road %s missing from roads dataset; skipping.", road_id)
                    continue
                tiles = road_to_tiles.get(road_id, [])
                logger.info("Processing silver road (osm_id=%s)", road_id)
                _, total_index_rows = _process_road(road_id, int(label), tiles, "silver", total_index_rows)

        _upload_index_if_needed(total_index_rows, force=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Patch extraction failed: %s", exc)
        raise
