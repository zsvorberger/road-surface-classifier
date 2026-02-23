import os, io, sys, math, random, logging, traceback, json, csv
import concurrent.futures
from collections import Counter
from pathlib import Path
import numpy as np
import time
import uuid
import geopandas as gpd
import pandas as pd
import rasterio
import cv2
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, LineString, Point
from shapely.errors import TopologicalError
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, losses, optimizers
from tensorflow.keras.applications import MobileNetV3Large, mobilenet_v3
from affine import Affine
from PIL import Image, ImageDraw
 

LOG_PATH = Path(__file__).resolve().parents[1] / "cnn_train.log"
logger = logging.getLogger("cnn_train")
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
MODEL_OUT = BASE_DIR / "output" / "cnn_test_model_zoomed_aug.h5"
TILE_SUMMARY = BASE_DIR / "output" / "tile_road_counts.csv"
GOLD_DATASET = BASE_DIR / "output" / "goldgpt.csv"
SILVER_DATASET = INPUT_DIR / "silver.csv"
NAIP_INDEX = BASE_DIR / "output" / "naip_index.csv"
VAL_ROADS_FILE = BASE_DIR / "output" / "val_roads.json"
EVAL_ROADS_FILE = BASE_DIR / "output" / "eval_roads.json"
PATCH_COORDS_CACHE = BASE_DIR / "output" / "patch_coords_cache.json"
PATCH_CACHE_DIR = BASE_DIR / "output" / "patch_cache"
PATCH_CACHE_INDEX = BASE_DIR / "output" / "patch_cache_index.csv"
USE_DISK_PATCH_CACHE = True
MAX_TOTAL_PATCHES = 40000
REUSE_PATCH_CACHE = True

BUFFER_METERS = 10.0
SEGMENT_LENGTH_M = 5.0
MIN_ROAD_LENGTH_M = 20.0
PATCH_SIZE = 96
MIN_MASK_COVERAGE = 0.0
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 5
EPOCHS = HEAD_EPOCHS + FINETUNE_EPOCHS
BATCH_SIZE = 16
SEED = 42
DEBUG_PATCH_DIR = BASE_DIR / "output" / "patch_debug"
DEBUG_PATCH_COUNT = 10
INCLUDE_HARD_NEGATIVES = True
HARD_NEGATIVE_OFFSET_M = 5.0
TILE_COUNT = 5
MAX_ROADS_PER_TILE = None
MIN_GRAVEL_COUNT = 10
MIN_GRAVEL_ROADS_PER_SPLIT = 10
SPLIT_ATTEMPTS = 50
DEBUG_MODE = True
DEBUG_ROADS_PER_CLASS = 10
SAMPLE_SPACING_M = 2.5
ROAD_BUFFER_M = 6.0
PATCH_SPACING_M = 6.0
PATCH_SIZE_M = 8.0
MIN_ROAD_PIXEL_RATIO = 0.60
DEBUG_SAVE_PER_CLASS = 50
GOLD_ROADS_PER_CLASS = 150
SILVER_ROADS_PER_CLASS = 200
EVAL_ROADS_PER_CLASS = 200
GOLD_ROAD_WEIGHT = 1.0
SILVER_ROAD_WEIGHT = 0.2
USE_ROAD_ONLY_PIXELS = True
ROAD_EDGE_DILATION_PX = 2
TILE_FETCH_LOG_EVERY = 25
MAX_PATCHES_PER_ROAD = 120
MIN_PATCH_VARIANCE = 50.0
MAX_PATCH_ATTEMPTS_PER_ROAD = 120
PREFLIGHT_BUFFER_M = 5.0
PREFLIGHT_INTERSECTION_AREA_M2 = 1.0
PATCH_EXTRACT_WORKERS = 1
ENABLE_DEBUG_PATCH_SAVES = False

AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT", "tilestorage01")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "patiles")
AZURE_STORAGE_KEY = os.environ.get("AZURE_STORAGE_KEY")

if not AZURE_STORAGE_ACCOUNT:
    raise EnvironmentError("Missing required environment variable: AZURE_STORAGE_ACCOUNT")
if not AZURE_STORAGE_CONTAINER:
    raise EnvironmentError("Missing required environment variable: AZURE_STORAGE_CONTAINER")
if not AZURE_STORAGE_KEY:
    raise EnvironmentError("Missing required environment variable: AZURE_STORAGE_KEY")

logger.info(
    "Azure credential loading: account=%s container=%s key_present=%s",
    AZURE_STORAGE_ACCOUNT,
    AZURE_STORAGE_CONTAINER,
    bool(AZURE_STORAGE_KEY),
)

logger.info("Azure credentials loaded (no container scan).")
logger.info(
    "Sampling config: spacing_m=%.1f max_attempts_per_road=%d max_patches_per_road=%d epochs=%d",
    PATCH_SPACING_M,
    MAX_PATCH_ATTEMPTS_PER_ROAD,
    MAX_PATCHES_PER_ROAD,
    EPOCHS,
)

# -----------------------------
# Helpers
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

def segmentize_line(line, segment_length):
    if line.length <= segment_length:
        return [line]
    segments = []
    distance = 0.0
    while distance < line.length:
        end = min(distance + segment_length, line.length)
        start_pt = line.interpolate(distance)
        end_pt = line.interpolate(end)
        if start_pt.distance(end_pt) > 0:
            segments.append(LineString([start_pt, end_pt]))
        distance = end
    return segments

def save_debug_patch(arr, out_path):
    try:
        from PIL import Image
        rgb = arr[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(out_path)
        return True
    except Exception:
        return False

def build_rotated_patch(src, segment, buffer_m):
    coords = list(segment.coords)
    if len(coords) < 2:
        return None
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return None
    angle_deg = math.degrees(math.atan2(dy, dx))
    mid = segment.interpolate(0.5, normalized=True)
    pixel_size = float(src.res[0])

    dst_transform = (
        Affine.translation(mid.x, mid.y)
        * Affine.rotation(angle_deg)
        * Affine.translation(-PATCH_SIZE / 2, -PATCH_SIZE / 2)
        * Affine.scale(pixel_size, -pixel_size)
    )

    dest = np.zeros((4, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for band in range(1, 5):
        reproject(
            source=rasterio.band(src, band),
            destination=dest[band - 1],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear,
        )

    mask = geometry_mask(
        [segment.buffer(buffer_m / 2.0)],
        transform=dst_transform,
        invert=True,
        out_shape=(PATCH_SIZE, PATCH_SIZE),
    )
    dest[:, ~mask] = 0
    rgb_patch = np.moveaxis(np.clip(dest[:3], 0, 255), 0, -1)
    road_mask = mask.astype(np.float32)
    return rgb_patch, road_mask

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

def _patch_cache_meta():
    return {
        "patch_spacing_m": float(PATCH_SPACING_M),
        "max_patch_attempts_per_road": int(MAX_PATCH_ATTEMPTS_PER_ROAD),
        "patch_size_m": float(PATCH_SIZE_M),
        "road_buffer_m": float(ROAD_BUFFER_M),
    }

def _load_patch_cache(path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("meta") != _patch_cache_meta():
            return {}
        roads = payload.get("roads", {})
        if not isinstance(roads, dict):
            return {}
        return {str(k): v for k, v in roads.items()}
    except Exception:
        return {}

def _save_patch_cache(path, road_cache):
    payload = {"meta": _patch_cache_meta(), "roads": road_cache}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

def _disk_patch_cache_meta():
    return {
        "patch_spacing_m": float(PATCH_SPACING_M),
        "max_patch_attempts_per_road": int(MAX_PATCH_ATTEMPTS_PER_ROAD),
        "max_patches_per_road": int(MAX_PATCHES_PER_ROAD),
        "patch_size_px": int(PATCH_SIZE),
        "patch_size_m": float(PATCH_SIZE_M),
        "road_buffer_m": float(ROAD_BUFFER_M),
        "min_road_length_m": float(MIN_ROAD_LENGTH_M),
        "min_patch_variance": float(MIN_PATCH_VARIANCE),
        "use_road_only_pixels": bool(USE_ROAD_ONLY_PIXELS),
        "road_edge_dilation_px": int(ROAD_EDGE_DILATION_PX),
        "seed": int(SEED),
    }

def _write_disk_cache_index(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _load_disk_cache_index(path):
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                if not row.get("path"):
                    continue
                rows.append(
                    {
                        "path": row["path"],
                        "label": int(row.get("label", 0)),
                        "road_id": str(row.get("road_id", "")),
                        "road_fraction": float(row.get("road_fraction", 0.0)),
                        "source": row.get("source", "gold"),
                    }
                )
            return rows
    except Exception:
        return []

def _save_patch_to_disk(patch, out_path):
    arr = patch.astype(np.float16, copy=False)
    np.savez_compressed(out_path, patch=arr)

def _load_patch_from_disk(path):
    with np.load(path) as data:
        return data["patch"].astype(np.float32)

def _process_road(args):
    if len(args) == 8:
        args = (*args, None, 0)
    (
        idx,
        road_id,
        label,
        road_lines,
        tiles,
        roads_crs,
        cache_entry,
        source,
        max_total_patches,
        starting_patch_count,
    ) = args
    X_local, y_local, road_ids_local, road_fractions_local = [], [], [], []
    kept_local = {0: 0, 1: 0}
    discarded_local = {0: 0, 1: 0}
    patches_for_road = 0
    rejected_for_road = 0
    attempts = 0
    reject_notes = []
    preflight_failed = False
    cache_out = []
    patch_rows = []
    if USE_DISK_PATCH_CACHE:
        PATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    patch_count = starting_patch_count

    def _write_patch(patch, road_fraction):
        nonlocal patch_count
        if max_total_patches is not None and patch_count >= max_total_patches:
            return False
        filename = f"{road_id}_{uuid.uuid4().hex}.npz"
        out_path = PATCH_CACHE_DIR / filename
        _save_patch_to_disk(patch, out_path)
        patch_rows.append(
            {
                "path": str(out_path),
                "label": int(label),
                "road_id": str(road_id),
                "road_fraction": float(road_fraction),
                "source": str(source),
            }
        )
        patch_count += 1
        return True

    if cache_entry:
        cached_by_tile = {}
        for entry in cache_entry:
            tile_url = entry.get("tile_url")
            points = entry.get("points", [])
            if tile_url:
                cached_by_tile[tile_url] = points
        for tile_url in tiles:
            points = cached_by_tile.get(tile_url, [])
            if not points:
                continue
            with open_remote_tiff(tile_url) as src:
                for point_entry in points:
                    if patches_for_road >= MAX_PATCHES_PER_ROAD:
                        break
                    if attempts >= MAX_PATCH_ATTEMPTS_PER_ROAD:
                        break
                    attempts += 1
                    dist, x, y, x0, y0, x1, y1 = point_entry
                    point = Point(float(x), float(y))
                    segment = LineString([(float(x0), float(y0)), (float(x1), float(y1))])
                    rgb_patch, road_mask, distance_channel = extract_patch_from_point(
                        src, point, PATCH_SIZE_M, segment
                    )
                    if rgb_patch is None or road_mask is None or distance_channel is None:
                        discarded_local[label] += 1
                        rejected_for_road += 1
                        if len(reject_notes) < 5:
                            reject_notes.append("patch_none")
                        continue
                    road_fraction = float(road_mask.mean())
                    patch_resized = resize_or_pad(rgb_patch, PATCH_SIZE)
                    if float(np.var(patch_resized[:, :, :3])) < MIN_PATCH_VARIANCE:
                        discarded_local[label] += 1
                        rejected_for_road += 1
                        if len(reject_notes) < 5:
                            reject_notes.append("low_variance")
                        continue
                    mask_resized = resize_or_pad(road_mask[..., np.newaxis], PATCH_SIZE)[..., 0]
                    dist_resized = resize_or_pad(distance_channel[..., np.newaxis], PATCH_SIZE)[..., 0]
                    rgb_weight = 0.5 + 0.5 * mask_resized
                    rgb_weighted = patch_resized[:, :, :3] * rgb_weight[..., np.newaxis]
                    patch = np.concatenate(
                        [rgb_weighted, mask_resized[..., np.newaxis], dist_resized[..., np.newaxis]],
                        axis=-1,
                    )
                    if USE_DISK_PATCH_CACHE:
                        if not _write_patch(patch, road_fraction):
                            break
                        X_local.append(patch)
                        y_local.append(label)
                        road_ids_local.append(road_id)
                        road_fractions_local.append(road_fraction)
                    kept_local[label] += 1
                    patches_for_road += 1
                if max_total_patches is not None and patch_count >= max_total_patches:
                    break
            if max_total_patches is not None and patch_count >= max_total_patches:
                break
        return {
            "idx": idx,
            "road_id": road_id,
            "X": X_local,
            "y": y_local,
            "road_ids": road_ids_local,
            "road_fractions": road_fractions_local,
            "kept": kept_local,
            "discarded": discarded_local,
            "patches_for_road": patches_for_road,
            "rejected_for_road": rejected_for_road,
            "reject_notes": reject_notes,
            "preflight_failed": preflight_failed,
            "cache_entry": cache_entry,
            "cache_generated": None,
            "patch_rows": patch_rows,
            "patch_count": patch_count,
        }

    cache_points_by_tile = {}
    for line in road_lines:
        if not line:
            if len(reject_notes) < 5:
                reject_notes.append("geometry_invalid_or_short")
            continue
        preflight_done = False
        for tile_idx, tile_url in enumerate(tiles, start=1):
            with open_remote_tiff(tile_url) as src:
                tile_crs = src.crs
                tile_poly = box(*src.bounds)
                line_proj = gpd.GeoSeries([line], crs=roads_crs).to_crs(tile_crs).iloc[0]
                if not preflight_done:
                    preflight_done = True
                    buffered = line_proj.buffer(PREFLIGHT_BUFFER_M)
                    bbox = buffered.envelope
                    intersection_area = bbox.intersection(tile_poly).area
                    if intersection_area < PREFLIGHT_INTERSECTION_AREA_M2:
                        preflight_failed = True
                        break
                if line_proj.length < MIN_ROAD_LENGTH_M:
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
                        break
                    attempts += 1
                    point = geom.interpolate(dist)
                    seg_start = max(0.0, dist - PATCH_SPACING_M)
                    seg_end = min(geom.length, dist + PATCH_SPACING_M)
                    segment = LineString([geom.interpolate(seg_start), geom.interpolate(seg_end)])
                    cache_points_by_tile.setdefault(tile_url, []).append(
                        [
                            float(dist),
                            float(point.x),
                            float(point.y),
                            float(segment.coords[0][0]),
                            float(segment.coords[0][1]),
                            float(segment.coords[-1][0]),
                            float(segment.coords[-1][1]),
                        ]
                    )
                    rgb_patch, road_mask, distance_channel = extract_patch_from_point(
                        src, point, PATCH_SIZE_M, segment
                    )
                    if rgb_patch is None or road_mask is None or distance_channel is None:
                        discarded_local[label] += 1
                        rejected_for_road += 1
                        if len(reject_notes) < 5:
                            reject_notes.append("patch_none")
                        continue
                    road_fraction = float(road_mask.mean())
                    patch_resized = resize_or_pad(rgb_patch, PATCH_SIZE)
                    if float(np.var(patch_resized[:, :, :3])) < MIN_PATCH_VARIANCE:
                        discarded_local[label] += 1
                        rejected_for_road += 1
                        if len(reject_notes) < 5:
                            reject_notes.append("low_variance")
                        continue
                    mask_resized = resize_or_pad(road_mask[..., np.newaxis], PATCH_SIZE)[..., 0]
                    dist_resized = resize_or_pad(distance_channel[..., np.newaxis], PATCH_SIZE)[..., 0]
                    rgb_weight = 0.5 + 0.5 * mask_resized
                    rgb_weighted = patch_resized[:, :, :3] * rgb_weight[..., np.newaxis]
                    patch = np.concatenate(
                        [rgb_weighted, mask_resized[..., np.newaxis], dist_resized[..., np.newaxis]],
                        axis=-1,
                    )
                    if USE_DISK_PATCH_CACHE:
                        if not _write_patch(patch, road_fraction):
                            break
                        X_local.append(patch)
                        y_local.append(label)
                        road_ids_local.append(road_id)
                        road_fractions_local.append(road_fraction)
                    kept_local[label] += 1
                    patches_for_road += 1
                if max_total_patches is not None and patch_count >= max_total_patches:
                    break
            if preflight_failed or patches_for_road >= MAX_PATCHES_PER_ROAD:
                break
            if max_total_patches is not None and patch_count >= max_total_patches:
                break
        if preflight_failed or patches_for_road >= MAX_PATCHES_PER_ROAD:
            break
        if max_total_patches is not None and patch_count >= max_total_patches:
            break

    for tile_url in tiles:
        points = cache_points_by_tile.get(tile_url)
        if points:
            cache_out.append({"tile_url": tile_url, "points": points})

    return {
        "idx": idx,
        "road_id": road_id,
        "X": X_local,
        "y": y_local,
        "road_ids": road_ids_local,
        "road_fractions": road_fractions_local,
        "kept": kept_local,
        "discarded": discarded_local,
        "patches_for_road": patches_for_road,
        "rejected_for_road": rejected_for_road,
        "reject_notes": reject_notes,
        "preflight_failed": preflight_failed,
        "cache_entry": cache_entry,
        "cache_generated": cache_out,
        "patch_rows": patch_rows,
        "patch_count": patch_count,
    }

# Debug overlay (disabled by default)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.imshow(rgb_patch.astype(np.uint8))
# ax.imshow(road_mask, alpha=0.3, cmap="Reds")
# plt.show()

def offset_segment(segment, offset_m):
    coords = list(segment.coords)
    if len(coords) < 2:
        return None
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return None
    nx, ny = -dy, dx
    scale = offset_m / math.hypot(nx, ny)
    shift_x, shift_y = nx * scale, ny * scale
    start = (x0 + shift_x, y0 + shift_y)
    end = (x1 + shift_x, y1 + shift_y)
    return LineString([start, end])

def balance_roads_equal(X, y, road_ids):
    road_labels = {}
    for lbl, rid in zip(y, road_ids):
        road_labels.setdefault(rid, int(lbl))
    paved_roads = [rid for rid, lbl in road_labels.items() if lbl == 0]
    gravel_roads = [rid for rid, lbl in road_labels.items() if lbl == 1]
    if not paved_roads or not gravel_roads:
        return X, y, road_ids
    target = min(len(paved_roads), len(gravel_roads))
    keep_roads = set(random.sample(paved_roads, target) + random.sample(gravel_roads, target))
    X_bal, y_bal, road_bal = [], [], []
    for x_i, y_i, r_i in zip(X, y, road_ids):
        if r_i in keep_roads:
            X_bal.append(x_i)
            y_bal.append(y_i)
            road_bal.append(r_i)
    return X_bal, y_bal, road_bal

def offset_point_along_normal(line, point, offset_m):
    try:
        if line.length == 0:
            return None
        proj = line.project(point)
        eps = min(1.0, max(0.1, line.length * 0.001))
        p0 = line.interpolate(max(0.0, proj - eps))
        p1 = line.interpolate(min(line.length, proj + eps))
        dx, dy = (p1.x - p0.x), (p1.y - p0.y)
        if dx == 0 and dy == 0:
            return None
        nx, ny = -dy, dx
        scale = offset_m / math.hypot(nx, ny)
        return point.__class__(point.x + nx * scale, point.y + ny * scale)
    except Exception:
        return None

# -----------------------------
# MobileNetV3 backbone
# -----------------------------
def make_cnn(input_shape=(PATCH_SIZE, PATCH_SIZE, 5)):
    inputs = layers.Input(shape=input_shape)
    rgb = inputs[..., :3] * 255.0
    x = mobilenet_v3.preprocess_input(rgb)
    backbone = MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=(PATCH_SIZE, PATCH_SIZE, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model, backbone

# -----------------------------
# Training logic
# -----------------------------
def train_on_tiles():
    logger.info("Script startup. python=%s cwd=%s", sys.version.split()[0], Path.cwd())
    random.seed(SEED)
    np.random.seed(SEED)
    if USE_DISK_PATCH_CACHE:
        logger.info("Disk patch cache enabled. dir=%s index=%s", PATCH_CACHE_DIR, PATCH_CACHE_INDEX)

    train_tiles = []
    val_tiles = []
    test_tiles = []

    logger.info("Tile fetch phase begin.")
    fetch_phase_start = time.time()
    logger.info(
        "Patch config: patch_px=%d, patch_m=%.1f, m_per_px=%.3f",
        PATCH_SIZE,
        PATCH_SIZE_M,
        PATCH_SIZE_M / float(PATCH_SIZE),
    )
    logger.info(
        "Augmentation enabled: yes (random_hflip, random_vflip, random_rot90, jitter_brightness_contrast)"
    )
    logger.info("Early stopping enabled (patience=3).")
    logger.info("Label smoothing enabled (0.05).")
    logger.info("Soft road emphasis applied to RGB channels.")
    if not ROADS_PARQUET.exists():
        sys.exit(f"Missing labels file: {ROADS_PARQUET}")

    if not GOLD_DATASET.exists():
        sys.exit(f"Missing gold dataset: {GOLD_DATASET}")
    gold_df = pd.read_csv(GOLD_DATASET)
    logger.info("Gold dataset loaded. rows=%d", len(gold_df))
    if "osm_id" not in gold_df.columns or "label" not in gold_df.columns:
        sys.exit("goldgpt.csv must include 'osm_id' and 'label' columns.")
    gold_df = gold_df[["osm_id", "label"]].copy()
    gold_df["osm_id"] = gold_df["osm_id"].astype(str)
    gold_df["label"] = gold_df["label"].astype(int)
    paved = gold_df[gold_df["label"] == 0]
    gravel = gold_df[gold_df["label"] == 1]
    logger.info("Gold class counts. paved=%d gravel=%d", len(paved), len(gravel))
    if len(paved) == 0 or len(gravel) == 0:
        sys.exit("Gold dataset missing at least one class; cannot train.")
    sample_start = time.time()
    max_count = max(len(paved), len(gravel))
    min_count = min(len(paved), len(gravel))
    diff_ratio = (max_count - min_count) / max_count if max_count else 0.0
    if diff_ratio > 0.10:
        target = min_count
        logger.info("Balancing classes by downsampling to %d per class.", target)
        paved_sample = paved.sample(n=target, random_state=SEED) if len(paved) > target else paved
        gravel_sample = gravel.sample(n=target, random_state=SEED) if len(gravel) > target else gravel
        logger.info("Class balance close; using all gold roads.")
        paved_sample = paved
        gravel_sample = gravel
    logger.info("Sampling complete. elapsed=%.2fs", time.time() - sample_start)
    gold_sample = pd.concat([paved_sample, gravel_sample], ignore_index=True)
    logger.info("Gold roads selected total=%d", len(gold_sample))
    sample_map = dict(zip(gold_sample["osm_id"], gold_sample["label"]))

    silver_rows = []
    if SILVER_DATASET.exists():
        silver_candidates = {0: [], 1: []}
        with open(SILVER_DATASET, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "osm_id" not in reader.fieldnames or "label" not in reader.fieldnames:
                logger.warning("silver.csv missing osm_id/label columns; skipping silver roads.")
                for row in reader:
                    road_id = str(row.get("osm_id", "")).strip()
                    if not road_id or road_id in sample_map:
                        continue
                    try:
                        label = int(row.get("label", ""))
                    except (TypeError, ValueError):
                        continue
                    if label not in (0, 1):
                        continue
                    silver_candidates[label].append(road_id)
        rng_silver = random.Random(SEED)
        for label in (0, 1):
            rng_silver.shuffle(silver_candidates[label])
        silver_rows = (
            [(rid, 0) for rid in silver_candidates[0][:SILVER_ROADS_PER_CLASS]]
            + [(rid, 1) for rid in silver_candidates[1][:SILVER_ROADS_PER_CLASS]]
        )
        if silver_rows:
            logger.info(
                "Silver candidate pool. paved=%d gravel=%d sampled=%d",
                len(silver_candidates[0]),
                len(silver_candidates[1]),
                len(silver_rows),
            )
        logger.warning("silver.csv not found; skipping silver roads.")

    silver_ids = {rid for rid, _ in silver_rows}
    all_target_ids = set(sample_map.keys()) | {rid for rid, _ in silver_rows}
    if not all_target_ids:
        sys.exit("No gold or silver roads available for sampling.")

    if not NAIP_INDEX.exists():
        sys.exit(f"Missing NAIP index: {NAIP_INDEX}")
    naip_cols = ["id", "tile_id", "tile_url", "signed_url"]
    naip_df = pd.read_csv(NAIP_INDEX, usecols=lambda c: c in naip_cols)
    if "id" not in naip_df.columns or "tile_id" not in naip_df.columns:
        sys.exit("naip_index.csv must include 'id' and 'tile_id' columns.")
    if "tile_url" not in naip_df.columns and "signed_url" not in naip_df.columns:
        sys.exit("naip_index.csv must include 'tile_url' or 'signed_url' columns.")
    naip_df["id"] = naip_df["id"].astype(str)
    naip_df = naip_df[naip_df["id"].isin(all_target_ids)].copy()
    if silver_ids:
        silver_in_naip = int(naip_df["id"].isin(silver_ids).sum())
        logger.info("Silver ids with NAIP tiles: %d/%d", silver_in_naip, len(silver_ids))
    gold_ids = set(sample_map.keys())
    if not gold_ids:
        sys.exit("No gold roads available for sampling.")
    naip_matches = naip_df[naip_df["id"].isin(gold_ids)].copy()
    if naip_matches.empty:
        sys.exit("No NAIP tiles matched sampled gold roads.")
    if "signed_url" in naip_matches.columns:
        naip_matches["tile_url"] = naip_matches["signed_url"].where(
            naip_matches["signed_url"].notna(),
            naip_matches.get("tile_url"),
        )
        naip_matches["tile_url"] = naip_matches["tile_url"]
    if "signed_url" in naip_df.columns:
        naip_df["tile_url"] = naip_df["signed_url"].where(
            naip_df["signed_url"].notna(),
            naip_df.get("tile_url"),
        )
    if naip_matches["tile_url"].isna().any():
        missing_urls = int(naip_matches["tile_url"].isna().sum())
        logger.warning(
            "NAIP index missing tile_url/signed_url for %d matched rows; skipping them.",
            missing_urls,
        )
        naip_matches = naip_matches[naip_matches["tile_url"].notna()].copy()
        if naip_matches.empty:
            sys.exit("No NAIP tiles with valid URLs after filtering missing tile_url/signed_url.")
    road_to_tiles = (
        naip_matches.groupby("id")["tile_url"].apply(list).to_dict()
    )
    logger.info(
        "Matched NAIP tiles. roads=%d tiles=%d",
        len(road_to_tiles),
        naip_matches["tile_id"].nunique(),
    )
    naip_df = naip_df[naip_df["tile_url"].notna()].copy()
    naip_tile_urls_by_id = naip_df.groupby("id")["tile_url"].apply(list).to_dict()

    roads = gpd.read_parquet(ROADS_PARQUET, columns=["id", "geometry"])
    if "id" not in roads.columns:
        sys.exit("Roads parquet missing required 'id' column for gold sampling.")
    roads = roads.copy()
    roads["id"] = roads["id"].astype(str)
    roads = roads[roads["id"].isin(all_target_ids)].copy()
    if silver_ids:
        silver_in_roads = int(roads["id"].isin(silver_ids).sum())
        logger.info("Silver ids in roads parquet: %d/%d", silver_in_roads, len(silver_ids))
    if roads.crs is None:
        logger.warning("Roads CRS is missing; assuming EPSG:4326.")
        roads = roads.set_crs("EPSG:4326")
    logger.info("Road geometries CRS: %s", roads.crs)

    debug_saved = 0
    debug_patches_dir = None
    debug_tiles_dir = None
    if ENABLE_DEBUG_PATCH_SAVES:
        DEBUG_PATCH_DIR.mkdir(parents=True, exist_ok=True)
        debug_patches_dir = BASE_DIR / "output" / "debug_patches"
        debug_tiles_dir = BASE_DIR / "output" / "debug_tiles"
        debug_patches_dir.mkdir(parents=True, exist_ok=True)
        debug_tiles_dir.mkdir(parents=True, exist_ok=True)
    debug_road_id = None
    debug_patch_saved_by_class = {0: 0, 1: 0}
    debug_tile_saved = False

    logger.info("Dataset construction start (gold/silver).")
    if ENABLE_DEBUG_PATCH_SAVES:
        DEBUG_PATCH_DIR.mkdir(parents=True, exist_ok=True)
    extract_start = time.time()

    X, y = [], []
    road_fractions = []
    road_ids = []
    patch_cache_rows = []
    patch_count = 0
    reuse_patch_cache = False
    kept = {0: 0, 1: 0}
    discarded = {0: 0, 1: 0}
    road_source = {rid: "gold" for rid in sample_map}
    cache_updates = {}
    cache_hit_count = 0
    cache_miss_count = 0
    skipped_missing_geometry = 0
    skipped_no_tiles = 0
    skipped_no_patches = 0
    gold_paved_used = 0
    gold_gravel_used = 0
    silver_paved_used = 0
    silver_gravel_used = 0

    roads = roads.set_index("id")
    if USE_DISK_PATCH_CACHE:
        PATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if PATCH_EXTRACT_WORKERS > 1:
            logger.warning("Disk patch cache enabled; forcing single-worker extraction.")
            logger.warning("Set PATCH_EXTRACT_WORKERS=1 to avoid contention.")
        if REUSE_PATCH_CACHE and PATCH_CACHE_INDEX.exists():
            patch_cache_rows = _load_disk_cache_index(PATCH_CACHE_INDEX)
            if patch_cache_rows:
                reuse_patch_cache = True
                logger.info(
                    "Reusing disk patch cache. rows=%d index=%s",
                    len(patch_cache_rows),
                    PATCH_CACHE_INDEX,
                )
    road_cache = _load_patch_cache(PATCH_COORDS_CACHE)
    if road_cache:
        logger.info("Loaded patch coordinate cache: roads=%d", len(road_cache))

    if reuse_patch_cache:
        total_patches = len(patch_cache_rows)
        y = [row["label"] for row in patch_cache_rows]
        road_ids = [row["road_id"] for row in patch_cache_rows]
        road_fractions = [row["road_fraction"] for row in patch_cache_rows]
        patch_paths = [row["path"] for row in patch_cache_rows]
        logger.info(
            "Patch extraction skipped. total_patches=%d elapsed=%.2fs",
            total_patches,
            time.time() - extract_start,
        )

    if not reuse_patch_cache:
        gold_ids = list(sample_map.keys())
        total_roads = len(gold_ids)
        tasks = []
        for idx, road_id in enumerate(gold_ids, start=1):
            if road_id not in roads.index:
                skipped_missing_geometry += 1
                logger.warning("Road %s missing from roads dataset; skipping.", road_id)
                continue
            tiles = road_to_tiles.get(road_id, [])
            if not tiles:
                skipped_no_tiles += 1
                logger.warning("No tiles found for road %s; skipping.", road_id)
                continue
            road_rows = roads.loc[road_id]
            if isinstance(road_rows, pd.Series):
                road_rows = [road_rows]
            else:
                road_rows = [row for _, row in road_rows.iterrows()]
            road_lines = [longest_linestring(row.geometry) for row in road_rows]
            if not any(road_lines):
                skipped_missing_geometry += 1
                logger.warning("Road %s missing valid geometry; skipping.", road_id)
                continue
            cache_entry = road_cache.get(road_id)
            if cache_entry:
                cache_hit_count += 1
                logger.info("cache hit for road %s", road_id)
            else:
                cache_miss_count += 1
                logger.info("cache miss for road %s -> building", road_id)
            tasks.append(
                (
                    idx,
                    road_id,
                    int(sample_map[road_id]),
                    road_lines,
                    tiles,
                    roads.crs,
                    cache_entry,
                    "gold",
                )
            )
    
        results_by_idx = {}
        use_parallel = PATCH_EXTRACT_WORKERS > 1
        if USE_DISK_PATCH_CACHE:
            use_parallel = False
        if use_parallel:
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=PATCH_EXTRACT_WORKERS
                ) as executor:
                    futures = {executor.submit(_process_road, task): task[0] for task in tasks}
                    for fut in concurrent.futures.as_completed(futures):
                        res = fut.result()
                        results_by_idx[res["idx"]] = res
            except Exception as exc:
                logger.warning("Parallel extraction failed; falling back to single-threaded: %s", exc)
                results_by_idx = {}
                use_parallel = False
    
        if not use_parallel:
            for task in tasks:
                res = _process_road(
                    (
                        *task,
                        MAX_TOTAL_PATCHES if USE_DISK_PATCH_CACHE else None,
                        patch_count,
                    )
                )
                results_by_idx[res["idx"]] = res
                if USE_DISK_PATCH_CACHE:
                    patch_count = res["patch_count"]
    
        for idx in sorted(results_by_idx):
            res = results_by_idx[idx]
            road_id = res["road_id"]
            label = int(sample_map.get(road_id, 0))
            logger.info("Processing road %d/%d (osm_id=%s)", idx, total_roads, road_id)
            if res["preflight_failed"]:
                logger.info("Road %s skipped by preflight check.", road_id)
            if res["patches_for_road"] == 0 and not res["preflight_failed"]:
                logger.warning("No valid patches for road %s; skipping.", road_id)
                if res["reject_notes"]:
                    logger.warning(
                        "Patch rejection sample for road %s: %s",
                        road_id,
                        ", ".join(res["reject_notes"]),
                    )
            else:
                logger.info("Extracted %d patches for road %s", res["patches_for_road"], road_id)
            if road_id == debug_road_id and res["patches_for_road"] == 0:
                logger.warning("Debug road %s produced zero patches.", road_id)
            if idx % 10 == 0:
                logger.info(
                    "Road progress %d/%d: kept=%d rejected=%d",
                    idx,
                    total_roads,
                    res["patches_for_road"],
                    res["rejected_for_road"],
                )
            if idx == 10 and (kept[0] + kept[1]) == 0:
                logger.warning("No patches after 10 roads; continuing run.")
            if res["patches_for_road"] == 0 or res["preflight_failed"]:
                skipped_no_patches += 1
                if res["cache_generated"] is not None:
                    cache_updates[road_id] = res["cache_generated"]
                continue
            if label == 0:
                gold_paved_used += 1
            else:
                gold_gravel_used += 1
            if USE_DISK_PATCH_CACHE:
                patch_cache_rows.extend(res["patch_rows"])
                patch_count = res["patch_count"]
            else:
                X.extend(res["X"])
                y.extend(res["y"])
                road_ids.extend(res["road_ids"])
                road_fractions.extend(res["road_fractions"])
            kept[0] += res["kept"][0]
            kept[1] += res["kept"][1]
            discarded[0] += res["discarded"][0]
            discarded[1] += res["discarded"][1]
            if res["cache_generated"] is not None:
                cache_updates[road_id] = res["cache_generated"]
            if USE_DISK_PATCH_CACHE and MAX_TOTAL_PATCHES is not None and patch_count >= MAX_TOTAL_PATCHES:
                logger.warning("Reached MAX_TOTAL_PATCHES=%d; stopping further extraction.", MAX_TOTAL_PATCHES)
                break
    
        if silver_rows:
            silver_seen = 0
            for road_id, label in silver_rows:
                if (
                    silver_paved_used >= SILVER_ROADS_PER_CLASS
                    and silver_gravel_used >= SILVER_ROADS_PER_CLASS
                ):
                    break
                if label == 0 and silver_paved_used >= SILVER_ROADS_PER_CLASS:
                    continue
                if label == 1 and silver_gravel_used >= SILVER_ROADS_PER_CLASS:
                    continue
                if road_id not in roads.index:
                    skipped_missing_geometry += 1
                    continue
                tiles = naip_tile_urls_by_id.get(road_id, [])
                if not tiles:
                    skipped_no_tiles += 1
                    continue
                road_rows = roads.loc[road_id]
                if isinstance(road_rows, pd.Series):
                    road_rows = [road_rows]
                else:
                    road_rows = [row for _, row in road_rows.iterrows()]
                road_lines = [longest_linestring(row.geometry) for row in road_rows]
                if not any(road_lines):
                    skipped_missing_geometry += 1
                    continue
                cache_entry = road_cache.get(road_id)
                if cache_entry:
                    cache_hit_count += 1
                    logger.info("cache hit for road %s", road_id)
                else:
                    cache_miss_count += 1
                    logger.info("cache miss for road %s -> building", road_id)
                silver_seen += 1
                res = _process_road(
                    (
                        silver_seen,
                        road_id,
                        int(label),
                        road_lines,
                        tiles,
                        roads.crs,
                        cache_entry,
                        "silver",
                        MAX_TOTAL_PATCHES if USE_DISK_PATCH_CACHE else None,
                        patch_count,
                    )
                )
                logger.info("Processing silver road %d (osm_id=%s)", silver_seen, road_id)
                if res["preflight_failed"]:
                    logger.info("Road %s skipped by preflight check.", road_id)
                if res["patches_for_road"] == 0 and not res["preflight_failed"]:
                    logger.warning("No valid patches for road %s; skipping.", road_id)
                    if res["reject_notes"]:
                        logger.warning(
                            "Patch rejection sample for road %s: %s",
                            road_id,
                            ", ".join(res["reject_notes"]),
                        )
                else:
                    logger.info("Extracted %d patches for road %s", res["patches_for_road"], road_id)
                if road_id == debug_road_id and res["patches_for_road"] == 0:
                    logger.warning("Debug road %s produced zero patches.", road_id)
                if res["patches_for_road"] == 0 or res["preflight_failed"]:
                    skipped_no_patches += 1
                    if res["cache_generated"] is not None:
                        cache_updates[road_id] = res["cache_generated"]
                    continue
                if label == 0:
                    silver_paved_used += 1
                else:
                    silver_gravel_used += 1
                sample_map[road_id] = int(label)
                road_source[road_id] = "silver"
                if USE_DISK_PATCH_CACHE:
                    patch_cache_rows.extend(res["patch_rows"])
                    patch_count = res["patch_count"]
                else:
                    X.extend(res["X"])
                    y.extend(res["y"])
                    road_ids.extend(res["road_ids"])
                    road_fractions.extend(res["road_fractions"])
                kept[0] += res["kept"][0]
                kept[1] += res["kept"][1]
                discarded[0] += res["discarded"][0]
                discarded[1] += res["discarded"][1]
                if res["cache_generated"] is not None:
                    cache_updates[road_id] = res["cache_generated"]
                if USE_DISK_PATCH_CACHE and MAX_TOTAL_PATCHES is not None and patch_count >= MAX_TOTAL_PATCHES:
                    logger.warning("Reached MAX_TOTAL_PATCHES=%d; stopping further extraction.", MAX_TOTAL_PATCHES)
                    break
            if (
                silver_paved_used < SILVER_ROADS_PER_CLASS
                or silver_gravel_used < SILVER_ROADS_PER_CLASS
            ):
                raise RuntimeError(
                    f"silver roads exhausted: target={SILVER_ROADS_PER_CLASS} "
                    f"accepted_paved={silver_paved_used} accepted_gravel={silver_gravel_used}"
                )
    
        logger.info(
            "Selected valid roads. gold_paved=%d gold_gravel=%d silver_paved=%d silver_gravel=%d",
            gold_paved_used,
            gold_gravel_used,
            silver_paved_used,
            silver_gravel_used,
        )
        logger.info(
            "Selection skips. missing_geometry=%d no_tiles=%d no_patches=%d cache_hits=%d cache_misses=%d",
            skipped_missing_geometry,
            skipped_no_tiles,
            skipped_no_patches,
            cache_hit_count,
            cache_miss_count,
        )
    
        if cache_updates:
            road_cache.update(cache_updates)
            try:
                _save_patch_cache(PATCH_COORDS_CACHE, road_cache)
                logger.info("Patch coordinate cache saved to %s", PATCH_COORDS_CACHE)
            except Exception as exc:
                logger.warning("Failed to save patch coordinate cache: %s", exc)
    
        if not reuse_patch_cache:
            total_patches = len(patch_cache_rows) if USE_DISK_PATCH_CACHE else len(X)
            logger.info(
                "Patch extraction end. total_patches=%d elapsed=%.2fs",
                total_patches,
                time.time() - extract_start,
            )
            if USE_DISK_PATCH_CACHE:
                _write_disk_cache_index(PATCH_CACHE_INDEX, patch_cache_rows)
                logger.info("Disk patch cache index saved to %s", PATCH_CACHE_INDEX)
                y = [row["label"] for row in patch_cache_rows]
                road_ids = [row["road_id"] for row in patch_cache_rows]
                road_fractions = [row["road_fraction"] for row in patch_cache_rows]
                patch_paths = [row["path"] for row in patch_cache_rows]
        if road_ids:
            per_road_counts = Counter(road_ids)
            avg_patches = len(road_ids) / len(per_road_counts)
            min_patches = min(per_road_counts.values())
            max_patches = max(per_road_counts.values())
            logger.info(
                "Patches per road after capping. avg=%.2f min=%d max=%d",
                avg_patches,
                min_patches,
                max_patches,
            )
    logger.info("Dataset construction end (gold/silver).")

    if USE_DISK_PATCH_CACHE:
        if total_patches == 0:
            sys.exit("No patches kept in gold/silver mode.")
    else:
        if not X:
            sys.exit("No patches kept in gold/silver mode.")

    logger.info("Dataset construction start (road-level split).")
    roads_with_patches = {}
    for rid, lbl in zip(road_ids, y):
        roads_with_patches.setdefault(rid, int(lbl))
    gravel_roads = [rid for rid, lbl in roads_with_patches.items() if lbl == 1]
    paved_roads = [rid for rid, lbl in roads_with_patches.items() if lbl == 0]
    if not gravel_roads or not paved_roads:
        sys.exit("Not enough roads with patches to build balanced split.")

    rng = random.Random(SEED)
    rng.shuffle(gravel_roads)
    rng.shuffle(paved_roads)
    val_roads = None
    if VAL_ROADS_FILE.exists():
        try:
            with open(VAL_ROADS_FILE, "r", encoding="utf-8") as handle:
                val_data = json.load(handle)
            val_roads = {str(rid) for rid in val_data}
            val_roads = {rid for rid in val_roads if rid in roads_with_patches}
            logger.info("Validation road set loaded from disk: %s", VAL_ROADS_FILE)
        except Exception as exc:
            logger.warning("Failed to load validation road set from %s: %s", VAL_ROADS_FILE, exc)
            val_roads = None
    if val_roads is None:
        gravel_split = int(len(gravel_roads) * 0.8)
        paved_split = int(len(paved_roads) * 0.8)
        train_roads = set(gravel_roads[:gravel_split] + paved_roads[:paved_split])
        val_roads = set(gravel_roads[gravel_split:] + paved_roads[paved_split:])
        try:
            with open(VAL_ROADS_FILE, "w", encoding="utf-8") as handle:
                json.dump(sorted(val_roads), handle)
            logger.info("Validation road set saved to disk: %s", VAL_ROADS_FILE)
        except Exception as exc:
            logger.warning("Failed to save validation road set to %s: %s", VAL_ROADS_FILE, exc)
        train_roads = set(roads_with_patches.keys()) - val_roads
    train_gravel = sum(1 for rid in train_roads if roads_with_patches[rid] == 1)
    train_paved = sum(1 for rid in train_roads if roads_with_patches[rid] == 0)
    val_gravel = sum(1 for rid in val_roads if roads_with_patches[rid] == 1)
    val_paved = sum(1 for rid in val_roads if roads_with_patches[rid] == 0)
    logger.info(
        "Road split complete. train_gravel=%d train_paved=%d val_gravel=%d val_paved=%d",
        train_gravel,
        train_paved,
        val_gravel,
        val_paved,
    )
    logger.info(
        "Roads per class (train/val). paved=%d/%d gravel=%d/%d",
        train_paved,
        val_paved,
        train_gravel,
        val_gravel,
    )

    eval_roads = None
    if EVAL_ROADS_FILE.exists():
        try:
            with open(EVAL_ROADS_FILE, "r", encoding="utf-8") as handle:
                eval_data = json.load(handle)
            eval_roads = {str(rid) for rid in eval_data}
            eval_roads = {rid for rid in eval_roads if rid in roads_with_patches}
            logger.info("Eval road set loaded from disk: %s", EVAL_ROADS_FILE)
        except Exception as exc:
            logger.warning("Failed to load eval road set from %s: %s", EVAL_ROADS_FILE, exc)
            eval_roads = None
    if eval_roads is None:
        eval_candidates = set(roads_with_patches.keys()) - train_roads - val_roads
        if not eval_candidates:
            eval_candidates = set(roads_with_patches.keys())
        eval_gravel = [rid for rid in eval_candidates if roads_with_patches[rid] == 1]
        eval_paved = [rid for rid in eval_candidates if roads_with_patches[rid] == 0]
        rng_eval = random.Random(SEED)
        rng_eval.shuffle(eval_gravel)
        rng_eval.shuffle(eval_paved)
        eval_take_gravel = min(len(eval_gravel), EVAL_ROADS_PER_CLASS)
        eval_take_paved = min(len(eval_paved), EVAL_ROADS_PER_CLASS)
        eval_roads = set(eval_gravel[:eval_take_gravel] + eval_paved[:eval_take_paved])
        try:
            with open(EVAL_ROADS_FILE, "w", encoding="utf-8") as handle:
                json.dump(sorted(eval_roads), handle)
            logger.info("Eval road set saved to disk: %s", EVAL_ROADS_FILE)
        except Exception as exc:
            logger.warning("Failed to save eval road set to %s: %s", EVAL_ROADS_FILE, exc)
    eval_gravel_used = sum(1 for rid in eval_roads if roads_with_patches.get(rid) == 1)
    eval_paved_used = sum(1 for rid in eval_roads if roads_with_patches.get(rid) == 0)
    logger.info(
        "Eval roads per class. paved=%d gravel=%d",
        eval_paved_used,
        eval_gravel_used,
    )

    train_idx = [i for i, rid in enumerate(road_ids) if rid in train_roads]
    val_idx = [i for i, rid in enumerate(road_ids) if rid in val_roads]
    eval_idx = [i for i, rid in enumerate(road_ids) if rid in eval_roads]
    y = np.array(y, dtype=np.int32)
    road_ids = np.array(road_ids, dtype=object)
    road_fractions = np.array(road_fractions, dtype=np.float32)
    if USE_DISK_PATCH_CACHE:
        patch_paths = np.array(patch_paths, dtype=object)
        train_paths, val_paths = patch_paths[train_idx], patch_paths[val_idx]
        eval_paths = patch_paths[eval_idx]
        X = np.array(X, dtype=np.float32)
        X[..., :3] = X[..., :3] / 255.0
        X_train, X_val = X[train_idx], X[val_idx]
        X_eval = X[eval_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    road_train = road_ids[train_idx]
    road_val = road_ids[val_idx]
    y_eval = y[eval_idx]
    road_eval = road_ids[eval_idx]
    train_weights = road_fractions[train_idx]
    val_weights = road_fractions[val_idx]
    eval_weights = road_fractions[eval_idx]

    def augment_training_data(X_arr):
        rng_local = np.random.default_rng(SEED)
        X_out = X_arr.copy()
        for idx in range(X_out.shape[0]):
            patch = X_out[idx]
            if rng_local.random() < 0.5:
                patch = patch[:, ::-1, :]
            if rng_local.random() < 0.5:
                patch = patch[::-1, :, :]
            rot_k = int(rng_local.integers(0, 4))
            if rot_k:
                patch = np.rot90(patch, k=rot_k, axes=(0, 1))
            brightness = float(rng_local.uniform(0.95, 1.05))
            contrast = float(rng_local.uniform(0.95, 1.05))
            patch = np.clip((patch - 0.5) * contrast + 0.5, 0.0, 1.0)
            patch = np.clip(patch * brightness, 0.0, 1.0)
            X_out[idx] = patch
        return X_out

    def _load_patch_tensor(path):
        def _py_load(p):
            p = p.numpy().decode("utf-8")
            arr = _load_patch_from_disk(p)
            arr[..., :3] = arr[..., :3] / 255.0
            return arr
        patch = tf.py_function(_py_load, [path], Tout=tf.float32)
        patch.set_shape((PATCH_SIZE, PATCH_SIZE, 5))
        return patch

    def _load_with_weight(path, label, weight):
        patch = _load_patch_tensor(path)
        return patch, label, weight

    def _load_no_weight(path, label):
        patch = _load_patch_tensor(path)
        return patch, label

    def _augment(patch, label, weight):
        patch = tf.image.random_flip_left_right(patch)
        patch = tf.image.random_flip_up_down(patch)
        rot = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        patch = tf.image.rot90(patch, k=rot)
        patch = tf.image.random_brightness(patch, max_delta=0.05)
        patch = tf.image.random_contrast(patch, lower=0.95, upper=1.05)
        patch = tf.clip_by_value(patch, 0.0, 1.0)
        return patch, label, weight

    if USE_DISK_PATCH_CACHE:
        train_paths_curr = train_paths[train_weights >= 0.15]
        y_train_curr = y_train[train_weights >= 0.15]
        w_train_curr = train_weights[train_weights >= 0.15]
        if train_paths_curr.size == 0:
            train_paths_curr = train_paths
            y_train_curr = y_train
            w_train_curr = train_weights
        ds_train_curr = tf.data.Dataset.from_tensor_slices(
            (train_paths_curr, y_train_curr, w_train_curr)
        )
        ds_train_curr = ds_train_curr.shuffle(min(10000, len(train_paths_curr)))
        ds_train_curr = ds_train_curr.map(
            _load_with_weight, num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_train_curr = ds_train_curr.map(
            _augment, num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_train_curr = ds_train_curr.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        ds_train = tf.data.Dataset.from_tensor_slices(
            (train_paths, y_train, train_weights)
        )
        ds_train = ds_train.shuffle(min(10000, len(train_paths)))
        ds_train = ds_train.map(_load_with_weight, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        ds_val = tf.data.Dataset.from_tensor_slices((val_paths, y_val))
        ds_val = ds_val.map(_load_no_weight, num_parallel_calls=tf.data.AUTOTUNE)
        ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        X_train = augment_training_data(X_train)
    logger.info("Dataset construction end (road-level split).")

    train_roads_used = len(set(road_train))
    val_roads_used = len(set(road_val))
    patches_per_road = len(road_ids) / len(set(road_ids)) if road_ids.size else 0.0
    logger.info(
        "Roads used. train=%d val=%d avg_patches_per_road=%.2f",
        train_roads_used,
        val_roads_used,
        patches_per_road,
    )

    logger.info(
        "Patches kept gravel=%d discarded=%d paved=%d discarded=%d",
        kept[1],
        discarded[1],
        kept[0],
        discarded[0],
    )

    train_counts = {0: int((y_train == 0).sum()), 1: int((y_train == 1).sum())}
    val_counts = {0: int((y_val == 0).sum()), 1: int((y_val == 1).sum())}
    logger.info(
        "Patch counts (train): %s | (val): %s",
        train_counts,
        val_counts,
    )
    logger.info(
        "Total patches generated (train/val). train=%d val=%d",
        len(train_idx),
        len(val_idx),
    )
    logger.info("Model initialization.")
    model, backbone = make_cnn()
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    logger.info("Training start.")
    train_start = time.time()
    curriculum_epochs = HEAD_EPOCHS
    if not USE_DISK_PATCH_CACHE:
        curriculum_mask = train_weights >= 0.15
        X_train_curr = X_train[curriculum_mask]
        y_train_curr = y_train[curriculum_mask]
        w_train_curr = train_weights[curriculum_mask]
        if X_train_curr.size == 0:
            X_train_curr = X_train
            y_train_curr = y_train
            w_train_curr = train_weights
    avg_rf_curr = float(w_train_curr.mean()) if w_train_curr.size else 0.0
    avg_rf_full = float(train_weights.mean()) if train_weights.size else 0.0

    def _epoch_log_callback(avg_road_fraction, avg_loss_weight):
        return callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logger.info(
                "Epoch %d end. loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f train_patches=%s val_patches=%s",
                epoch + 1,
                float(logs.get("loss", 0.0)),
                float(logs.get("accuracy", 0.0)),
                float(logs.get("val_loss", 0.0)),
                float(logs.get("val_accuracy", 0.0)),
                train_counts,
                val_counts,
            )
        )

    def _weight_log_callback(avg_road_fraction, avg_loss_weight):
        return callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logger.info(
                "Epoch %d stats. avg_road_fraction_per_batch=%.4f avg_loss_weight=%.4f",
                epoch + 1,
                avg_road_fraction,
                avg_loss_weight,
            )
        )

    if USE_DISK_PATCH_CACHE:
        model.fit(
            ds_train_curr,
            validation_data=ds_val,
            epochs=curriculum_epochs,
            verbose=1,
            callbacks=[
                _epoch_log_callback(avg_rf_curr, avg_rf_curr),
                _weight_log_callback(avg_rf_curr, avg_rf_curr),
            ],
        )
        model.fit(
            X_train_curr,
            y_train_curr,
            validation_data=(X_val, y_val),
            epochs=curriculum_epochs,
            batch_size=BATCH_SIZE,
            verbose=1,
            sample_weight=w_train_curr,
            callbacks=[
                _epoch_log_callback(avg_rf_curr, avg_rf_curr),
                _weight_log_callback(avg_rf_curr, avg_rf_curr),
            ],
        )

    total_layers = len(backbone.layers)
    fine_tune_start = int(total_layers * 0.8)
    for layer in backbone.layers[:fine_tune_start]:
        layer.trainable = False
    for layer in backbone.layers[fine_tune_start:]:
        layer.trainable = True
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    logger.info(
        "Fine-tuning MobileNetV3 layers. trainable=%d total=%d",
        total_layers - fine_tune_start,
        total_layers,
    )

    if USE_DISK_PATCH_CACHE:
        model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=EPOCHS,
            initial_epoch=curriculum_epochs,
            verbose=1,
            callbacks=[
                early_stop,
                _epoch_log_callback(avg_rf_full, avg_rf_full),
                _weight_log_callback(avg_rf_full, avg_rf_full),
            ],
        )
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            initial_epoch=curriculum_epochs,
            batch_size=BATCH_SIZE,
            verbose=1,
            sample_weight=train_weights,
            callbacks=[
                early_stop,
                _epoch_log_callback(avg_rf_full, avg_rf_full),
                _weight_log_callback(avg_rf_full, avg_rf_full),
            ],
        )
    logger.info("Training complete. elapsed=%.2fs", time.time() - train_start)
    model.save(MODEL_OUT)
    logger.info("Model saved to %s", MODEL_OUT)

    logger.info("Evaluation start.")
    if USE_DISK_PATCH_CACHE:
        preds = model.predict(ds_val, verbose=0).ravel()
        preds = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).ravel()
    patch_pred = (preds >= 0.5).astype(int)
    patch_acc = float((patch_pred == y_val).mean()) if len(y_val) else 0.0

    road_probs = {}
    road_weights = {}
    road_labels = {}
    road_sources = {}
    for prob, rid, true_label, weight in zip(preds, road_val, y_val, val_weights):
        road_probs.setdefault(rid, []).append(float(prob))
        road_weights.setdefault(rid, []).append(float(weight))
        road_labels.setdefault(rid, int(true_label))
        road_sources.setdefault(rid, road_source.get(rid, "gold"))
    road_pred = {}
    road_pred_weighted = {}
    per_road_counts = []
    per_road_counts_trimmed = []
    for rid, probs in road_probs.items():
        weights = road_weights.get(rid, [])
        count = len(probs)
        per_road_counts.append(count)
        if count > 40:
            keep_count = max(1, int(round(count * 0.6)))
            ranked = sorted(
                zip(probs, weights),
                key=lambda item: abs(item[0] - 0.5),
                reverse=True,
            )[:keep_count]
            probs = [p for p, _ in ranked]
            weights = [w for _, w in ranked]
        per_road_counts_trimmed.append(len(probs))
        road_weight = GOLD_ROAD_WEIGHT if road_sources.get(rid) == "gold" else SILVER_ROAD_WEIGHT
        road_pred[rid] = (sum(probs) / len(probs) if probs else 0.0) * road_weight
        clipped = [float(np.clip(w, 0.2, 1.0)) for w in weights] if weights else []
        weight_sum = sum(clipped)
        road_pred_weighted[rid] = (
            sum(p * w for p, w in zip(probs, clipped)) / weight_sum
            if weight_sum
            else 0.0
        ) * road_weight
    if per_road_counts:
        logger.info(
            "Avg patches per road (pre/post trim): %.2f / %.2f",
            sum(per_road_counts) / len(per_road_counts),
            sum(per_road_counts_trimmed) / len(per_road_counts_trimmed),
        )
    road_pred_bin = {rid: int(prob >= 0.5) for rid, prob in road_pred.items()}
    road_pred_weighted_bin = {
        rid: int(prob >= 0.5) for rid, prob in road_pred_weighted.items()
    }
    road_true = [road_labels[rid] for rid in road_labels]
    road_est = [road_pred_bin[rid] for rid in road_labels]
    road_est_weighted = [road_pred_weighted_bin[rid] for rid in road_labels]
    road_acc = float(
        sum(int(a == b) for a, b in zip(road_true, road_est)) / len(road_true)
    ) if road_true else 0.0
    road_acc_weighted = float(
        sum(int(a == b) for a, b in zip(road_true, road_est_weighted)) / len(road_true)
    ) if road_true else 0.0
    tn = sum(1 for a, b in zip(road_true, road_est) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(road_true, road_est) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(road_true, road_est) if a == 1 and b == 0)
    tp = sum(1 for a, b in zip(road_true, road_est) if a == 1 and b == 1)
    logger.info("Patch-level accuracy: %.4f", patch_acc)
    logger.info("Road-level accuracy (unweighted): %.4f", road_acc)
    logger.info("Road-level accuracy (weighted): %.4f", road_acc_weighted)
    logger.info("Road-level confusion matrix (tn, fp, fn, tp): %d, %d, %d, %d", tn, fp, fn, tp)
    for label_name, pred_map in (
        ("unweighted", road_pred_bin),
        ("weighted", road_pred_weighted_bin),
    ):
        gold_ids = [rid for rid, src in road_sources.items() if src == "gold"]
        silver_ids = [rid for rid, src in road_sources.items() if src == "silver"]
        if gold_ids:
            gold_acc = sum(
                int(road_labels[rid] == pred_map[rid]) for rid in gold_ids
            ) / len(gold_ids)
            logger.info("Road-level accuracy (%s, gold-only): %.4f", label_name, gold_acc)
        if silver_ids:
            silver_acc = sum(
                int(road_labels[rid] == pred_map[rid]) for rid in silver_ids
            ) / len(silver_ids)
            logger.info("Road-level accuracy (%s, silver-only): %.4f", label_name, silver_acc)

    eval_size = len(eval_idx)
    if eval_size:
        logger.info(
            "Eval roads per class. paved=%d gravel=%d",
            eval_paved_used,
            eval_gravel_used,
        )
        if USE_DISK_PATCH_CACHE:
            ds_eval = tf.data.Dataset.from_tensor_slices((eval_paths, y_eval))
            ds_eval = ds_eval.map(_load_no_weight, num_parallel_calls=tf.data.AUTOTUNE)
            ds_eval = ds_eval.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            eval_preds = model.predict(ds_eval, verbose=0).ravel()
            eval_preds = model.predict(X_eval, batch_size=BATCH_SIZE, verbose=0).ravel()
        eval_patch_acc = float(((eval_preds >= 0.5).astype(int) == y_eval).mean())
        eval_road_probs = {}
        eval_road_labels = {}
        eval_road_sources = {}
        eval_road_counts = []
        eval_road_counts_trimmed = []
        for prob, rid, true_label in zip(eval_preds, road_eval, y_eval):
            eval_road_probs.setdefault(rid, []).append(float(prob))
            eval_road_labels.setdefault(rid, int(true_label))
            eval_road_sources.setdefault(rid, road_source.get(rid, "gold"))
        eval_road_pred = {}
        for rid, probs in eval_road_probs.items():
            count = len(probs)
            eval_road_counts.append(count)
            if count > 40:
                keep_count = max(1, int(round(count * 0.6)))
                ranked = sorted(
                    probs,
                    key=lambda p: abs(p - 0.5),
                    reverse=True,
                )[:keep_count]
                probs = ranked
            eval_road_counts_trimmed.append(len(probs))
            road_weight = (
                GOLD_ROAD_WEIGHT if eval_road_sources.get(rid) == "gold" else SILVER_ROAD_WEIGHT
            )
            eval_road_pred[rid] = (sum(probs) / len(probs) if probs else 0.0) * road_weight
        if eval_road_counts:
            logger.info(
                "Eval avg patches per road (pre/post trim): %.2f / %.2f",
                sum(eval_road_counts) / len(eval_road_counts),
                sum(eval_road_counts_trimmed) / len(eval_road_counts_trimmed),
            )
        eval_road_pred_bin = {rid: int(prob >= 0.5) for rid, prob in eval_road_pred.items()}
        eval_road_true = [eval_road_labels[rid] for rid in eval_road_labels]
        eval_road_est = [eval_road_pred_bin[rid] for rid in eval_road_labels]
        eval_road_acc = float(
            sum(int(a == b) for a, b in zip(eval_road_true, eval_road_est)) / len(eval_road_true)
        ) if eval_road_true else 0.0
        eval_tn = sum(1 for a, b in zip(eval_road_true, eval_road_est) if a == 0 and b == 0)
        eval_fp = sum(1 for a, b in zip(eval_road_true, eval_road_est) if a == 0 and b == 1)
        eval_fn = sum(1 for a, b in zip(eval_road_true, eval_road_est) if a == 1 and b == 0)
        eval_tp = sum(1 for a, b in zip(eval_road_true, eval_road_est) if a == 1 and b == 1)
        logger.info("Eval patch-level accuracy: %.4f", eval_patch_acc)
        logger.info("Eval road-level accuracy: %.4f", eval_road_acc)
        logger.info(
            "Eval road-level confusion matrix (tn, fp, fn, tp): %d, %d, %d, %d",
            eval_tn,
            eval_fp,
            eval_fn,
            eval_tp,
        )
        eval_gold_ids = [rid for rid, src in eval_road_sources.items() if src == "gold"]
        eval_silver_ids = [rid for rid, src in eval_road_sources.items() if src == "silver"]
        if eval_gold_ids:
            eval_gold_acc = sum(
                int(eval_road_labels[rid] == eval_road_pred_bin[rid]) for rid in eval_gold_ids
            ) / len(eval_gold_ids)
            logger.info("Eval road-level accuracy (gold-only): %.4f", eval_gold_acc)
        if eval_silver_ids:
            eval_silver_acc = sum(
                int(eval_road_labels[rid] == eval_road_pred_bin[rid]) for rid in eval_silver_ids
            ) / len(eval_silver_ids)
            logger.info("Eval road-level accuracy (silver-only): %.4f", eval_silver_acc)
        logger.info("Eval set empty; skipping eval metrics.")
    logger.info("Evaluation end.")

    return

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    try:
        train_on_tiles()
    except Exception:
        logger.error("Training failed with exception.")
        logger.error(traceback.format_exc())
        raise
