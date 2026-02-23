import os, io, sys, math, random, logging, traceback, json
from collections import Counter
from pathlib import Path
import numpy as np
import time
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, LineString
from shapely.errors import TopologicalError
from tensorflow.keras import layers, models, callbacks, losses
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
NAIP_INDEX = BASE_DIR / "output" / "naip_index.csv"
VAL_ROADS_FILE = BASE_DIR / "output" / "val_roads.json"

BUFFER_METERS = 10.0
SEGMENT_LENGTH_M = 5.0
MIN_ROAD_LENGTH_M = 20.0
PATCH_SIZE = 96
MIN_MASK_COVERAGE = 0.0
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
EPOCHS = 7
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
ROAD_BUFFER_M = 5.0
PATCH_SPACING_M = 3.0
PATCH_SIZE_M = 16.0
MIN_ROAD_PIXEL_RATIO = 0.60
DEBUG_SAVE_PER_CLASS = 50
GOLD_ROADS_PER_CLASS = 75
TILE_FETCH_LOG_EVERY = 25
MAX_PATCHES_PER_ROAD = 150
MIN_PATCH_VARIANCE = 50.0
MAX_PATCH_ATTEMPTS_PER_ROAD = 120
PREFLIGHT_BUFFER_M = 5.0
PREFLIGHT_INTERSECTION_AREA_M2 = 1.0

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
    distance_channel = _distance_channel(window_transform, road_mask.shape, segment, size_m)
    if distance_channel is None:
        return None, None, None
    return rgb_patch, road_mask, distance_channel

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
# Simple CNN
# -----------------------------
def make_cnn(input_shape=(PATCH_SIZE, PATCH_SIZE, 5)):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss=losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy'],
    )
    return model

# -----------------------------
# Training logic
# -----------------------------
def train_on_tiles():
    logger.info("Script startup. python=%s cwd=%s", sys.version.split()[0], Path.cwd())
    random.seed(SEED)
    np.random.seed(SEED)

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
    else:
        logger.info("Class balance close; using all gold roads.")
        paved_sample = paved
        gravel_sample = gravel
    gold_sample = pd.concat([paved_sample, gravel_sample], ignore_index=True)
    logger.info("Sampling complete. elapsed=%.2fs", time.time() - sample_start)
    sample_map = dict(zip(gold_sample["osm_id"], gold_sample["label"]))
    logger.info("Gold roads selected total=%d", len(sample_map))

    if not NAIP_INDEX.exists():
        sys.exit(f"Missing NAIP index: {NAIP_INDEX}")
    naip_df = pd.read_csv(NAIP_INDEX)
    if "id" not in naip_df.columns or "tile_id" not in naip_df.columns:
        sys.exit("naip_index.csv must include 'id' and 'tile_id' columns.")
    if "tile_url" not in naip_df.columns and "signed_url" not in naip_df.columns:
        sys.exit("naip_index.csv must include 'tile_url' or 'signed_url' columns.")
    naip_df["id"] = naip_df["id"].astype(str)
    naip_matches = naip_df[naip_df["id"].isin(sample_map.keys())].copy()
    if naip_matches.empty:
        sys.exit("No NAIP tiles matched sampled gold roads.")
    if "signed_url" in naip_matches.columns:
        naip_matches["tile_url"] = naip_matches["signed_url"].where(
            naip_matches["signed_url"].notna(),
            naip_matches.get("tile_url"),
        )
    else:
        naip_matches["tile_url"] = naip_matches["tile_url"]
    if naip_matches["tile_url"].isna().any():
        sys.exit("NAIP index missing tile_url/signed_url for some matched roads.")
    road_to_tiles = (
        naip_matches.groupby("id")["tile_url"].apply(list).to_dict()
    )
    logger.info(
        "Matched NAIP tiles. roads=%d tiles=%d",
        len(road_to_tiles),
        naip_matches["tile_id"].nunique(),
    )
    logger.info(
        "Selected roads per class. paved=%d gravel=%d",
        len(paved_sample),
        len(gravel_sample),
    )

    roads = gpd.read_parquet(ROADS_PARQUET)
    if "id" not in roads.columns:
        sys.exit("Roads parquet missing required 'id' column for gold sampling.")
    roads = roads.copy()
    roads["id"] = roads["id"].astype(str)
    if roads.crs is None:
        logger.warning("Roads CRS is missing; assuming EPSG:4326.")
        roads = roads.set_crs("EPSG:4326")
    logger.info("Road geometries CRS: %s", roads.crs)
    roads = roads[roads["id"].isin(sample_map.keys())].copy()
    roads["label"] = roads["id"].map(sample_map).astype(int)
    if "label" not in roads.columns:
        sys.exit("Missing 'label' column in merged_labels.parquet")

    debug_saved = 0
    DEBUG_PATCH_DIR.mkdir(parents=True, exist_ok=True)
    debug_patches_dir = BASE_DIR / "output" / "debug_patches"
    debug_tiles_dir = BASE_DIR / "output" / "debug_tiles"
    debug_patches_dir.mkdir(parents=True, exist_ok=True)
    debug_tiles_dir.mkdir(parents=True, exist_ok=True)
    debug_road_id = None
    debug_patch_saved_by_class = {0: 0, 1: 0}
    debug_tile_saved = False

    logger.info("Dataset construction start (gold subset).")
    DEBUG_PATCH_DIR.mkdir(parents=True, exist_ok=True)
    extract_start = time.time()

    X, y = [], []
    road_fractions = []
    road_ids = []
    kept = {0: 0, 1: 0}
    discarded = {0: 0, 1: 0}

    roads = roads.set_index("id")
    sampled_ids = list(sample_map.keys())
    total_roads = len(sampled_ids)
    for idx, road_id in enumerate(sampled_ids, start=1):
        if road_id not in roads.index:
            logger.warning("Road %s missing from roads dataset; skipping.", road_id)
            continue
        tiles = road_to_tiles.get(road_id, [])
        if not tiles:
            logger.warning("No tiles found for road %s; skipping.", road_id)
            continue
        logger.info("Processing road %d/%d (osm_id=%s)", idx, total_roads, road_id)
        label = int(sample_map[road_id])
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
                    if tile_idx == 1:
                        logger.info("Tile CRS for road %s: %s", road_id, tile_crs)
                    tile_poly = box(*src.bounds)
                    line_proj = gpd.GeoSeries([line], crs=roads.crs).to_crs(tile_crs).iloc[0]
                    if debug_road_id is None and not debug_tile_saved:
                        # Assumes NAIP tiles store RGB in bands 1-3 and align with tile CRS.
                        try:
                            rgb = src.read([1, 2, 3])
                            rgb = np.moveaxis(rgb, 0, -1)
                            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                            img = Image.fromarray(rgb)
                            draw = ImageDraw.Draw(img)
                            coords = list(line_proj.coords)
                            if len(coords) >= 2:
                                px = [rasterio.transform.rowcol(src.transform, x, y) for x, y in coords]
                                points = [(c, r) for r, c in px]
                                draw.line(points, fill=(255, 0, 0), width=2)
                            out_path = debug_tiles_dir / f"{road_id}_tile.png"
                            img.save(out_path)
                            debug_tile_saved = True
                            logger.info("Saved debug tile overlay to %s", out_path)
                        except Exception as exc:
                            logger.warning("Debug tile save failed for road %s: %s", road_id, exc)
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
                            discarded[label] += 1
                            rejected_for_road += 1
                            if len(reject_notes) < 5:
                                reject_notes.append("patch_none")
                            continue
                        road_fraction = float(road_mask.mean())
                        patch_resized = resize_or_pad(rgb_patch, PATCH_SIZE)
                        if float(np.var(patch_resized[:, :, :3])) < MIN_PATCH_VARIANCE:
                            discarded[label] += 1
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
                        X.append(patch)
                        y.append(label)
                        road_ids.append(road_id)
                        road_fractions.append(road_fraction)
                        kept[label] += 1
                        patches_for_road += 1
                        if patches_for_road % TILE_FETCH_LOG_EVERY == 0:
                            logger.info(
                                "Road %s patch progress: %d",
                                road_id,
                                patches_for_road,
                            )
                        if debug_road_id is None:
                            debug_road_id = road_id
                        if debug_patch_saved_by_class[label] < 3:
                            try:
                                rgb = np.clip(patch_resized[:, :, :3], 0, 255).astype(np.uint8)
                                out_path = debug_patches_dir / f"{road_id}_patch{debug_patch_saved_by_class[label]+1}_label{label}.png"
                                Image.fromarray(rgb).save(out_path)
                                debug_patch_saved_by_class[label] += 1
                                logger.info("Saved debug patch to %s", out_path)
                            except Exception as exc:
                                logger.warning("Debug patch save failed for road %s: %s", road_id, exc)
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
        if road_id == debug_road_id and patches_for_road == 0:
            logger.warning("Debug road %s produced zero patches.", road_id)
        if idx % 10 == 0:
            logger.info(
                "Road progress %d/%d: kept=%d rejected=%d",
                idx,
                total_roads,
                patches_for_road,
                rejected_for_road,
            )
        if idx == 10 and (kept[0] + kept[1]) == 0:
            logger.warning("No patches after 10 roads; continuing run.")

    logger.info(
        "Patch extraction end. total_patches=%d elapsed=%.2fs",
        len(X),
        time.time() - extract_start,
    )
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
    logger.info("Dataset construction end (gold subset).")

    if not X:
        sys.exit("No patches kept in gold mode.")

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
    else:
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

    train_idx = [i for i, rid in enumerate(road_ids) if rid in train_roads]
    val_idx = [i for i, rid in enumerate(road_ids) if rid in val_roads]
    X = np.array(X, dtype=np.float32)
    X[..., :3] = X[..., :3] / 255.0
    y = np.array(y, dtype=np.int32)
    road_ids = np.array(road_ids, dtype=object)
    road_fractions = np.array(road_fractions, dtype=np.float32)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    road_train = road_ids[train_idx]
    road_val = road_ids[val_idx]
    train_weights = road_fractions[train_idx]
    val_weights = road_fractions[val_idx]

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
        len(X_train),
        len(X_val),
    )
    logger.info("Model initialization.")
    model = make_cnn()
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    logger.info("Training start.")
    train_start = time.time()
    curriculum_epochs = max(1, int(EPOCHS * 0.25))
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
    preds = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).ravel()
    patch_pred = (preds >= 0.5).astype(int)
    patch_acc = float((patch_pred == y_val).mean()) if len(y_val) else 0.0

    road_probs = {}
    road_labels = {}
    for prob, rid, true_label in zip(preds, road_val, y_val):
        road_probs.setdefault(rid, []).append(float(prob))
        road_labels.setdefault(rid, int(true_label))
    road_pred = {}
    per_road_counts = []
    for rid, probs in road_probs.items():
        road_pred[rid] = sum(probs) / len(probs)
        per_road_counts.append(len(probs))
    if per_road_counts:
        logger.info("Avg patches per road: %.2f", sum(per_road_counts) / len(per_road_counts))
    road_pred_bin = {rid: int(prob >= 0.5) for rid, prob in road_pred.items()}
    road_true = [road_labels[rid] for rid in road_labels]
    road_est = [road_pred_bin[rid] for rid in road_labels]
    road_acc = float(
        sum(int(a == b) for a, b in zip(road_true, road_est)) / len(road_true)
    ) if road_true else 0.0
    tn = sum(1 for a, b in zip(road_true, road_est) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(road_true, road_est) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(road_true, road_est) if a == 1 and b == 0)
    tp = sum(1 for a, b in zip(road_true, road_est) if a == 1 and b == 1)
    logger.info("Patch-level accuracy: %.4f", patch_acc)
    logger.info("Road-level accuracy: %.4f", road_acc)
    logger.info("Road-level confusion matrix (tn, fp, fn, tp): %d, %d, %d, %d", tn, fp, fn, tp)
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
