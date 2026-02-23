import os, io, sys, math, random, logging, traceback, json
from collections import Counter
from pathlib import Path
import numpy as np
import time
import geopandas as gpd
import pandas as pd
import rasterio
import cv2
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, LineString
from shapely.errors import TopologicalError
from tensorflow.keras import layers, models, callbacks, losses, utils
from affine import Affine
from PIL import Image, ImageDraw
from azure.storage.blob import BlobServiceClient
 

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
EVAL_ROADS_FILE = BASE_DIR / "output" / "eval_roads.json"

BUFFER_METERS = 10.0
SEGMENT_LENGTH_M = 5.0
MIN_ROAD_LENGTH_M = 20.0
PATCH_SIZE = 96
MIN_MASK_COVERAGE = 0.0
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
EPOCHS = 5
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
EVAL_ROADS_PER_CLASS = 200
USE_ROAD_ONLY_PIXELS = True
ROAD_EDGE_DILATION_PX = 2
TILE_FETCH_LOG_EVERY = 25
MAX_PATCHES_PER_ROAD = 120
MIN_PATCH_VARIANCE = 50.0
MAX_PATCH_ATTEMPTS_PER_ROAD = 120
PREFLIGHT_BUFFER_M = 5.0
PREFLIGHT_INTERSECTION_AREA_M2 = 1.0

AZURE_PATCH_ACCOUNT = os.environ.get("AZURE_PATCH_ACCOUNT", "maskedpatches")
AZURE_PATCH_KEY = os.environ.get("AZURE_PATCH_KEY")
PATCHES_CONTAINER = "patches"
MASKS_CONTAINER = "masks"
INDEXES_CONTAINER = "index"
INDEX_BLOB_NAME = "patch_index.csv"
INDEX_BLOB_NAME_SILVER = "patch_index_silver.csv"

if not AZURE_PATCH_ACCOUNT:
    raise EnvironmentError("Missing required environment variable: AZURE_PATCH_ACCOUNT")
if not AZURE_PATCH_KEY:
    raise EnvironmentError("Missing required environment variable: AZURE_PATCH_KEY")

logger.info(
    "Azure patch credential loading: account=%s key_present=%s",
    AZURE_PATCH_ACCOUNT,
    bool(AZURE_PATCH_KEY),
)

logger.info("Azure patch credentials loaded (no container scan).")
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
def connect_patch_storage():
    account_url = f"https://{AZURE_PATCH_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=AZURE_PATCH_KEY)


def _blob_name_from_path(path, container_name):
    prefix = f"{container_name}/"
    if path.startswith(prefix):
        return path[len(prefix):]
    return path


def _read_index_blob(indexes_container, blob_name):
    index_blob = indexes_container.get_blob_client(blob_name)
    if not index_blob.exists():
        logger.warning("Index blob missing: %s/%s", INDEXES_CONTAINER, blob_name)
        return pd.DataFrame()
    data = index_blob.download_blob().readall().decode("utf-8")
    return pd.read_csv(io.StringIO(data))


def load_index_from_blob():
    logger.info(
        "Loading patch indexes from blob: container=%s blobs=%s,%s",
        INDEXES_CONTAINER,
        INDEX_BLOB_NAME,
        INDEX_BLOB_NAME_SILVER,
    )
    service = connect_patch_storage()
    indexes_container = service.get_container_client(INDEXES_CONTAINER)
    gold_df = _read_index_blob(indexes_container, INDEX_BLOB_NAME)
    silver_df = _read_index_blob(indexes_container, INDEX_BLOB_NAME_SILVER)
    index_df = pd.concat([gold_df, silver_df], ignore_index=True)
    if index_df.empty:
        sys.exit("Patch index is empty; cannot train.")
    required_cols = {"patch_id", "patch_path", "mask_path", "label", "road_id", "dataset"}
    missing_cols = required_cols - set(index_df.columns)
    if missing_cols:
        sys.exit(f"Patch index missing required columns: {', '.join(sorted(missing_cols))}")
    index_df = index_df[index_df["dataset"].isin(["gold", "silver"])].copy()
    if index_df.empty:
        sys.exit("No gold or silver rows found in patch index; cannot train.")
    index_df["label"] = index_df["label"].astype(str).str.strip().str.lower()
    label_map = {"gravel": 1, "1": 1, "paved": 0, "0": 0}
    index_df["label_int"] = index_df["label"].map(label_map)
    index_df = index_df[index_df["label_int"].isin([0, 1])].copy()
    if index_df.empty:
        sys.exit("No rows with valid labels found in patch index; cannot train.")
    index_df["road_id"] = index_df["road_id"].astype(str)
    index_df["dataset_weight"] = index_df["dataset"].map({"gold": 1.0, "silver": 0.5}).fillna(1.0)
    gold_count = int((index_df["dataset"] == "gold").sum())
    silver_count = int((index_df["dataset"] == "silver").sum())
    gold_paved = int(((index_df["dataset"] == "gold") & (index_df["label_int"] == 0)).sum())
    gold_gravel = int(((index_df["dataset"] == "gold") & (index_df["label_int"] == 1)).sum())
    silver_paved = int(((index_df["dataset"] == "silver") & (index_df["label_int"] == 0)).sum())
    silver_gravel = int(((index_df["dataset"] == "silver") & (index_df["label_int"] == 1)).sum())
    logger.info(
        "Loaded patch index rows. gold=%d (paved=%d gravel=%d) silver=%d (paved=%d gravel=%d) total=%d",
        gold_count,
        gold_paved,
        gold_gravel,
        silver_count,
        silver_paved,
        silver_gravel,
        len(index_df),
    )
    return index_df


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


class PatchBlobSequence(utils.Sequence):
    def __init__(self, rows, batch_size, shuffle, augment, service):
        self.rows = rows
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.service = service
        self.patches_container = service.get_container_client(PATCHES_CONTAINER)
        self.masks_container = service.get_container_client(MASKS_CONTAINER)
        self.indices = np.arange(len(self.rows))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(math.ceil(len(self.rows) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_patch(self, patch_path, mask_path):
        patch_blob_name = _blob_name_from_path(patch_path, PATCHES_CONTAINER)
        mask_blob_name = _blob_name_from_path(mask_path, MASKS_CONTAINER)
        patch_bytes = self.patches_container.download_blob(patch_blob_name).readall()
        mask_bytes = self.masks_container.download_blob(mask_blob_name).readall()
        patch_img = Image.open(io.BytesIO(patch_bytes)).convert("RGB")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        patch_arr = np.array(patch_img, dtype=np.float32)
        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
        if patch_arr.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected patch size {patch_arr.shape}")
        if mask_arr.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Unexpected mask size {mask_arr.shape}")
        patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 5), dtype=np.float32)
        patch[:, :, :3] = patch_arr
        patch[:, :, 3] = mask_arr
        patch[:, :, 4] = 0.0
        return patch, float(mask_arr.mean())

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_rows = self.rows.iloc[batch_idx]
        X_batch = np.zeros((len(batch_rows), PATCH_SIZE, PATCH_SIZE, 5), dtype=np.float32)
        y_batch = np.zeros((len(batch_rows),), dtype=np.int32)
        w_batch = np.zeros((len(batch_rows),), dtype=np.float32)
        for i, row in enumerate(batch_rows.itertuples(index=False)):
            try:
                patch, road_fraction = self._load_patch(row.patch_path, row.mask_path)
            except Exception as exc:
                logger.warning("Failed to load patch %s: %s", row.patch_path, exc)
                patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 5), dtype=np.float32)
                road_fraction = 0.0
            X_batch[i] = patch
            y_batch[i] = int(row.label_int)
            w_batch[i] = road_fraction * float(getattr(row, "dataset_weight", 1.0))
        X_batch[..., :3] = X_batch[..., :3] / 255.0
        if self.augment:
            X_batch = augment_training_data(X_batch)
        return X_batch, y_batch, w_batch


def compute_mask_means(rows, service, log_every=500):
    masks_container = service.get_container_client(MASKS_CONTAINER)
    weights = []
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        mask_path = str(row.mask_path)
        mask_blob_name = _blob_name_from_path(mask_path, MASKS_CONTAINER)
        try:
            mask_bytes = masks_container.download_blob(mask_blob_name).readall()
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
            weights.append(float(mask_arr.mean()) * float(getattr(row, "dataset_weight", 1.0)))
        except Exception as exc:
            logger.warning("Failed to load mask for weight %s: %s", mask_path, exc)
            weights.append(0.0)
        if idx % log_every == 0:
            logger.info("Computed mask weights %d/%d", idx, len(rows))
    return np.array(weights, dtype=np.float32)

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
    logger.info("Dataset construction start (gold+silver subset from blob).")
    index_df = load_index_from_blob()
    y = index_df["label_int"].astype(int).tolist()
    road_ids = index_df["road_id"].astype(str).tolist()
    kept = {0: int(sum(1 for lbl in y if lbl == 0)), 1: int(sum(1 for lbl in y if lbl == 1))}
    discarded = {0: 0, 1: 0}
    logger.info("Dataset construction end (gold+silver subset from blob).")

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
    y_train, y_val = y[train_idx], y[val_idx]
    road_train = road_ids[train_idx]
    road_val = road_ids[val_idx]
    y_eval = y[eval_idx]
    road_eval = road_ids[eval_idx]

    patch_service = connect_patch_storage()
    train_rows = index_df.iloc[train_idx].reset_index(drop=True)
    val_rows = index_df.iloc[val_idx].reset_index(drop=True)
    eval_rows = index_df.iloc[eval_idx].reset_index(drop=True)

    def _log_split_counts(name, rows):
        gold_rows = int((rows["dataset"] == "gold").sum())
        silver_rows = int((rows["dataset"] == "silver").sum())
        paved_rows = int((rows["label_int"] == 0).sum())
        gravel_rows = int((rows["label_int"] == 1).sum())
        gold_paved_rows = int(((rows["dataset"] == "gold") & (rows["label_int"] == 0)).sum())
        gold_gravel_rows = int(((rows["dataset"] == "gold") & (rows["label_int"] == 1)).sum())
        silver_paved_rows = int(((rows["dataset"] == "silver") & (rows["label_int"] == 0)).sum())
        silver_gravel_rows = int(((rows["dataset"] == "silver") & (rows["label_int"] == 1)).sum())
        logger.info(
            "%s split patches. total=%d gold=%d (paved=%d gravel=%d) silver=%d (paved=%d gravel=%d) paved=%d gravel=%d",
            name,
            len(rows),
            gold_rows,
            gold_paved_rows,
            gold_gravel_rows,
            silver_rows,
            silver_paved_rows,
            silver_gravel_rows,
            paved_rows,
            gravel_rows,
        )

    _log_split_counts("Train", train_rows)
    _log_split_counts("Val", val_rows)
    if len(eval_rows):
        _log_split_counts("Eval", eval_rows)

    train_weights = compute_mask_means(train_rows, patch_service)
    val_weights = compute_mask_means(val_rows, patch_service)
    eval_weights = compute_mask_means(eval_rows, patch_service)

    train_seq = PatchBlobSequence(train_rows, BATCH_SIZE, shuffle=True, augment=True, service=patch_service)
    val_seq = PatchBlobSequence(val_rows, BATCH_SIZE, shuffle=False, augment=False, service=patch_service)
    eval_seq = PatchBlobSequence(eval_rows, BATCH_SIZE, shuffle=False, augment=False, service=patch_service)

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
        len(train_rows),
        len(val_rows),
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
    train_rows_curr = train_rows[curriculum_mask].reset_index(drop=True)
    if train_rows_curr.empty:
        train_rows_curr = train_rows
    avg_rf_curr = float(train_weights[curriculum_mask].mean()) if train_rows_curr is not train_rows else float(train_weights.mean())
    avg_rf_full = float(train_weights.mean()) if train_weights.size else 0.0
    curriculum_seq = PatchBlobSequence(train_rows_curr, BATCH_SIZE, shuffle=True, augment=True, service=patch_service)

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
        curriculum_seq,
        validation_data=val_seq,
        epochs=curriculum_epochs,
        verbose=1,
        callbacks=[
            _epoch_log_callback(avg_rf_curr, avg_rf_curr),
            _weight_log_callback(avg_rf_curr, avg_rf_curr),
        ],
    )

    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        initial_epoch=curriculum_epochs,
        verbose=1,
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
    preds = model.predict(val_seq, verbose=0).ravel()
    patch_pred = (preds >= 0.5).astype(int)
    patch_acc = float((patch_pred == y_val).mean()) if len(y_val) else 0.0

    road_probs = {}
    road_weights = {}
    road_labels = {}
    for prob, rid, true_label, weight in zip(preds, road_val, y_val, val_weights):
        road_probs.setdefault(rid, []).append(float(prob))
        road_weights.setdefault(rid, []).append(float(weight))
        road_labels.setdefault(rid, int(true_label))
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
        road_pred[rid] = sum(probs) / len(probs) if probs else 0.0
        clipped = [float(np.clip(w, 0.2, 1.0)) for w in weights] if weights else []
        weight_sum = sum(clipped)
        road_pred_weighted[rid] = (
            sum(p * w for p, w in zip(probs, clipped)) / weight_sum
            if weight_sum
            else 0.0
        )
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

    if len(eval_rows):
        logger.info(
            "Eval roads per class. paved=%d gravel=%d",
            eval_paved_used,
            eval_gravel_used,
        )
        eval_preds = model.predict(eval_seq, verbose=0).ravel()
        eval_patch_acc = float(((eval_preds >= 0.5).astype(int) == y_eval).mean())
        eval_road_probs = {}
        eval_road_labels = {}
        eval_road_counts = []
        eval_road_counts_trimmed = []
        for prob, rid, true_label in zip(eval_preds, road_eval, y_eval):
            eval_road_probs.setdefault(rid, []).append(float(prob))
            eval_road_labels.setdefault(rid, int(true_label))
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
            eval_road_pred[rid] = sum(probs) / len(probs) if probs else 0.0
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
    else:
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
