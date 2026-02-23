import os, io, sys, math, random, logging, traceback, json, csv, hashlib
from collections import Counter
from pathlib import Path
import time
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from azure.storage.blob import BlobServiceClient

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# CONFIG (env overrides)
# -----------------------------

def _env(name, default):
    return os.environ.get(name, str(default))


def _env_int(name, default):
    return int(_env(name, default))


def _env_float(name, default):
    return float(_env(name, default))


def _env_bool(name, default):
    raw = _env(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_optional_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "none", "null"}:
        return None
    return int(raw)


def _env_optional_float(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "none", "null"}:
        return None
    return float(raw)


def _env_optional_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "none", "null"}:
        return None
    return raw in {"1", "true", "yes", "y", "on"}


def _env_optional_str(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else None


def _env_optional_float_list(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "none", "null"}:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(float(p) for p in parts)


@dataclass
class TrainConfig:
    # =========================
    # 1) DATA & SAMPLING (most important)
    # =========================
    batch_size: int
    effective_batch_size: int
    grad_accum_steps: int
    class_weights: Optional[Tuple[float, float]]
    sampling_strategy: str
    num_workers: int
    pin_memory: Optional[bool]
    prefetch_factor: int

    # Additional data/sampling controls
    seed: int
    patch_size: int
    patch_size_m: float
    patch_spacing_m: float
    sample_spacing_m: float
    road_buffer_m: float
    buffer_meters: float
    segment_length_m: float
    min_road_length_m: float
    min_mask_coverage: float
    preflight_buffer_m: float
    preflight_intersection_area_m2: float

    include_hard_negatives: bool
    hard_negative_offset_m: float
    min_gravel_count: int
    min_gravel_roads_per_split: int
    max_patches_per_road: int
    min_patch_variance: float
    max_patch_attempts_per_road: int
    min_road_pixel_ratio: float
    use_road_only_pixels: bool
    road_edge_dilation_px: int

    val_split: float
    test_split: float
    split_attempts: int
    gold_roads_per_class: int
    eval_roads_per_class: int
    resplit: bool

    tile_count: int
    max_roads_per_tile: Optional[int]
    tile_fetch_log_every: int

    debug_mode: bool
    debug_roads_per_class: int
    debug_save_per_class: int
    debug_patch_dir: Path
    debug_patch_count: int

    patch_cache_dir: Optional[str]
    mask_weight_cache_dir: Path
    mask_weight_log_every: int

    curriculum_fraction: float
    curriculum_min_weight: float

    augmentations_enabled: bool
    augment_hflip_prob: float
    augment_vflip_prob: float
    augment_rot90_max_k: int
    augment_brightness_min: float
    augment_brightness_max: float
    augment_contrast_min: float
    augment_contrast_max: float

    # =========================
    # 2) LEARNING RATE & SCHEDULING
    # =========================
    base_lr: float
    max_lr: float
    min_lr: float
    scheduler_type: str
    warmup_pct: float
    scheduler_step_unit: str
    scheduler_t_max: Optional[int]

    # =========================
    # 3) OPTIMIZER
    # =========================
    optimizer_type: str
    betas: Tuple[float, float]
    momentum: float
    weight_decay: float
    nesterov: bool

    # =========================
    # 4) LOSS FUNCTION
    # =========================
    loss_type: str
    label_smoothing: float
    label_smoothing_target: float
    label_positive_threshold: float
    focal_gamma: float
    loss_reduction: str

    # =========================
    # 5) REGULARIZATION
    # =========================
    dropout_rate: float
    stochastic_depth: float
    early_stopping_patience: int
    early_stopping_min_epoch: int
    early_stopping_min_delta: float

    # =========================
    # 6) NORMALIZATION CONTROL
    # =========================
    norm_type: str
    freeze_batchnorm: bool
    batchnorm_momentum: float
    batchnorm_eps: float

    # =========================
    # 7) PRECISION & NUMERICS
    # =========================
    use_amp: bool
    amp_dtype: str
    grad_scaler_enabled: bool

    # =========================
    # 8) GRADIENT CONTROL
    # =========================
    grad_clip_norm: float
    grad_clip_value: Optional[float]
    zero_grad_strategy: str
    gradient_accumulation_steps: int

    # =========================
    # 9) INITIALIZATION & TRANSFER
    # =========================
    pretrained: bool
    reinit_head: bool
    init_type: str

    # =========================
    # 10) TRAINING CONTROL
    # =========================
    epochs: int
    validation_frequency: int
    checkpoint_frequency: int
    save_best_only: bool
    resume_training: bool
    resume_path: Optional[Path]

    # =========================
    # 11) EVALUATION
    # =========================
    eval_mode: str
    threshold: float
    use_tta: bool
    ema_weights: bool
    eval_trim_count_threshold: int
    eval_trim_keep_ratio: float
    eval_weight_clip_min: float
    eval_weight_clip_max: float

    # =========================
    # 12) SYSTEM / CUDA
    # =========================
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    tf32_allowed: bool
    channels_last: bool
    device_preference: str
    device: str

    # =========================
    # OTHER / IO / LOGGING
    # =========================
    base_dir: Path
    input_dir: Path
    roads_parquet: Path
    model_out: Path
    tile_summary: Path
    gold_dataset: Path
    naip_index: Path
    val_roads_file: Path
    eval_roads_file: Path
    split_roads_file: Path

    log_path: Path
    perf_log_every: int
    perf_csv_path: Optional[str]
    perf_log_gpu_mem: bool

    azure_patch_account: str
    azure_patch_key: Optional[str]
    patches_container: str
    masks_container: str
    indexes_container: str
    index_blob_name: str
    index_blob_name_silver: str

    dataset_weight_gold: float
    dataset_weight_silver: float

    # Model architecture (SimpleCNN)
    model_in_channels: int
    model_conv1_out: int
    model_conv2_out: int
    model_conv_kernel: int
    model_conv_padding: int
    model_pool_kernel: int
    model_fc1_out: int


def build_config() -> TrainConfig:
    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    epochs = _env_int("EPOCHS", 10)
    batch_size = _env_int("BATCH_SIZE", 256)
    grad_accum_steps = _env_int("GRAD_ACCUM_STEPS", 1)
    effective_batch_size = _env_optional_int("EFFECTIVE_BATCH_SIZE", None)
    if effective_batch_size is None:
        effective_batch_size = batch_size * max(1, grad_accum_steps)

    scheduler_type_env = _env_optional_str("SCHEDULER_TYPE", None)
    use_lr_scheduler_env = _env_bool("USE_LR_SCHEDULER", True)
    if scheduler_type_env:
        scheduler_type = scheduler_type_env.lower()
    else:
        scheduler_type = "cosine" if use_lr_scheduler_env else "none"

    device_preference = _env("DEVICE", "auto").strip().lower()
    cuda_available = torch.cuda.is_available()
    if device_preference == "cpu":
        device = "cpu"
    elif device_preference == "cuda":
        device = "cuda"
    else:
        device = "cuda" if cuda_available else "cpu"

    resume_path_raw = _env_optional_str("RESUME_PATH", None)

    cfg = TrainConfig(
        # =========================
        # 1) DATA & SAMPLING
        # =========================
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        grad_accum_steps=grad_accum_steps,
        class_weights=_env_optional_float_list("CLASS_WEIGHTS", None),
        sampling_strategy=_env("SAMPLING_STRATEGY", "standard"),
        num_workers=_env_int("MAX_DATA_WORKERS", 0),
        pin_memory=_env_optional_bool("PIN_MEMORY", None),
        prefetch_factor=_env_int("PREFETCH_FACTOR", 2),

        seed=_env_int("SEED", 42),
        patch_size=_env_int("PATCH_SIZE", 96),
        patch_size_m=_env_float("PATCH_SIZE_M", 8.0),
        patch_spacing_m=_env_float("PATCH_SPACING_M", 6.0),
        sample_spacing_m=_env_float("SAMPLE_SPACING_M", 2.5),
        road_buffer_m=_env_float("ROAD_BUFFER_M", 6.0),
        buffer_meters=_env_float("BUFFER_METERS", 10.0),
        segment_length_m=_env_float("SEGMENT_LENGTH_M", 5.0),
        min_road_length_m=_env_float("MIN_ROAD_LENGTH_M", 20.0),
        min_mask_coverage=_env_float("MIN_MASK_COVERAGE", 0.0),
        preflight_buffer_m=_env_float("PREFLIGHT_BUFFER_M", 5.0),
        preflight_intersection_area_m2=_env_float("PREFLIGHT_INTERSECTION_AREA_M2", 1.0),

        include_hard_negatives=_env_bool("INCLUDE_HARD_NEGATIVES", True),
        hard_negative_offset_m=_env_float("HARD_NEGATIVE_OFFSET_M", 2.0),
        min_gravel_count=_env_int("MIN_GRAVEL_COUNT", 10),
        min_gravel_roads_per_split=_env_int("MIN_GRAVEL_ROADS_PER_SPLIT", 10),
        max_patches_per_road=_env_int("MAX_PATCHES_PER_ROAD", 150),
        min_patch_variance=_env_float("MIN_PATCH_VARIANCE", 70.0),
        max_patch_attempts_per_road=_env_int("MAX_PATCH_ATTEMPTS_PER_ROAD", 120),
        min_road_pixel_ratio=_env_float("MIN_ROAD_PIXEL_RATIO", 0.60),
        use_road_only_pixels=_env_bool("USE_ROAD_ONLY_PIXELS", True),
        road_edge_dilation_px=_env_int("ROAD_EDGE_DILATION_PX", 2),

        val_split=_env_float("VAL_SPLIT", 0.2),
        test_split=_env_float("TEST_SPLIT", 0.2),
        split_attempts=_env_int("SPLIT_ATTEMPTS", 50),
        gold_roads_per_class=_env_int("GOLD_ROADS_PER_CLASS", 175),
        eval_roads_per_class=_env_int("EVAL_ROADS_PER_CLASS", 200),
        resplit=_env_bool("RESPLIT", False),

        tile_count=_env_int("TILE_COUNT", 5),
        max_roads_per_tile=_env_optional_int("MAX_ROADS_PER_TILE", None),
        tile_fetch_log_every=_env_int("TILE_FETCH_LOG_EVERY", 25),

        debug_mode=_env_bool("DEBUG_MODE", True),
        debug_roads_per_class=_env_int("DEBUG_ROADS_PER_CLASS", 10),
        debug_save_per_class=_env_int("DEBUG_SAVE_PER_CLASS", 50),
        debug_patch_dir=Path(_env("DEBUG_PATCH_DIR", output_dir / "patch_debug")),
        debug_patch_count=_env_int("DEBUG_PATCH_COUNT", 10),

        patch_cache_dir=_env_optional_str("PATCH_CACHE_DIR", "/tmp/patch_cache"),
        mask_weight_cache_dir=Path(_env("MASK_WEIGHT_CACHE_DIR", Path(__file__).resolve().parent / "cache")),
        mask_weight_log_every=_env_int("MASK_WEIGHT_LOG_EVERY", 500),

        curriculum_fraction=_env_float("CURRICULUM_FRACTION", 0.25),
        curriculum_min_weight=_env_float("CURRICULUM_MIN_WEIGHT", 0.15),

        augmentations_enabled=_env_bool("AUGMENTATIONS_ENABLED", True),
        augment_hflip_prob=_env_float("AUGMENT_HFLIP_PROB", 0.5),
        augment_vflip_prob=_env_float("AUGMENT_VFLIP_PROB", 0.5),
        augment_rot90_max_k=_env_int("AUGMENT_ROT90_MAX_K", 3),
        augment_brightness_min=_env_float("AUGMENT_BRIGHTNESS_MIN", 0.95),
        augment_brightness_max=_env_float("AUGMENT_BRIGHTNESS_MAX", 1.05),
        augment_contrast_min=_env_float("AUGMENT_CONTRAST_MIN", 0.95),
        augment_contrast_max=_env_float("AUGMENT_CONTRAST_MAX", 1.05),

        # =========================
        # 2) LEARNING RATE & SCHEDULING
        # =========================
        base_lr=_env_float("LEARNING_RATE", 3e-3),
        max_lr=_env_float("MAX_LR", _env_float("LEARNING_RATE", 4e-4)),
        min_lr=_env_float("MIN_LR", 0.0),
        scheduler_type=scheduler_type,
        warmup_pct=_env_float("WARMUP_PCT", 0.0),
        scheduler_step_unit=_env("SCHEDULER_STEP_UNIT", "epoch"),
        scheduler_t_max=_env_optional_int("SCHEDULER_T_MAX", None),

        # =========================
        # 3) OPTIMIZER
        # =========================
        optimizer_type=_env("OPTIMIZER_NAME", "adamw"),
        betas=(
            _env_float("ADAM_BETA1", 0.90),
            _env_float("ADAM_BETA2", 0.999),
        ),
        momentum=_env_float("SGD_MOMENTUM", 0.9),
        weight_decay=_env_float("WEIGHT_DECAY", 1e-3),
        nesterov=_env_bool("SGD_NESTEROV", False),

        # =========================
        # 4) LOSS FUNCTION
        # =========================
        loss_type=_env("LOSS_TYPE", "bce_with_logits"),
        label_smoothing=_env_float("LABEL_SMOOTHING", 0.05),
        label_smoothing_target=_env_float("LABEL_SMOOTHING_TARGET", 0.5),
        label_positive_threshold=_env_float("LABEL_POSITIVE_THRESHOLD", 0.5),
        focal_gamma=_env_float("FOCAL_GAMMA", 0.0),
        loss_reduction=_env("LOSS_REDUCTION", "none"),

        # =========================
        # 5) REGULARIZATION
        # =========================
        dropout_rate=_env_float("DROPOUT_RATE", 0.0),
        stochastic_depth=_env_float("STOCHASTIC_DEPTH", 0.0),
        early_stopping_patience=max(10, _env_int("EARLY_STOPPING_PATIENCE", 10)),
        early_stopping_min_epoch=_env_int("EARLY_STOPPING_MIN_EPOCH", 15),
        early_stopping_min_delta=_env_float("EARLY_STOPPING_MIN_DELTA", 0.001),

        # =========================
        # 6) NORMALIZATION CONTROL
        # =========================
        norm_type=_env("NORM_TYPE", "none"),
        freeze_batchnorm=_env_bool("FREEZE_BATCHNORM", False),
        batchnorm_momentum=_env_float("BATCHNORM_MOMENTUM", 0.1),
        batchnorm_eps=_env_float("BATCHNORM_EPS", 1e-5),

        # =========================
        # 7) PRECISION & NUMERICS
        # =========================
        use_amp=_env_bool("USE_AMP", False),
        amp_dtype=_env("AMP_DTYPE", "float16"),
        grad_scaler_enabled=_env_bool("GRAD_SCALER_ENABLED", True),

        # =========================
        # 8) GRADIENT CONTROL
        # =========================
        grad_clip_norm=_env_float("GRAD_CLIP_NORM", 0.0),
        grad_clip_value=_env_optional_float("GRAD_CLIP_VALUE", None),
        zero_grad_strategy=_env("ZERO_GRAD_STRATEGY", "set_to_none"),
        gradient_accumulation_steps=grad_accum_steps,

        # =========================
        # 9) INITIALIZATION & TRANSFER
        # =========================
        pretrained=_env_bool("PRETRAINED", False),
        reinit_head=_env_bool("REINIT_HEAD", False),
        init_type=_env("INIT_TYPE", "kaiming"),

        # =========================
        # 10) TRAINING CONTROL
        # =========================
        epochs=epochs,
        validation_frequency=_env_int("VALIDATION_FREQUENCY", 1),
        checkpoint_frequency=_env_int("CHECKPOINT_FREQUENCY", 1),
        save_best_only=_env_bool("SAVE_BEST_ONLY", True),
        resume_training=_env_bool("RESUME_TRAINING", False),
        resume_path=Path(resume_path_raw) if resume_path_raw else None,

        # =========================
        # 11) EVALUATION
        # =========================
        eval_mode=_env("EVAL_MODE", "standard"),
        threshold=_env_float("EVAL_THRESHOLD", 0.5),
        use_tta=_env_bool("USE_TTA", False),
        ema_weights=_env_bool("EMA_WEIGHTS", False),
        eval_trim_count_threshold=_env_int("EVAL_TRIM_COUNT_THRESHOLD", 40),
        eval_trim_keep_ratio=_env_float("EVAL_TRIM_KEEP_RATIO", 0.6),
        eval_weight_clip_min=_env_float("EVAL_WEIGHT_CLIP_MIN", 0.2),
        eval_weight_clip_max=_env_float("EVAL_WEIGHT_CLIP_MAX", 1.0),

        # =========================
        # 12) SYSTEM / CUDA
        # =========================
        cudnn_benchmark=_env_bool("CUDNN_BENCHMARK", True),
        cudnn_deterministic=_env_bool("CUDNN_DETERMINISTIC", False),
        tf32_allowed=_env_bool("TF32_ALLOWED", True),
        channels_last=_env_bool("CHANNELS_LAST", False),
        device_preference=device_preference,
        device=device,

        # =========================
        # OTHER / IO / LOGGING
        # =========================
        base_dir=base_dir,
        input_dir=input_dir,
        roads_parquet=output_dir / "merged_labels.parquet",
        model_out=output_dir / "cnn_test_model_zoomed_aug_pytorch.pt",
        tile_summary=output_dir / "tile_road_counts.csv",
        gold_dataset=output_dir / "goldgpt.csv",
        naip_index=output_dir / "naip_index.csv",
        val_roads_file=output_dir / "val_roads.json",
        eval_roads_file=output_dir / "eval_roads.json",
        split_roads_file=output_dir / "road_splits.json",

        log_path=base_dir / "cnn_train.log",
        perf_log_every=_env_int("PERF_LOG_EVERY", 100),
        perf_csv_path=_env_optional_str("PERF_CSV_PATH", ""),
        perf_log_gpu_mem=_env_bool("PERF_LOG_GPU_MEM", True),

        azure_patch_account=os.environ.get("AZURE_PATCH_ACCOUNT", "maskedpatches"),
        azure_patch_key=os.environ.get("AZURE_PATCH_KEY"),
        patches_container=_env("PATCHES_CONTAINER", "patches"),
        masks_container=_env("MASKS_CONTAINER", "masks"),
        indexes_container=_env("INDEXES_CONTAINER", "index"),
        index_blob_name=_env("INDEX_BLOB_NAME", "patch_index.csv"),
        index_blob_name_silver=_env("INDEX_BLOB_NAME_SILVER", "patch_index_silver.csv"),

        dataset_weight_gold=_env_float("DATASET_WEIGHT_GOLD", 1.0),
        dataset_weight_silver=_env_float("DATASET_WEIGHT_SILVER", 0.5),

        model_in_channels=_env_int("MODEL_IN_CHANNELS", 5),
        model_conv1_out=_env_int("MODEL_CONV1_OUT", 16),
        model_conv2_out=_env_int("MODEL_CONV2_OUT", 32),
        model_conv_kernel=_env_int("MODEL_CONV_KERNEL", 3),
        model_conv_padding=_env_int("MODEL_CONV_PADDING", 0),
        model_pool_kernel=_env_int("MODEL_POOL_KERNEL", 2),
        model_fc1_out=_env_int("MODEL_FC1_OUT", 64),
    )

    if cfg.class_weights is not None and len(cfg.class_weights) != 2:
        raise ValueError("CLASS_WEIGHTS must have exactly two values: neg,pos")

    return cfg


CFG = build_config()


LOG_PATH = CFG.log_path
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


if not CFG.azure_patch_account:
    raise EnvironmentError("Missing required environment variable: AZURE_PATCH_ACCOUNT")
if not CFG.azure_patch_key:
    raise EnvironmentError("Missing required environment variable: AZURE_PATCH_KEY")

logger.info(
    "Azure patch credential loading: account=%s key_present=%s",
    CFG.azure_patch_account,
    bool(CFG.azure_patch_key),
)

logger.info("Azure patch credentials loaded (no container scan).")
logger.info(
    "Sampling config: spacing_m=%.1f max_attempts_per_road=%d max_patches_per_road=%d epochs=%d",
    CFG.patch_spacing_m,
    CFG.max_patch_attempts_per_road,
    CFG.max_patches_per_road,
    CFG.epochs,
)

# -----------------------------
# Helpers
# -----------------------------


def connect_patch_storage(cfg: TrainConfig):
    account_url = f"https://{cfg.azure_patch_account}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=cfg.azure_patch_key)


def _blob_name_from_path(path, container_name):
    prefix = f"{container_name}/"
    if path.startswith(prefix):
        return path[len(prefix):]
    return path


def _read_index_blob(indexes_container, blob_name, cfg: TrainConfig):
    index_blob = indexes_container.get_blob_client(blob_name)
    if not index_blob.exists():
        logger.warning("Index blob missing: %s/%s", cfg.indexes_container, blob_name)
        return pd.DataFrame()
    data = index_blob.download_blob().readall().decode("utf-8")
    return pd.read_csv(io.StringIO(data))


def load_index_from_blob(cfg: TrainConfig):
    logger.info(
        "Loading patch indexes from blob: container=%s blobs=%s,%s",
        cfg.indexes_container,
        cfg.index_blob_name,
        cfg.index_blob_name_silver,
    )
    service = connect_patch_storage(cfg)
    indexes_container = service.get_container_client(cfg.indexes_container)
    gold_df = _read_index_blob(indexes_container, cfg.index_blob_name, cfg)
    silver_df = _read_index_blob(indexes_container, cfg.index_blob_name_silver, cfg)
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
    index_df["dataset_weight"] = index_df["dataset"].map(
        {"gold": cfg.dataset_weight_gold, "silver": cfg.dataset_weight_silver}
    ).fillna(1.0)
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


def augment_patch(patch, cfg: TrainConfig):
    if random.random() < cfg.augment_hflip_prob:
        patch = patch[:, ::-1, :]
    if random.random() < cfg.augment_vflip_prob:
        patch = patch[::-1, :, :]
    if cfg.augment_rot90_max_k > 0:
        rot_k = random.randint(0, cfg.augment_rot90_max_k)
        if rot_k:
            patch = np.rot90(patch, k=rot_k, axes=(0, 1))
    brightness = random.uniform(cfg.augment_brightness_min, cfg.augment_brightness_max)
    contrast = random.uniform(cfg.augment_contrast_min, cfg.augment_contrast_max)
    patch = np.clip((patch - 0.5) * contrast + 0.5, 0.0, 1.0)
    patch = np.clip(patch * brightness, 0.0, 1.0)
    return patch


class PatchBlobDataset(Dataset):
    def __init__(self, rows, cfg: TrainConfig, augment=False):
        self.rows = rows.reset_index(drop=True)
        self.cfg = cfg
        self.augment = augment
        self.patches_container = None
        self.masks_container = None
        self.cache_dir = Path(cfg.patch_cache_dir) if cfg.patch_cache_dir else None

    def _ensure_clients(self):
        if self.patches_container is None or self.masks_container is None:
            service = connect_patch_storage(self.cfg)
            self.patches_container = service.get_container_client(self.cfg.patches_container)
            self.masks_container = service.get_container_client(self.cfg.masks_container)

    def _read_blob_cached(self, container, container_name, blob_name):
        if not self.cache_dir:
            return container.download_blob(blob_name).readall()
        cache_path = self.cache_dir / container_name / blob_name
        try:
            if cache_path.exists():
                return cache_path.read_bytes()
        except Exception as exc:
            logger.warning("Cache read failed %s: %s", cache_path, exc)
        data = container.download_blob(blob_name).readall()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            tmp_path.write_bytes(data)
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            logger.warning("Cache write failed %s: %s", cache_path, exc)
        return data

    def __len__(self):
        return len(self.rows)

    def _load_patch(self, patch_path, mask_path):
        self._ensure_clients()
        patch_blob_name = _blob_name_from_path(str(patch_path), self.cfg.patches_container)
        mask_blob_name = _blob_name_from_path(str(mask_path), self.cfg.masks_container)
        patch_bytes = self._read_blob_cached(
            self.patches_container, self.cfg.patches_container, patch_blob_name
        )
        mask_bytes = self._read_blob_cached(
            self.masks_container, self.cfg.masks_container, mask_blob_name
        )
        patch_img = Image.open(io.BytesIO(patch_bytes)).convert("RGB")
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        patch_arr = np.array(patch_img, dtype=np.float32)
        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
        if patch_arr.shape[:2] != (self.cfg.patch_size, self.cfg.patch_size):
            raise ValueError(f"Unexpected patch size {patch_arr.shape}")
        if mask_arr.shape[:2] != (self.cfg.patch_size, self.cfg.patch_size):
            raise ValueError(f"Unexpected mask size {mask_arr.shape}")
        patch = np.zeros((self.cfg.patch_size, self.cfg.patch_size, self.cfg.model_in_channels), dtype=np.float32)
        patch[:, :, :3] = patch_arr
        patch[:, :, 3] = mask_arr
        if self.cfg.model_in_channels > 4:
            patch[:, :, 4:] = 0.0
        patch[:, :, :3] = patch[:, :, :3] / 255.0
        return patch, float(mask_arr.mean())

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        try:
            patch, road_fraction = self._load_patch(row.patch_path, row.mask_path)
        except Exception as exc:
            logger.warning("Failed to load patch %s: %s", row.patch_path, exc)
            patch = np.zeros((self.cfg.patch_size, self.cfg.patch_size, self.cfg.model_in_channels), dtype=np.float32)
            road_fraction = 0.0
        if self.augment and self.cfg.augmentations_enabled:
            patch = augment_patch(patch, self.cfg)
        weight = road_fraction * float(getattr(row, "dataset_weight", 1.0))
        x = torch.from_numpy(patch).permute(2, 0, 1).contiguous()
        y = torch.tensor(float(row.label_int), dtype=torch.float32)
        w = torch.tensor(float(weight), dtype=torch.float32)
        return x, y, w, str(row.road_id)


def compute_mask_means(rows, service, cfg: TrainConfig, split_hash: str, split_name: str):
    cache_dir = cfg.mask_weight_cache_dir
    cache_map = {
        "train": cache_dir / "mask_weights_10897773539278805867_65180.npy",
        "val": cache_dir / "mask_weights_14465285412936174450_41514.npy",
        "eval": cache_dir / "mask_weights_1048541919256952097_3563.npy",
    }
    cache_path = cache_map.get(split_name)
    if cache_path is None:
        raise ValueError(f"Unsupported split name for mask weights: {split_name}")
    if not cache_path.exists():
        raise FileNotFoundError(f"Mask weight cache missing: {cache_path}")
    weights = np.array(np.load(cache_path), dtype=np.float32)
    logger.info(
        "Using cached mask weights for split=%s (%d patches)",
        split_name,
        len(weights),
    )
    return weights


# -----------------------------
# BUILDERS
# -----------------------------


def build_model(cfg: TrainConfig) -> nn.Module:
    conv_kernel = cfg.model_conv_kernel
    conv_padding = cfg.model_conv_padding
    pool_kernel = cfg.model_pool_kernel
    conv1_out = cfg.model_conv1_out
    conv2_out = cfg.model_conv2_out
    fc1_out = cfg.model_fc1_out

    def _calc_flatten_dim():
        size = cfg.patch_size
        size = size + 2 * conv_padding - conv_kernel + 1
        size = size // pool_kernel
        size = size + 2 * conv_padding - conv_kernel + 1
        size = size // pool_kernel
        return conv2_out * size * size

    flatten_dim = _calc_flatten_dim()
    layers = [
        nn.Conv2d(cfg.model_in_channels, conv1_out, kernel_size=conv_kernel, padding=conv_padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool_kernel),
        nn.Conv2d(conv1_out, conv2_out, kernel_size=conv_kernel, padding=conv_padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool_kernel),
        nn.Flatten(),
        nn.Linear(flatten_dim, fc1_out),
        nn.ReLU(inplace=True),
    ]
    if cfg.dropout_rate and cfg.dropout_rate > 0.0:
        layers.append(nn.Dropout(p=cfg.dropout_rate))
    layers.append(nn.Linear(fc1_out, 1))
    return nn.Sequential(*layers)


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    opt_name = cfg.optimizer_type.strip().lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
    if opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )
    raise ValueError(f"Unsupported optimizer_type: {cfg.optimizer_type}")


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    sched = cfg.scheduler_type.strip().lower()
    if sched in {"none", "", "off"}:
        return None
    if sched == "cosine":
        t_max = cfg.scheduler_t_max if cfg.scheduler_t_max else cfg.epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=cfg.min_lr,
        )
    if sched == "onecycle":
        total_steps = max(1, steps_per_epoch * cfg.epochs)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            total_steps=total_steps,
            pct_start=cfg.warmup_pct,
        )
    raise ValueError(f"Unsupported scheduler_type: {cfg.scheduler_type}")


def build_loss(cfg: TrainConfig):
    loss_name = cfg.loss_type.strip().lower()

    if loss_name == "bce_with_logits":
        criterion = nn.BCEWithLogitsLoss(reduction=cfg.loss_reduction)
    elif loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(reduction=cfg.loss_reduction)
    else:
        raise ValueError(f"Unsupported loss_type: {cfg.loss_type}")

    def _loss_fn(logits, targets):
        if loss_name == "bce_with_logits" and logits.dim() == 2 and logits.size(-1) == 1 and targets.dim() == 1:
            targets = targets.unsqueeze(1)
        return criterion(logits, targets)

    return _loss_fn


def build_dataloader(cfg: TrainConfig, rows, augment: bool, shuffle: bool, device: torch.device):
    cpu_count = multiprocessing.cpu_count()
    if cfg.num_workers and cfg.num_workers > 0:
        data_workers = max(1, min(cfg.num_workers, cpu_count))
    else:
        data_workers = max(1, min(16, cpu_count))

    pin_memory = cfg.pin_memory if cfg.pin_memory is not None else (device.type == "cuda")
    prefetch_factor = cfg.prefetch_factor
    if data_workers > 8:
        prefetch_factor = min(prefetch_factor, 1)
    elif data_workers > 4:
        prefetch_factor = min(prefetch_factor, 2)
    persistent_workers = data_workers > 0 and data_workers <= 8
    return DataLoader(
        PatchBlobDataset(rows, cfg, augment=augment),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=data_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )


def build_scaler(cfg: TrainConfig):
    enabled = bool(cfg.use_amp and cfg.grad_scaler_enabled and cfg.device == "cuda")
    return torch.cuda.amp.GradScaler(enabled=enabled)


# -----------------------------
# LOOP
# -----------------------------


def _epoch_log(epoch, loss, acc, val_loss, val_acc, train_counts, val_counts):
    logger.info(
        "Epoch %d end. loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f train_patches=%s val_patches=%s",
        epoch,
        loss,
        acc,
        val_loss,
        val_acc,
        train_counts,
        val_counts,
    )


def _weight_log(epoch, avg_road_fraction, avg_loss_weight):
    logger.info(
        "Epoch %d stats. avg_road_fraction_per_batch=%.4f avg_loss_weight=%.4f",
        epoch,
        avg_road_fraction,
        avg_loss_weight,
    )


_PERF_CSV_HEADER = [
    "phase",
    "epoch",
    "batch",
    "data_time_s",
    "transfer_time_s",
    "compute_time_s",
    "total_time_s",
    "gpu_mem_alloc_mb",
    "gpu_mem_reserved_mb",
]


def _maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _write_perf_row(row, cfg: TrainConfig):
    if not cfg.perf_csv_path:
        return
    path = Path(cfg.perf_csv_path)
    try:
        new_file = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(_PERF_CSV_HEADER)
            writer.writerow(row)
    except Exception as exc:
        logger.warning("Perf CSV write failed %s: %s", path, exc)


def _apply_label_smoothing(targets, cfg: TrainConfig):
    if cfg.label_smoothing <= 0.0:
        return targets
    return targets * (1.0 - cfg.label_smoothing) + cfg.label_smoothing_target * cfg.label_smoothing


def _class_weight_tensor(labels, cfg: TrainConfig):
    if cfg.class_weights is None:
        return None
    neg_w, pos_w = cfg.class_weights
    return torch.where(
        labels >= cfg.label_positive_threshold,
        torch.tensor(pos_w, device=labels.device),
        torch.tensor(neg_w, device=labels.device),
    )


def _apply_focal(loss_raw, logits, targets, cfg: TrainConfig):
    if cfg.focal_gamma <= 0.0:
        return loss_raw
    if cfg.loss_type.strip().lower() != "bce_with_logits":
        return loss_raw
    probs = torch.sigmoid(logits)
    pt = torch.where(targets >= cfg.label_positive_threshold, probs, 1.0 - probs)
    return loss_raw * torch.pow(1.0 - pt, cfg.focal_gamma)


def _stable_hash_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _seed_for_road(road_id: str, seed: int) -> int:
    return _stable_hash_int(f"{road_id}:{seed}") % (2**32)


def _deterministic_eval_order(rows: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    if rows.empty:
        return rows
    ordered_idx = []
    for rid, group in rows.groupby("road_id", sort=False):
        idxs = list(group.index)
        rng = random.Random(_seed_for_road(str(rid), cfg.seed))
        rng.shuffle(idxs)
        ordered_idx.extend(idxs)
    return rows.loc[ordered_idx].reset_index(drop=True)


def _split_fingerprint(name: str, roads: set, seed: int):
    roads_sorted = sorted(str(rid) for rid in roads)
    h = hashlib.sha256(("\n".join(roads_sorted)).encode("utf-8")).hexdigest()[:12]
    logger.info("Split fingerprint %s: seed=%d roads=%d hash=%s", name, seed, len(roads_sorted), h)


def _split_hash(train_roads: set, val_roads: set, eval_roads: set) -> str:
    def _join(tag, roads):
        return f"{tag}:" + "\n".join(sorted(str(rid) for rid in roads))
    payload = "\n".join([_join("train", train_roads), _join("val", val_roads), _join("eval", eval_roads)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _amp_dtype(cfg: TrainConfig):
    if cfg.amp_dtype.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, cfg: TrainConfig, log_every=None):
    if log_every is None:
        log_every = cfg.perf_log_every
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    total_weight = 0.0
    total_road_fraction = 0.0
    loss_fn = build_loss(cfg)
    epoch_start = time.time()
    last_iter_end = epoch_start
    logged_device = False

    optimizer.zero_grad(set_to_none=(cfg.zero_grad_strategy == "set_to_none"))
    accum_steps = max(1, cfg.grad_accum_steps)

    for batch_idx, (xb, yb, wb, _) in enumerate(loader, start=1):
        batch_start = time.time()
        data_time = batch_start - last_iter_end
        transfer_start = time.time()
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)
        transfer_end = time.time()
        compute_start = time.time()
        if not logged_device:
            logger.info(
                "First train batch device: xb=%s yb=%s wb=%s",
                xb.device,
                yb.device,
                wb.device,
            )
            logged_device = True

        yb_loss = _apply_label_smoothing(yb, cfg)
        class_w = _class_weight_tensor(yb, cfg)

        with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=cfg.use_amp):
            logits = model(xb)
            loss_raw = loss_fn(logits, yb_loss)
            loss_raw = _apply_focal(loss_raw, logits, yb, cfg)
            if class_w is not None:
                loss_raw = loss_raw * class_w
            loss = (loss_raw * wb).mean()

        loss_for_backward = loss / float(accum_steps)
        if cfg.use_amp:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if batch_idx % accum_steps == 0 or batch_idx == len(loader):
            if cfg.use_amp:
                if cfg.grad_clip_norm > 0.0 or (cfg.grad_clip_value is not None and cfg.grad_clip_value > 0.0):
                    scaler.unscale_(optimizer)
                if cfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                if cfg.grad_clip_value is not None and cfg.grad_clip_value > 0.0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                if cfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                if cfg.grad_clip_value is not None and cfg.grad_clip_value > 0.0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip_value)
                optimizer.step()
            optimizer.zero_grad(set_to_none=(cfg.zero_grad_strategy == "set_to_none"))
            if scheduler is not None and cfg.scheduler_step_unit.lower() == "batch":
                scheduler.step()

        compute_end = time.time()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            probs_flat = probs.view(-1)
            yb_flat = yb.view(-1)
            preds = (probs_flat >= cfg.threshold).float()
            total_correct += int((preds == yb_flat).sum().item())
            total_seen += int(yb_flat.numel())
            total_loss += float(loss.item()) * int(yb.numel())
            total_weight += float(wb.mean().item()) * int(yb.numel())
            total_road_fraction += float(wb.mean().item()) * int(yb.numel())

        if log_every and batch_idx % log_every == 0:
            if device.type == "cuda":
                _maybe_sync(device)
            elapsed = time.time() - epoch_start
            transfer_time = transfer_end - transfer_start
            compute_time = compute_end - compute_start
            total_time = compute_end - batch_start
            gpu_mem_alloc = 0.0
            gpu_mem_reserved = 0.0
            if cfg.perf_log_gpu_mem and device.type == "cuda":
                gpu_mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**2)
            logger.info(
                "Train batch %d elapsed=%.1fs data=%.3fs transfer=%.3fs compute=%.3fs total=%.3fs avg_batch=%.3fs gpu_mem=%.0f/%.0fMB",
                batch_idx,
                elapsed,
                data_time,
                transfer_time,
                compute_time,
                total_time,
                elapsed / float(batch_idx),
                gpu_mem_alloc,
                gpu_mem_reserved,
            )
            _write_perf_row(
                [
                    "train",
                    epoch,
                    batch_idx,
                    f"{data_time:.6f}",
                    f"{transfer_time:.6f}",
                    f"{compute_time:.6f}",
                    f"{total_time:.6f}",
                    f"{gpu_mem_alloc:.2f}",
                    f"{gpu_mem_reserved:.2f}",
                ],
                cfg,
            )
        last_iter_end = time.time()

    avg_loss = total_loss / total_seen if total_seen else 0.0
    avg_acc = total_correct / total_seen if total_seen else 0.0
    avg_weight = total_weight / total_seen if total_seen else 0.0
    avg_road_fraction = total_road_fraction / total_seen if total_seen else 0.0
    return avg_loss, avg_acc, avg_road_fraction, avg_weight


def evaluate(model, loader, device, epoch, cfg: TrainConfig, log_every=None):
    if log_every is None:
        log_every = cfg.perf_log_every * 2
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    loss_fn = build_loss(cfg)
    all_probs = []
    all_labels = []
    all_roads = []
    all_weights = []
    eval_start = time.time()
    last_iter_end = eval_start
    with torch.no_grad():
        for batch_idx, (xb, yb, wb, road_ids) in enumerate(loader, start=1):
            batch_start = time.time()
            data_time = batch_start - last_iter_end
            transfer_start = time.time()
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            transfer_end = time.time()
            compute_start = time.time()
            yb_loss = _apply_label_smoothing(yb, cfg)
            class_w = _class_weight_tensor(yb, cfg)

            with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg), enabled=cfg.use_amp):
                logits = model(xb)
                loss_raw = loss_fn(logits, yb_loss)
                loss_raw = _apply_focal(loss_raw, logits, yb, cfg)
                if class_w is not None:
                    loss_raw = loss_raw * class_w
                loss = (loss_raw * wb).mean()

            probs = torch.sigmoid(logits)
            probs_flat = probs.view(-1)
            yb_flat = yb.view(-1)
            preds = (probs_flat >= cfg.threshold).float()
            total_correct += int((preds == yb_flat).sum().item())
            total_seen += int(yb_flat.numel())
            total_loss += float(loss.item()) * int(yb.numel())
            all_probs.extend(probs_flat.cpu().numpy().tolist())
            all_labels.extend(yb_flat.cpu().numpy().tolist())
            all_weights.extend(wb.view(-1).cpu().numpy().tolist())
            all_roads.extend(list(road_ids))
            compute_end = time.time()
            if log_every and batch_idx % log_every == 0:
                if device.type == "cuda":
                    _maybe_sync(device)
                elapsed = time.time() - eval_start
                transfer_time = transfer_end - transfer_start
                compute_time = compute_end - compute_start
                total_time = compute_end - batch_start
                gpu_mem_alloc = 0.0
                gpu_mem_reserved = 0.0
                if cfg.perf_log_gpu_mem and device.type == "cuda":
                    gpu_mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                    gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                logger.info(
                    "Eval batch %d elapsed=%.1fs data=%.3fs transfer=%.3fs compute=%.3fs total=%.3fs avg_batch=%.3fs gpu_mem=%.0f/%.0fMB",
                    batch_idx,
                    elapsed,
                    data_time,
                    transfer_time,
                    compute_time,
                    total_time,
                    elapsed / float(batch_idx),
                    gpu_mem_alloc,
                    gpu_mem_reserved,
                )
                _write_perf_row(
                    [
                        "eval",
                        epoch,
                        batch_idx,
                        f"{data_time:.6f}",
                        f"{transfer_time:.6f}",
                        f"{compute_time:.6f}",
                        f"{total_time:.6f}",
                        f"{gpu_mem_alloc:.2f}",
                        f"{gpu_mem_reserved:.2f}",
                    ],
                    cfg,
                )
            last_iter_end = time.time()
    avg_loss = total_loss / total_seen if total_seen else 0.0
    avg_acc = total_correct / total_seen if total_seen else 0.0
    return avg_loss, avg_acc, np.array(all_probs), np.array(all_labels), all_roads, np.array(all_weights)


def _compute_road_accuracy(probs, labels, road_ids, weights, cfg: TrainConfig):
    road_probs = {}
    road_weights = {}
    road_labels = {}
    for prob, rid, true_label, weight in zip(probs, road_ids, labels, weights):
        road_probs.setdefault(rid, []).append(float(prob))
        road_weights.setdefault(rid, []).append(float(weight))
        road_labels.setdefault(rid, int(true_label))

    road_pred = {}
    road_pred_weighted = {}
    for rid, probs_for_road in road_probs.items():
        weights_for_road = road_weights.get(rid, [])
        road_pred[rid] = sum(probs_for_road) / len(probs_for_road) if probs_for_road else 0.0
        if weights_for_road:
            weight_sum = sum(weights_for_road)
            if weight_sum > 0:
                road_pred_weighted[rid] = (
                    sum(p * w for p, w in zip(probs_for_road, weights_for_road)) / weight_sum
                )
            else:
                road_pred_weighted[rid] = road_pred[rid]
        else:
            road_pred_weighted[rid] = road_pred[rid]

    road_pred_bin = {rid: int(prob >= cfg.threshold) for rid, prob in road_pred.items()}
    road_pred_weighted_bin = {
        rid: int(prob >= cfg.threshold) for rid, prob in road_pred_weighted.items()
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
    return road_acc, road_acc_weighted


# -----------------------------
# Training logic
# -----------------------------


def train_on_tiles(cfg: TrainConfig):
    logger.info("Script startup. python=%s cwd=%s", sys.version.split()[0], Path.cwd())
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    logger.info("Global seed set: %d", cfg.seed)

    device = torch.device(cfg.device)
    logger.info("PyTorch device: %s", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32_allowed
        torch.backends.cudnn.allow_tf32 = cfg.tf32_allowed
    if device.type == "cuda":
        try:
            logger.info(
                "CUDA devices: count=%d current=%d name=%s",
                torch.cuda.device_count(),
                torch.cuda.current_device(),
                torch.cuda.get_device_name(torch.cuda.current_device()),
            )
        except Exception as exc:
            logger.warning("Failed to query CUDA device info: %s", exc)

    cpu_count = multiprocessing.cpu_count()
    if cfg.num_workers and cfg.num_workers > 0:
        data_workers = max(1, min(cfg.num_workers, cpu_count))
    else:
        data_workers = max(1, cpu_count)
    logger.info("DataLoader workers: %d (cpu_count=%d)", data_workers, cpu_count)

    logger.info("Tile fetch phase begin.")
    fetch_phase_start = time.time()
    logger.info(
        "Patch config: patch_px=%d, patch_m=%.1f, m_per_px=%.3f",
        cfg.patch_size,
        cfg.patch_size_m,
        cfg.patch_size_m / float(cfg.patch_size),
    )
    logger.info(
        "Augmentation enabled: %s (random_hflip, random_vflip, random_rot90, jitter_brightness_contrast)",
        "yes" if cfg.augmentations_enabled else "no",
    )
    logger.info("Early stopping enabled (patience=%d).", cfg.early_stopping_patience)
    logger.info("Label smoothing enabled (%.4f).", cfg.label_smoothing)
    logger.info("Soft road emphasis applied to RGB channels.")
    logger.info("Dataset construction start (gold+silver subset from blob).")

    index_df = load_index_from_blob(cfg)
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

    rng = random.Random(cfg.seed)
    rng.shuffle(gravel_roads)
    rng.shuffle(paved_roads)
    split_loaded = False
    train_roads = set()
    val_roads = None
    eval_roads = None
    if cfg.split_roads_file.exists() and not cfg.resplit:
        try:
            with open(cfg.split_roads_file, "r", encoding="utf-8") as handle:
                split_data = json.load(handle)
            train_roads = {str(rid) for rid in split_data.get("train", [])}
            val_roads = {str(rid) for rid in split_data.get("val", [])}
            eval_roads = {str(rid) for rid in split_data.get("eval", [])}
            split_loaded = True
            logger.info("Road split loaded from disk: %s", cfg.split_roads_file)
        except Exception as exc:
            logger.warning("Failed to load split file %s: %s", cfg.split_roads_file, exc)
            split_loaded = False
            train_roads = set()
            val_roads = None
            eval_roads = None
    elif cfg.resplit:
        logger.info("RESPLIT enabled; regenerating road splits.")

    if not split_loaded:
        def _split_class(roads, val_frac, eval_frac):
            total = len(roads)
            val_count = max(1, int(round(total * val_frac))) if total else 0
            eval_count = max(1, int(round(total * eval_frac))) if total else 0
            val_count = min(val_count, total)
            eval_count = min(eval_count, total - val_count)
            train_count = max(0, total - val_count - eval_count)
            train_ids = roads[:train_count]
            val_ids = roads[train_count:train_count + val_count]
            eval_ids = roads[train_count + val_count:train_count + val_count + eval_count]
            return set(train_ids), set(val_ids), set(eval_ids)

        train_g, val_g, eval_g = _split_class(gravel_roads, cfg.val_split, cfg.test_split)
        train_p, val_p, eval_p = _split_class(paved_roads, cfg.val_split, cfg.test_split)
        train_roads = train_g | train_p
        val_roads = val_g | val_p
        eval_roads = eval_g | eval_p

    if eval_roads is None:
        if cfg.eval_roads_file.exists():
            try:
                with open(cfg.eval_roads_file, "r", encoding="utf-8") as handle:
                    eval_data = json.load(handle)
                eval_roads = {str(rid) for rid in eval_data}
                eval_roads = {rid for rid in eval_roads if rid in roads_with_patches}
                logger.info("Eval road set loaded from disk: %s", cfg.eval_roads_file)
            except Exception as exc:
                logger.warning("Failed to load eval road set from %s: %s", cfg.eval_roads_file, exc)
                eval_roads = None

    if eval_roads is None:
        eval_candidates = set(roads_with_patches.keys()) - train_roads - val_roads
        if not eval_candidates:
            eval_candidates = set(roads_with_patches.keys())
        eval_gravel = [rid for rid in eval_candidates if roads_with_patches[rid] == 1]
        eval_paved = [rid for rid in eval_candidates if roads_with_patches[rid] == 0]
        rng_eval = random.Random(cfg.seed)
        rng_eval.shuffle(eval_gravel)
        rng_eval.shuffle(eval_paved)
        eval_take_gravel = min(len(eval_gravel), cfg.eval_roads_per_class)
        eval_take_paved = min(len(eval_paved), cfg.eval_roads_per_class)
        eval_roads = set(eval_gravel[:eval_take_gravel] + eval_paved[:eval_take_paved])

    train_roads = {rid for rid in train_roads if rid in roads_with_patches}
    val_roads = {rid for rid in val_roads if rid in roads_with_patches}
    eval_roads = {rid for rid in eval_roads if rid in roads_with_patches}
    if not eval_roads:
        sys.exit("Eval split has 0 roads. Delete split file or set RESPLIT=1 to regenerate.")
    overlap = (train_roads & val_roads) | (train_roads & eval_roads) | (val_roads & eval_roads)
    if overlap:
        logger.warning("Overlap detected in road splits; de-duplicating %d roads", len(overlap))
        val_roads -= train_roads
        eval_roads -= (train_roads | val_roads)

    if not split_loaded:
        try:
            with open(cfg.split_roads_file, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "train": sorted(train_roads),
                        "val": sorted(val_roads),
                        "eval": sorted(eval_roads),
                    },
                    handle,
                )
            logger.info("Road split saved to disk: %s", cfg.split_roads_file)
        except Exception as exc:
            logger.warning("Failed to save road split to %s: %s", cfg.split_roads_file, exc)

        try:
            with open(cfg.val_roads_file, "w", encoding="utf-8") as handle:
                json.dump(sorted(val_roads), handle)
            logger.info("Validation road set saved to disk: %s", cfg.val_roads_file)
        except Exception as exc:
            logger.warning("Failed to save validation road set to %s: %s", cfg.val_roads_file, exc)
        try:
            with open(cfg.eval_roads_file, "w", encoding="utf-8") as handle:
                json.dump(sorted(eval_roads), handle)
            logger.info("Eval road set saved to disk: %s", cfg.eval_roads_file)
        except Exception as exc:
            logger.warning("Failed to save eval road set to %s: %s", cfg.eval_roads_file, exc)

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

    eval_gravel_used = sum(1 for rid in eval_roads if roads_with_patches.get(rid) == 1)
    eval_paved_used = sum(1 for rid in eval_roads if roads_with_patches.get(rid) == 0)
    logger.info(
        "Eval roads per class. paved=%d gravel=%d",
        eval_paved_used,
        eval_gravel_used,
    )
    _split_fingerprint("train", train_roads, cfg.seed)
    _split_fingerprint("val", val_roads, cfg.seed)
    _split_fingerprint("eval", eval_roads, cfg.seed)
    split_hash = _split_hash(train_roads, val_roads, eval_roads)

    train_idx = [i for i, rid in enumerate(road_ids) if rid in train_roads]
    val_idx = [i for i, rid in enumerate(road_ids) if rid in val_roads]
    eval_idx = [i for i, rid in enumerate(road_ids) if rid in eval_roads]
    road_ids_arr = np.array(road_ids, dtype=object)

    patch_service = connect_patch_storage(cfg)
    train_rows = index_df.iloc[train_idx].reset_index(drop=True)
    val_rows = index_df.iloc[val_idx].reset_index(drop=True)
    eval_rows = index_df.iloc[eval_idx].reset_index(drop=True)
    eval_rows = _deterministic_eval_order(eval_rows, cfg)
    if eval_rows.empty:
        sys.exit("Eval split is empty or missing; cannot compute evaluation metrics.")
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
    _log_split_counts("Eval", eval_rows)

    train_weights = compute_mask_means(train_rows, patch_service, cfg, split_hash, "train")
    val_weights = compute_mask_means(val_rows, patch_service, cfg, split_hash, "val")
    eval_weights = compute_mask_means(eval_rows, patch_service, cfg, split_hash, "eval")

    def _trim_to_cache(rows, weights, split_label):
        if len(rows) > len(weights):
            logger.info(
                "Trimming %s_rows to cached dataset size (ignoring %d extra Azure rows).",
                split_label,
                len(rows) - len(weights),
            )
            rows = rows.iloc[:len(weights)].copy()
        else:
            rows = rows.copy()
        rows["sample_weight"] = weights[:len(rows)]
        return rows

    train_rows = _trim_to_cache(train_rows, train_weights, "train")
    val_rows = _trim_to_cache(val_rows, val_weights, "val")
    eval_rows = _trim_to_cache(eval_rows, eval_weights, "eval")

    y_train = train_rows["label_int"].to_numpy(dtype=np.int32)
    y_val = val_rows["label_int"].to_numpy(dtype=np.int32)
    y_eval = eval_rows["label_int"].to_numpy(dtype=np.int32)
    road_train = train_rows["road_id"].astype(str).to_numpy()
    road_val = val_rows["road_id"].astype(str).to_numpy()
    road_eval = eval_rows["road_id"].astype(str).to_numpy()

    train_loader = build_dataloader(cfg, train_rows, augment=True, shuffle=True, device=device)
    val_loader = build_dataloader(cfg, val_rows, augment=False, shuffle=False, device=device)
    eval_loader = build_dataloader(cfg, eval_rows, augment=False, shuffle=False, device=device)

    logger.info("Dataset construction end (road-level split).")

    train_roads_used = len(set(road_train))
    val_roads_used = len(set(road_val))
    patches_per_road = len(road_ids_arr) / len(set(road_ids_arr)) if road_ids_arr.size else 0.0
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
    model = build_model(cfg).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(model, cfg)
    accum_steps = max(1, cfg.grad_accum_steps)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / float(accum_steps)))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = build_scaler(cfg)

    if cfg.resume_training and cfg.resume_path and cfg.resume_path.exists():
        try:
            model.load_state_dict(torch.load(cfg.resume_path, map_location=device))
            logger.info("Resumed model from %s", cfg.resume_path)
        except Exception as exc:
            logger.warning("Failed to resume model from %s: %s", cfg.resume_path, exc)

    logger.info("Training start.")
    train_start = time.time()
    curriculum_epochs = max(1, int(cfg.epochs * cfg.curriculum_fraction))
    curriculum_mask = train_rows["sample_weight"] >= cfg.curriculum_min_weight
    train_rows_curr = train_rows[curriculum_mask].reset_index(drop=True)
    if train_rows_curr.empty:
        train_rows_curr = train_rows
    avg_rf_curr = float(train_rows_curr["sample_weight"].mean()) if len(train_rows_curr) else 0.0
    avg_rf_full = float(train_rows["sample_weight"].mean()) if len(train_rows) else 0.0

    curriculum_loader = build_dataloader(cfg, train_rows_curr, augment=True, shuffle=True, device=device)

    best_val_road_acc = None
    patience_left = cfg.early_stopping_patience

    for epoch in range(1, curriculum_epochs + 1):
        train_loss, train_acc, avg_road_fraction, avg_loss_weight = train_one_epoch(
            model, curriculum_loader, optimizer, scheduler, scaler, device, epoch, cfg
        )
        do_val = cfg.validation_frequency > 0 and epoch % cfg.validation_frequency == 0
        if do_val:
            val_loss, val_acc, val_probs, val_labels, val_road_ids, val_weights = evaluate(
                model, val_loader, device, epoch, cfg
            )
            val_road_acc, _ = _compute_road_accuracy(
                val_probs, val_labels, val_road_ids, val_weights, cfg
            )
        else:
            val_loss, val_acc = 0.0, 0.0
        _epoch_log(epoch, train_loss, train_acc, val_loss, val_acc, train_counts, val_counts)
        _weight_log(epoch, avg_road_fraction, avg_loss_weight)
        if scheduler is not None and cfg.scheduler_step_unit.lower() == "epoch":
            scheduler.step()

        if do_val:
            improved = (
                best_val_road_acc is None
                or val_road_acc > best_val_road_acc + cfg.early_stopping_min_delta
            )
            if improved:
                best_val_road_acc = val_road_acc
                patience_left = cfg.early_stopping_patience
                if cfg.save_best_only:
                    torch.save(model.state_dict(), cfg.model_out)
            elif epoch >= cfg.early_stopping_min_epoch:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("Early stopping triggered during curriculum.")
                    break
        if cfg.checkpoint_frequency > 0 and epoch % cfg.checkpoint_frequency == 0 and not cfg.save_best_only:
            torch.save(model.state_dict(), cfg.model_out)

    for epoch in range(curriculum_epochs + 1, cfg.epochs + 1):
        train_loss, train_acc, avg_road_fraction, avg_loss_weight = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch, cfg
        )
        do_val = cfg.validation_frequency > 0 and epoch % cfg.validation_frequency == 0
        if do_val:
            val_loss, val_acc, val_probs, val_labels, val_road_ids, val_weights = evaluate(
                model, val_loader, device, epoch, cfg
            )
            val_road_acc, _ = _compute_road_accuracy(
                val_probs, val_labels, val_road_ids, val_weights, cfg
            )
        else:
            val_loss, val_acc = 0.0, 0.0
        _epoch_log(epoch, train_loss, train_acc, val_loss, val_acc, train_counts, val_counts)
        _weight_log(epoch, avg_road_fraction, avg_loss_weight)
        if scheduler is not None and cfg.scheduler_step_unit.lower() == "epoch":
            scheduler.step()

        if do_val:
            improved = (
                best_val_road_acc is None
                or val_road_acc > best_val_road_acc + cfg.early_stopping_min_delta
            )
            if improved:
                best_val_road_acc = val_road_acc
                patience_left = cfg.early_stopping_patience
                if cfg.save_best_only:
                    torch.save(model.state_dict(), cfg.model_out)
            elif epoch >= cfg.early_stopping_min_epoch:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info("Early stopping triggered.")
                    break
        if cfg.checkpoint_frequency > 0 and epoch % cfg.checkpoint_frequency == 0 and not cfg.save_best_only:
            torch.save(model.state_dict(), cfg.model_out)

    logger.info("Training complete. elapsed=%.2fs", time.time() - train_start)

    # Load best model for evaluation
    if cfg.model_out.exists():
        model.load_state_dict(torch.load(cfg.model_out, map_location=device))
        logger.info("Model saved to %s", cfg.model_out)

    logger.info("Evaluation start.")
    _, _, preds, y_val_probs, road_val_ids, val_weights = evaluate(
        model, val_loader, device, cfg.epochs, cfg
    )
    patch_pred = (preds >= cfg.threshold).astype(int)
    patch_acc = float((patch_pred == y_val).mean()) if len(y_val) else 0.0

    road_probs = {}
    road_weights = {}
    road_labels = {}
    for prob, rid, true_label, weight in zip(preds, road_val_ids, y_val, val_weights):
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
        if count > cfg.eval_trim_count_threshold:
            keep_count = max(1, int(round(count * cfg.eval_trim_keep_ratio)))
            ranked = sorted(
                zip(probs, weights),
                key=lambda item: abs(item[0] - cfg.threshold),
                reverse=True,
            )[:keep_count]
            probs = [p for p, _ in ranked]
            weights = [w for _, w in ranked]
        per_road_counts_trimmed.append(len(probs))
        road_pred[rid] = sum(probs) / len(probs) if probs else 0.0
        clipped = [float(np.clip(w, cfg.eval_weight_clip_min, cfg.eval_weight_clip_max)) for w in weights] if weights else []
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
    road_pred_bin = {rid: int(prob >= cfg.threshold) for rid, prob in road_pred.items()}
    road_pred_weighted_bin = {
        rid: int(prob >= cfg.threshold) for rid, prob in road_pred_weighted.items()
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

    eval_loss, eval_acc, eval_preds, eval_labels, eval_road_ids, _ = evaluate(
        model, eval_loader, device, cfg.epochs, cfg
    )
    eval_patch_acc = float(((eval_preds >= cfg.threshold).astype(int) == y_eval).mean())
    eval_road_probs = {}
    eval_road_labels = {}
    eval_road_counts = []
    eval_road_counts_trimmed = []
    for prob, rid, true_label in zip(eval_preds, eval_road_ids, y_eval):
        eval_road_probs.setdefault(rid, []).append(float(prob))
        eval_road_labels.setdefault(rid, int(true_label))
    eval_road_pred = {}
    for rid, probs in eval_road_probs.items():
        count = len(probs)
        eval_road_counts.append(count)
        if count > cfg.eval_trim_count_threshold:
            keep_count = max(1, int(round(count * cfg.eval_trim_keep_ratio)))
            ranked = sorted(
                probs,
                key=lambda p: abs(p - cfg.threshold),
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
    eval_road_pred_bin = {rid: int(prob >= cfg.threshold) for rid, prob in eval_road_pred.items()}
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
    logger.info("Evaluation end.")

    logger.info("==== FINAL METRICS SUMMARY ====")
    logger.info("Val patch acc: %.4f", patch_acc)
    logger.info("Val road acc: %.4f (weighted: %.4f)", road_acc, road_acc_weighted)
    logger.info("Val confusion (tn, fp, fn, tp): %d, %d, %d, %d", tn, fp, fn, tp)
    logger.info("Eval patch acc: %.4f", eval_patch_acc)
    logger.info("Eval road acc: %.4f", eval_road_acc)
    logger.info("Eval confusion (tn, fp, fn, tp): %d, %d, %d, %d", eval_tn, eval_fp, eval_fn, eval_tp)
    logger.info("==== END SUMMARY ====")

    return


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    try:
        train_on_tiles(CFG)
    except Exception:
        logger.error("Training failed with exception.")
        logger.error(traceback.format_exc())
        raise
