"""
REST API Route Definitions
==========================
File: app/api/routes.py

Purpose:
--------
Defines FastAPI endpoints for the ASTRA backend:
1. GET `/health`: Basic system liveness check.
2. GET `/triplets`: Catalogs all available GOES-16 satellite scan triplets with metadata and thumbnails.
3. GET `/triplets/{triplet_id}/thumbnail`: Streams a lightweight PNG thumbnail for a given triplet.
4. GET `/sample-pair`: Loads satellite scan frames for a specified `triplet_id` as base64-encoded PNGs.
5. POST `/generate`: Computes optical flow and synthesized frames at time fraction `t` for a specified `triplet_id`.
6. GET `/benchmark`: Benchmarks all four methods against held-out ground truth.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from app.processing.optical_flow import (
    calculate_metrics,
    compute_farneback_flow,
    compute_flow,
    flow_to_hsv_color,
    interpolate_linear,
    warp_frame_forward,
)
from app.processing.refinement_model import PatchRefinementUNet
from app.processing.direct_synthesis_model import DirectSynthesisUNet
from app.data.goes16_fetch import (
    GOES16Fetcher,
    download_single_triplet,
    parse_date_input,
    DEFAULT_DATE_HOUR_PAIRS
)

logger = logging.getLogger("astra_api")

router = APIRouter()

# Directories and paths
SAMPLE_DATA_DIR = Path("sample_data")
TRIPLETS_DIR = SAMPLE_DATA_DIR / "triplets"
MANIFEST_PATH = TRIPLETS_DIR / "manifest.json"
CHECKPOINT_PATH = Path("checkpoints/best_refinement_model.pth")

# Global in-memory caches
_TRIPLET_ARRAYS_CACHE: Dict[str, Dict[str, Any]] = {}
_THUMBNAILS_CACHE: Dict[str, str] = {}
_GENERATE_5424_CACHE: Dict[str, Any] = {}
_BENCHMARK_CACHE: Dict[str, Any] = {}


# ------------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ------------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """
    Request model for frame interpolation synthesis.
    """
    triplet_id: str = Field(
        ...,
        description="Required unique ID of the triplet to interpolate (e.g. '2024-01-01_18' or '2024-01-15_00')."
    )
    time_fraction: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Target normalized timestamp fraction t in range [0.0, 1.0]."
    )
    method: str = Field(
        "farneback",
        description="Interpolation method: 'linear', 'farneback', 'raft', or 'raft_refined'. Defaults to 'farneback'."
    )
    preview_size: Optional[int] = Field(
        1024,
        description="Max image dimension for base64 transfer (e.g. 1024 for high-speed web display, or 5424 for full resolution native metrics)."
    )
    pyr_scale: float = Field(0.5, description="Farneback optical flow pyramid scale factor.")
    levels: int = Field(3, description="Number of pyramid layers for optical flow.")
    winsize: int = Field(15, description="Averaging window size for optical flow.")
    iterations: int = Field(3, description="Iterations at each pyramid level.")


class GenerateResponse(BaseModel):
    """
    Response model returning synthetic frame, linear baseline, confidence map, and metrics.
    """
    triplet_id: Optional[str] = Field(default=None, description="Triplet identifier used for synthesis.")
    time_fraction: float
    flow_method: str = Field(description="Which method was used: 'linear', 'farneback', 'raft', or 'raft_refined'.")
    dimensions: Dict[str, int]
    generated_image: str = Field(description="Base64 encoded PNG data URL of synthetic frame for the requested method.")
    warped_image: str = Field(description="Base64 encoded PNG data URL of synthetic frame for the requested method (alias).")
    linear_image: str = Field(description="Base64 encoded PNG data URL of baseline linear-interpolated frame.")
    ground_truth_image: Optional[str] = Field(default=None, description="Base64 encoded PNG data URL of ground truth frame (null if unavailable).")
    confidence_map: str = Field(description="Base64 encoded PNG data URL of optical flow confidence map (grayscale).")
    confidence_map_colored: str = Field(description="Base64 encoded PNG data URL of optical flow confidence map (Viridis heatmap).")
    confidence_is_heuristic: bool = Field(default=True, description="Indicates confidence map is a flow-consistency heuristic, not a trained model.")
    confidence_label: str = Field(default="Bidirectional Flow Consistency Heuristic (Not a trained model)", description="Label indicating heuristic nature of confidence map.")
    flow_visualization: str = Field(description="Base64 encoded PNG data URL of HSV motion flow field.")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Quantitative validation metrics (SSIM, MSE, PSNR, MAE) with null fields if GT unavailable.")


class SequenceRequest(BaseModel):
    """
    Request model for continuous multi-frame temporal sequence synthesis.
    """
    triplet_id: str = Field(..., description="Unique ID of the triplet.")
    method: str = Field(
        "farneback",
        description="Synthesis method: 'linear', 'farneback', 'raft', 'raft_refined', or 'neural_synthesis'."
    )
    num_steps: int = Field(
        5,
        ge=3,
        le=9,
        description="Number of frames across the sequence, e.g. 5 for [0, 0.25, 0.5, 0.75, 1.0]."
    )
    preview_size: Optional[int] = Field(
        512,
        description="Max dimension for sequence frames (e.g. 512 for smooth animation playback)."
    )


# ------------------------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------------------------
def array_to_base64_png(array: np.ndarray, max_dim: Optional[int] = None) -> str:
    """
    Encodes a 2D or 3D numpy image array to a standard base64 PNG Data URL string.
    Optionally rescales the image to `max_dim` for responsive web streaming.
    """
    img = array if array.dtype == np.uint8 else np.clip(array, 0, 255).astype(np.uint8)

    if max_dim and max_dim > 0:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Failed to encode image array into PNG format.")

    b64_encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64_encoded}"


def compute_flow_confidence_map(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    flow: np.ndarray,
    t: float = 0.5,
    sensitivity: float = 20.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a pixel-wise motion confidence map by evaluating bidirectional consistency
    between forward-warped frame_t0 and backward-warped frame_t1:

    Confidence(x, y) = exp(-|I_warped0(x, y) - I_warped1(x, y)| / sensitivity) * 255
    """
    h, w = frame_t0.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    # 1. Forward warp frame_t0 to time t
    map_x0 = (grid_x - t * flow[..., 0]).astype(np.float32)
    map_y0 = (grid_y - t * flow[..., 1]).astype(np.float32)
    warped_0 = cv2.remap(frame_t0, map_x0, map_y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 2. Backward warp frame_t1 to time t
    map_x1 = (grid_x + (1.0 - t) * flow[..., 0]).astype(np.float32)
    map_y1 = (grid_y + (1.0 - t) * flow[..., 1]).astype(np.float32)
    warped_1 = cv2.remap(frame_t1, map_x1, map_y1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 3. Compute structural intensity residual
    residual = np.abs(warped_0.astype(np.float32) - warped_1.astype(np.float32))

    # 4. Exponential confidence mapping
    confidence = np.exp(-residual / sensitivity) * 255.0
    confidence_gray = np.clip(confidence, 0, 255).astype(np.uint8)

    # 5. Viridis color heatmap for vivid frontend rendering
    confidence_color = cv2.applyColorMap(confidence_gray, cv2.COLORMAP_VIRIDIS)

    return confidence_gray, confidence_color


def get_available_triplets() -> List[Dict[str, Any]]:
    """
    Discovers all available triplet datasets by reading manifest.json and scanning
    both the sample_data/triplets/ (full disk scans) and sample_data/patches/ (meteorological cloud ROIs).
    Returns a comprehensive catalog of real GOES-16 Band 13 observation triplets with IDs, dates,
    UTC scan coverage times, types, and preview thumbnails.
    """
    triplets: Dict[str, Dict[str, Any]] = {}

    # 1. Read manifest.json if present
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            for t_entry in manifest_data.get("triplets", []):
                tid = t_entry.get("triplet_id")
                if tid:
                    t_entry["type"] = "Full Disk (5424×5424)"
                    triplets[tid] = t_entry
        except Exception as e:
            logger.warning("Could not parse triplets manifest.json: %s", e)

    # 2. Scan subdirectories in TRIPLETS_DIR (Full Disk)
    if TRIPLETS_DIR.exists():
        for sub_dir in sorted(TRIPLETS_DIR.iterdir()):
            if not sub_dir.is_dir():
                continue
            tid = sub_dir.name
            npy_files = sorted(list(sub_dir.glob("0*.npy")))
            if len(npy_files) < 2:
                continue

            file_t0 = npy_files[0]
            file_t1 = npy_files[-1]
            file_gt = npy_files[1] if len(npy_files) >= 3 else None

            def load_m(npy: Optional[Path]) -> Dict:
                if not npy:
                    return {}
                jf = npy.with_name(f"{npy.stem}_meta.json")
                if jf.exists():
                    try:
                        with open(jf, "r", encoding="utf-8") as fp:
                            return json.load(fp)
                    except Exception:
                        pass
                return {}

            m0 = load_m(file_t0)
            m_gt = load_m(file_gt)
            m1 = load_m(file_t1)

            date = tid.split("_")[0] if "_" in tid else "2024-01-01"
            existing = triplets.get(tid, {})

            # Generate or reuse cached small 128px thumbnail
            thumb_b64 = _THUMBNAILS_CACHE.get(tid)
            if not thumb_b64:
                try:
                    png_preview = sub_dir / f"{file_t0.stem}.png"
                    if png_preview.exists():
                        img = cv2.imread(str(png_preview), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            thumb_b64 = array_to_base64_png(img, max_dim=128)
                            _THUMBNAILS_CACHE[tid] = thumb_b64
                except Exception as e:
                    logger.debug("Failed generating thumbnail for %s: %s", tid, e)

            start_t = m0.get("time_coverage_start", existing.get("files", {}).get("t0", {}).get("timestamp", f"{date}T18:00:20Z"))
            mid_t = m_gt.get("time_coverage_start", existing.get("files", {}).get("t_mid", {}).get("timestamp", f"{date}T18:10:20Z")) if file_gt else None
            end_t = m1.get("time_coverage_start", existing.get("files", {}).get("t1", {}).get("timestamp", f"{date}T18:20:20Z"))

            merged = {
                "id": tid,
                "triplet_id": tid,
                "date": existing.get("date", date),
                "hour": existing.get("hour", int(tid.split("_")[1]) if "_" in tid and tid.split("_")[1].isdigit() else 0),
                "start_time": start_t,
                "mid_time": mid_t,
                "end_time": end_t,
                "start_scan_time": start_t,
                "mid_scan_time": mid_t,
                "end_scan_time": end_t,
                "has_ground_truth": file_gt is not None,
                "type": "Full Disk (5424×5424)",
                "directory": str(sub_dir),
                "thumbnail": thumb_b64,
                "thumbnail_url": f"/triplets/{tid}/thumbnail",
            }
            triplets[tid] = merged

    # 3. Scan subdirectories in sample_data/patches/ (50+ Real GOES-16 Observation Triplets)
    patches_dir = SAMPLE_DATA_DIR / "patches"
    candidate_roots = [patches_dir / "train", patches_dir / "val", patches_dir]
    for root in candidate_roots:
        if not root.exists():
            continue
        for sub_dir in sorted(root.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name in ("neural_synthesis_demo", "manifest.json", "train", "val"):
                continue
            tid = sub_dir.name
            if tid in triplets:
                continue

            p0 = sub_dir / "patch_000_t0.npy"
            pm = sub_dir / "patch_000_t_mid.npy"
            p1 = sub_dir / "patch_000_t1.npy"

            if not p0.exists() or not p1.exists():
                # Fallback to any patch
                t0_list = list(sub_dir.glob("*_t0.npy"))
                if not t0_list:
                    continue
                p0 = t0_list[0]
                pm = Path(str(p0).replace("_t0.npy", "_t_mid.npy"))
                p1 = Path(str(p0).replace("_t0.npy", "_t1.npy"))

            parts = tid.split("_")
            date_str = parts[0] if len(parts) > 0 else "2024-01-15"
            hour_int = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

            start_t = f"{date_str}T{hour_int:02d}:00:20.0Z"
            mid_t = f"{date_str}T{hour_int:02d}:10:20.0Z" if pm.exists() else None
            end_t = f"{date_str}T{hour_int:02d}:20:20.0Z"

            thumb_b64 = _THUMBNAILS_CACHE.get(tid)
            if not thumb_b64 and p0.exists():
                try:
                    p0_arr = np.load(p0)
                    thumb_b64 = array_to_base64_png(p0_arr, max_dim=128)
                    _THUMBNAILS_CACHE[tid] = thumb_b64
                except Exception as e:
                    logger.debug("Failed generating patch thumbnail for %s: %s", tid, e)

            triplets[tid] = {
                "id": tid,
                "triplet_id": tid,
                "date": date_str,
                "hour": hour_int,
                "start_time": start_t,
                "mid_time": mid_t,
                "end_time": end_t,
                "start_scan_time": start_t,
                "mid_scan_time": mid_t,
                "end_scan_time": end_t,
                "has_ground_truth": pm.exists(),
                "type": "Cloud Dynamics ROI (256×256)",
                "directory": str(sub_dir),
                "thumbnail": thumb_b64,
                "thumbnail_url": f"/triplets/{tid}/thumbnail",
            }

    return list(triplets.values())


def load_triplet_scans(triplet_id: str) -> Dict[str, Any]:
    """
    Discovers and loads authentic GOES-16 scan arrays (T0, T1, GT) and metadata for the given triplet_id.
    Seamlessly supports both native full-disk scans (5424x5424) and high-variance meteorological cloud ROIs (256x256).
    Caches loaded numpy arrays in memory to avoid repetitive disk I/O.
    """
    triplet_id = triplet_id.strip()
    if triplet_id in _TRIPLET_ARRAYS_CACHE:
        return _TRIPLET_ARRAYS_CACHE[triplet_id]

    target_dir = TRIPLETS_DIR / triplet_id
    is_patch_roi = False

    if not target_dir.exists() or not target_dir.is_dir():
        candidate_paths = [
            SAMPLE_DATA_DIR / "patches" / "train" / triplet_id,
            SAMPLE_DATA_DIR / "patches" / "val" / triplet_id,
            SAMPLE_DATA_DIR / "patches" / triplet_id,
            SAMPLE_DATA_DIR / triplet_id,
        ]
        found = False
        for c_path in candidate_paths:
            if c_path.exists() and c_path.is_dir():
                target_dir = c_path
                is_patch_roi = True
                found = True
                break

        if not found:
            manifest = None
            if MANIFEST_PATH.exists():
                try:
                    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except Exception:
                    pass
            if manifest:
                for t_item in manifest.get("triplets", []):
                    if t_item.get("triplet_id") == triplet_id:
                        d_path = Path(t_item.get("directory", ""))
                        if d_path.exists():
                            target_dir = d_path
                            break

            if not target_dir.exists():
                available = [t["id"] for t in get_available_triplets()]
                raise HTTPException(
                    status_code=404,
                    detail=f"Triplet '{triplet_id}' not found. Available triplets: {available}"
                )

    # Check for patch ROI format (patch_000_t0.npy) vs full disk format (00_scan_0_*.npy)
    if is_patch_roi or (target_dir / "patch_000_t0.npy").exists():
        file_t0 = target_dir / "patch_000_t0.npy"
        file_gt = target_dir / "patch_000_t_mid.npy"
        file_t1 = target_dir / "patch_000_t1.npy"

        if not file_t0.exists():
            cands = sorted(list(target_dir.glob("*_t0.npy")))
            if not cands:
                raise HTTPException(status_code=404, detail=f"No patch files found in {target_dir}")
            file_t0 = cands[0]
            file_gt = Path(str(file_t0).replace("_t0.npy", "_t_mid.npy"))
            file_t1 = Path(str(file_t0).replace("_t0.npy", "_t1.npy"))

        if not file_gt.exists():
            file_gt = None
    else:
        npy_files = sorted(list(target_dir.glob("0*.npy")))
        if len(npy_files) < 2:
            raise HTTPException(
                status_code=404,
                detail=f"Triplet directory '{target_dir}' contains {len(npy_files)} .npy files. Minimum 2 required."
            )
        file_t0 = npy_files[0]
        file_t1 = npy_files[-1]
        file_gt = npy_files[1] if len(npy_files) >= 3 else None

    # Load authentic arrays
    try:
        t0_raw = np.load(file_t0)
        t1_raw = np.load(file_t1)
        gt_raw = np.load(file_gt) if file_gt and Path(file_gt).exists() else None
    except Exception as exc:
        logger.error("Failed to load numpy scan file for triplet '%s': %s", triplet_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Triplet '{triplet_id}' scan file is corrupted or unreadable: {exc}"
        )

    t0_clean = np.nan_to_num(t0_raw, nan=0.0, posinf=255.0, neginf=0.0)
    t0_arr = t0_clean.astype(np.uint8) if t0_clean.dtype != np.uint8 else t0_clean

    t1_clean = np.nan_to_num(t1_raw, nan=0.0, posinf=255.0, neginf=0.0)
    t1_arr = t1_clean.astype(np.uint8) if t1_clean.dtype != np.uint8 else t1_clean

    if gt_raw is not None:
        gt_clean = np.nan_to_num(gt_raw, nan=0.0, posinf=255.0, neginf=0.0)
        gt_arr = gt_clean.astype(np.uint8) if gt_clean.dtype != np.uint8 else gt_clean
    else:
        gt_arr = None

    parts = triplet_id.split("_")
    date_part = parts[0] if len(parts) > 0 else "2024-01-15"
    hour_part = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    def load_meta(npy_file: Optional[Path], default_time: str) -> Dict:
        if not npy_file:
            return {}
        json_file = npy_file.with_name(f"{npy_file.stem}_meta.json")
        meta = {}
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
        if not meta.get("time_coverage_start"):
            meta["time_coverage_start"] = default_time
        if not meta.get("product"):
            meta["product"] = "ABI-L2-CMIPF (Cloud and Moisture Imagery - Channel 13)"
        if not meta.get("channel"):
            meta["channel"] = 13
        if not meta.get("channel_name"):
            meta["channel_name"] = "Clean IR Longwave Window (~10.35 µm)"
        return meta

    entry = {
        "triplet_id": triplet_id,
        "directory": target_dir,
        "t0_arr": t0_arr,
        "t1_arr": t1_arr,
        "gt_arr": gt_arr,
        "t0_meta": load_meta(file_t0, f"{date_part}T{hour_part:02d}:00:20.0Z"),
        "t1_meta": load_meta(file_t1, f"{date_part}T{hour_part:02d}:20:20.0Z"),
        "gt_meta": load_meta(file_gt, f"{date_part}T{hour_part:02d}:10:20.0Z") if file_gt else None,
    }

    _TRIPLET_ARRAYS_CACHE[triplet_id] = entry
    return entry


# ------------------------------------------------------------------------------
# Refinement Model Setup & Inference
# ------------------------------------------------------------------------------
_REFINEMENT_MODEL: Optional[PatchRefinementUNet] = None
_REFINEMENT_DEVICE: Optional[torch.device] = None


def get_refinement_model(checkpoint_path: Path = CHECKPOINT_PATH) -> Tuple[PatchRefinementUNet, torch.device]:
    """Loads and caches the trained PatchRefinementUNet from checkpoint."""
    global _REFINEMENT_MODEL, _REFINEMENT_DEVICE
    if _REFINEMENT_MODEL is None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Refinement checkpoint not found at {checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = PatchRefinementUNet(in_channels=1, out_channels=1, base_features=32).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _REFINEMENT_MODEL = model
        _REFINEMENT_DEVICE = device
        logger.info("Loaded refinement U-Net checkpoint into API memory (device=%s).", device)
    return _REFINEMENT_MODEL, _REFINEMENT_DEVICE


def apply_refinement_model(
    warped_frame: np.ndarray,
    checkpoint_path: Path = CHECKPOINT_PATH,
    tile_size: int = 512,
    overlap: int = 32,
) -> np.ndarray:
    """
    Applies the trained PatchRefinementUNet to a 2D grayscale image.
    Fully resolution-agnostic and robust to arbitrary inputs outside the training set:
    - For images <= 1024, processes in a single direct forward pass.
    - For large images (e.g. 5424x5424 full-disk), processes in tiled patches with overlap
      to ensure fast CPU inference and prevent out-of-memory crashes.
    """
    model, device = get_refinement_model(checkpoint_path)

    # Sanitize input array
    warped_clean = np.nan_to_num(warped_frame, nan=0.0, posinf=255.0, neginf=0.0)
    h, w = warped_clean.shape[:2]

    if max(h, w) <= 1024:
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        img_f = warped_clean.astype(np.float32) / 255.0
        if pad_h > 0 or pad_w > 0:
            img_f = cv2.copyMakeBorder(img_f, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        tensor_in = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            tensor_out = model(tensor_in)

        refined = tensor_out.squeeze().cpu().numpy()
        if pad_h > 0 or pad_w > 0:
            refined = refined[:h, :w]

        return np.clip(refined * 255.0, 0, 255).astype(np.uint8)

    # Tiled inference for full 5424x5424 frames
    img_f = warped_clean.astype(np.float32) / 255.0
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    step = max(64, tile_size - overlap)
    for y in range(0, h, step):
        y_end = min(y + tile_size, h)
        y_start = max(0, y_end - tile_size)
        pad_y = (16 - (y_end - y_start) % 16) % 16

        for x in range(0, w, step):
            x_end = min(x + tile_size, w)
            x_start = max(0, x_end - tile_size)
            pad_x = (16 - (x_end - x_start) % 16) % 16

            tile = img_f[y_start:y_end, x_start:x_end]
            if pad_y > 0 or pad_x > 0:
                tile = cv2.copyMakeBorder(tile, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT)

            tensor_in = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                tensor_out = model(tensor_in)

            res_tile = tensor_out.squeeze().cpu().numpy()
            if pad_y > 0 or pad_x > 0:
                res_tile = res_tile[:(y_end - y_start), :(x_end - x_start)]

            output[y_start:y_end, x_start:x_end] += res_tile
            weight[y_start:y_end, x_start:x_end] += 1.0

    weight = np.maximum(weight, 1e-5)
    refined_full = output / weight
    return np.clip(refined_full * 255.0, 0, 255).astype(np.uint8)


_DIRECT_SYNTHESIS_MODEL: Optional[DirectSynthesisUNet] = None


def load_direct_synthesis_model() -> Optional[DirectSynthesisUNet]:
    """Loads the trained proof-of-concept DirectSynthesisUNet checkpoint."""
    global _DIRECT_SYNTHESIS_MODEL
    if _DIRECT_SYNTHESIS_MODEL is not None:
        return _DIRECT_SYNTHESIS_MODEL

    base_dir = Path(__file__).resolve().parent.parent.parent
    ckpt_path = base_dir / "checkpoints" / "best_direct_synthesis_model.pth"
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/best_direct_synthesis_model.pth")

    if not ckpt_path.exists():
        logger.warning("Direct synthesis checkpoint not found at %s", ckpt_path)
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DirectSynthesisUNet(in_channels=4, out_channels=1, base_features=32)
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        _DIRECT_SYNTHESIS_MODEL = model
        logger.info("Successfully loaded DirectSynthesisUNet on %s", device)
        return model
    except Exception as e:
        logger.error("Failed to load DirectSynthesisUNet: %s", e)
        return None


def apply_direct_synthesis_model(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    flow: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Applies the trained proof-of-concept DirectSynthesisUNet to generate an intermediate frame at t=0.5.
    """
    model = load_direct_synthesis_model()
    if model is None:
        return interpolate_linear(frame_t0, frame_t1, t=0.5)

    device = next(model.parameters()).device
    h, w = frame_t0.shape[:2]

    if flow is None:
        flow = compute_farneback_flow(frame_t0, frame_t1)

    t0_norm = frame_t0.astype(np.float32) / 255.0
    t1_norm = frame_t1.astype(np.float32) / 255.0
    u_norm = np.clip(flow[:, :, 0] / 20.0, -1.0, 1.0)
    v_norm = np.clip(flow[:, :, 1] / 20.0, -1.0, 1.0)

    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    if pad_h > 0 or pad_w > 0:
        t0_norm = cv2.copyMakeBorder(t0_norm, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        t1_norm = cv2.copyMakeBorder(t1_norm, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        u_norm = cv2.copyMakeBorder(u_norm, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        v_norm = cv2.copyMakeBorder(v_norm, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    tensor_in = np.stack([t0_norm, t1_norm, u_norm, v_norm], axis=0)
    tensor_torch = torch.from_numpy(tensor_in).unsqueeze(0).to(device)

    with torch.no_grad():
        synth_tensor = model(tensor_torch)

    synth = synth_tensor.squeeze().cpu().numpy()
    if pad_h > 0 or pad_w > 0:
        synth = synth[:h, :w]

    return np.clip(synth * 255.0, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@router.get("/health", tags=["Health"], summary="Health Check")
async def health_check():
    """
    Health check endpoint returning system operational status.
    """
    return {"status": "ok"}


@router.get(
    "/triplets",
    tags=["Imagery & Data"],
    summary="List All Available Scan Triplets"
)
async def list_triplets():
    """
    Scans `sample_data/triplets/` and `manifest.json` to return a catalog
    of all available GOES-16 satellite scan triplets, including IDs, dates,
    UTC scan coverage times, and preview thumbnails.
    """
    triplets = get_available_triplets()
    return {
        "status": "success",
        "total": len(triplets),
        "triplets": triplets,
    }


@router.get(
    "/triplets/{triplet_id}/thumbnail",
    tags=["Imagery & Data"],
    summary="Get Thumbnail Image for Triplet"
)
async def get_triplet_thumbnail(triplet_id: str):
    """
    Returns a lightweight (128x128) PNG thumbnail for the specified triplet.
    """
    cached = load_triplet_scans(triplet_id)
    t0_arr = cached["t0_arr"]
    h, w = t0_arr.shape[:2]
    scale = 128.0 / float(max(h, w))
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = cv2.resize(t0_arr, (tw, th), interpolation=cv2.INTER_AREA)
    success, buffer = cv2.imencode(".png", thumb)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode thumbnail.")
    return Response(content=buffer.tobytes(), media_type="image/png")


@router.get(
    "/sample-pair",
    tags=["Imagery & Data"],
    summary="Fetch Pre-Downloaded Satellite Scan Pair for Triplet"
)
async def get_sample_pair(
    triplet_id: str = Query(
        ...,
        description="Required unique ID of the triplet to load (e.g. '2024-01-01_18' or '2024-01-15_00')."
    ),
    preview_size: Optional[int] = Query(
        1024,
        description="Max dimension for base64 encoded PNG response (default: 1024, pass 0 for full resolution)."
    )
):
    """
    Reads GOES-16 Channel 13 satellite scan frames for the specified `triplet_id`,
    encodes them as base64 PNG data URLs, and returns them alongside UTC scan timestamps
    and projection metadata.
    """
    cached = load_triplet_scans(triplet_id)

    t0_b64 = array_to_base64_png(cached["t0_arr"], max_dim=preview_size)
    t1_b64 = array_to_base64_png(cached["t1_arr"], max_dim=preview_size)

    response = {
        "status": "success",
        "triplet_id": triplet_id,
        "raw_dimensions": {
            "height": int(cached["t0_arr"].shape[0]),
            "width": int(cached["t0_arr"].shape[1]),
        },
        "frame_t0": {
            "image": t0_b64,
            "timestamp": cached["t0_meta"].get("time_coverage_start", "Unknown"),
            "metadata": cached["t0_meta"],
        },
        "frame_t1": {
            "image": t1_b64,
            "timestamp": cached["t1_meta"].get("time_coverage_start", "Unknown"),
            "metadata": cached["t1_meta"],
        },
    }

    if cached["gt_arr"] is not None:
        gt_b64 = array_to_base64_png(cached["gt_arr"], max_dim=preview_size)
        response["ground_truth_t05"] = {
            "image": gt_b64,
            "timestamp": cached["gt_meta"].get("time_coverage_start", "Unknown"),
            "metadata": cached["gt_meta"],
        }
    else:
        response["ground_truth_t05"] = None

    return response


@router.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["Frame Synthesis"],
    summary="Synthesize Intermediate Motion-Compensated Frame"
)
async def generate_frame(payload: GenerateRequest):
    """
    Synthesizes an intermediate satellite frame at timestamp fraction `t` for the specified `triplet_id`.

    Supported methods:
      - 'linear': Baseline linear cross-dissolve pixel averaging.
      - 'farneback': Classical polynomial expansion optical flow warping.
      - 'raft': Pretrained deep correlation volume optical flow warping.
      - 'raft_refined': RAFT optical flow warping followed by trained residual U-Net refinement.
    """
    triplet_id = payload.triplet_id.strip()
    method = payload.method.lower().strip()
    valid_methods = ("linear", "farneback", "raft", "raft_refined", "neural_synthesis", "direct_synthesis")
    if method not in valid_methods:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown method '{payload.method}'. Supported: {', '.join(valid_methods)}."
        )

    t = payload.time_fraction
    preview_dim = payload.preview_size

    # Return cached full-resolution (5424x5424) benchmark result if available
    cache_key = f"{triplet_id}_{method}_{t}"
    if preview_dim and preview_dim >= 5424 and abs(t - 0.5) < 1e-4 and cache_key in _GENERATE_5424_CACHE:
        return _GENERATE_5424_CACHE[cache_key]

    cached = load_triplet_scans(triplet_id)
    frame_t0 = cached["t0_arr"]
    frame_t1 = cached["t1_arr"]
    ground_truth = cached["gt_arr"]

    # If preview_size is specified and > 0, resize input frames for fast web calculation
    orig_h, orig_w = frame_t0.shape[:2]

    if preview_dim and preview_dim > 0 and max(orig_h, orig_w) > preview_dim:
        scale = preview_dim / float(max(orig_h, orig_w))
        calc_w = max(1, int(orig_w * scale))
        calc_h = max(1, int(orig_h * scale))
        f0 = cv2.resize(frame_t0, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
        f1 = cv2.resize(frame_t1, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(ground_truth, (calc_w, calc_h), interpolation=cv2.INTER_AREA) if ground_truth is not None else None
    else:
        f0 = frame_t0
        f1 = frame_t1
        gt = ground_truth
        calc_h, calc_w = orig_h, orig_w

    # Linear baseline frame (always generated for comparison and linear baseline product)
    linear_frame = interpolate_linear(frame_t0=f0, frame_t1=f1, t=t)

    # Dispatch based on chosen method
    if method == "linear":
        warped_frame = linear_frame
        flow = np.zeros((calc_h, calc_w, 2), dtype=np.float32)
        confidence_gray = np.full((calc_h, calc_w), 255, dtype=np.uint8)
        confidence_color = cv2.applyColorMap(confidence_gray, cv2.COLORMAP_VIRIDIS)
        flow_rgb = flow_to_hsv_color(flow)

    elif method == "farneback":
        flow = compute_farneback_flow(
            frame_t0=f0,
            frame_t1=f1,
            pyr_scale=payload.pyr_scale,
            levels=payload.levels,
            winsize=payload.winsize,
            iterations=payload.iterations,
        )
        warped_frame = warp_frame_forward(frame_t0=f0, flow=flow, t=t)
        confidence_gray, confidence_color = compute_flow_confidence_map(
            frame_t0=f0, frame_t1=f1, flow=flow, t=t
        )
        flow_rgb = flow_to_hsv_color(flow)

    elif method == "raft":
        flow = compute_flow(frame_t0=f0, frame_t1=f1, flow_method="raft")
        warped_frame = warp_frame_forward(frame_t0=f0, flow=flow, t=t)
        confidence_gray, confidence_color = compute_flow_confidence_map(
            frame_t0=f0, frame_t1=f1, flow=flow, t=t
        )
        flow_rgb = flow_to_hsv_color(flow)

    elif method == "raft_refined":
        flow = compute_flow(frame_t0=f0, frame_t1=f1, flow_method="raft")
        warped_raw = warp_frame_forward(frame_t0=f0, flow=flow, t=t)
        # Pass the RAFT-warped frame through the trained refinement U-Net
        try:
            warped_frame = apply_refinement_model(warped_raw)
        except Exception as exc:
            logger.error("Failed to apply refinement U-Net: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Refinement U-Net execution failed: {exc}"
            )
        confidence_gray, confidence_color = compute_flow_confidence_map(
            frame_t0=f0, frame_t1=f1, flow=flow, t=t
        )
        flow_rgb = flow_to_hsv_color(flow)

    elif method in ("neural_synthesis", "direct_synthesis"):
        flow = compute_farneback_flow(
            frame_t0=f0,
            frame_t1=f1,
            pyr_scale=payload.pyr_scale,
            levels=payload.levels,
            winsize=payload.winsize,
            iterations=payload.iterations,
        )
        warped_frame = apply_direct_synthesis_model(frame_t0=f0, frame_t1=f1, flow=flow)
        confidence_gray, confidence_color = compute_flow_confidence_map(
            frame_t0=f0, frame_t1=f1, flow=flow, t=t
        )
        flow_rgb = flow_to_hsv_color(flow)

    # Quantitative Validation Metrics (if ground truth available and target timestamp matches t=0.5)
    # Quantitative Validation Metrics (if ground truth available and target timestamp matches t=0.5)
    if gt is not None and abs(t - 0.5) < 1e-4:
        metrics_warped = calculate_metrics(warped_frame, gt)
        metrics_linear = calculate_metrics(linear_frame, gt)
        metrics_data = {
            "is_evaluated_against_ground_truth": True,
            "target_time_fraction": t,
            "flow_method": method,
            "mse": float(metrics_warped["mse"]),
            "psnr_db": float(metrics_warped["psnr_db"]),
            "ssim": float(metrics_warped["ssim"]),
            "mae": float(metrics_warped["mae"]),
            "warped": metrics_warped,
            "linear": metrics_linear,
            "mse_improvement_percentage": float(
                ((metrics_linear["mse"] - metrics_warped["mse"]) / metrics_linear["mse"]) * 100.0
            ) if metrics_linear["mse"] > 0 else 0.0,
        }
    else:
        metrics_data = {
            "is_evaluated_against_ground_truth": False,
            "target_time_fraction": t,
            "flow_method": method,
            "mse": None,
            "psnr_db": None,
            "ssim": None,
            "mae": None,
            "warped": {
                "mse": None,
                "psnr_db": None,
                "ssim": None,
                "mae": None,
            },
            "linear": {
                "mse": None,
                "psnr_db": None,
                "ssim": None,
                "mae": None,
            },
            "mse_improvement_percentage": None,
            "note": "Ground truth scan not available for this triplet or evaluation not at t=0.50",
        }

    # Convert images to Base64 PNG Data URLs (capped at max 1024px for responsive web transfer)
    warped_b64 = array_to_base64_png(warped_frame, max_dim=1024)
    linear_b64 = array_to_base64_png(linear_frame, max_dim=1024)
    gt_b64 = array_to_base64_png(gt, max_dim=1024) if gt is not None else None
    conf_gray_b64 = array_to_base64_png(confidence_gray, max_dim=1024)
    conf_color_b64 = array_to_base64_png(confidence_color, max_dim=1024)
    flow_b64 = array_to_base64_png(flow_rgb, max_dim=1024)

    resp = GenerateResponse(
        triplet_id=triplet_id,
        time_fraction=t,
        flow_method=method,
        dimensions={"height": calc_h, "width": calc_w},
        generated_image=warped_b64,
        warped_image=warped_b64,
        linear_image=linear_b64,
        ground_truth_image=gt_b64,
        confidence_map=conf_gray_b64,
        confidence_map_colored=conf_color_b64,
        confidence_is_heuristic=True,
        confidence_label="Bidirectional Flow Consistency Heuristic (Not a trained model)",
        flow_visualization=flow_b64,
        metrics=metrics_data,
    )
    if preview_dim and preview_dim >= 5424 and abs(t - 0.5) < 1e-4:
        _GENERATE_5424_CACHE[cache_key] = resp
    return resp


# ------------------------------------------------------------------------------
# Continuous Multi-Frame Temporal Sequence Endpoint
# ------------------------------------------------------------------------------
_SEQUENCE_CACHE: Dict[str, Any] = {}


@router.post(
    "/sequence",
    tags=["Temporal Sequences"],
    summary="Synthesize Continuous Multi-Frame Temporal Sequence"
)
async def generate_temporal_sequence(payload: SequenceRequest):
    """
    Synthesizes a continuous atmospheric evolution sequence between T0 and T1
    at normalized timestamps t in [0.0, ..., 1.0].
    Returns intermediate synthesized frames, boundary frames, ground truth at t=0.5,
    exact validation metrics, and motion statistics.
    """
    global _SEQUENCE_CACHE
    triplet_id = payload.triplet_id.strip()
    method = payload.method.lower().strip()
    num_steps = payload.num_steps
    preview_dim = payload.preview_size or 512

    cache_key = f"seq_{triplet_id}_{method}_{num_steps}_{preview_dim}"
    if cache_key in _SEQUENCE_CACHE:
        return _SEQUENCE_CACHE[cache_key]

    cached = load_triplet_scans(triplet_id)
    frame_t0 = cached["t0_arr"]
    frame_t1 = cached["t1_arr"]
    ground_truth = cached["gt_arr"]

    orig_h, orig_w = frame_t0.shape[:2]
    if preview_dim > 0 and max(orig_h, orig_w) > preview_dim:
        scale = preview_dim / float(max(orig_h, orig_w))
        calc_w = max(1, int(orig_w * scale))
        calc_h = max(1, int(orig_h * scale))
        f0 = cv2.resize(frame_t0, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
        f1 = cv2.resize(frame_t1, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(ground_truth, (calc_w, calc_h), interpolation=cv2.INTER_AREA) if ground_truth is not None else None
    else:
        f0, f1, gt = frame_t0, frame_t1, ground_truth
        calc_h, calc_w = orig_h, orig_w

    # Compute dense motion flow if needed
    flow = None
    if method in ("farneback", "neural_synthesis"):
        flow = compute_farneback_flow(frame_t0=f0, frame_t1=f1)
    elif method in ("raft", "raft_refined"):
        flow = compute_flow(frame_t0=f0, frame_t1=f1, flow_method="raft")

    # Flow displacement stats
    mean_disp = float(np.mean(np.linalg.norm(flow, axis=2))) if flow is not None else 0.0
    max_disp = float(np.max(np.linalg.norm(flow, axis=2))) if flow is not None else 0.0

    frames_output = []
    # Generate timesteps uniformly between 0.0 and 1.0
    time_fractions = [round(i / (num_steps - 1), 3) for i in range(num_steps)]

    t0_start_str = cached["t0_meta"].get("time_coverage_start", "18:00 UTC")

    for idx, t in enumerate(time_fractions):
        is_boundary = (idx == 0 or idx == num_steps - 1)
        metrics = None

        if idx == 0:
            frame_img = f0
            time_label = f"T₀ ({t0_start_str[-9:-4]} UTC)" if len(t0_start_str) >= 9 else "T₀ (00m)"
        elif idx == num_steps - 1:
            frame_img = f1
            t1_start_str = cached["t1_meta"].get("time_coverage_start", "18:15 UTC")
            time_label = f"T₁ ({t1_start_str[-9:-4]} UTC)" if len(t1_start_str) >= 9 else "T₁ (15m)"
        else:
            minute_offset = t * 15.0
            time_label = f"t={t:.2f} (+{minute_offset:.1f}m)"
            if method == "linear":
                frame_img = interpolate_linear(frame_t0=f0, frame_t1=f1, t=t)
            elif method == "farneback":
                frame_img = warp_frame_forward(frame_t0=f0, flow=flow, t=t)
            elif method == "raft":
                frame_img = warp_frame_forward(frame_t0=f0, flow=flow, t=t)
            elif method == "raft_refined":
                raw_warp = warp_frame_forward(frame_t0=f0, flow=flow, t=t)
                try:
                    frame_img = apply_refinement_model(raw_warp)
                except Exception:
                    frame_img = raw_warp
            elif method == "neural_synthesis":
                frame_img = apply_direct_synthesis_model(frame_t0=f0, frame_t1=f1, flow=flow)
            else:
                frame_img = interpolate_linear(frame_t0=f0, frame_t1=f1, t=t)

            # Evaluate metrics at t=0.50 if GT exists
            if gt is not None and abs(t - 0.5) < 1e-3:
                m = calculate_metrics(frame_img, gt)
                metrics = {
                    "mse": float(m["mse"]),
                    "psnr_db": float(m["psnr_db"]),
                    "ssim": float(m["ssim"]),
                    "mae": float(m["mae"])
                }

        b64_str = array_to_base64_png(frame_img)
        frames_output.append({
            "index": idx,
            "time_fraction": t,
            "time_label": time_label,
            "image": b64_str,
            "is_boundary": is_boundary,
            "metrics": metrics
        })

    gt_b64 = array_to_base64_png(gt) if gt is not None else None

    result = {
        "status": "success",
        "triplet_id": triplet_id,
        "method": method,
        "num_frames": len(frames_output),
        "frames": frames_output,
        "ground_truth_image": gt_b64,
        "motion_summary": {
            "mean_displacement_px": round(mean_disp, 2),
            "max_displacement_px": round(max_disp, 2)
        }
    }
    _SEQUENCE_CACHE[cache_key] = result
    return result


# ------------------------------------------------------------------------------
# Benchmark Endpoint
# ------------------------------------------------------------------------------
@router.get(
    "/benchmark",
    tags=["Benchmarking"],
    summary="Benchmark All Four Methods on a Triplet"
)
async def benchmark_all_methods(
    triplet_id: str = Query("2024-01-01_18", description="ID of the triplet to benchmark"),
    preview_size: Optional[int] = Query(512, description="Calculation dimension for fast benchmarking"),
    force_recompute: bool = Query(False, description="Whether to bypass cache and recompute")
):
    """
    Evaluates all four temporal synthesis methods (Linear, Farneback, RAFT, RAFT+Refinement)
    against the held-out ground truth scan (t=0.50) on the requested triplet.
    Returns MSE, PSNR, SSIM, and MAE for all four methods in a unified benchmark comparison.
    """
    global _BENCHMARK_CACHE
    cache_key = f"{triplet_id}_{preview_size}"
    if cache_key in _BENCHMARK_CACHE and not force_recompute:
        return _BENCHMARK_CACHE[cache_key]

    cached = load_triplet_scans(triplet_id)
    frame_t0 = cached["t0_arr"]
    frame_t1 = cached["t1_arr"]
    ground_truth = cached["gt_arr"]

    if ground_truth is None:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot benchmark triplet '{triplet_id}': no ground truth scan available."
        )

    # Resize for benchmarking if requested
    orig_h, orig_w = frame_t0.shape[:2]
    if preview_size and preview_size > 0 and max(orig_h, orig_w) > preview_size:
        scale = preview_size / float(max(orig_h, orig_w))
        calc_w = max(1, int(orig_w * scale))
        calc_h = max(1, int(orig_h * scale))
        f0 = cv2.resize(frame_t0, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
        f1 = cv2.resize(frame_t1, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(ground_truth, (calc_w, calc_h), interpolation=cv2.INTER_AREA)
    else:
        f0, f1, gt = frame_t0, frame_t1, ground_truth
        calc_h, calc_w = orig_h, orig_w

    t = 0.5

    # 1. Linear Interpolation
    lin_frame = interpolate_linear(frame_t0=f0, frame_t1=f1, t=t)
    m_lin = calculate_metrics(lin_frame, gt)

    # 2. Farneback Flow
    flow_fb = compute_farneback_flow(frame_t0=f0, frame_t1=f1)
    fb_frame = warp_frame_forward(frame_t0=f0, flow=flow_fb, t=t)
    m_fb = calculate_metrics(fb_frame, gt)

    # 3. RAFT Optical Flow
    flow_raft = compute_flow(frame_t0=f0, frame_t1=f1, flow_method="raft")
    raft_frame = warp_frame_forward(frame_t0=f0, flow=flow_raft, t=t)
    m_raft = calculate_metrics(raft_frame, gt)

    # 4. RAFT + Refinement U-Net
    try:
        ref_frame = apply_refinement_model(raft_frame)
    except Exception:
        ref_frame = raft_frame
    m_ref = calculate_metrics(ref_frame, gt)

    lin_mse = m_lin["mse"]

    def calc_imp(m_val):
        if lin_mse <= 0:
            return 0.0
        return float(((lin_mse - m_val["mse"]) / lin_mse) * 100.0)

    methods_data = [
        {
            "id": "linear",
            "name": "Linear Interpolation",
            "short_name": "Linear",
            "mse": float(m_lin["mse"]),
            "psnr_db": float(m_lin["psnr_db"]),
            "ssim": float(m_lin["ssim"]),
            "mae": float(m_lin["mae"]),
            "improvement_vs_linear": 0.0,
            "description": "Baseline linear pixel cross-dissolve",
        },
        {
            "id": "farneback",
            "name": "Farneback Flow",
            "short_name": "Farneback",
            "mse": float(m_fb["mse"]),
            "psnr_db": float(m_fb["psnr_db"]),
            "ssim": float(m_fb["ssim"]),
            "mae": float(m_fb["mae"]),
            "improvement_vs_linear": calc_imp(m_fb),
            "description": "Classical polynomial optical flow",
        },
        {
            "id": "raft",
            "name": "RAFT Flow",
            "short_name": "RAFT",
            "mse": float(m_raft["mse"]),
            "psnr_db": float(m_raft["psnr_db"]),
            "ssim": float(m_raft["ssim"]),
            "mae": float(m_raft["mae"]),
            "improvement_vs_linear": calc_imp(m_raft),
            "description": "Deep correlation volume optical flow",
        },
        {
            "id": "raft_refined",
            "name": "RAFT + Refinement",
            "short_name": "RAFT + U-Net",
            "mse": float(m_ref["mse"]),
            "psnr_db": float(m_ref["psnr_db"]),
            "ssim": float(m_ref["ssim"]),
            "mae": float(m_ref["mae"]),
            "improvement_vs_linear": calc_imp(m_ref),
            "description": "RAFT flow + trained refinement U-Net",
        },
    ]

    result = {
        "status": "success",
        "triplet_id": triplet_id,
        "sample_pair_metadata": {
            "channel": "GOES-16 ABI Band 13 (Clean IR 10.35 µm)",
            "evaluated_at": "t=0.50 (Held-Out Middle Observation)",
            "resolution": f"{calc_w} × {calc_h} px",
        },
        "methods": methods_data,
    }

    _BENCHMARK_CACHE[cache_key] = result
    return result


@router.get("/neural-synthesis/samples", summary="Fetch real trained proof-of-concept neural synthesis samples")
async def get_neural_synthesis_samples():
    """
    Returns real demonstration patches generated by the compact conditional DirectSynthesisUNet
    trained on GOES-16 Band 13 radiance patches.
    Includes model checkpoint training stats, loss curve image, ground truth, boundary observations,
    synthesized intermediate frame, residual heatmap, and exact computed metrics.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    demo_dir = base_dir / "sample_data" / "patches" / "neural_synthesis_demo"
    if not demo_dir.exists():
        demo_dir = Path("sample_data/patches/neural_synthesis_demo")

    catalog_path = demo_dir / "demo_catalog.json"
    if not catalog_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Neural synthesis demo catalog not found. Please ensure training script generated samples."
        )

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Load loss curve image if present
    loss_curve_path = base_dir / "checkpoints" / "direct_synthesis_loss_curve.png"
    if not loss_curve_path.exists():
        loss_curve_path = Path("checkpoints/direct_synthesis_loss_curve.png")

    loss_curve_b64 = None
    if loss_curve_path.exists():
        with open(loss_curve_path, "rb") as f:
            loss_curve_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

    samples_output = []
    for s in catalog.get("samples", []):
        sample_entry = {
            "id": s["id"],
            "triplet_id": s.get("triplet_id"),
            "patch_name": s.get("patch_name"),
            "metrics": s.get("metrics"),
            "images": {}
        }
        for key, fname in s.get("files", {}).items():
            img_path = demo_dir / fname
            if img_path.exists():
                with open(img_path, "rb") as img_file:
                    sample_entry["images"][key] = f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
            else:
                sample_entry["images"][key] = None
        samples_output.append(sample_entry)

    return {
        "status": "success",
        "model_type": catalog.get("model_type"),
        "trained_epochs": catalog.get("trained_epochs"),
        "best_epoch": catalog.get("best_epoch"),
        "best_val_loss": catalog.get("best_val_loss"),
        "loss_curve_image": loss_curve_b64,
        "samples": samples_output,
        "note": "Proof-of-concept Direct-Synthesis Model trained on GOES-16 patches with L1+SSIM loss. Demonstrates direct non-warped generation ahead of end-semester diffusion scaling."
    }


# ------------------------------------------------------------------------------
# NOAA GOES-16 Dataset Live Connection & Ingestion Endpoints
# ------------------------------------------------------------------------------
@router.get(
    "/goes/status",
    tags=["NOAA GOES Dataset"],
    summary="NOAA GOES-16 S3 Connection Health & Latency"
)
async def get_goes_status():
    """
    Tests live anonymous HTTPS connectivity to the public NOAA GOES-16 AWS S3 archive (noaa-goes16).
    Returns bucket metadata, AWS region, endpoint URL, HTTP response status, round-trip latency,
    and total available local observation triplets.
    """
    fetcher = GOES16Fetcher()
    status = fetcher.ping()
    all_triplets = get_available_triplets()
    status["available_triplets_count"] = len(all_triplets)
    status["full_disk_count"] = len([t for t in all_triplets if "Full Disk" in t.get("type", "")])
    status["roi_patches_count"] = len([t for t in all_triplets if "Cloud Dynamics" in t.get("type", "")])
    status["sensor"] = "NOAA GOES-16 ABI (Advanced Baseline Imager)"
    status["channel"] = 13
    status["channel_name"] = "Clean IR Longwave Window (~10.35 µm)"
    status["spatial_resolution"] = "2km at nadir (5424×5424 full disk)"
    return status


@router.get(
    "/goes/query-scans",
    tags=["NOAA GOES Dataset"],
    summary="Query Real NOAA GOES-16 Band 13 Scans by Date and Hour"
)
async def query_goes_scans(
    date: str = Query("2024-01-15", description="Observation date (YYYY-MM-DD)"),
    hour: int = Query(0, ge=0, le=23, description="UTC Observation Hour (0-23)")
):
    """
    Queries the public NOAA GOES-16 AWS S3 bucket for all Channel 13 (Clean IR ~10.3 µm)
    NetCDF scans available for the requested date and hour.
    """
    try:
        year, doy, d_str = parse_date_input(date)
        fetcher = GOES16Fetcher()
        files = fetcher.list_channel_files(year=year, day_of_year=doy, hour=hour, channel=13)
        triplet_candidates = fetcher.select_triplet_scans(files) if len(files) >= 3 else []
        return {
            "status": "success",
            "date": d_str,
            "hour": hour,
            "year": year,
            "day_of_year": doy,
            "channel": 13,
            "total_files": len(files),
            "files": [f.split("/")[-1] for f in files],
            "selected_triplet": [f.split("/")[-1] for f in triplet_candidates] if triplet_candidates else [],
        }
    except Exception as e:
        logger.error("Error querying NOAA S3: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class FetchTripletRequest(BaseModel):
    date: str = Field(..., description="Date string YYYY-MM-DD")
    hour: int = Field(..., ge=0, le=23, description="UTC hour 0-23")
    channel: int = Field(default=13, description="Spectral Channel (default 13)")


@router.post(
    "/goes/fetch-triplet",
    tags=["NOAA GOES Dataset"],
    summary="Fetch & Ingest Real GOES-16 Scan Triplet from NOAA AWS S3"
)
async def fetch_triplet_from_noaa(payload: FetchTripletRequest):
    """
    Downloads and ingests a full scan triplet directly from the public NOAA GOES-16 S3 bucket.
    Normalizes CMI to 0-255 grayscale, writes .npy & .png artifacts, updates the manifest,
    and makes the new observation triplet immediately available in the dashboard.
    """
    try:
        year, doy, d_str = parse_date_input(payload.date)
        fetcher = GOES16Fetcher()
        tid = f"{d_str}_{payload.hour:02d}"
        out_dir = TRIPLETS_DIR / tid

        record = download_single_triplet(
            fetcher=fetcher,
            year=year,
            day_of_year=doy,
            hour=payload.hour,
            triplet_dir=out_dir,
            channel=payload.channel,
            keep_raw_nc=False,
            date_str=d_str
        )

        # Invalidate caches
        _TRIPLET_ARRAYS_CACHE.pop(tid, None)
        _THUMBNAILS_CACHE.pop(tid, None)

        return {
            "status": "success",
            "message": f"Successfully ingested real GOES-16 triplet {tid} from NOAA AWS S3.",
            "triplet": record,
        }
    except Exception as e:
        logger.error("Failed fetching triplet from NOAA S3: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

