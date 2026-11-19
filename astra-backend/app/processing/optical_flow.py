"""
Dense Optical Flow & Motion-Compensated Frame Interpolation Module
==================================================================
File: app/processing/optical_flow.py

Purpose:
--------
Provides computational algorithms to calculate dense optical flow motion fields
between consecutive satellite/scientific imagery frames, synthesize intermediate
frames using forward flow warping (`cv2.remap`), and compare against plain linear
pixel averaging (alpha blending).

Supported Flow Estimation Backends:
-----------------------------------
  flow_method="farneback"  — Classical Gunnar Farneback polynomial expansion (OpenCV).
  flow_method="raft"       — RAFT pretrained deep neural network (torchvision).

Scientific & Theoretical Background:
------------------------------------
1. What is Optical Flow?
   - Optical flow is the vector field representing the apparent motion of brightness
     patterns between two consecutive 2D image frames:
         Flow(x, y) = (u(x, y), v(x, y))
     where:
       * u = horizontal displacement (dx / dt, east-west pixel velocity)
       * v = vertical displacement (dy / dt, north-south pixel velocity)
   - In meteorology and atmospheric science, optical flow captures atmospheric advection,
     wind drift, storm rotation, jet-stream shear, and cloud formation/dissipation.

2. Gunnar Farneback's Algorithm (cv2.calcOpticalFlowFarneback):
   - Approximates local pixel neighborhoods with quadratic polynomials:
         f(x) ~ x^T A x + b^T x + c
   - Computes displacement by solving for translation between polynomial surfaces
     across multi-scale Gaussian pyramids to capture both large-scale global motions
     and fine-scale local vortices.
   - Limitation: The fixed polynomial window size limits its ability to track
     large displacements (e.g. > 15-20 px) that exceed the window radius, causing
     flow underestimation for fast-moving features like jet stream cirrus.

3. RAFT (Recurrent All-Pairs Field Transforms) — Teed & Deng, ECCV 2020:
   - A deep neural network architecture that builds a 4D correlation volume
     over ALL pixel pairs between two frames, then iteratively refines the
     flow estimate through a lightweight recurrent (GRU) update operator.
   - KEY ADVANTAGE: Because the 4D correlation volume captures correspondences
     at ALL displacements simultaneously (not just within a local window),
     RAFT handles large, fast, and non-rigid motions far better than Farneback.
     This is critical for meteorological imagery where tropical cyclone eyewall
     rotation, explosive convective overshooting tops, and jet-stream shear
     can produce displacements of 30-100+ pixels between 10-minute scans.
   - PRETRAINED MODEL NOTE: We load `raft_small` with `Raft_Small_Weights.DEFAULT`
     pretrained on the FlyingChairs + FlyingThings3D + Sintel + KITTI datasets.
     No training or fine-tuning happens at inference time — the model weights are
     frozen and used purely for forward-pass prediction. This is analogous to
     using a pretrained ResNet for feature extraction.

4. Why Motion-Compensated Warping Differs from Pixel Averaging (Linear Blend):
   - Linear Averaging / Pixel Blending:
         I_linear(x, y) = (1 - t) * I_0(x, y) + t * I_1(x, y)
     * Does NOT model motion or physics. It simply fades out features from frame 0
       while fading in features from frame 1.
     * Causes "ghosting" (phantom double-clouds) and severe blurriness because moving
       storm edges are present in two separate spatial locations simultaneously.
   - Optical Flow Motion Warping (Advection):
         I_warped(x, y) = I_0(x - t*u, y - t*v)
     * Physically transports (advects) pixel intensities along estimated velocity
       trajectories using backward coordinate remap (`cv2.remap`).
     * Preserves sharp cloud boundaries, eye walls, brightness temperature gradients,
       and structural sharpness without ghosting artifacts.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("optical_flow")


def compute_farneback_flow(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
) -> np.ndarray:
    """
    Computes dense optical flow from frame_t0 to frame_t1 using the Farneback algorithm.

    Args:
        frame_t0 (np.ndarray): 2D grayscale image at time t=0 (uint8 or float32).
        frame_t1 (np.ndarray): 2D grayscale image at time t=1 (uint8 or float32).
        pyr_scale (float): Image scale (< 1) to build pyramids for each image.
                           0.5 represents a classical Gaussian pyramid.
        levels (int): Number of pyramid layers including the initial image.
        winsize (int): Averaging window size; larger values increase robustness to noise
                       and yield smoother motion fields.
        iterations (int): Number of iterations at each pyramid level.
        poly_n (int): Size of the pixel neighborhood used to find polynomial expansion (e.g. 5 or 7).
        poly_sigma (float): Standard deviation of Gaussian used to smooth derivatives.
        flags (int): Operation flags (e.g., cv2.OPTFLOW_FARNEBACK_GAUSSIAN).

    Returns:
        np.ndarray: 2-channel float32 flow array of shape (H, W, 2) where
                    flow[..., 0] is horizontal motion (u) and
                    flow[..., 1] is vertical motion (v).
    """
    # Ensure inputs are 8-bit unsigned integers for OpenCV Farneback
    f0 = frame_t0 if frame_t0.dtype == np.uint8 else np.clip(frame_t0, 0, 255).astype(np.uint8)
    f1 = frame_t1 if frame_t1.dtype == np.uint8 else np.clip(frame_t1, 0, 255).astype(np.uint8)

    logger.info("Computing Farneback optical flow on frame shape: %s", f0.shape)
    flow = cv2.calcOpticalFlowFarneback(
        prev=f0,
        next=f1,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )
    logger.info(
        "Flow computed. Displacement range: u in [%.2f, %.2f], v in [%.2f, %.2f]",
        flow[..., 0].min(), flow[..., 0].max(),
        flow[..., 1].min(), flow[..., 1].max()
    )
    return flow


def compute_raft_flow(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    num_iterations: int = 12,
    max_size: int = 1024,
) -> np.ndarray:
    """
    Computes dense optical flow from frame_t0 to frame_t1 using the pretrained
    RAFT (Recurrent All-Pairs Field Transforms) deep neural network.

    IMPORTANT — PRETRAINED INFERENCE ONLY:
    This function loads a pretrained RAFT-Small model from torchvision with
    weights trained on synthetic (FlyingChairs, FlyingThings3D) and real
    (Sintel, KITTI) optical flow benchmarks.  No training, fine-tuning,
    or gradient computation happens here — torch.no_grad() is used and
    the model is set to eval() mode.  The pretrained weights are cached
    after the first download.

    Memory-Aware Downscaling:
    RAFT's 4D correlation volume is O(N²) in the number of pixels.
    For a 5424×5424 full-disk GOES-16 frame that would be ~845 GB of RAM.
    To avoid out-of-memory crashes, frames are downscaled so that the longest
    side does not exceed `max_size` (default 1024).  The flow field is then
    bilinearly upscaled back to the original resolution and the displacement
    vectors are proportionally rescaled (since a 2 px shift at 1/4 resolution
    corresponds to an 8 px shift at full resolution).

    Why RAFT Outperforms Farneback on Large / Fast Motion:
    - Farneback uses a fixed-size local polynomial window (typically 15x15 px).
      Motions larger than ~half the window size (roughly 7-8 px) cannot be
      reliably captured, leading to systematic flow underestimation for fast-
      moving satellite features.
    - RAFT builds a 4D correlation volume over ALL pairs of pixels between
      both frames, allowing it to find correspondences at any displacement
      magnitude.  A recurrent GRU unit then iteratively refines the estimate,
      producing sub-pixel accurate flow even for displacements > 100 px.
    - This makes RAFT especially valuable for GOES-16 full-disk imagery
      where 10-minute inter-scan intervals can produce cloud displacements
      of 30-100+ pixels at native 2 km resolution.

    Args:
        frame_t0 (np.ndarray): 2D grayscale image at time t=0 (uint8, shape H×W).
        frame_t1 (np.ndarray): 2D grayscale image at time t=1 (uint8, shape H×W).
        num_iterations (int): Number of RAFT recurrent refinement iterations
                              (more iterations = more accurate but slower; default 12).
        max_size (int): Maximum allowed dimension (height or width) before
                        downscaling.  Set to 0 or None to disable downscaling
                        (WARNING: will OOM on large frames).  Default 1024.

    Returns:
        np.ndarray: 2-channel float32 flow array of shape (H, W, 2) — identical
                    format to compute_farneback_flow, so it is a drop-in replacement
                    for all downstream warping and visualization functions.
    """
    import torch
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    orig_h, orig_w = frame_t0.shape[:2]

    # ---- 0. Memory-aware downscaling for large frames ----
    # RAFT correlation volume is O(N²) in pixel count.  A 5424×5424 frame would
    # need ~845 GB.  We cap the longest dimension at `max_size` and rescale
    # the flow vectors back up afterward.
    scale_factor = 1.0
    work_t0, work_t1 = frame_t0, frame_t1
    if max_size and max(orig_h, orig_w) > max_size:
        scale_factor = max_size / max(orig_h, orig_w)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        work_t0 = cv2.resize(frame_t0, (new_w, new_h), interpolation=cv2.INTER_AREA)
        work_t1 = cv2.resize(frame_t1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(
            "RAFT: downscaled %dx%d → %dx%d (scale=%.4f) to fit in memory",
            orig_h, orig_w, new_h, new_w, scale_factor
        )

    work_h, work_w = work_t0.shape[:2]

    # ---- 1. Convert grayscale uint8 (H, W) → 3-channel float32 tensor [1, 3, H, W] ----
    # RAFT expects RGB images normalized to [0, 1] (or using the weight-specific
    # transform).  For grayscale satellite imagery we replicate the single channel
    # across R, G, B since the pretrained model was trained on RGB data.
    def gray_to_raft_tensor(gray: np.ndarray) -> "torch.Tensor":
        img = gray.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0  # normalize to [0, 1]
        # (H, W) → (H, W, 3) → (3, H, W) → (1, 3, H, W)
        rgb = np.stack([img, img, img], axis=-1)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor

    t0_tensor = gray_to_raft_tensor(work_t0)
    t1_tensor = gray_to_raft_tensor(work_t1)

    # ---- 2. Pad to dimensions divisible by 8 (RAFT architectural requirement) ----
    # RAFT's correlation pyramid and feature encoder use strides of 8,
    # so spatial dimensions must be multiples of 8.
    pad_h = (8 - work_h % 8) % 8
    pad_w = (8 - work_w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        t0_tensor = torch.nn.functional.pad(t0_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        t1_tensor = torch.nn.functional.pad(t1_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    # ---- 3. Select compute device (GPU if available, otherwise CPU) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0_tensor = t0_tensor.to(device)
    t1_tensor = t1_tensor.to(device)

    # ---- 4. Load pretrained RAFT-Small model ----
    # Raft_Small_Weights.DEFAULT loads weights pretrained on:
    #   FlyingChairs → FlyingThings3D → Sintel → KITTI
    # The model is set to eval() mode — no dropout, no batch-norm updates,
    # no gradient tracking.  This is pure inference.
    logger.info(
        "Computing RAFT optical flow on frame shape: %s (device: %s, iters: %d)",
        (work_h, work_w), device, num_iterations
    )
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(device)
    model.eval()

    # ---- 5. Run forward pass (inference only, no gradients) ----
    with torch.no_grad():
        # RAFT returns a list of flow predictions (one per refinement iteration).
        # The final element is the most refined estimate.
        flow_predictions = model(t0_tensor, t1_tensor, num_flow_updates=num_iterations)
        flow_tensor = flow_predictions[-1]  # shape: (1, 2, H_padded, W_padded)

    # ---- 6. Crop padding and convert back to numpy (H, W, 2) float32 ----
    # flow_tensor channels: [0] = horizontal (u), [1] = vertical (v)
    flow_np = flow_tensor[0].cpu().numpy()          # shape: (2, H_padded, W_padded)
    flow_np = flow_np[:, :work_h, :work_w]          # crop away padding
    flow_np = flow_np.transpose(1, 2, 0)            # (2, H, W) → (H, W, 2)

    # ---- 7. Upscale flow back to original resolution if we downscaled ----
    # Flow vectors are in pixel units at the *working* resolution, so when we
    # bilinearly upscale the flow field back to the original resolution we must
    # also multiply the displacement magnitudes by (1 / scale_factor) to convert
    # from downscaled-pixel-space to original-pixel-space.
    if scale_factor < 1.0:
        flow_u = cv2.resize(flow_np[..., 0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        flow_v = cv2.resize(flow_np[..., 1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        inv_scale = 1.0 / scale_factor
        flow_np = np.stack([flow_u * inv_scale, flow_v * inv_scale], axis=-1)
        logger.info(
            "RAFT: upscaled flow %dx%d → %dx%d (displacement ×%.2f)",
            work_h, work_w, orig_h, orig_w, inv_scale
        )

    logger.info(
        "RAFT flow computed. Displacement range: u in [%.2f, %.2f], v in [%.2f, %.2f]",
        flow_np[..., 0].min(), flow_np[..., 0].max(),
        flow_np[..., 1].min(), flow_np[..., 1].max()
    )
    return flow_np.astype(np.float32)


def compute_flow(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    flow_method: str = "farneback",
    **kwargs,
) -> np.ndarray:
    """
    Unified dispatcher that selects the flow estimation backend.

    Args:
        frame_t0 (np.ndarray): Grayscale frame at t=0.
        frame_t1 (np.ndarray): Grayscale frame at t=1.
        flow_method (str): One of "farneback" or "raft".
        **kwargs: Forwarded to the chosen backend function.

    Returns:
        np.ndarray: (H, W, 2) float32 flow field.
    """
    method = flow_method.lower().strip()
    if method == "farneback":
        return compute_farneback_flow(frame_t0, frame_t1, **kwargs)
    elif method == "raft":
        return compute_raft_flow(frame_t0, frame_t1, **kwargs)
    else:
        raise ValueError(
            f"Unknown flow_method '{flow_method}'. Supported: 'farneback', 'raft'."
        )


def warp_frame_forward(
    frame_t0: np.ndarray,
    flow: np.ndarray,
    t: float = 0.5,
    border_mode: int = cv2.BORDER_REFLECT,
) -> np.ndarray:
    """
    Warps frame_t0 forward to an intermediate time fraction t in [0.0, 1.0]
    using the optical flow field via backward lookup with cv2.remap.

    Mathematical Formulation:
    -------------------------
    If a feature at position (x0, y0) in frame_t0 moves by vector (u, v) to (x0 + u, y0 + v)
    in frame_t1, then at intermediate time t, its position is (x0 + t*u, y0 + t*v).
    To render destination pixel (x, y) at time t, we look back to the source coordinate:
        x_source = x - t * u(x, y)
        y_source = y - t * v(x, y)
    `cv2.remap` applies bilinear interpolation to fetch sub-pixel intensity from (x_source, y_source).

    Args:
        frame_t0 (np.ndarray): 2D grayscale image at time t=0.
        flow (np.ndarray): (H, W, 2) optical flow field from frame_t0 to frame_t1.
        t (float): Normalized time step (default 0.5 for halfway interpolation).
        border_mode (int): Pixel extrapolation method (default: cv2.BORDER_REFLECT).

    Returns:
        np.ndarray: Warped intermediate 2D array of same shape and dtype as frame_t0.
    """
    h, w = frame_t0.shape[:2]

    # Generate reference coordinate meshgrid
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    # Compute backward sampling map
    map_x = (grid_x - t * flow[..., 0]).astype(np.float32)
    map_y = (grid_y - t * flow[..., 1]).astype(np.float32)

    # Perform bilinear texture mapping / resampling
    warped = cv2.remap(
        src=frame_t0,
        map1=map_x,
        map2=map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=border_mode,
    )
    return warped


def interpolate_linear(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    t: float = 0.5
) -> np.ndarray:
    """
    Computes plain linear interpolation (pixel averaging / cross-dissolve) baseline:
        I_linear = (1.0 - t) * frame_t0 + t * frame_t1

    Args:
        frame_t0 (np.ndarray): 2D grayscale frame at t=0.
        frame_t1 (np.ndarray): 2D grayscale frame at t=1.
        t (float): Intermediate time weight in [0.0, 1.0].

    Returns:
        np.ndarray: Linearly blended intermediate frame (uint8 or float32).
    """
    f0_f = frame_t0.astype(np.float32)
    f1_f = frame_t1.astype(np.float32)

    linear_blend = (1.0 - t) * f0_f + t * f1_f

    if frame_t0.dtype == np.uint8:
        return np.clip(linear_blend, 0, 255).astype(np.uint8)
    return linear_blend


def flow_to_hsv_color(flow: np.ndarray) -> np.ndarray:
    """
    Visualizes optical flow vector field using the standard Middlebury color wheel in HSV space:
    - Hue: Represents the direction / angle of motion (0 to 360 deg).
    - Saturation: Set to maximum (255) for vivid representation.
    - Value: Proportional to the speed / magnitude of motion.

    Args:
        flow (np.ndarray): (H, W, 2) optical flow array.

    Returns:
        np.ndarray: (H, W, 3) BGR image suitable for cv2.imwrite or visualization.
    """
    u = flow[..., 0]
    v = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(u, v)

    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # Angle to Hue: angle is in radians, convert to [0, 180] for 8-bit HSV
    hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255

    # Normalize magnitude to [0, 255]
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = mag_norm.astype(np.uint8)

    # Convert HSV to BGR color space for standard image saving
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def compute_ssim_robust(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) using scikit-image if available,
    with an exact OpenCV/NumPy Gaussian filter fallback if skimage is not installed.
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        return float(ssim_fn(img1, img2, data_range=255))
    except ImportError:
        x = img1.astype(np.float32)
        y = img2.astype(np.float32)
        C1 = (0.01 * 255.0) ** 2
        C2 = (0.03 * 255.0) ** 2
        kernel = (11, 11)
        sigma = 1.5

        mu1 = cv2.GaussianBlur(x, kernel, sigma)
        mu2 = cv2.GaussianBlur(y, kernel, sigma)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(x ** 2, kernel, sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(y ** 2, kernel, sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(x * y, kernel, sigma) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(np.mean(ssim_map))


def calculate_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Calculates quantitative image reconstruction metrics against ground truth:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Peak Signal-to-Noise Ratio (PSNR in dB)
    - Structural Similarity Index (SSIM)

    Args:
        predicted (np.ndarray): Generated synthetic frame.
        ground_truth (np.ndarray): Actual observed ground truth scan frame.

    Returns:
        Dict[str, float]: Evaluation metrics dictionary containing mse, mae, psnr_db, and ssim.
    """
    pred_f = predicted.astype(np.float64)
    gt_f = ground_truth.astype(np.float64)

    mse = float(np.mean((pred_f - gt_f) ** 2))
    mae = float(np.mean(np.abs(pred_f - gt_f)))

    if mse == 0:
        psnr = float("inf")
    else:
        max_val = 255.0 if predicted.dtype == np.uint8 else np.max(gt_f)
        psnr = float(10.0 * np.log10((max_val ** 2) / mse))

    pred_u8 = predicted if predicted.dtype == np.uint8 else np.clip(predicted, 0, 255).astype(np.uint8)
    gt_u8 = ground_truth if ground_truth.dtype == np.uint8 else np.clip(ground_truth, 0, 255).astype(np.uint8)
    ssim_val = compute_ssim_robust(pred_u8, gt_u8)

    return {"mse": mse, "mae": mae, "psnr_db": psnr, "ssim": ssim_val}


def generate_motion_interpolations(
    frame_t0: np.ndarray,
    frame_t1: np.ndarray,
    t: float = 0.5,
    ground_truth: Optional[np.ndarray] = None,
    flow_method: str = "farneback",
) -> Dict:
    """
    High-level pipeline executing optical flow calculation, forward warping,
    and linear interpolation comparison.

    Args:
        frame_t0 (np.ndarray): Grayscale array at t=0.
        frame_t1 (np.ndarray): Grayscale array at t=1.
        t (float): Target time fraction (default 0.5).
        ground_truth (np.ndarray, optional): Ground truth frame at time t.
        flow_method (str): Flow estimation backend — "farneback" or "raft".

    Returns:
        Dict: Dictionary containing:
            - "warped_frame": Optical-flow motion compensated frame.
            - "linear_frame": Baseline pixel averaged frame.
            - "flow_field": Raw (H, W, 2) flow array.
            - "flow_rgb": Colorized flow visualization image.
            - "flow_method": Which backend was used.
            - "metrics_warped": (Optional) Metrics vs ground truth.
            - "metrics_linear": (Optional) Metrics vs ground truth.
    """
    # 1. Compute Dense Optical Flow (dispatcher selects Farneback or RAFT)
    flow = compute_flow(frame_t0, frame_t1, flow_method=flow_method)

    # 2. Forward Warping via cv2.remap
    warped_frame = warp_frame_forward(frame_t0, flow, t=t)

    # 3. Plain Linear Pixel Averaging Baseline
    linear_frame = interpolate_linear(frame_t0, frame_t1, t=t)

    # 4. Colorized Optical Flow Map
    flow_rgb = flow_to_hsv_color(flow)

    method_label = flow_method.upper()
    output = {
        "warped_frame": warped_frame,
        "linear_frame": linear_frame,
        "flow_field": flow,
        "flow_rgb": flow_rgb,
        "flow_method": flow_method,
    }

    # 5. Evaluate against ground truth if provided
    if ground_truth is not None:
        metrics_warped = calculate_metrics(warped_frame, ground_truth)
        metrics_linear = calculate_metrics(linear_frame, ground_truth)
        output["metrics_warped"] = metrics_warped
        output["metrics_linear"] = metrics_linear

        logger.info(
            "Evaluation Metrics at t=%.2f (flow_method=%s):\n"
            "  [Warped (%s)]: SSIM=%.4f | MSE=%.2f | MAE=%.2f | PSNR=%.2f dB\n"
            "  [Linear Averaging   ]: SSIM=%.4f | MSE=%.2f | MAE=%.2f | PSNR=%.2f dB",
            t, method_label, method_label,
            metrics_warped["ssim"], metrics_warped["mse"], metrics_warped["mae"], metrics_warped["psnr_db"],
            metrics_linear["ssim"], metrics_linear["mse"], metrics_linear["mae"], metrics_linear["psnr_db"]
        )

    return output


def run_sample_data_test(
    sample_dir: str = "sample_data",
    output_dir: Optional[str] = None
) -> Dict:
    """
    Test routine that automatically finds downloaded scans in sample_data/,
    runs optical flow warping and linear interpolation at t=0.5,
    and writes visual PNG artifacts.

    Args:
        sample_dir (str): Folder where scan numpy arrays are located.
        output_dir (str, optional): Folder to save output PNGs (defaults to sample_dir).

    Returns:
        Dict: Processed results and file paths.
    """
    sample_path = Path(sample_dir).resolve()
    out_path = Path(output_dir or sample_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Discover sample .npy files
    npy_files = sorted(list(sample_path.glob("*.npy")))
    if len(npy_files) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 .npy files in '{sample_path}', found {len(npy_files)}. "
            "Please run app.data.goes16_fetch first."
        )

    logger.info("Found %d sample .npy scan files.", len(npy_files))
    file_t0 = npy_files[0]
    file_t1 = npy_files[-1]
    file_gt = npy_files[len(npy_files) // 2] if len(npy_files) >= 3 else None

    logger.info("Loading frame T0: %s", file_t0.name)
    frame_t0 = np.load(file_t0)

    logger.info("Loading frame T1: %s", file_t1.name)
    frame_t1 = np.load(file_t1)

    ground_truth = None
    if file_gt and file_gt not in (file_t0, file_t1):
        logger.info("Loading Ground Truth intermediate frame: %s", file_gt.name)
        ground_truth = np.load(file_gt)

    # Run interpolation
    result = generate_motion_interpolations(
        frame_t0=frame_t0,
        frame_t1=frame_t1,
        t=0.5,
        ground_truth=ground_truth
    )

    # Save output artifacts
    warped_png = out_path / "synthetic_warped_t05.png"
    linear_png = out_path / "baseline_linear_t05.png"
    flow_png = out_path / "optical_flow_hsv.png"

    cv2.imwrite(str(warped_png), result["warped_frame"])
    cv2.imwrite(str(linear_png), result["linear_frame"])
    cv2.imwrite(str(flow_png), result["flow_rgb"])

    logger.info("Saved Warped frame PNG: %s", warped_png.name)
    logger.info("Saved Linear frame PNG: %s", linear_png.name)
    logger.info("Saved Optical Flow HSV visualization: %s", flow_png.name)

    # Save absolute difference error maps if ground truth is available
    if ground_truth is not None:
        diff_warped = np.abs(result["warped_frame"].astype(float) - ground_truth.astype(float))
        diff_linear = np.abs(result["linear_frame"].astype(float) - ground_truth.astype(float))

        # Scale difference maps (amplify by 3x for clear visual inspection)
        diff_warped_viz = np.clip(diff_warped * 3.0, 0, 255).astype(np.uint8)
        diff_linear_viz = np.clip(diff_linear * 3.0, 0, 255).astype(np.uint8)

        # Apply heatmap colormap (Inferno / Jet)
        heatmap_warped = cv2.applyColorMap(diff_warped_viz, cv2.COLORMAP_INFERNO)
        heatmap_linear = cv2.applyColorMap(diff_linear_viz, cv2.COLORMAP_INFERNO)

        diff_warped_png = out_path / "error_map_warped_vs_gt.png"
        diff_linear_png = out_path / "error_map_linear_vs_gt.png"

        cv2.imwrite(str(diff_warped_png), heatmap_warped)
        cv2.imwrite(str(diff_linear_png), heatmap_linear)
        logger.info("Saved Warped Error Map: %s", diff_warped_png.name)
        logger.info("Saved Linear Error Map: %s", diff_linear_png.name)

    return result


def main():
    """Command Line Interface Entry Point."""
    parser = argparse.ArgumentParser(
        description="Farneback Optical Flow Motion Warping vs Linear Averaging Pipeline"
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="sample_data",
        help="Directory containing input .npy satellite scan files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sample_data",
        help="Target folder for output images and flow visualizations"
    )
    args = parser.parse_args()

    run_sample_data_test(sample_dir=args.sample_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
