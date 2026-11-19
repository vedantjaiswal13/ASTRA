"""
Farneback vs RAFT — Side-by-Side Comparison Script
====================================================
File: compare_flow_methods.py

Loads the three pre-downloaded GOES-16 satellite scans from sample_data/,
generates a synthetic midpoint frame using both Farneback (classical) and
RAFT (deep learning) optical flow, then evaluates each against the real
ground-truth mid-scan and prints a comparison table.

Usage:
    python compare_flow_methods.py
"""

from pathlib import Path

import cv2
import numpy as np

from app.processing.optical_flow import (
    compute_flow,
    warp_frame_forward,
    interpolate_linear,
    calculate_metrics,
)

# ---------------------------------------------------------------------------
# 1. Locate and load the three sample scans (t0, ground-truth mid, t1)
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path("sample_data")
npy_files = sorted(SAMPLE_DIR.glob("*.npy"))

if len(npy_files) < 3:
    raise FileNotFoundError(
        f"Expected 3 .npy files in {SAMPLE_DIR}, found {len(npy_files)}. "
        "Run app.data.goes16_fetch first."
    )

file_t0 = npy_files[0]   # earliest scan
file_gt = npy_files[1]   # real mid-time ground-truth scan
file_t1 = npy_files[2]   # latest scan

print(f"T0 (start) : {file_t0.name}")
print(f"GT (mid)   : {file_gt.name}")
print(f"T1 (end)   : {file_t1.name}")
print()

frame_t0 = np.load(file_t0)
frame_gt = np.load(file_gt)
frame_t1 = np.load(file_t1)

print(f"Frame shape: {frame_t0.shape}  dtype: {frame_t0.dtype}")
print()

# ---------------------------------------------------------------------------
# 2. Generate intermediate frames with each method
# ---------------------------------------------------------------------------
T = 0.5  # midpoint

# --- Farneback (classical polynomial expansion, CPU, fast) -----------------
print("Computing Farneback flow …")
flow_fb = compute_flow(frame_t0, frame_t1, flow_method="farneback")
warped_fb = warp_frame_forward(frame_t0, flow_fb, t=T)

# --- RAFT (pretrained deep model, CPU or GPU, slower but handles big motion)
print("Computing RAFT flow …")
flow_raft = compute_flow(frame_t0, frame_t1, flow_method="raft")
warped_raft = warp_frame_forward(frame_t0, flow_raft, t=T)

# --- Linear Averaging baseline (no flow, just pixel blending) --------------
linear_blend = interpolate_linear(frame_t0, frame_t1, t=T)

# ---------------------------------------------------------------------------
# 3. Evaluate all three against the real ground-truth mid frame
# ---------------------------------------------------------------------------
metrics_fb     = calculate_metrics(warped_fb, frame_gt)
metrics_raft   = calculate_metrics(warped_raft, frame_gt)
metrics_linear = calculate_metrics(linear_blend, frame_gt)

# ---------------------------------------------------------------------------
# 4. Print comparison table
# ---------------------------------------------------------------------------
header = f"{'Method':<22} {'MSE':>10} {'PSNR (dB)':>10} {'SSIM':>10} {'MAE':>10}"
sep    = "-" * len(header)

print()
print("=" * len(header))
print("  Interpolation Quality vs Real Ground-Truth Mid-Scan")
print("=" * len(header))
print(header)
print(sep)

for label, m in [
    ("Farneback (warp)",   metrics_fb),
    ("RAFT (warp)",        metrics_raft),
    ("Linear Averaging",   metrics_linear),
]:
    print(
        f"{label:<22} "
        f"{m['mse']:>10.2f} "
        f"{m['psnr_db']:>10.2f} "
        f"{m['ssim']:>10.4f} "
        f"{m['mae']:>10.2f}"
    )

print(sep)
print()

# ---------------------------------------------------------------------------
# 5. Save the generated frames for visual inspection (optional)
# ---------------------------------------------------------------------------
OUT = SAMPLE_DIR / "comparison"
OUT.mkdir(exist_ok=True)

cv2.imwrite(str(OUT / "warped_farneback.png"), warped_fb)
cv2.imwrite(str(OUT / "warped_raft.png"),      warped_raft)
cv2.imwrite(str(OUT / "linear_blend.png"),     linear_blend)
cv2.imwrite(str(OUT / "ground_truth.png"),     frame_gt)

print(f"Output PNGs saved to: {OUT.resolve()}")
