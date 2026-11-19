"""
ASTRA Four-Method Benchmark Evaluation
=======================================
File: evaluate_all_methods.py

Purpose:
--------
Evaluates and benchmarks all four temporal frame synthesis methods on the
held-out ground truth validation patches:
  1. Linear Interpolation (Alpha blending baseline)
  2. Farneback (Classical polynomial expansion optical flow)
  3. RAFT (Pretrained deep correlation volume optical flow)
  4. RAFT + Refinement U-Net (Deep optical flow with trained residual refinement)

Metrics evaluated:
  - Mean Squared Error (MSE)
  - Peak Signal-to-Noise Ratio (PSNR in dB)
  - Structural Similarity Index (SSIM)
  - Mean Absolute Error (MAE)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from app.processing.optical_flow import (
    calculate_metrics,
    compute_flow,
    interpolate_linear,
    warp_frame_forward,
)
from app.processing.refinement_model import PatchRefinementUNet

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluate_methods")


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[PatchRefinementUNet, int, float]:
    """Loads the best trained refinement model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PatchRefinementUNet(in_channels=1, out_channels=1, base_features=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("val_loss", 0.0)
    logger.info("Loaded refinement model from Epoch %d (Validation Loss: %.4f)", best_epoch, best_loss)
    return model, best_epoch, best_loss


def evaluate_patch(
    p0: np.ndarray,
    pm_gt: np.ndarray,
    p1: np.ndarray,
    model: PatchRefinementUNet,
    device: torch.device,
    cached_raft_path: Path = None,
) -> Dict[str, Dict[str, float]]:
    """
    Runs all 4 methods on a single patch triplet and calculates metrics against ground truth pm_gt.
    """
    # 1. Linear Interpolation
    pred_linear = interpolate_linear(p0, p1, t=0.5)

    # 2. Farneback Optical Flow
    flow_fb = compute_flow(p0, p1, flow_method="farneback")
    pred_fb = warp_frame_forward(p0, flow_fb, t=0.5)

    # 3. RAFT Optical Flow
    if cached_raft_path and cached_raft_path.exists():
        pred_raft = np.load(cached_raft_path)
    else:
        flow_raft = compute_flow(p0, p1, flow_method="raft")
        pred_raft = warp_frame_forward(p0, flow_raft, t=0.5)

    # 4. RAFT + Refinement U-Net
    x_tensor = (
        torch.from_numpy(pred_raft.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        refined_tensor = model(x_tensor)
    pred_refined = np.clip(
        refined_tensor.squeeze().cpu().numpy() * 255.0, 0, 255
    ).astype(np.uint8)

    # Compute pixel statistics right before comparison
    refined_min = float(np.min(pred_refined))
    refined_max = float(np.max(pred_refined))
    refined_mean = float(np.mean(pred_refined))

    gt_min = float(np.min(pm_gt))
    gt_max = float(np.max(pm_gt))
    gt_mean = float(np.mean(pm_gt))

    pixel_stats = {
        "refined": {"min": refined_min, "max": refined_max, "mean": refined_mean},
        "gt": {"min": gt_min, "max": gt_max, "mean": gt_mean},
    }

    # Compute metrics for all four methods
    return {
        "Linear Interpolation": calculate_metrics(pred_linear, pm_gt),
        "Farneback (Warp)": calculate_metrics(pred_fb, pm_gt),
        "RAFT (Warp)": calculate_metrics(pred_raft, pm_gt),
        "RAFT + Refinement U-Net": calculate_metrics(pred_refined, pm_gt),
    }, {
        "linear": pred_linear,
        "farneback": pred_fb,
        "raft": pred_raft,
        "refined": pred_refined,
        "gt": pm_gt,
    }, pixel_stats


def run_benchmark(
    manifest_path: Path = Path("sample_data/patches/patches_manifest.json"),
    checkpoint_path: Path = Path("checkpoints/best_refinement_model.pth"),
    split: str = "val",
    save_visuals: bool = True,
    output_dir: Path = Path("sample_data/comparison_patches"),
) -> Dict[str, Dict[str, float]]:
    """Runs the 4-method comparison across all held-out validation patches."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation compute device: %s", device)

    # Load manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    patches = [p for p in manifest.get("patches", []) if split == "all" or p.get("split") == split]
    if not patches:
        raise ValueError(f"No patches found for split='{split}' in {manifest_path}")

    # Load trained model
    model, best_epoch, best_loss = load_model(checkpoint_path, device)

    base_dir = manifest_path.parent
    if save_visuals:
        output_dir.mkdir(parents=True, exist_ok=True)

    method_names = [
        "Linear Interpolation",
        "Farneback (Warp)",
        "RAFT (Warp)",
        "RAFT + Refinement U-Net",
    ]

    all_metrics: Dict[str, List[Dict[str, float]]] = {m: [] for m in method_names}

    print("\n" + "=" * 78)
    print(f" ASTRA 4-Method Benchmark on Held-Out Ground Truth Patches (Split: {split.upper()})")
    print(f" Total Evaluation Patches: {len(patches)} (from source triplet: {manifest.get('triplets', {}).get(split, 'N/A')})")
    print(f" Checkpoint Loaded:        {checkpoint_path.name} (Epoch {best_epoch})")
    print("=" * 78 + "\n", flush=True)

    for idx, patch_info in enumerate(patches, start=1):
        pid = patch_info["patch_id"]
        files = patch_info["files"]

        p0 = np.load(base_dir / files["t0"])
        pm = np.load(base_dir / files["t_mid"])
        p1 = np.load(base_dir / files["t1"])

        t0_path = base_dir / files["t0"]
        cached_raft = t0_path.parent / f"{t0_path.stem.replace('_t0', '')}_warped_raft.npy"

    all_pixel_stats: List[Dict] = []

    for idx, patch_info in enumerate(patches, start=1):
        pid = patch_info["patch_id"]
        files = patch_info["files"]

        p0 = np.load(base_dir / files["t0"])
        pm = np.load(base_dir / files["t_mid"])
        p1 = np.load(base_dir / files["t1"])

        t0_path = base_dir / files["t0"]
        cached_raft = t0_path.parent / f"{t0_path.stem.replace('_t0', '')}_warped_raft.npy"

        metrics_map, images_map, pixel_stats = evaluate_patch(
            p0=p0, pm_gt=pm, p1=p1,
            model=model, device=device,
            cached_raft_path=cached_raft
        )

        all_pixel_stats.append(pixel_stats)

        for m_name in method_names:
            all_metrics[m_name].append(metrics_map[m_name])

        ref_s = pixel_stats["refined"]
        gt_s = pixel_stats["gt"]
        logger.info(
            "[%d/%d] %s -> Refined: [min=%.1f, max=%.1f, mean=%.1f] | GT: [min=%.1f, max=%.1f, mean=%.1f] | Refined SSIM: %.4f",
            idx, len(patches), pid,
            ref_s["min"], ref_s["max"], ref_s["mean"],
            gt_s["min"], gt_s["max"], gt_s["mean"],
            metrics_map["RAFT + Refinement U-Net"]["ssim"],
        )

        # Save side-by-side visual strip for inspection
        if save_visuals and idx <= 4:
            panel = np.hstack([
                images_map["linear"],
                images_map["farneback"],
                images_map["raft"],
                images_map["refined"],
                images_map["gt"],
            ])
            cv2.imwrite(str(output_dir / f"{pid}_comparison_strip.png"), panel)

    # ---------------------------------------------------------------------------
    # Summary Table Construction
    # ---------------------------------------------------------------------------
    summary: Dict[str, Dict[str, float]] = {}

    header = f"{'Method':<26} | {'MSE':>10} | {'PSNR (dB)':>10} | {'SSIM':>10} | {'MAE':>10}"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  HELD-OUT VALIDATION BENCHMARK RESULTS (Average across all patches)")
    print("=" * len(header))
    print(header)
    print(sep)

    for m_name in method_names:
        list_m = all_metrics[m_name]
        mean_mse = float(np.mean([m["mse"] for m in list_m]))
        mean_psnr = float(np.mean([m["psnr_db"] for m in list_m]))
        mean_ssim = float(np.mean([m["ssim"] for m in list_m]))
        mean_mae = float(np.mean([m["mae"] for m in list_m]))

        summary[m_name] = {
            "mse": mean_mse,
            "psnr_db": mean_psnr,
            "ssim": mean_ssim,
            "mae": mean_mae,
        }

        print(
            f"{m_name:<26} | "
            f"{mean_mse:>10.2f} | "
            f"{mean_psnr:>10.2f} | "
            f"{mean_ssim:>10.4f} | "
            f"{mean_mae:>10.2f}"
        )

    print(sep)

    # ---------------------------------------------------------------------------
    # Detailed Pixel Statistics for RAFT + Refinement U-Net vs Ground Truth
    # ---------------------------------------------------------------------------
    avg_ref_min = float(np.mean([ps["refined"]["min"] for ps in all_pixel_stats]))
    avg_ref_max = float(np.mean([ps["refined"]["max"] for ps in all_pixel_stats]))
    avg_ref_mean = float(np.mean([ps["refined"]["mean"] for ps in all_pixel_stats]))

    avg_gt_min = float(np.mean([ps["gt"]["min"] for ps in all_pixel_stats]))
    avg_gt_max = float(np.mean([ps["gt"]["max"] for ps in all_pixel_stats]))
    avg_gt_mean = float(np.mean([ps["gt"]["mean"] for ps in all_pixel_stats]))

    overall_ref_min = float(np.min([ps["refined"]["min"] for ps in all_pixel_stats]))
    overall_ref_max = float(np.max([ps["refined"]["max"] for ps in all_pixel_stats]))
    overall_gt_min = float(np.min([ps["gt"]["min"] for ps in all_pixel_stats]))
    overall_gt_max = float(np.max([ps["gt"]["max"] for ps in all_pixel_stats]))

    print(f"\n  PIXEL VALUE STATISTICS (Before Comparison against Ground Truth):")
    print(f"  * RAFT + Refinement Output : Min = {overall_ref_min:5.1f} (avg min: {avg_ref_min:5.1f}) | "
          f"Max = {overall_ref_max:5.1f} (avg max: {avg_ref_max:5.1f}) | "
          f"Mean = {avg_ref_mean:6.2f}")
    print(f"  * Ground Truth (Target)    : Min = {overall_gt_min:5.1f} (avg min: {avg_gt_min:5.1f}) | "
          f"Max = {overall_gt_max:5.1f} (avg max: {avg_gt_max:5.1f}) | "
          f"Mean = {avg_gt_mean:6.2f}")
    print(sep)

    # Relative improvement
    base_mse = summary["Linear Interpolation"]["mse"]
    base_ssim = summary["Linear Interpolation"]["ssim"]
    refined_mse = summary["RAFT + Refinement U-Net"]["mse"]
    refined_ssim = summary["RAFT + Refinement U-Net"]["ssim"]

    mse_reduction = ((base_mse - refined_mse) / base_mse) * 100.0 if base_mse > 0 else 0.0
    ssim_boost = ((refined_ssim - base_ssim) / base_ssim) * 100.0 if base_ssim > 0 else 0.0

    print(f"\n  Improvement over Linear Baseline:")
    print(f"  * MSE Error Reduction:   {mse_reduction:+.1f}%")
    print(f"  * SSIM Quality Increase: {ssim_boost:+.1f}%")
    print(sep)

    # ---------------------------------------------------------------------------
    # Evaluation Scale & Combination Methodology Confirmation
    # ---------------------------------------------------------------------------
    print("\n" + "=" * len(header))
    print("  EVALUATION SCALE & COMBINATION METHODOLOGY CONFIRMATION")
    print("=" * len(header))
    print("  1. Execution Scale: INDIVIDUAL 256x256 PATCHES")
    print("     The metrics comparison is computed directly on individual 256x256")
    print("     spatiotemporal patches extracted from cloud-active regions.")
    print("     Each patch is evaluated in isolation against its corresponding")
    print("     held-out ground truth 256x256 window. The table values represent")
    print("     the arithmetic mean across all evaluated validation patches.")
    print()
    print("  2. Patch Combination Methodology (How patches relate to full frames):")
    print("     - In this benchmark: Patches are evaluated independently, not stitched.")
    print("     - Full-Canvas Mosaicking: Each patch record retains its original full-disk")
    print("       bounding box coordinates (ymin, xmin, ymax, xmax). To reconstruct full")
    print("       5424x5424 frames, patches are tiled with an overlap margin (e.g. 32-64px)")
    print("       and blended using 2D Hann / raised-cosine distance weights to ensure")
    print("       seamless transitions between adjacent patches.")
    print("     - Fully Convolutional Direct Inference: Because PatchRefinementUNet has no")
    print("       dense/linear layers, it can also process arbitrary-sized dimensions")
    print("       (multiples of 16) directly without mandatory patch tiling.")
    print("=" * len(header) + "\n", flush=True)


    # Save benchmark JSON summary
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": split,
            "patches_count": len(patches),
            "summary": summary,
            "per_patch_metrics": {m: all_metrics[m] for m in method_names},
        }, f, indent=2)
    logger.info("Saved benchmark summary JSON to: %s", summary_path)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate 4 frame interpolation methods on held-out patches")
    parser.add_argument(
        "--manifest",
        type=str,
        default="sample_data/patches/patches_manifest.json",
        help="Path to patches_manifest.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_refinement_model.pth",
        help="Path to best_refinement_model.pth"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train", "all"],
        help="Split to evaluate on (default: val)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sample_data/comparison_patches",
        help="Directory to save comparison visuals and summary JSON"
    )

    args = parser.parse_args()

    run_benchmark(
        manifest_path=Path(args.manifest),
        checkpoint_path=Path(args.checkpoint),
        split=args.split,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
