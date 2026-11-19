"""
Training Pipeline for ASTRA Direct Synthesis Model
==================================================
File: train_direct_synthesis.py

# This is a lightweight proof-of-concept direct-synthesis model, explicitly
# not the full diffusion architecture described in the project proposal,
# trained on a small dataset.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.processing.direct_synthesis_model import DirectSynthesisUNet
from app.processing.optical_flow import calculate_metrics, compute_farneback_flow
from app.processing.refinement_model import CombinedL1SSIMLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_direct_synthesis")


class DirectSynthesisDataset(Dataset):
    """
    Dataset for direct neural synthesis.
    Loads T0, T1, T_mid patches, computes Farneback flow, and constructs 4-channel input tensor.
    """

    def __init__(self, t0_files: List[Path], max_samples: int = None):
        self.samples = []
        if max_samples and max_samples > 0:
            t0_files = t0_files[:max_samples]

        logger.info("Loading and preparing %d patch samples...", len(t0_files))
        for p0_path in t0_files:
            pm_path = Path(str(p0_path).replace("_t0.npy", "_t_mid.npy"))
            p1_path = Path(str(p0_path).replace("_t0.npy", "_t1.npy"))
            if not pm_path.exists() or not p1_path.exists():
                continue

            try:
                p0 = np.load(p0_path).astype(np.float32)
                pm = np.load(pm_path).astype(np.float32)
                p1 = np.load(p1_path).astype(np.float32)

                # Compute Farneback flow on uint8 arrays
                f0_u8 = np.clip(p0, 0, 255).astype(np.uint8)
                f1_u8 = np.clip(p1, 0, 255).astype(np.uint8)
                flow = compute_farneback_flow(f0_u8, f1_u8, pyr_scale=0.5, levels=3, winsize=15, iterations=3)

                # Normalize inputs
                t0_norm = p0 / 255.0
                t1_norm = p1 / 255.0
                u_norm = np.clip(flow[..., 0] / 20.0, -1.0, 1.0)
                v_norm = np.clip(flow[..., 1] / 20.0, -1.0, 1.0)
                target_norm = pm / 255.0

                # 4-channel input: [4, 256, 256]
                inp = np.stack([t0_norm, t1_norm, u_norm, v_norm], axis=0).astype(np.float32)
                tgt = target_norm[np.newaxis, ...].astype(np.float32)

                self.samples.append({
                    "input": inp,
                    "target": tgt,
                    "id": p0_path.stem.replace("_t0", ""),
                    "triplet": p0_path.parent.name,
                    "p0_raw": f0_u8,
                    "p1_raw": f1_u8,
                    "pm_raw": np.clip(pm, 0, 255).astype(np.uint8),
                    "flow_raw": flow,
                })
            except Exception as e:
                logger.warning("Error processing %s: %s", p0_path.name, e)

        logger.info("Dataset successfully prepared with %d valid samples.", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.from_numpy(s["input"]), torch.from_numpy(s["target"])


def run_training(
    train_dir: Path = Path("sample_data/patches/train"),
    val_dir: Path = Path("sample_data/patches/val"),
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    max_train: int = 96,
    max_val: int = 32,
    checkpoint_dir: Path = Path("checkpoints"),
    demo_dir: Path = Path("sample_data/patches/neural_synthesis_demo"),
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover patch files
    train_files = sorted(list(train_dir.glob("*/*_t0.npy")))
    val_files = sorted(list(val_dir.glob("*/*_t0.npy")))
    logger.info("Discovered %d train patches, %d val patches.", len(train_files), len(val_files))

    train_ds = DirectSynthesisDataset(train_files, max_samples=max_train)
    val_ds = DirectSynthesisDataset(val_files, max_samples=max_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Setup model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectSynthesisUNet(in_channels=4, out_channels=1, base_features=32).to(device)
    criterion = CombinedL1SSIMLoss(alpha=0.8)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("Initialized DirectSynthesisUNet on %s (Parameters: %d)", device, sum(p.numel() for p in model.parameters()))

    history = {"train_loss": [], "val_loss": [], "val_l1": [], "val_ssim": [], "epochs": []}
    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best_direct_synthesis_model.pth"

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss, _, _ = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss, val_l1, val_ssim = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss, l1_val, ssim_val = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_l1 += l1_val.item() * x.size(0)
                val_ssim += ssim_val.item() * x.size(0)
        val_loss /= len(val_ds)
        val_l1 /= len(val_ds)
        val_ssim /= len(val_ds)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_l1"].append(val_l1)
        history["val_ssim"].append(val_ssim)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_l1": val_l1,
                "val_ssim": val_ssim,
            }, best_checkpoint_path)

        logger.info(
            "Epoch [%2d/%2d] | Train Loss: %.4f | Val Loss: %.4f (L1: %.4f, SSIM: %.4f) %s",
            epoch, epochs, train_loss, val_loss, val_l1, val_ssim, "(★ Best)" if is_best else ""
        )

    # 4. Save Loss Curve Plot
    loss_curve_path = checkpoint_dir / "direct_synthesis_loss_curve.png"
    plt.figure(figsize=(9, 5))
    plt.plot(history["epochs"], history["train_loss"], label="Train Loss (L1+SSIM)", color="#00f0ff", lw=2)
    plt.plot(history["epochs"], history["val_loss"], label="Val Loss (L1+SSIM)", color="#fbbf24", lw=2)
    plt.title("ASTRA Direct Synthesis U-Net: Training & Validation Loss Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()
    logger.info("Saved loss curve plot to %s", loss_curve_path)

    # 5. Generate and Save Demo Output Samples for Interactive UI
    logger.info("Generating demo frames using best trained checkpoint...")
    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    demo_catalog = []
    # Take first 4 distinct validation samples across triplets
    for i in range(min(4, len(val_ds))):
        s = val_ds.samples[i]
        inp_tensor = torch.from_numpy(s["input"]).unsqueeze(0).to(device)
        with torch.no_grad():
            out_tensor = model(inp_tensor)

        pred_f = out_tensor.squeeze().cpu().numpy()
        pred_u8 = np.clip(pred_f * 255.0, 0, 255).astype(np.uint8)
        gt_u8 = s["pm_raw"]
        p0_u8 = s["p0_raw"]
        p1_u8 = s["p1_raw"]

        # Calculate real metrics
        m = calculate_metrics(pred_u8, gt_u8)

        # Compute L1 absolute difference residual
        diff_u8 = np.abs(pred_u8.astype(np.int32) - gt_u8.astype(np.int32)).astype(np.uint8)
        diff_color = cv2.applyColorMap(diff_u8 * 3, cv2.COLORMAP_TURBO)

        # Save PNG images
        patch_tag = f"demo_patch_{i:02d}_{s['triplet']}"
        f_pred = demo_dir / f"{patch_tag}_synth.png"
        f_gt = demo_dir / f"{patch_tag}_gt.png"
        f_t0 = demo_dir / f"{patch_tag}_t0.png"
        f_t1 = demo_dir / f"{patch_tag}_t1.png"
        f_diff = demo_dir / f"{patch_tag}_residual.png"

        cv2.imwrite(str(f_pred), pred_u8)
        cv2.imwrite(str(f_gt), gt_u8)
        cv2.imwrite(str(f_t0), p0_u8)
        cv2.imwrite(str(f_t1), p1_u8)
        cv2.imwrite(str(f_diff), diff_color)

        demo_catalog.append({
            "id": patch_tag,
            "triplet_id": s["triplet"],
            "patch_name": f"Cloud ROI #{i+1} ({s['triplet']})",
            "metrics": {
                "mse": float(m["mse"]),
                "psnr_db": float(m["psnr_db"]),
                "ssim": float(m["ssim"]),
                "mae": float(m["mae"]),
            },
            "files": {
                "t0": f"{patch_tag}_t0.png",
                "t1": f"{patch_tag}_t1.png",
                "gt": f"{patch_tag}_gt.png",
                "synth": f"{patch_tag}_synth.png",
                "residual": f"{patch_tag}_residual.png",
            }
        })

    demo_manifest_file = demo_dir / "demo_catalog.json"
    with open(demo_manifest_file, "w", encoding="utf-8") as fp:
        json.dump({
            "model_type": "DirectSynthesisUNet (Compact Conditional Generative Model)",
            "trained_epochs": epochs,
            "best_epoch": best_ckpt["epoch"],
            "best_val_loss": float(best_val_loss),
            "samples": demo_catalog,
        }, fp, indent=2)

    logger.info("Demo catalog and generated frames saved to %s", demo_dir)
    print("\nTraining and Demo Frame Generation Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-train", type=int, default=96)
    parser.add_argument("--max-val", type=int, default=32)
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        max_train=args.max_train,
        max_val=args.max_val
    )
