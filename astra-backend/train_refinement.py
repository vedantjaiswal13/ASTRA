"""
ASTRA Neural Optical Flow Refinement Training Pipeline
======================================================
File: train_refinement.py

DATASET SPECIFICATION & RESEARCH PROTOTYPE NOTE:
------------------------------------------------
- Exact Dataset Size: 2 GOES-16 triplets (2024-01-01_18, 2024-01-15_00),
  16 patches total of dimension 256x256 (8 train patches / 8 validation patches).
- RESEARCH PROTOTYPE STATEMENT: This is a small-scale research prototype
  demonstrating the deep residual refinement concept on motion-compensated
  satellite cloud imagery. It is not intended as a production-scale
  operational meteorological forecasting model.

Requirements Implemented:
1. Loads train/validation patch splits from sample_data/patches/patches_manifest.json.
2. Uses a weighted combination of L1 pixel loss and SSIM loss against real withheld t_mid.
3. Trains for a few dozen epochs with early stopping based on validation loss.
4. Prints train and validation loss after every epoch.
5. Saves a loss curve plot (train vs val loss over epochs) as a PNG.
6. Saves the best checkpoint by validation loss (not just the final epoch).
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.processing.optical_flow import compute_raft_flow, warp_frame_forward
from app.processing.refinement_model import CombinedL1SSIMLoss, PatchRefinementUNet

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_refinement")


class SatellitePatchDataset(Dataset):
    """
    PyTorch Dataset for satellite cloud patch refinement.
    Loads (t0, t_mid, t1), retrieves or computes the RAFT-warped intermediate patch,
    and returns:
        x: RAFT-warped patch [1, 256, 256] in [0.0, 1.0]
        y: Ground-truth withheld middle patch [1, 256, 256] in [0.0, 1.0]
    """

    def __init__(
        self,
        records: List[Dict],
        base_dir: Path,
        split_name: str = "train",
        cache_warped: bool = True
    ):
        self.records = records
        self.base_dir = base_dir
        self.split_name = split_name
        self.cache_warped = cache_warped
        self.items: List[Tuple[np.ndarray, np.ndarray, str]] = []

        logger.info("Initializing %s dataset with %d patches...", split_name.upper(), len(records))
        self._prepare_data()

    def _prepare_data(self):
        for idx, rec in enumerate(self.records):
            pid = rec["patch_id"]
            files = rec["files"]

            t0_path = self.base_dir / files["t0"]
            tm_path = self.base_dir / files["t_mid"]
            t1_path = self.base_dir / files["t1"]

            p0 = np.load(t0_path)
            pm = np.load(tm_path)
            p1 = np.load(t1_path)

            # Check for cached RAFT-warped patch
            parent_dir = t0_path.parent
            warped_cache_path = parent_dir / f"{t0_path.stem.replace('_t0', '')}_warped_raft.npy"

            if warped_cache_path.exists():
                warped = np.load(warped_cache_path)
            else:
                logger.info(
                    "[%s %d/%d] Precomputing RAFT flow and warp for patch %s...",
                    self.split_name.upper(), idx + 1, len(self.records), pid
                )
                # Compute flow with RAFT-Small
                flow = compute_raft_flow(p0, p1, num_iterations=8)
                warped = warp_frame_forward(p0, flow, t=0.5)
                if self.cache_warped:
                    np.save(warped_cache_path, warped)

            # Convert to float32 in [0.0, 1.0]
            warped_norm = warped.astype(np.float32) / 255.0
            target_norm = pm.astype(np.float32) / 255.0

            self.items.append((warped_norm, target_norm, pid))

        logger.info("%s dataset ready: %d samples loaded.", self.split_name.upper(), len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        warped_norm, target_norm, pid = self.items[idx]
        # Shape: [1, H, W]
        x = torch.from_numpy(warped_norm).unsqueeze(0)
        y = torch.from_numpy(target_norm).unsqueeze(0)
        return x, y, pid


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedL1SSIMLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """Runs one training epoch, returns (total_loss, l1_loss, ssim_loss)."""
    model.train()
    total_loss, total_l1, total_ssim = 0.0, 0.0, 0.0

    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss, l1_val, ssim_val = criterion(pred, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_l1 += l1_val.item() * batch_size
        total_ssim += ssim_val.item() * batch_size

    n = len(dataloader.dataset)
    return total_loss / n, total_l1 / n, total_ssim / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedL1SSIMLoss,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluates model on validation set, returns (total_loss, l1_loss, ssim_loss)."""
    model.eval()
    total_loss, total_l1, total_ssim = 0.0, 0.0, 0.0

    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss, l1_val, ssim_val = criterion(pred, y)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_l1 += l1_val.item() * batch_size
        total_ssim += ssim_val.item() * batch_size

    n = len(dataloader.dataset)
    return total_loss / n, total_l1 / n, total_ssim / n


def plot_loss_curves(history: Dict[str, List[float]], best_epoch: int, output_path: Path):
    """Generates and saves train vs validation loss curve plot as PNG."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs, history["train_loss"], label="Train Loss (L1 + SSIM)", color="#2563eb", lw=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss (L1 + SSIM)", color="#ea580c", lw=2)

    # Highlight best epoch
    best_val_loss = history["val_loss"][best_epoch - 1]
    plt.scatter(
        [best_epoch], [best_val_loss],
        color="#16a34a", s=100, zorder=5,
        label=f"Best Checkpoint (Ep {best_epoch}: {best_val_loss:.4f})"
    )

    plt.title("ASTRA Neural Refinement Model: Training & Validation Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (Weighted L1 + SSIM)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved loss curve plot to: %s", output_path)


def save_validation_preview(
    model: nn.Module,
    val_dataset: SatellitePatchDataset,
    device: torch.device,
    output_path: Path
):
    """Saves side-by-side visual comparison: [Warped RAFT | Refined U-Net | Real Ground Truth]."""
    if len(val_dataset) == 0:
        return

    model.eval()
    x, y, pid = val_dataset[0]
    with torch.no_grad():
        x_dev = x.unsqueeze(0).to(device)
        pred = model(x_dev).squeeze(0).cpu().numpy()[0]

    warped_img = (x.squeeze(0).numpy() * 255.0).astype(np.uint8)
    refined_img = (pred * 255.0).astype(np.uint8)
    target_img = (y.squeeze(0).numpy() * 255.0).astype(np.uint8)

    # Difference maps (magnified x3 for visual clarity)
    diff_warped = np.clip(np.abs(warped_img.astype(float) - target_img.astype(float)) * 3, 0, 255).astype(np.uint8)
    diff_refined = np.clip(np.abs(refined_img.astype(float) - target_img.astype(float)) * 3, 0, 255).astype(np.uint8)

    # Annotate panels
    def add_label(img, text):
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return out

    panel = np.hstack([
        add_label(warped_img, "RAFT Warped"),
        add_label(refined_img, "Refined U-Net"),
        add_label(target_img, "Ground Truth Mid"),
        add_label(diff_refined, "Refinement Error x3")
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)
    logger.info("Saved validation preview visual to: %s", output_path)


def run_training(
    manifest_path: Path,
    epochs: int = 25,
    lr: float = 1e-3,
    l1_weight: float = 0.8,
    patience: int = 6,
    batch_size: int = 4,
    checkpoint_dir: Path = Path("checkpoints"),
    plot_path: Path = Path("checkpoints/loss_curve.png"),
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the complete training workflow:
    1. Loads train/validation patch splits.
    2. Trains compact U-Net on RAFT-warped patches with L1 + SSIM loss.
    3. Employs early stopping and best checkpoint persistence.
    4. Generates loss curve plot and validation preview.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # Load dataset manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    patches = manifest.get("patches", [])
    train_records = [p for p in patches if p["split"] == "train"]
    val_records = [p for p in patches if p["split"] == "val"]

    # Research prototype check and logging
    triplets_count = manifest.get("summary", {}).get("total_triplets", len(manifest.get("triplets", {})))
    print("\n" + "=" * 75)
    print(" ASTRA NEURAL REFINEMENT MODEL TRAINING")
    print("=" * 75)
    print(f" Dataset Manifest:         {manifest_path}")
    print(f" Triplets Used:            {triplets_count} ({manifest.get('triplets', {})})")
    print(f" Total Patches:            {len(patches)} (Train: {len(train_records)}, Val: {len(val_records)})")
    print(f" Architecture:             Compact 4-Level Residual U-Net")
    print(f" Loss Function:            {l1_weight:.1f} * L1 + {1.0 - l1_weight:.1f} * (1 - SSIM)")
    print(f" Target Epochs:            {epochs} (Early Stopping Patience: {patience})")
    print(f" Compute Device:           {device}")
    print(" NOTE: Small-scale research prototype, not a production-scale model.")
    print("=" * 75 + "\n", flush=True)

    if len(train_records) == 0:
        raise ValueError("Train split contains 0 patches!")

    base_dir = manifest_path.parent
    train_dataset = SatellitePatchDataset(train_records, base_dir=base_dir, split_name="train")

    # If val records is empty (e.g. 1 triplet), use train as eval fallback with notice
    eval_dataset = (
        SatellitePatchDataset(val_records, base_dir=base_dir, split_name="val")
        if len(val_records) > 0
        else train_dataset
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, optimizer, and scheduler
    model = PatchRefinementUNet(in_channels=1, out_channels=1, base_features=32).to(device)
    criterion = CombinedL1SSIMLoss(alpha=l1_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_refinement_model.pth"

    history = {
        "train_loss": [], "train_l1": [], "train_ssim": [],
        "val_loss": [], "val_l1": [], "val_ssim": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    patience_counter = 0

    print(f"{'Epoch':<10} | {'Train Loss':<12} | {'Val Loss':<12} | {'Train (L1/SSIM)':<18} | {'Val (L1/SSIM)':<18} | Status")
    print("-" * 88)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_l1, tr_ssim = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_l1, v_ssim = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_l1"].append(tr_l1)
        history["train_ssim"].append(tr_ssim)
        history["val_loss"].append(v_loss)
        history["val_l1"].append(v_l1)
        history["val_ssim"].append(v_ssim)

        is_best = False
        if v_loss < best_val_loss - 1e-4:
            best_val_loss = v_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            is_best = True
            status = f"BEST ({best_val_loss:.4f}) *"
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{patience}"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"{tr_loss:.4f}       | "
            f"{v_loss:.4f}     | "
            f"{tr_l1:.4f} / {tr_ssim:.4f}    | "
            f"{v_l1:.4f} / {v_ssim:.4f}  | {status}"
        )

        # Early stopping check
        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Validation loss did not improve for {patience} consecutive epochs.")
            break

    # Save the best checkpoint
    logger.info("Saving best model checkpoint from Epoch %d (Val Loss: %.4f)...", best_epoch, best_val_loss)
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_weights,
        "val_loss": best_val_loss,
        "config": {
            "patch_size": 256,
            "in_channels": 1,
            "out_channels": 1,
            "base_features": 32,
            "l1_weight": l1_weight,
            "ssim_weight": 1.0 - l1_weight,
            "epochs_run": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "triplets_count": triplets_count,
            "total_patches": len(patches),
        },
        "history": history,
    }, best_checkpoint_path)

    # Save loss curves plot
    plot_loss_curves(history, best_epoch=best_epoch, output_path=plot_path)

    # Save visual validation preview
    val_preview_path = checkpoint_dir / "val_refinement_preview.png"
    save_validation_preview(model, eval_dataset, device, val_preview_path)

    print("\n" + "=" * 75)
    print(" TRAINING COMPLETE")
    print("=" * 75)
    print(f" Best Epoch:               {best_epoch}/{len(history['train_loss'])}")
    print(f" Best Validation Loss:     {best_val_loss:.4f}")
    print(f" Best Checkpoint Saved:    {best_checkpoint_path}")
    print(f" Loss Curve Plot Saved:    {plot_path}")
    print(f" Visual Preview Saved:     {val_preview_path}")
    print("=" * 75 + "\n", flush=True)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": str(best_checkpoint_path),
        "loss_curve": str(plot_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train ASTRA Neural Optical Flow Refinement U-Net"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="sample_data/patches/patches_manifest.json",
        help="Path to patches_manifest.json (default: sample_data/patches/patches_manifest.json)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs (default: 25)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=0.8,
        help="Weight for L1 loss term (SSIM gets 1 - l1_weight) (default: 0.8)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early stopping patience in epochs (default: 6)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="checkpoints/loss_curve.png",
        help="Path to save loss curve PNG (default: checkpoints/loss_curve.png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    run_training(
        manifest_path=Path(args.manifest),
        epochs=args.epochs,
        lr=args.lr,
        l1_weight=args.l1_weight,
        patience=args.patience,
        batch_size=args.batch_size,
        checkpoint_dir=Path(args.checkpoint_dir),
        plot_path=Path(args.plot_path),
        seed=args.seed
    )


if __name__ == "__main__":
    main()
