"""
ASTRA Satellite Cloud Patch Extraction and Preprocessing Pipeline
==================================================================
File: extract_patches.py

Purpose:
--------
Extracts 256x256 spatiotemporal patches from cloud-active regions across
GOES-16 Channel 13 triplets (t0, t_mid, t1):
1. Discards low-variance / flat ocean and off-Earth space regions.
2. Extracts 5-10 high-variance cloud patches per triplet.
3. Groups and saves patches as numpy arrays by triplet.
4. Splits the dataset 80/20 train/validation BY TRIPLET to prevent
   temporal/spatial data leakage.
5. Emits a comprehensive patches_manifest.json and prints dataset metrics.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("patch_extractor")


def find_triplets(triplets_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Discovers all valid triplets inside `triplets_dir`.
    Checks manifest.json first if present, then scans subfolders for 3 .npy files.

    Returns:
        Dict[str, Dict[str, Path]]: Map of triplet_id -> {"t0": Path, "t_mid": Path, "t1": Path}
    """
    triplets: Dict[str, Dict[str, Path]] = {}

    # Check manifest.json
    manifest_path = triplets_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            for item in manifest_data.get("triplets", []):
                tid = item["triplet_id"]
                files = item.get("files", {})
                t0_p = Path(files.get("t0", {}).get("npy_path", ""))
                tm_p = Path(files.get("t_mid", {}).get("npy_path", ""))
                t1_p = Path(files.get("t1", {}).get("npy_path", ""))
                if t0_p.exists() and tm_p.exists() and t1_p.exists():
                    triplets[tid] = {"t0": t0_p, "t_mid": tm_p, "t1": t1_p}
        except Exception as e:
            logger.warning("Error reading manifest.json: %s. Falling back to folder scan.", e)

    # Scan directories in triplets_dir
    for sub in triplets_dir.iterdir():
        if sub.is_dir() and sub.name not in ("patches", "raw_netcdf", "comparison"):
            if sub.name in triplets:
                continue
            npys = sorted([f for f in sub.glob("*.npy") if not f.name.startswith("patch_")])
            if len(npys) >= 3:
                t0 = next((f for f in npys if "00_" in f.name or "start" in f.name), npys[0])
                tm = next((f for f in npys if "01_" in f.name or "halfway" in f.name or "mid" in f.name), npys[1])
                t1 = next((f for f in npys if "02_" in f.name or "end" in f.name), npys[2])
                triplets[sub.name] = {"t0": t0, "t_mid": tm, "t1": t1}

    # Also check if triplets_dir itself is a single triplet folder (e.g. sample_data/)
    if not triplets:
        npys = sorted([f for f in triplets_dir.glob("*.npy") if not f.name.startswith("patch_")])
        if len(npys) >= 3:
            t0 = next((f for f in npys if "00_" in f.name or "start" in f.name), npys[0])
            tm = next((f for f in npys if "01_" in f.name or "halfway" in f.name or "mid" in f.name), npys[1])
            t1 = next((f for f in npys if "02_" in f.name or "end" in f.name), npys[2])
            triplets[triplets_dir.name] = {"t0": t0, "t_mid": tm, "t1": t1}

    return triplets


def extract_candidate_patches(
    frame_t0: np.ndarray,
    frame_mid: np.ndarray,
    frame_t1: np.ndarray,
    patch_size: int = 256,
    target_count: int = 8,
    variance_threshold: float = 200.0,
    max_zero_fraction: float = 0.05,
    min_distance: int = 128,
    max_attempts: int = 500,
    rng: Optional[np.random.RandomState] = None
) -> List[Dict[str, Any]]:
    """
    Samples candidate patches, checks pixel variance, and returns high-variance cloud patches.

    Args:
        frame_t0 (np.ndarray): Full-disk scan at t0.
        frame_mid (np.ndarray): Full-disk scan at t_mid.
        frame_t1 (np.ndarray): Full-disk scan at t1.
        patch_size (int): Dimensions of square patch (default 256).
        target_count (int): Desired number of patches (roughly 5-10, default 8).
        variance_threshold (float): Minimum variance to qualify as cloud-active.
        max_zero_fraction (float): Max fraction of space/off-Earth pixels allowed.
        min_distance (int): Minimum Euclidean distance between accepted patch centers.
        max_attempts (int): Maximum random proposals to test.
        rng (RandomState): Reproducible random number generator.

    Returns:
        List[Dict[str, Any]]: Accepted patches with coordinates, variances, and arrays.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    H, W = frame_t0.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Frame dimensions ({H}, {W}) smaller than patch size {patch_size}")

    accepted_patches: List[Dict[str, Any]] = []
    accepted_centers: List[Tuple[float, float]] = []

    # Two-pass adaptive strategy: first with strict variance, second with relaxed if needed
    thresholds = [variance_threshold, variance_threshold * 0.6]

    for current_thresh in thresholds:
        if len(accepted_patches) >= target_count:
            break

        for _ in range(max_attempts):
            if len(accepted_patches) >= target_count:
                break

            y = rng.randint(0, H - patch_size)
            x = rng.randint(0, W - patch_size)
            cy, cx = y + patch_size / 2.0, x + patch_size / 2.0

            # Spatial separation check
            too_close = False
            for acy, acx in accepted_centers:
                dist = np.hypot(cy - acy, cx - acx)
                if dist < min_distance:
                    too_close = True
                    break
            if too_close:
                continue

            p0 = frame_t0[y:y + patch_size, x:x + patch_size]
            pm = frame_mid[y:y + patch_size, x:x + patch_size]
            p1 = frame_t1[y:y + patch_size, x:x + patch_size]

            # 1. Filter out space / off-disk boundary pixels (value 0)
            if (
                np.mean(p0 == 0) > max_zero_fraction
                or np.mean(pm == 0) > max_zero_fraction
                or np.mean(p1 == 0) > max_zero_fraction
            ):
                continue

            # 2. Compute pixel variance across all three frames
            var_0 = float(np.var(p0))
            var_m = float(np.var(pm))
            var_1 = float(np.var(p1))
            min_var = min(var_0, var_m, var_1)

            # Discard low-variance patches (uniform ocean / clear calm regions)
            if min_var < current_thresh:
                continue

            accepted_centers.append((cy, cx))
            accepted_patches.append({
                "bbox": [int(y), int(x), int(y + patch_size), int(x + patch_size)],
                "center": [float(cy), float(cx)],
                "variance": {
                    "t0": round(var_0, 2),
                    "t_mid": round(var_m, 2),
                    "t1": round(var_1, 2),
                    "min_var": round(min_var, 2),
                },
                "p0": p0,
                "pm": pm,
                "p1": p1,
            })

    return accepted_patches


def split_triplets(
    triplet_ids: List[str],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Splits triplet IDs into train and validation sets (e.g. 80/20) BY TRIPLET.
    This guarantees that all patches from any given triplet are in the same split,
    strictly preventing validation leakage.
    """
    sorted_ids = sorted(triplet_ids)
    rng = np.random.RandomState(seed)
    shuffled = list(sorted_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    if n_total <= 1:
        # If only 1 triplet is available, assign to train
        return shuffled, []

    n_train = int(round(train_ratio * n_total))
    n_train = max(1, min(n_train, n_total - 1))  # Ensure both train and val have >= 1 if n_total >= 2

    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train:])
    return train_ids, val_ids


def run_patch_preprocessing(
    triplets_dir: Path,
    output_dir: Path,
    patches_per_triplet: int = 8,
    patch_size: int = 256,
    variance_threshold: float = 200.0,
    train_ratio: float = 0.8,
    save_png_preview: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the complete preprocessing and patch extraction pipeline:
    1. Locates all triplets.
    2. Performs 80/20 train/validation split by triplet.
    3. Extracts 5-10 cloud-active 256x256 patches per triplet.
    4. Saves patches as numpy arrays (.npy) grouped by triplet.
    5. Saves manifest.json and reports metrics.
    """
    triplets = find_triplets(triplets_dir)
    if not triplets:
        raise FileNotFoundError(f"No valid triplets found in {triplets_dir}")

    triplet_ids = sorted(list(triplets.keys()))
    train_ids, val_ids = split_triplets(triplet_ids, train_ratio=train_ratio, seed=seed)

    output_dir = output_dir.resolve()
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(" ASTRA Satellite Cloud Patch Preprocessing Pipeline")
    print(f" Source Triplets Directory: {triplets_dir}")
    print(f" Target Output Directory:   {output_dir}")
    print(f" Found Triplets:            {len(triplets)}")
    print(f" Train Triplets (80%):      {len(train_ids)} {train_ids}")
    print(f" Validation Triplets (20%): {len(val_ids)} {val_ids}")
    print(f" Patches Per Triplet:       {patches_per_triplet} (256x256)")
    print(f" Variance Threshold:        {variance_threshold}")
    print("=" * 75 + "\n", flush=True)

    split_assignment: Dict[str, str] = {}
    for tid in train_ids:
        split_assignment[tid] = "train"
    for tid in val_ids:
        split_assignment[tid] = "val"

    patch_records: List[Dict[str, Any]] = []
    total_patches_extracted = 0
    train_patches_count = 0
    val_patches_count = 0

    rng = np.random.RandomState(seed)

    for idx, (tid, paths) in enumerate(triplets.items(), start=1):
        split = split_assignment[tid]
        logger.info(
            "[%d/%d] Processing triplet '%s' -> Split: %s",
            idx, len(triplets), tid, split.upper()
        )

        # Load frames
        t0 = np.load(paths["t0"])
        tm = np.load(paths["t_mid"])
        t1 = np.load(paths["t1"])

        # Extract cloud-active patches
        patches = extract_candidate_patches(
            frame_t0=t0,
            frame_mid=tm,
            frame_t1=t1,
            patch_size=patch_size,
            target_count=patches_per_triplet,
            variance_threshold=variance_threshold,
            rng=rng
        )

        # Triplet-specific destination directory
        triplet_out_dir = output_dir / split / tid
        triplet_out_dir.mkdir(parents=True, exist_ok=True)

        for p_idx, patch in enumerate(patches):
            patch_name = f"patch_{p_idx:03d}"

            # 1. Save individual numpy arrays (.npy)
            p0_path = triplet_out_dir / f"{patch_name}_t0.npy"
            pm_path = triplet_out_dir / f"{patch_name}_t_mid.npy"
            p1_path = triplet_out_dir / f"{patch_name}_t1.npy"

            np.save(p0_path, patch["p0"])
            np.save(pm_path, patch["pm"])
            np.save(p1_path, patch["p1"])

            # 2. Save 3-frame stack (3, 256, 256) for easy PyTorch Dataset ingestion
            stack_path = triplet_out_dir / f"{patch_name}_stack.npy"
            stack_arr = np.stack([patch["p0"], patch["pm"], patch["p1"]], axis=0)
            np.save(stack_path, stack_arr)

            # 3. Optional visual preview PNG (side-by-side [t0 | t_mid | t1])
            preview_path = None
            if save_png_preview:
                preview_img = np.hstack([patch["p0"], patch["pm"], patch["p1"]])
                preview_path = triplet_out_dir / f"{patch_name}_preview.png"
                cv2.imwrite(str(preview_path), preview_img)

            # Record patch metadata
            record = {
                "patch_id": f"{tid}_{patch_name}",
                "triplet_id": tid,
                "split": split,
                "bbox": patch["bbox"],
                "center": patch["center"],
                "variance": patch["variance"],
                "files": {
                    "t0": str(p0_path.relative_to(output_dir)),
                    "t_mid": str(pm_path.relative_to(output_dir)),
                    "t1": str(p1_path.relative_to(output_dir)),
                    "stack": str(stack_path.relative_to(output_dir)),
                    "preview": str(preview_path.relative_to(output_dir)) if preview_path else None,
                }
            }
            patch_records.append(record)

        num_extracted = len(patches)
        total_patches_extracted += num_extracted
        if split == "train":
            train_patches_count += num_extracted
        else:
            val_patches_count += num_extracted

        logger.info(
            "Extracted %d patches for triplet '%s' (Running total: %d patches)",
            num_extracted, tid, total_patches_extracted
        )

    # Compile manifest
    manifest_data = {
        "title": "ASTRA Cloud-Active Patch Dataset Manifest",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "patch_size": patch_size,
            "patches_per_triplet_target": patches_per_triplet,
            "variance_threshold": variance_threshold,
            "train_ratio": train_ratio,
            "seed": seed,
        },
        "summary": {
            "total_triplets": len(triplets),
            "train_triplets_count": len(train_ids),
            "val_triplets_count": len(val_ids),
            "total_patches": total_patches_extracted,
            "train_patches": train_patches_count,
            "val_patches": val_patches_count,
        },
        "triplets": {
            "train": train_ids,
            "val": val_ids,
        },
        "patches": patch_records,
    }

    manifest_file = output_dir / "patches_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Print required final counts
    print("\n" + "=" * 75)
    print(" ASTRA PATCH EXTRACTION SUMMARY")
    print("=" * 75)
    print(f" Total Triplets Processed: {len(triplets)}")
    print(f"   - Train Triplets:       {len(train_ids)} ({len(train_ids)/len(triplets)*100:.1f}%)")
    print(f"   - Validation Triplets:  {len(val_ids)} ({len(val_ids)/len(triplets)*100:.1f}%)")
    print("-" * 75)
    print(f" Total Patches Extracted:  {total_patches_extracted}")
    print(f"   - Train Patches:        {train_patches_count}")
    print(f"   - Validation Patches:   {val_patches_count}")
    print("-" * 75)
    print(f" Output Directory:         {output_dir}")
    print(f" Manifest Saved:           {manifest_file}")
    print("=" * 75 + "\n", flush=True)

    return manifest_data


def main():
    parser = argparse.ArgumentParser(
        description="ASTRA Satellite Cloud Patch Extraction and Preprocessing"
    )
    parser.add_argument(
        "--triplets-dir",
        type=str,
        default="sample_data/triplets",
        help="Path to folder containing downloaded triplets (default: sample_data/triplets)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sample_data/patches",
        help="Path to output patches directory (default: sample_data/patches)"
    )
    parser.add_argument(
        "--patches-per-triplet",
        type=int,
        default=8,
        help="Number of 256x256 patches to extract per triplet (default: 8, range 5-10)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Square patch size in pixels (default: 256)"
    )
    parser.add_argument(
        "--variance-thresh",
        type=float,
        default=200.0,
        help="Variance threshold for cloud activity filter (default: 200.0)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio by triplet (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Disable saving PNG visual preview images"
    )

    args = parser.parse_args()

    run_patch_preprocessing(
        triplets_dir=Path(args.triplets_dir),
        output_dir=Path(args.output_dir),
        patches_per_triplet=args.patches_per_triplet,
        patch_size=args.patch_size,
        variance_threshold=args.variance_thresh,
        train_ratio=args.train_ratio,
        save_png_preview=not args.no_png,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
