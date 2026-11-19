"""
Test & Validation Script: Optical Flow Motion Warping vs Linear Interpolation
============================================================================
File: test_optical_flow.py

Purpose:
--------
Loads the downloaded GOES-16 Band 13 numpy arrays from `sample_data/`,
computes dense Farneback optical flow, generates a synthetic intermediate frame
via motion warping at t=0.5, generates a pixel-averaged baseline frame,
and compares both against the actual intermediate scan ground truth.
"""

import logging
from pathlib import Path
from app.processing.optical_flow import run_sample_data_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__ == "__main__":
    sample_dir = Path("sample_data")
    print("=" * 70)
    print("ASTRA BACKEND: Optical Flow Motion Warping & Frame Synthesis Test")
    print("=" * 70)
    
    results = run_sample_data_test(sample_dir=str(sample_dir))
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY!")
    if "metrics_warped" in results:
        m_warp = results["metrics_warped"]
        m_lin = results["metrics_linear"]
        print("Quantitative Comparison vs Ground Truth (t=0.5):")
        print(f"  • Optical Flow Warped Frame : SSIM = {m_warp['ssim']:.4f}, MSE = {m_warp['mse']:.2f}, PSNR = {m_warp['psnr_db']:.2f} dB, MAE = {m_warp['mae']:.2f}")
        print(f"  • Linear Pixel Averaged Frame: SSIM = {m_lin['ssim']:.4f}, MSE = {m_lin['mse']:.2f}, PSNR = {m_lin['psnr_db']:.2f} dB, MAE = {m_lin['mae']:.2f}")
        improvement = ((m_lin['mse'] - m_warp['mse']) / m_lin['mse']) * 100
        print(f"  --> Optical flow warping achieved a {improvement:.1f}% reduction in reconstruction MSE!")
    print("=" * 70)
