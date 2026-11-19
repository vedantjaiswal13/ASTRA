"""
Image & Data Processing Package
===============================
File: app/processing/__init__.py

Purpose:
--------
Houses modules dedicated to scientific matrix transformations, computer vision pipelines,
optical flow motion estimation, frame synthesis, and image filtering routines.

Exports:
--------
- compute_farneback_flow: Dense optical flow field calculation.
- warp_frame_forward: Motion-compensated forward frame warping using cv2.remap.
- interpolate_linear: Baseline pixel-averaging cross-dissolve frame blending.
- flow_to_hsv_color: Middlebury color wheel representation of optical flow vector fields.
- generate_motion_interpolations: Full pipeline for synthesizing intermediate frames.
- ImageProcessor: General utility pipeline for image filtering and normalization.
"""

from app.processing.optical_flow import (
    compute_farneback_flow,
    compute_raft_flow,
    compute_flow,
    warp_frame_forward,
    interpolate_linear,
    flow_to_hsv_color,
    generate_motion_interpolations,
    run_sample_data_test,
)
from app.processing.image_processor import ImageProcessor
from app.processing.refinement_model import (
    CombinedL1SSIMLoss,
    PatchRefinementUNet,
    SSIMLoss,
)

__all__ = [
    "compute_farneback_flow",
    "compute_raft_flow",
    "compute_flow",
    "warp_frame_forward",
    "interpolate_linear",
    "flow_to_hsv_color",
    "generate_motion_interpolations",
    "run_sample_data_test",
    "ImageProcessor",
    "PatchRefinementUNet",
    "CombinedL1SSIMLoss",
    "SSIMLoss",
]
