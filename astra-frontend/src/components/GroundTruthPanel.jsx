import React, { useState, useRef, useEffect, useCallback } from "react";

export default function GroundTruthPanel({
  activeImage,
  groundTruthImage,
  confidenceImage,
  method,
  metrics,
}) {
  const [splitPercent, setSplitPercent] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef(null);

  const updateSplit = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    setSplitPercent(percent);
  }, []);

  const handleMouseDown = (e) => {
    setIsDragging(true);
    updateSplit(e.clientX);
  };

  const handleTouchStart = (e) => {
    setIsDragging(true);
    if (e.touches && e.touches[0]) {
      updateSplit(e.touches[0].clientX);
    }
  };

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging) return;
      updateSplit(e.clientX);
    };

    const handleMouseUp = () => {
      if (isDragging) setIsDragging(false);
    };

    const handleTouchMove = (e) => {
      if (!isDragging) return;
      if (e.touches && e.touches[0]) {
        updateSplit(e.touches[0].clientX);
      }
    };

    const handleTouchEnd = () => {
      if (isDragging) setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      window.addEventListener("touchmove", handleTouchMove, { passive: false });
      window.addEventListener("touchend", handleTouchEnd);
    }

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleTouchEnd);
    };
  }, [isDragging, updateSplit]);

  const currentMetrics =
    method === "linear" ? metrics?.linear : metrics?.warped;
  const ssimText =
    currentMetrics?.ssim != null ? currentMetrics.ssim.toFixed(4) : "--";
  const maeText =
    currentMetrics?.mae != null ? currentMetrics.mae.toFixed(2) : "--";

  const getMethodTitle = (m) => {
    switch (m) {
      case "farneback":
        return "Farneback Flow Warping";
      case "raft":
        return "Deep RAFT Flow Warping";
      case "raft_refined":
        return "RAFT + Trained Refinement U-Net";
      case "linear":
        return "Linear Pixel Averaging";
      default:
        return "Optical Flow Warping";
    }
  };

  return (
    <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-4">
      {/* Section Header */}
      <div className="flex items-center justify-between pb-3 border-b border-outline-variant/30">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-secondary text-[18px]">
            split_scene
          </span>
          <h2 className="font-headline-sm text-sm font-semibold text-primary tracking-wide uppercase">
            Ground Truth Direct Evaluation &amp; Residual Inspector
          </h2>
        </div>
        <div className="font-mono text-xs text-on-surface-variant flex items-center gap-3">
          <span>Confidence Scale: Low (Purple) to High (Green/Yellow)</span>
          <div className="h-2 w-28 rounded-full bg-gradient-to-r from-purple-900 via-emerald-600 to-yellow-400 border border-outline-variant/30" />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Comparison View A: Generated vs Actual Interactive Split-Screen */}
        <div className="bg-surface-container-lowest rounded-lg border border-outline-variant/40 p-3 flex flex-col gap-2">
          <div className="flex items-center justify-between font-mono text-xs">
            <span className="text-primary font-semibold">Generated (Reconstruction)</span>
            <span className="text-outline">vs</span>
            <span className="text-secondary font-semibold">Ground Truth Observation</span>
          </div>

          <div
            id="split-container"
            ref={containerRef}
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
            className="relative aspect-[16/9] bg-surface-dim rounded overflow-hidden split-slider-container cursor-ew-resize select-none"
          >
            {/* Background Image: Ground Truth Reference */}
            {groundTruthImage ? (
              <img
                id="img-split-gt"
                src={groundTruthImage}
                className="absolute inset-0 w-full h-full object-cover pointer-events-none"
                alt="Ground Truth Scan"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-xs text-outline font-mono">
                No Ground Truth Frame Available
              </div>
            )}

            {/* Foreground Clipped Image: Generated Reconstruction */}
            {activeImage && (
              <div
                id="split-clipper"
                className="absolute inset-0 overflow-hidden pointer-events-none"
                style={{ width: `${splitPercent}%` }}
              >
                <img
                  id="img-split-gen"
                  src={activeImage}
                  className="absolute inset-0 h-full object-cover max-w-none pointer-events-none"
                  style={{
                    width: containerRef.current
                      ? `${containerRef.current.clientWidth}px`
                      : "100%",
                  }}
                  alt="Generated Frame"
                />
              </div>
            )}

            {/* Interactive Split Line & Handle */}
            <div
              id="split-handle"
              className="absolute inset-y-0 w-0.5 bg-primary-container shadow-[0_0_8px_rgba(0,240,255,0.9)] pointer-events-none"
              style={{ left: `${splitPercent}%` }}
            >
              <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-primary-container text-on-primary-container flex items-center justify-center font-mono text-[10px] font-bold shadow-lg pointer-events-auto">
                ↔
              </div>
            </div>

            <span className="absolute bottom-2 left-2 px-2 py-0.5 bg-surface-container-lowest/80 font-mono text-[10px] text-primary rounded z-10 pointer-events-none">
              GENERATED
            </span>
            <span className="absolute bottom-2 right-2 px-2 py-0.5 bg-surface-container-lowest/80 font-mono text-[10px] text-secondary rounded z-10 pointer-events-none">
              GROUND TRUTH
            </span>
          </div>

          <div className="flex justify-between font-mono text-[11px] text-on-surface-variant pt-1">
            <span id="label-split-method">Method: {getMethodTitle(method)}</span>
            <span id="label-split-ssim" className="text-primary font-semibold">
              SSIM Index: {ssimText}
            </span>
          </div>
        </div>

        {/* Comparison View B: Error Residual / Confidence Field */}
        <div className="bg-surface-container-lowest rounded-lg border border-outline-variant/40 p-3 flex flex-col gap-2">
          <div className="flex items-center justify-between font-mono text-xs">
            <span className="text-primary font-semibold">
              Motion Confidence &amp; Consistency Map
            </span>
            <span id="label-res-method" className="text-outline text-[11px]">
              Viridis Representation
            </span>
          </div>

          <div className="relative aspect-[16/9] bg-[#02050e] rounded overflow-hidden flex items-center justify-center border border-outline-variant/30">
            {confidenceImage ? (
              <img
                id="img-confidence"
                src={confidenceImage}
                className="w-full h-full object-cover"
                alt="Optical Flow Confidence Map"
              />
            ) : (
              <div className="text-xs text-outline font-mono">
                Awaiting confidence / residual map...
              </div>
            )}
            <div className="absolute top-2 left-2 px-2 py-0.5 bg-surface-container-lowest/80 font-mono text-[10px] text-on-surface-variant rounded border border-outline-variant/30">
              BIDIRECTIONAL CONSISTENCY
            </div>
            <div className="absolute bottom-2 right-2 px-2 py-0.5 bg-surface-container-lowest/80 font-mono text-[10px] text-primary-container rounded border border-outline-variant/30">
              EXPONENTIAL RESIDUAL WEIGHTING
            </div>
          </div>

          <div className="flex justify-between font-mono text-[11px] text-on-surface-variant pt-1">
            <span>High Confidence: Smooth Advection</span>
            <span id="label-res-mae">MAE: {maeText}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
