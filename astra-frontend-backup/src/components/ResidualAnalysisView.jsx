import React, { useState, useEffect, useRef } from "react";

// Perceptually distinct scientific colormap (Turbo / High-Contrast Thermal)
// Maps normalized error t in [0, 1] to [r, g, b]
function getTurboColor(t) {
  t = Math.max(0, Math.min(1, t));
  // High contrast spectral polynomial (Turbo colormap approximation)
  const r = Math.round(
    255 *
      Math.max(
        0,
        Math.min(
          1,
          0.1357 +
            t *
              (4.5974 -
                t * (42.3277 - t * (130.5887 - t * (150.5667 - t * 58.1375))))
        )
      )
  );
  const g = Math.round(
    255 *
      Math.max(
        0,
        Math.min(
          1,
          0.0914 +
            t *
              (2.1856 +
                t * (4.8052 - t * (14.0195 - t * (4.2109 + t * 2.7747))))
        )
      )
  );
  const b = Math.round(
    255 *
      Math.max(
        0,
        Math.min(
          1,
          0.1067 +
            t *
              (12.5732 -
                t * (85.1927 - t * (244.49 - t * (313.136 - t * 139.605))))
        )
      )
  );
  return [r, g, b];
}

// Predefined cloud-active regions (derived from patch manifest coordinates on GOES-16 full-disk)
const CLOUD_REGIONS = [
  {
    id: "patch_000",
    name: "Convective Storm Cell (Patch 000)",
    desc: "Active cloud cluster with high spatial variance",
    crop: { x: 0.69, y: 0.16, size: 0.18 }, // ~ [3772, 860] on full-disk
  },
  {
    id: "central_core",
    name: "Tropical Storm Core (Central Crop)",
    desc: "Central dynamic cloud advection zone",
    crop: { x: 0.40, y: 0.40, size: 0.22 },
  },
  {
    id: "frontal_boundary",
    name: "Cold Frontal Shear Boundary",
    desc: "Rapid shear divergence and cloud edge deformation",
    crop: { x: 0.25, y: 0.55, size: 0.20 },
  },
  {
    id: "marine_stratocumulus",
    name: "Off-Coast Cloud Band",
    desc: "Marine boundary layer cloud formation",
    crop: { x: 0.55, y: 0.65, size: 0.18 },
  },
];

export default function ResidualAnalysisView({
  activeImage,
  groundTruthImage,
  method,
  onMethodChange = () => {},
  timeFraction,
  onGenerate,
  isGenerating,
}) {
  const [selectedRegionId, setSelectedRegionId] = useState("patch_000");
  const [gain, setGain] = useState(2.0); // Contrast boost multiplier for subtle error visibility
  const [metrics, setMetrics] = useState({
    cropMae: 0,
    cropMaxDiff: 0,
    cropMse: 0,
    cropPsnr: 0,
    totalPixels: 0,
    highErrorFraction: 0,
  });
  const [isProcessing, setIsProcessing] = useState(false);

  const canvasGenRef = useRef(null);
  const canvasGtRef = useRef(null);
  const canvasDiffRef = useRef(null);

  const activeRegion =
    CLOUD_REGIONS.find((r) => r.id === selectedRegionId) || CLOUD_REGIONS[0];

  const getMethodName = (m) => {
    switch (m) {
      case "linear":
        return "Linear Interpolation";
      case "farneback":
        return "Farneback Flow";
      case "raft":
        return "RAFT Optical Flow";
      case "raft_refined":
        return "RAFT + Refinement U-Net";
      default:
        return m ? m.toUpperCase() : "GENERATED";
    }
  };

  // Compute crops and per-pixel residual heatmap whenever images or region change
  useEffect(() => {
    if (!activeImage || !groundTruthImage) return;

    let isCancelled = false;
    setIsProcessing(true);

    const imgGen = new Image();
    const imgGt = new Image();
    imgGen.crossOrigin = "anonymous";
    imgGt.crossOrigin = "anonymous";

    let loadedCount = 0;
    const onBothLoaded = () => {
      loadedCount++;
      if (loadedCount < 2 || isCancelled) return;

      const canvasGen = canvasGenRef.current;
      const canvasGt = canvasGtRef.current;
      const canvasDiff = canvasDiffRef.current;
      if (!canvasGen || !canvasGt || !canvasDiff) return;

      const ctxGen = canvasGen.getContext("2d", { willReadFrequently: true });
      const ctxGt = canvasGt.getContext("2d", { willReadFrequently: true });
      const ctxDiff = canvasDiff.getContext("2d", { willReadFrequently: true });

      const cropSizePx = 384; // Standardized square resolution for high-detail comparison
      canvasGen.width = cropSizePx;
      canvasGen.height = cropSizePx;
      canvasGt.width = cropSizePx;
      canvasGt.height = cropSizePx;
      canvasDiff.width = cropSizePx;
      canvasDiff.height = cropSizePx;

      // Calculate source bounding box from normalized coordinates
      const crop = activeRegion.crop;
      const sxGen = Math.floor(crop.x * imgGen.naturalWidth);
      const syGen = Math.floor(crop.y * imgGen.naturalHeight);
      const sSizeGen = Math.floor(crop.size * imgGen.naturalWidth);

      const sxGt = Math.floor(crop.x * imgGt.naturalWidth);
      const syGt = Math.floor(crop.y * imgGt.naturalHeight);
      const sSizeGt = Math.floor(crop.size * imgGt.naturalWidth);

      // Draw cropped regions
      ctxGen.drawImage(imgGen, sxGen, syGen, sSizeGen, sSizeGen, 0, 0, cropSizePx, cropSizePx);
      ctxGt.drawImage(imgGt, sxGt, syGt, sSizeGt, sSizeGt, 0, 0, cropSizePx, cropSizePx);

      // Read pixel arrays
      const dataGen = ctxGen.getImageData(0, 0, cropSizePx, cropSizePx);
      const dataGt = ctxGt.getImageData(0, 0, cropSizePx, cropSizePx);
      const diffImgData = ctxDiff.createImageData(cropSizePx, cropSizePx);

      const pGen = dataGen.data;
      const pGt = dataGt.data;
      const pDiff = diffImgData.data;

      let sumDiff = 0;
      let sumSqDiff = 0;
      let maxDiff = 0;
      let highErrorCount = 0;
      const totalPixels = cropSizePx * cropSizePx;

      for (let i = 0; i < pGen.length; i += 4) {
        // Luminance / intensity difference
        const valGen = pGen[i]; // IR channel is grayscale (R=G=B)
        const valGt = pGt[i];
        const absDiff = Math.abs(valGen - valGt);

        sumDiff += absDiff;
        sumSqDiff += absDiff * absDiff;
        if (absDiff > maxDiff) maxDiff = absDiff;
        if (absDiff > 25) highErrorCount++;

        // Apply contrast gain and map to Turbo colormap
        const normalized = Math.min(1.0, (absDiff / 100.0) * gain);
        const [r, g, b] = getTurboColor(normalized);

        pDiff[i] = r;
        pDiff[i + 1] = g;
        pDiff[i + 2] = b;
        pDiff[i + 3] = 255;
      }

      ctxDiff.putImageData(diffImgData, 0, 0);

      const mse = sumSqDiff / totalPixels;
      const mae = sumDiff / totalPixels;
      const psnr = mse > 0 ? 10 * Math.log10((255 * 255) / mse) : 99.9;

      setMetrics({
        cropMae: mae,
        cropMaxDiff: maxDiff,
        cropMse: mse,
        cropPsnr: psnr,
        totalPixels,
        highErrorFraction: (highErrorCount / totalPixels) * 100,
      });

      setIsProcessing(false);
    };

    imgGen.onload = onBothLoaded;
    imgGt.onload = onBothLoaded;
    imgGen.src = activeImage;
    imgGt.src = groundTruthImage;

    return () => {
      isCancelled = true;
    };
  }, [activeImage, groundTruthImage, activeRegion, gain]);

  if (!activeImage || !groundTruthImage) {
    return (
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-10 shadow-lg flex flex-col items-center justify-center text-center gap-4 min-h-[450px]">
        <span className="material-symbols-outlined text-primary-container text-5xl animate-pulse">
          difference
        </span>
        <div className="flex flex-col gap-1">
          <h3 className="font-headline-sm text-lg font-bold text-primary">
            Residual Analysis Pipeline Ready
          </h3>
          <p className="text-on-surface-variant font-body-md text-sm max-w-md">
            Synthesize an intermediate frame using the workbench to generate real
            per-pixel residuals against held-out ground truth observation.
          </p>
        </div>
        <button
          onClick={onGenerate}
          disabled={isGenerating}
          className="mt-2 flex items-center gap-2 bg-primary-container text-on-primary-container px-5 py-2.5 rounded-lg font-label-md text-xs font-bold shadow-md hover:brightness-110 cursor-pointer transition-all"
        >
          <span className="material-symbols-outlined text-[18px]">
            {isGenerating ? "sync" : "play_arrow"}
          </span>
          <span>{isGenerating ? "Synthesizing..." : "Generate Frame Now"}</span>
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Top Banner & Region Selector */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-surface-container-high flex items-center justify-center border border-primary-container/30">
            <span className="material-symbols-outlined text-primary-container text-[22px]">
              difference
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-headline-sm text-sm font-bold text-primary tracking-wide uppercase">
                High-Resolution Residual Error Analysis
              </h2>
              <span className="px-2 py-0.5 rounded bg-primary-container/20 text-primary-container font-mono text-[10px] font-semibold border border-primary-container/30">
                {getMethodName(method)}
              </span>
            </div>
            <p className="text-on-surface-variant font-mono text-xs mt-0.5">
              Per-pixel absolute error metric |I_pred(x, y) - I_gt(x, y)| across cloud-active ROI
            </p>
          </div>
        </div>

        {/* Controls: Method Selector, Region Selector & Contrast Boost */}
        <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
          {/* 4 Methods Selector Pills */}
          <div className="flex items-center gap-1 bg-surface-container-lowest border border-outline-variant/40 p-1 rounded-lg">
            <span className="text-outline uppercase text-[10px] px-1.5">METHOD:</span>
            {[
              { id: "linear", label: "Linear" },
              { id: "farneback", label: "Farneback" },
              { id: "raft", label: "RAFT" },
              { id: "raft_refined", label: "RAFT + U-Net" },
            ].map((m) => {
              const isActive = method === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => onMethodChange(m.id)}
                  disabled={isGenerating}
                  className={`px-2 py-0.5 rounded text-[10px] font-bold transition-all cursor-pointer ${
                    isActive
                      ? "bg-primary-container text-on-primary-container shadow-sm"
                      : "text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high"
                  }`}
                >
                  {m.label}
                </button>
              );
            })}
          </div>

          <div className="flex items-center gap-2 bg-surface-container-lowest border border-outline-variant/40 px-3 py-1.5 rounded-lg">
            <span className="text-outline uppercase text-[10px]">Cloud ROI:</span>
            <select
              value={selectedRegionId}
              onChange={(e) => setSelectedRegionId(e.target.value)}
              className="bg-transparent text-primary text-xs font-semibold focus:outline-none cursor-pointer"
            >
              {CLOUD_REGIONS.map((r) => (
                <option key={r.id} value={r.id} className="bg-surface text-on-surface">
                  {r.name}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2 bg-surface-container-lowest border border-outline-variant/40 px-3 py-1.5 rounded-lg">
            <span className="text-outline uppercase text-[10px]">Error Gain:</span>
            <button
              onClick={() => setGain(1.0)}
              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                gain === 1.0
                  ? "bg-primary-container text-on-primary-container"
                  : "text-on-surface-variant hover:text-on-surface"
              }`}
            >
              1×
            </button>
            <button
              onClick={() => setGain(2.0)}
              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                gain === 2.0
                  ? "bg-primary-container text-on-primary-container"
                  : "text-on-surface-variant hover:text-on-surface"
              }`}
            >
              2×
            </button>
            <button
              onClick={() => setGain(3.5)}
              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                gain === 3.5
                  ? "bg-primary-container text-on-primary-container"
                  : "text-on-surface-variant hover:text-on-surface"
              }`}
            >
              3.5×
            </button>
          </div>
        </div>
      </div>

      {/* Main Side-by-Side Trio: Cropped Gen | Cropped Ground Truth | Residual Heatmap */}
      <div className="relative">
        {(isGenerating || isProcessing) && (
          <div className="absolute inset-0 bg-surface/75 backdrop-blur-sm z-20 flex flex-col items-center justify-center gap-3 rounded-xl border border-primary-container/30">
            <span className="material-symbols-outlined text-primary-container text-4xl animate-spin">
              sync
            </span>
            <span className="font-mono text-xs text-primary font-bold tracking-wide">
              {isGenerating ? `Synthesizing ${getMethodName(method)}...` : "Computing Residual Difference..."}
            </span>
          </div>
        )}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Panel 1: Cropped Generated Frame */}
        <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 overflow-hidden shadow-lg flex flex-col">
          <div className="px-4 py-3 bg-surface-container-low border-b border-outline-variant/30 flex items-center justify-between">
            <div className="flex items-center gap-2 font-mono">
              <span className="w-2 h-2 rounded-full bg-primary-container"></span>
              <span className="text-xs font-bold text-primary">CROPPED GENERATED</span>
            </div>
            <span className="font-mono text-[10px] text-outline">
              t = {timeFraction.toFixed(2)}
            </span>
          </div>

          <div className="relative aspect-square bg-[#030611] flex items-center justify-center p-2">
            <canvas
              ref={canvasGenRef}
              className="w-full h-full object-contain rounded border border-outline-variant/20 shadow-inner"
            />
            <div className="absolute top-4 left-4 bg-surface-container-lowest/85 backdrop-blur-md px-2 py-1 rounded border border-outline-variant/30 font-mono text-[10px] text-primary">
              {getMethodName(method)}
            </div>
          </div>

          <div className="p-3 bg-surface-container-low/60 border-t border-outline-variant/30 flex items-center justify-between font-mono text-[11px] text-on-surface-variant">
            <span>ROI: {activeRegion.name}</span>
            <span className="text-secondary font-semibold">384 × 384 px</span>
          </div>
        </div>

        {/* Panel 2: Cropped Ground Truth Frame */}
        <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 overflow-hidden shadow-lg flex flex-col">
          <div className="px-4 py-3 bg-surface-container-low border-b border-outline-variant/30 flex items-center justify-between">
            <div className="flex items-center gap-2 font-mono">
              <span className="w-2 h-2 rounded-full bg-secondary"></span>
              <span className="text-xs font-bold text-secondary">CROPPED GROUND TRUTH</span>
            </div>
            <span className="font-mono text-[10px] text-outline">GOES-16 Verified</span>
          </div>

          <div className="relative aspect-square bg-[#030611] flex items-center justify-center p-2">
            <canvas
              ref={canvasGtRef}
              className="w-full h-full object-contain rounded border border-outline-variant/20 shadow-inner"
            />
            <div className="absolute top-4 left-4 bg-surface-container-lowest/85 backdrop-blur-md px-2 py-1 rounded border border-outline-variant/30 font-mono text-[10px] text-secondary">
              OBSERVED KEYFRAME
            </div>
          </div>

          <div className="p-3 bg-surface-container-low/60 border-t border-outline-variant/30 flex items-center justify-between font-mono text-[11px] text-on-surface-variant">
            <span>Band 13 Clean IR</span>
            <span className="text-secondary font-semibold">Target Truth</span>
          </div>
        </div>

        {/* Panel 3: Perceptually Distinct Residual Heatmap */}
        <div className="bg-surface/80 backdrop-blur-md rounded-xl border-2 border-primary-container/60 shadow-[0_0_20px_rgba(0,240,255,0.12)] overflow-hidden flex flex-col">
          <div className="px-4 py-3 bg-surface-container border-b border-primary-container/30 flex items-center justify-between">
            <div className="flex items-center gap-2 font-mono">
              <span className="w-2 h-2 rounded-full bg-rose-400 animate-pulse"></span>
              <span className="text-xs font-bold text-primary">PER-PIXEL RESIDUAL HEATMAP</span>
            </div>
            <span className="font-mono text-[10px] text-primary-container font-semibold">
              Turbo Colormap
            </span>
          </div>

          <div className="relative aspect-square bg-[#030611] flex items-center justify-center p-2">
            <canvas
              ref={canvasDiffRef}
              className="w-full h-full object-contain rounded border border-primary-container/30 shadow-inner"
            />
            <div className="absolute top-4 left-4 bg-surface-container-lowest/85 backdrop-blur-md px-2 py-1 rounded border border-primary-container/40 font-mono text-[10px] text-primary-container">
              |GEN - GT| ERROR
            </div>
            {isProcessing && (
              <div className="absolute inset-0 flex items-center justify-center bg-surface-dim/80 backdrop-blur-sm">
                <span className="material-symbols-outlined text-primary-container text-3xl animate-spin">
                  sync
                </span>
              </div>
            )}
          </div>

          <div className="p-3 bg-surface-container-low/60 border-t border-primary-container/30 flex items-center justify-between font-mono text-[11px]">
            <span className="text-on-surface-variant">Max Local Diff:</span>
            <span className="text-rose-400 font-bold">
              {metrics.cropMaxDiff} / 255
            </span>
          </div>
        </div>
      </div>
      </div>

      {/* Residual Colormap Legend & Detailed Error Analysis Bar */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-outline-variant/30 pb-3">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-[18px]">palette</span>
            <span className="font-headline-sm text-xs font-bold text-primary uppercase tracking-wide">
              Scientific Error Colormap Legend
            </span>
          </div>
          <span className="font-mono text-[11px] text-outline">
            Turbo High-Contrast Colormap with {gain}× Gain Calibration
          </span>
        </div>

        {/* Continuous Gradient Bar with Ticks */}
        <div className="flex flex-col gap-2">
          <div
            className="h-5 w-full rounded-md shadow-inner border border-outline-variant/40 relative overflow-hidden"
            style={{
              background:
                "linear-gradient(to right, #23171b, #3047a9, #2bb07f, #b3e02b, #f68919, #cf1616)",
            }}
          ></div>

          {/* Ticks & Labels */}
          <div className="flex justify-between font-mono text-[11px] text-on-surface-variant px-1">
            <div className="flex flex-col items-start">
              <span className="font-bold text-emerald-400">0.0 (Zero Error)</span>
              <span className="text-[9px] text-outline">Perfect reconstruction</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="font-bold text-cyan-400">~15.0</span>
              <span className="text-[9px] text-outline">Minor deviation</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="font-bold text-yellow-400">~35.0</span>
              <span className="text-[9px] text-outline">Moderate shear</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="font-bold text-orange-400">~60.0</span>
              <span className="text-[9px] text-outline">High advective gap</span>
            </div>
            <div className="flex flex-col items-end">
              <span className="font-bold text-rose-400">100+ (Severe Error)</span>
              <span className="text-[9px] text-outline">Significant artifact</span>
            </div>
          </div>
        </div>

        {/* Region Crop Quantitative Metrics Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
          <div className="bg-surface-container-lowest p-3 rounded-lg border border-outline-variant/30 flex flex-col">
            <span className="text-outline font-mono text-[10px] uppercase">ROI MAE (Mean Absolute Error)</span>
            <span className="font-mono text-lg font-bold text-primary mt-1">
              {metrics.cropMae.toFixed(2)}
            </span>
            <span className="text-[9px] font-mono text-on-surface-variant/70 mt-0.5">Average pixel intensity offset</span>
          </div>

          <div className="bg-surface-container-lowest p-3 rounded-lg border border-outline-variant/30 flex flex-col">
            <span className="text-outline font-mono text-[10px] uppercase">ROI PSNR</span>
            <span className="font-mono text-lg font-bold text-primary mt-1">
              {metrics.cropPsnr.toFixed(2)} <span className="text-xs font-normal text-on-surface-variant">dB</span>
            </span>
            <span className="text-[9px] font-mono text-on-surface-variant/70 mt-0.5">Local signal-to-noise</span>
          </div>

          <div className="bg-surface-container-lowest p-3 rounded-lg border border-outline-variant/30 flex flex-col">
            <span className="text-outline font-mono text-[10px] uppercase">Peak Pixel Difference</span>
            <span className="font-mono text-lg font-bold text-rose-400 mt-1">
              {metrics.cropMaxDiff} <span className="text-xs font-normal text-on-surface-variant">/ 255</span>
            </span>
            <span className="text-[9px] font-mono text-on-surface-variant/70 mt-0.5">Worst-case error point</span>
          </div>

          <div className="bg-surface-container-lowest p-3 rounded-lg border border-outline-variant/30 flex flex-col">
            <span className="text-outline font-mono text-[10px] uppercase">Discrepancy Zone (&gt;25 L1)</span>
            <span className="font-mono text-lg font-bold text-secondary mt-1">
              {metrics.highErrorFraction.toFixed(1)}%
            </span>
            <span className="text-[9px] font-mono text-on-surface-variant/70 mt-0.5">Percentage of active ROI</span>
          </div>
        </div>
      </div>
    </div>
  );
}
