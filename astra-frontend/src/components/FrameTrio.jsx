import React from "react";

export default function FrameTrio({
  sampleData,
  generatedData,
  activeImage,
  timeFraction,
  onTimeFractionChange,
  onTimeFractionCommit,
  method,
  isLoadingSample,
  sampleError,
  isGenerating,
  generationError,
  isPlaying,
  onTogglePlay,
  onStep,
  onRefresh,
}) {
  const formatTimestamp = (isoStr) => {
    if (!isoStr) return "--";
    try {
      const d = new Date(isoStr);
      return isNaN(d.getTime()) ? isoStr : d.toUTCString().replace("GMT", "UTC");
    } catch {
      return isoStr;
    }
  };

  const getMethodLabel = (m) => {
    switch (m) {
      case "farneback":
        return "Farneback Flow";
      case "raft":
        return "RAFT Flow";
      case "raft_refined":
        return "RAFT + Refinement";
      case "linear":
        return "Linear Interpolation";
      default:
        return m ? m.toUpperCase() : "INTERPOLATED";
    }
  };

  const psnrValue =
    generatedData?.metrics?.warped?.psnr_db != null
      ? (method === "linear"
          ? generatedData?.metrics?.linear?.psnr_db
          : generatedData?.metrics?.warped?.psnr_db
        )?.toFixed(2)
      : null;

  return (
    <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-4">
      {/* Section Header */}
      <div className="flex items-center justify-between pb-3 border-b border-outline-variant/30">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary-container text-[18px]">
            view_column
          </span>
          <h2 className="font-headline-sm text-sm font-semibold text-primary tracking-wide uppercase">
            Temporal Keyframe Interpolation Trio
          </h2>
          <span
            id="scan-interval-badge"
            className="font-label-sm text-[11px] text-outline font-mono"
          >
            GOES-16 ABI Scans
          </span>
        </div>

        {/* Viewport Tools */}
        <div className="flex items-center gap-2 font-mono text-xs">
          <button
            id="btn-refresh"
            onClick={onRefresh}
            className="flex items-center gap-1 bg-surface-container px-2.5 py-1 rounded border border-outline-variant/30 text-on-surface-variant hover:text-primary transition-colors text-[11px] cursor-pointer"
            title="Reload Data"
          >
            <span className="material-symbols-outlined text-[14px]">refresh</span>
            <span>Reload</span>
          </button>
        </div>
      </div>

      {/* 3-Column Viewport Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Viewport 1: Observed T0 */}
        <div className="flex flex-col bg-surface-container-lowest rounded-lg border border-outline-variant/40 overflow-hidden group">
          <div className="flex items-center justify-between px-3 py-2 bg-surface-container-low border-b border-outline-variant/30">
            <div className="flex items-center gap-2 font-mono">
              <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-secondary text-[10px] font-semibold">
                T₀ OBSERVED
              </span>
              <span id="label-t0-time" className="text-xs text-on-surface font-semibold">
                {sampleData ? formatTimestamp(sampleData.frame_t0?.timestamp) : sampleError ? "Error" : "Loading..."}
              </span>
            </div>
            <span className="text-[10px] font-mono text-outline">Start Scan</span>
          </div>
          <div className="relative aspect-square bg-surface-dim overflow-hidden flex items-center justify-center">
            {sampleError ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-4 bg-surface-dim/95 text-center z-10">
                <span className="material-symbols-outlined text-error text-3xl mb-1">cloud_off</span>
                <span className="text-xs text-error font-mono font-medium">Failed to load scan</span>
                <span className="text-[10px] text-on-surface-variant/70 font-mono mt-1 max-w-[200px] truncate">{sampleError}</span>
              </div>
            ) : isLoadingSample ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface-dim/80 z-10 gap-2">
                <span className="material-symbols-outlined text-primary-container text-3xl animate-spin">
                  sync
                </span>
                <span className="font-mono text-[10px] text-primary-container tracking-wider uppercase">Loading T₀ Scan...</span>
              </div>
            ) : sampleData?.frame_t0?.image ? (
              <img
                id="img-t0"
                src={sampleData.frame_t0.image}
                className="w-full h-full object-cover"
                alt="GOES-16 T0 Observed Scan"
              />
            ) : (
              <div className="text-xs text-outline font-mono">No T₀ image</div>
            )}
            <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-2">
              <div className="flex justify-between font-mono text-[9px] text-outline/80">
                <span>GOES-16 Band 13</span>
                <span>T₀ KEYFRAME</span>
              </div>
            </div>
          </div>
          <div className="p-2.5 bg-surface-container-low/60 flex items-center justify-between font-mono text-[11px] text-on-surface-variant">
            <span>Product: ABI-L2-CMIPF</span>
            <span id="badge-t0-dim" className="text-secondary font-medium">
              {sampleData?.raw_dimensions
                ? `${sampleData.raw_dimensions.width} × ${sampleData.raw_dimensions.height} px`
                : "--"}
            </span>
          </div>
        </div>

        {/* Viewport 2: Generated Intermediate */}
        <div className="flex flex-col bg-surface-container-lowest rounded-lg border-2 border-primary-container/60 shadow-[0_0_16px_rgba(0,240,255,0.1)] overflow-hidden group">
          <div className="flex items-center justify-between px-3 py-2 bg-surface-container border-b border-primary-container/30">
            <div className="flex items-center gap-2 font-mono">
              <span className="px-1.5 py-0.5 rounded bg-primary-container text-on-primary-container text-[10px] font-bold tracking-wide">
                GENERATED
              </span>
              <span id="label-gen-time" className="text-xs text-primary font-semibold">
                t = {timeFraction.toFixed(2)}
              </span>
            </div>
            <span id="label-gen-fraction" className="text-[10px] font-mono text-primary-container font-semibold">
              Active Synth
            </span>
          </div>
          <div className="relative aspect-square bg-surface-dim overflow-hidden flex items-center justify-center">
            {generationError ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-4 bg-surface-dim/95 text-center z-10">
                <span className="material-symbols-outlined text-error text-3xl mb-1">error</span>
                <span className="text-xs text-error font-mono font-medium">Generation Failed</span>
                <span className="text-[10px] text-on-surface-variant/70 font-mono mt-1 max-w-[220px] line-clamp-2">{generationError}</span>
              </div>
            ) : isGenerating ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface-dim/90 z-10 gap-2">
                <span className="material-symbols-outlined text-primary-container text-3xl animate-spin">
                  sync
                </span>
                <span className="font-mono text-[10px] text-primary-container tracking-wider uppercase animate-pulse">
                  Synthesizing {getMethodLabel(method)}...
                </span>
              </div>
            ) : activeImage ? (
              <img
                id="img-gen"
                src={activeImage}
                className="w-full h-full object-cover"
                alt="Synthesized Intermediate Frame"
              />
            ) : (
              <div className="flex flex-col items-center justify-center p-4 text-center">
                <span className="material-symbols-outlined text-outline text-2xl mb-1">auto_awesome</span>
                <span className="text-xs text-on-surface-variant font-mono">
                  Click 'Generate Frame' to synthesize
                </span>
              </div>
            )}

            {/* Crosshair Center Overlay */}
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
              <div className="w-8 h-8 border border-primary-container/40 rounded-full flex items-center justify-center">
                <div className="w-1 h-1 bg-primary-container rounded-full"></div>
              </div>
            </div>
            <div className="absolute inset-0 pointer-events-none border border-primary-container/20 flex flex-col justify-between p-2">
              <div className="flex justify-between font-mono text-[9px] text-primary-container/80">
                <span>METHOD: {getMethodLabel(method).toUpperCase()}</span>
                <span>INTERPOLATED</span>
              </div>
            </div>
          </div>
          <div className="p-2.5 bg-surface-container-low/60 flex items-center justify-between font-mono text-[11px]">
            <span id="label-active-method-info" className="text-on-surface-variant truncate max-w-[200px]">
              Method: {getMethodLabel(method)}
            </span>
            <span id="badge-gen-psnr" className="text-primary-container font-semibold whitespace-nowrap">
              PSNR: {psnrValue ? `${psnrValue} dB` : "-- dB"}
            </span>
          </div>
        </div>

        {/* Viewport 3: Observed T1 */}
        <div className="flex flex-col bg-surface-container-lowest rounded-lg border border-outline-variant/40 overflow-hidden group">
          <div className="flex items-center justify-between px-3 py-2 bg-surface-container-low border-b border-outline-variant/30">
            <div className="flex items-center gap-2 font-mono">
              <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-secondary text-[10px] font-semibold">
                T₁ OBSERVED
              </span>
              <span id="label-t1-time" className="text-xs text-on-surface font-semibold">
                {sampleData ? formatTimestamp(sampleData.frame_t1?.timestamp) : sampleError ? "Error" : "Loading..."}
              </span>
            </div>
            <span className="text-[10px] font-mono text-outline">Target Scan</span>
          </div>
          <div className="relative aspect-square bg-surface-dim overflow-hidden flex items-center justify-center">
            {sampleError ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-4 bg-surface-dim/95 text-center z-10">
                <span className="material-symbols-outlined text-error text-3xl mb-1">cloud_off</span>
                <span className="text-xs text-error font-mono font-medium">Failed to load scan</span>
                <span className="text-[10px] text-on-surface-variant/70 font-mono mt-1 max-w-[200px] truncate">{sampleError}</span>
              </div>
            ) : isLoadingSample ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface-dim/80 z-10 gap-2">
                <span className="material-symbols-outlined text-primary-container text-3xl animate-spin">
                  sync
                </span>
                <span className="font-mono text-[10px] text-primary-container tracking-wider uppercase">Loading T₁ Scan...</span>
              </div>
            ) : sampleData?.frame_t1?.image ? (
              <img
                id="img-t1"
                src={sampleData.frame_t1.image}
                className="w-full h-full object-cover"
                alt="GOES-16 T1 Observed Scan"
              />
            ) : (
              <div className="text-xs text-outline font-mono">No T₁ image</div>
            )}
            <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-2">
              <div className="flex justify-between font-mono text-[9px] text-outline/80">
                <span>GOES-16 Band 13</span>
                <span>T₁ KEYFRAME</span>
              </div>
            </div>
          </div>
          <div className="p-2.5 bg-surface-container-low/60 flex items-center justify-between font-mono text-[11px] text-on-surface-variant">
            <span>Product: ABI-L2-CMIPF</span>
            <span id="badge-t1-dim" className="text-secondary font-medium">
              {sampleData?.raw_dimensions
                ? `${sampleData.raw_dimensions.width} × ${sampleData.raw_dimensions.height} px`
                : "--"}
            </span>
          </div>
        </div>
      </div>

      {/* Temporal Sub-Frame Stepper / Scrubber Line */}
      <div className="bg-surface-container-lowest px-4 py-3 rounded-lg border border-outline-variant/30 flex items-center gap-4">
        <div className="flex items-center gap-1 font-mono text-xs">
          <button
            id="btn-step-prev"
            onClick={() => onStep(-0.25)}
            className="w-7 h-7 rounded bg-surface-container hover:bg-surface-bright flex items-center justify-center text-on-surface transition-colors cursor-pointer"
            title="Previous Step (t - 0.25)"
          >
            <span className="material-symbols-outlined text-[16px]">skip_previous</span>
          </button>
          <button
            id="btn-play-toggle"
            onClick={onTogglePlay}
            className={`w-7 h-7 rounded flex items-center justify-center font-bold transition-all cursor-pointer ${
              isPlaying
                ? "bg-secondary text-on-secondary"
                : "bg-primary-container text-on-primary-container hover:brightness-110"
            }`}
            title="Toggle Auto-Animation"
          >
            <span id="play-icon" className="material-symbols-outlined text-[16px]">
              {isPlaying ? "pause" : "play_arrow"}
            </span>
          </button>
          <button
            id="btn-step-next"
            onClick={() => onStep(0.25)}
            className="w-7 h-7 rounded bg-surface-container hover:bg-surface-bright flex items-center justify-center text-on-surface transition-colors cursor-pointer"
            title="Next Step (t + 0.25)"
          >
            <span className="material-symbols-outlined text-[16px]">skip_next</span>
          </button>
        </div>

        <div className="flex-1 relative flex items-center h-6">
          <input
            id="time-slider"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={timeFraction}
            onChange={(e) => onTimeFractionChange(parseFloat(e.target.value))}
            onMouseUp={onTimeFractionCommit}
            onTouchEnd={onTimeFractionCommit}
            className="w-full accent-primary-container cursor-pointer z-20 opacity-0 absolute inset-0"
          />

          <div className="w-full h-1.5 bg-surface-container-highest rounded-full pointer-events-none relative">
            <div
              id="slider-progress-bar"
              className="absolute left-0 top-0 h-full bg-primary-container rounded-full"
              style={{ width: `${timeFraction * 100}%` }}
            />
          </div>

          {/* Stepper Node Markers */}
          <div className="absolute left-0 -translate-x-1/2 flex flex-col items-center pointer-events-none">
            <div className="w-2.5 h-2.5 rounded-sm bg-secondary" />
            <span className="font-mono text-[9px] text-secondary mt-1">t=0.00</span>
          </div>
          <div className="absolute left-1/4 -translate-x-1/2 flex flex-col items-center pointer-events-none">
            <div
              className={`w-2 h-2 rounded-full ${
                Math.abs(timeFraction - 0.25) < 0.03
                  ? "bg-primary-container ring-2 ring-primary-container/40"
                  : "bg-surface-variant"
              }`}
            />
            <span className="font-mono text-[9px] text-outline mt-1">t=0.25</span>
          </div>
          <div className="absolute left-2/4 -translate-x-1/2 flex flex-col items-center pointer-events-none">
            <div
              className={`w-3.5 h-3.5 rounded-full ${
                Math.abs(timeFraction - 0.5) < 0.03
                  ? "bg-primary-container ring-2 ring-primary-container/40"
                  : "bg-surface-variant"
              }`}
            />
            <span className="font-mono text-[9px] text-primary-container font-bold mt-1">
              t=0.50
            </span>
          </div>
          <div className="absolute left-3/4 -translate-x-1/2 flex flex-col items-center pointer-events-none">
            <div
              className={`w-2 h-2 rounded-full ${
                Math.abs(timeFraction - 0.75) < 0.03
                  ? "bg-primary-container ring-2 ring-primary-container/40"
                  : "bg-surface-variant"
              }`}
            />
            <span className="font-mono text-[9px] text-outline mt-1">t=0.75</span>
          </div>
          <div className="absolute right-0 translate-x-1/2 flex flex-col items-center pointer-events-none">
            <div className="w-2.5 h-2.5 rounded-sm bg-secondary" />
            <span className="font-mono text-[9px] text-secondary mt-1">t=1.00</span>
          </div>
        </div>

        <div className="font-mono text-[11px] text-on-surface-variant pl-4">
          Fraction:{" "}
          <span id="label-slider-val" className="text-primary font-bold">
            t = {timeFraction.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}
