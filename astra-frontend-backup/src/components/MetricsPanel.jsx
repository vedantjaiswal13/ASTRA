import React from "react";

export default function MetricsPanel({ metrics, dimensions, method }) {
  const currentMetrics =
    method === "linear" ? metrics?.linear : metrics?.warped;

  const mse = currentMetrics?.mse != null ? currentMetrics.mse.toFixed(2) : "--";
  const psnr =
    currentMetrics?.psnr_db != null ? currentMetrics.psnr_db.toFixed(2) : "--";
  const ssim =
    currentMetrics?.ssim != null ? currentMetrics.ssim.toFixed(4) : "--";
  const mae = currentMetrics?.mae != null ? currentMetrics.mae.toFixed(2) : "--";

  const improvement = metrics?.mse_improvement_percentage;
  const showImprovement =
    method !== "linear" && improvement != null && improvement > 0;

  return (
    <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-3">
      <div className="flex items-center justify-between pb-2 border-b border-outline-variant/30">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary-container text-[18px]">
            analytics
          </span>
          <h2 className="font-headline-sm text-sm font-semibold text-primary tracking-wide uppercase">
            Evaluation Metrics
          </h2>
        </div>
        <span
          id="metrics-grid-info"
          className="font-label-sm text-[11px] text-outline font-mono"
        >
          {dimensions
            ? `Calculated over full 2D spatial grid (${dimensions.width} × ${dimensions.height} px)`
            : "Calculated over full 2D spatial grid"}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1: MSE */}
        <div className="bg-surface-container-lowest p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="font-label-sm text-xs font-mono uppercase text-on-surface-variant">
              MSE
            </span>
            <span className="text-[10px] font-mono text-outline">
              Mean Squared Error
            </span>
          </div>
          <div className="my-2">
            <div
              id="metric-mse"
              className="font-mono text-2xl font-bold text-primary tracking-tight"
            >
              {mse}
            </div>
          </div>
          <div
            id="metric-mse-sub"
            className={`text-[11px] font-mono border-t border-outline-variant/20 pt-1.5 ${
              showImprovement
                ? "text-primary-container font-semibold"
                : "text-on-surface-variant/80"
            }`}
          >
            {showImprovement
              ? `${improvement.toFixed(1)}% error reduction vs linear`
              : "L2 intensity variance"}
          </div>
        </div>

        {/* Metric 2: PSNR */}
        <div className="bg-surface-container-lowest p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="font-label-sm text-xs font-mono uppercase text-on-surface-variant">
              PSNR
            </span>
            <span className="text-[10px] font-mono text-outline">
              Peak Signal-to-Noise
            </span>
          </div>
          <div className="my-2">
            <div
              id="metric-psnr"
              className="font-mono text-2xl font-bold text-primary tracking-tight"
            >
              {psnr}{" "}
              <span className="text-sm font-normal text-on-surface-variant">dB</span>
            </div>
          </div>
          <div className="text-[11px] font-mono text-on-surface-variant/80 border-t border-outline-variant/20 pt-1.5">
            10 · log₁₀(255² / MSE)
          </div>
        </div>

        {/* Metric 3: SSIM */}
        <div className="bg-surface-container-lowest p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="font-label-sm text-xs font-mono uppercase text-on-surface-variant">
              SSIM
            </span>
            <span className="text-[10px] font-mono text-outline">
              Structural Similarity
            </span>
          </div>
          <div className="my-2">
            <div
              id="metric-ssim"
              className="font-mono text-2xl font-bold text-primary tracking-tight"
            >
              {ssim}
            </div>
          </div>
          <div className="text-[11px] font-mono text-on-surface-variant/80 border-t border-outline-variant/20 pt-1.5">
            Scale [0, 1], luminance &amp; structure
          </div>
        </div>

        {/* Metric 4: MAE */}
        <div className="bg-surface-container-lowest p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="font-label-sm text-xs font-mono uppercase text-on-surface-variant">
              MAE
            </span>
            <span className="text-[10px] font-mono text-outline">
              Mean Absolute Error
            </span>
          </div>
          <div className="my-2">
            <div
              id="metric-mae"
              className="font-mono text-2xl font-bold text-primary tracking-tight"
            >
              {mae}
            </div>
          </div>
          <div className="text-[11px] font-mono text-on-surface-variant/80 border-t border-outline-variant/20 pt-1.5">
            Average L1 pixel difference
          </div>
        </div>
      </div>
    </div>
  );
}
