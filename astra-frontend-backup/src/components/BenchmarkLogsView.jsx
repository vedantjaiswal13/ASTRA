import React, { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

// Distinct method color scheme adhering to ASTRA dark glassmorphic palette
const METHOD_COLORS = {
  linear: {
    bg: "bg-sky-400",
    border: "border-sky-400",
    text: "text-sky-400",
    hex: "#38bdf8",
    badge: "bg-sky-500/20 text-sky-300 border-sky-500/40",
  },
  farneback: {
    bg: "bg-cyan-400",
    border: "border-cyan-400",
    text: "text-cyan-400",
    hex: "#00f0ff",
    badge: "bg-cyan-500/20 text-cyan-300 border-cyan-500/40",
  },
  raft: {
    bg: "bg-indigo-400",
    border: "border-indigo-400",
    text: "text-indigo-400",
    hex: "#818cf8",
    badge: "bg-indigo-500/20 text-indigo-300 border-indigo-500/40",
  },
  raft_refined: {
    bg: "bg-emerald-400",
    border: "border-emerald-400",
    text: "text-emerald-400",
    hex: "#34d399",
    badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  },
};

export default function BenchmarkLogsView({
  activeMethod,
  onSelectMethod,
  activeTripletId = "2024-01-01_18",
  availableTriplets = [],
  onSelectTriplet = () => {},
}) {
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [progressStep, setProgressStep] = useState("");
  const [error, setError] = useState(null);

  // 1. Fetch benchmark metrics by calling POST /generate for each method
  const fetchBenchmark = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const methodConfigs = [
        {
          id: "linear",
          name: "Linear Interpolation",
          short_name: "Linear",
          description: "Baseline linear pixel cross-dissolve",
        },
        {
          id: "farneback",
          name: "Farneback Flow",
          short_name: "Farneback",
          description: "Classical polynomial optical flow",
        },
        {
          id: "raft",
          name: "RAFT Flow",
          short_name: "RAFT",
          description: "Deep correlation volume optical flow",
        },
        {
          id: "raft_refined",
          name: "RAFT + Refinement",
          short_name: "RAFT + U-Net",
          description: "RAFT flow + trained refinement U-Net",
        },
      ];

      // Execute POST /generate sequentially for each method against the active triplet at FULL native resolution (5424x5424)
      const results = [];
      for (let i = 0; i < methodConfigs.length; i++) {
        const cfg = methodConfigs[i];
        setProgressStep(
          `Computing ${cfg.name} at Native Full Resolution (5424 × 5424 px) [${i + 1}/${methodConfigs.length}]...`
        );
        const res = await fetch(`${API_BASE}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            triplet_id: activeTripletId,
            time_fraction: 0.5,
            method: cfg.id,
            preview_size: 5424, // Full Native Resolution (5424x5424)
            pyr_scale: 0.5,
            levels: 3,
            winsize: 15,
            iterations: 3,
          }),
        });

        if (!res.ok) {
          throw new Error(`Failed to generate ${cfg.name} (HTTP ${res.status})`);
        }

        const data = await res.json();
        const curMetrics =
          cfg.id === "linear" ? data.metrics?.linear : data.metrics?.warped;

        results.push({
          ...cfg,
          has_ground_truth: data.metrics?.is_evaluated_against_ground_truth ?? false,
          mse: curMetrics?.mse ?? null,
          psnr_db: curMetrics?.psnr_db ?? null,
          ssim: curMetrics?.ssim ?? null,
          mae: curMetrics?.mae ?? null,
          improvement_vs_linear:
            data.metrics?.mse_improvement_percentage ?? null,
          dimensions: data.dimensions,
        });
      }

      setBenchmarkData({
        status: "success",
        endpoint: "POST /generate",
        sample_pair_metadata: {
          channel: "GOES-16 ABI Band 13 (Clean IR 10.35 µm)",
          evaluated_at: "t=0.50 (Held-Out Middle Observation)",
          resolution: `${results[0]?.dimensions?.width || 5424} × ${
            results[0]?.dimensions?.height || 5424
          } px (Native Full Resolution)`,
        },
        methods: results,
      });
    } catch (err) {
      console.error("Failed to load benchmark:", err);
      setError(
        `Failed to retrieve benchmark logs from ${API_BASE}/generate: ${err.message}`
      );
    } finally {
      setIsLoading(false);
      setProgressStep("");
    }
  };

  useEffect(() => {
    fetchBenchmark();
  }, [activeTripletId]);

  const methods = benchmarkData?.methods || [];

  // Group metric keys for comparison charts
  const metricGroups = [
    {
      key: "mse",
      title: "MSE (Mean Squared Error)",
      unit: "",
      direction: "Lower is better",
      format: (v) => (v != null ? v.toFixed(2) : "N/A"),
      isLowerBetter: true,
    },
    {
      key: "psnr_db",
      title: "PSNR (Peak Signal-to-Noise)",
      unit: "dB",
      direction: "Higher is better",
      format: (v) => (v != null ? `${v.toFixed(2)} dB` : "N/A"),
      isLowerBetter: false,
    },
    {
      key: "ssim",
      title: "SSIM (Structural Similarity)",
      unit: "",
      direction: "Higher is better (max 1.0)",
      format: (v) => (v != null ? v.toFixed(4) : "N/A"),
      isLowerBetter: false,
    },
    {
      key: "mae",
      title: "MAE (Mean Absolute Error)",
      unit: "",
      direction: "Lower is better",
      format: (v) => (v != null ? v.toFixed(2) : "N/A"),
      isLowerBetter: true,
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* Top Header Card */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-surface-container-high flex items-center justify-center border border-primary-container/30">
            <span className="material-symbols-outlined text-primary-container text-[22px]">
              query_stats
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-headline-sm text-sm font-bold text-primary tracking-wide uppercase">
                Quantitative Method Benchmark Suite
              </h2>
              <span className="px-2 py-0.5 rounded bg-primary-container/20 text-primary-container font-mono text-[10px] font-semibold border border-primary-container/30">
                5424 × 5424 px Native Full Resolution
              </span>
              <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono text-[10px] font-semibold border border-amber-500/30">
                Triplet: {activeTripletId}
              </span>
            </div>
            <p className="text-on-surface-variant font-mono text-xs mt-0.5">
              Full-resolution evaluation (5424 × 5424 px) on identical GOES-16 satellite scan pair (held-out t=0.50 observation)
            </p>
          </div>
        </div>

        {/* View Controls */}
        <div className="flex items-center gap-3 font-mono text-xs">
          {availableTriplets && availableTriplets.length > 0 && (
            <div className="flex items-center gap-1.5 bg-surface-container px-3 py-1.5 rounded-lg border border-outline-variant/40">
              <span className="text-on-surface-variant text-[11px] uppercase">Active Triplet:</span>
              <select
                id="benchmark-triplet-selector"
                value={activeTripletId}
                onChange={(e) => onSelectTriplet(e.target.value)}
                className="bg-transparent text-primary-fixed font-bold focus:outline-none cursor-pointer text-xs"
              >
                {availableTriplets.map((t) => (
                  <option key={t.id} value={t.id} className="bg-surface-container-lowest text-on-surface">
                    {t.date} {t.hour !== undefined ? `${String(t.hour).padStart(2, '0')}:00 UTC` : t.id}
                  </option>
                ))}
              </select>
            </div>
          )}
          <button
            onClick={() => fetchBenchmark()}
            disabled={isLoading}
            className="flex items-center gap-1.5 bg-surface-container px-3 py-1.5 rounded-lg border border-outline-variant/40 text-on-surface-variant hover:text-primary transition-colors cursor-pointer disabled:opacity-50"
            title="Re-run POST /generate for all four methods"
          >
            <span
              className={`material-symbols-outlined text-[16px] ${
                isLoading ? "animate-spin" : ""
              }`}
            >
              sync
            </span>
            <span>{isLoading ? "Running Suite..." : "Re-run Benchmark"}</span>
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-error-container/40 border border-error/50 text-error backdrop-blur-md font-mono text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[20px]">error</span>
            <span>{error}</span>
          </div>
          <button
            onClick={() => fetchBenchmark()}
            className="px-3 py-1 bg-error/20 hover:bg-error/30 rounded border border-error/40 text-error font-semibold cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading State Skeleton */}
      {isLoading && !benchmarkData && (
        <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-12 flex flex-col items-center justify-center gap-4 text-center">
          <span className="material-symbols-outlined text-primary-container text-4xl animate-spin">
            sync
          </span>
          <div className="flex flex-col gap-1">
            <span className="font-headline-sm text-sm font-bold text-primary">
              Computing 4-Method Benchmark Suite
            </span>
            <span className="text-on-surface-variant font-mono text-xs">
              {progressStep || "Evaluating Linear, Farneback, RAFT, and RAFT + U-Net against ground truth at full native resolution (5424 × 5424 px)..."}
            </span>
          </div>
        </div>
      )}

      {benchmarkData && (
        <>
          {/* Section 1: Benchmark Table */}
          <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 overflow-hidden shadow-lg flex flex-col">
            <div className="px-5 py-3.5 bg-surface-container-low border-b border-outline-variant/30 flex items-center justify-between">
              <div className="flex items-center gap-2 font-mono">
                <span className="material-symbols-outlined text-primary-container text-[18px]">
                  table_chart
                </span>
                <span className="text-xs font-bold text-primary uppercase">
                  Benchmark Metrics Comparison Table
                </span>
              </div>
              <span className="text-[11px] font-mono text-on-surface-variant">
                Target: t = 0.50 (Held-Out Middle Scan)
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-outline-variant/40 bg-surface-container-lowest/60 text-outline text-[11px] uppercase tracking-wider">
                    <th className="py-3 px-5">Method</th>
                    <th className="py-3 px-4">MSE ↓</th>
                    <th className="py-3 px-4">PSNR ↑</th>
                    <th className="py-3 px-4">SSIM ↑</th>
                    <th className="py-3 px-4">MAE ↓</th>
                    <th className="py-3 px-4">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/20">
                  {methods.map((m) => {
                    const isSelected = activeMethod === m.id;
                    const colors = METHOD_COLORS[m.id] || METHOD_COLORS.linear;

                    return (
                      <tr
                        key={m.id}
                        className={`transition-colors ${
                          isSelected
                            ? "bg-primary-container/10 border-l-4 border-l-primary-container font-semibold"
                            : "hover:bg-surface-container-high/30"
                        }`}
                      >
                        <td className="py-3.5 px-5">
                          <div className="flex items-center gap-3">
                            <span
                              className={`w-2.5 h-2.5 rounded-full ${colors.bg}`}
                            />
                            <div className="flex flex-col">
                              <span className="text-on-surface font-bold">
                                {m.name}
                              </span>
                              <span className="text-[10px] text-on-surface-variant/70 font-normal">
                                {m.description}
                              </span>
                            </div>
                          </div>
                        </td>

                        <td className="py-3.5 px-4 text-primary font-bold">
                          {m.mse != null ? m.mse.toFixed(2) : "N/A"}
                        </td>

                        <td className="py-3.5 px-4 text-primary font-bold">
                          {m.psnr_db != null ? (
                            <>
                              {m.psnr_db.toFixed(2)}{" "}
                              <span className="text-[10px] text-outline font-normal">
                                dB
                              </span>
                            </>
                          ) : (
                            "N/A"
                          )}
                        </td>

                        <td className="py-3.5 px-4 text-secondary font-bold">
                          {m.ssim != null ? m.ssim.toFixed(4) : "N/A"}
                        </td>

                        <td className="py-3.5 px-4 text-primary font-bold">
                          {m.mae != null ? m.mae.toFixed(2) : "N/A"}
                        </td>

                        <td className="py-3.5 px-4">
                          {isSelected ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-primary-container/20 text-primary-container border border-primary-container/40 text-[10px] font-bold">
                              <span>★</span> ACTIVE IN WORKBENCH
                            </span>
                          ) : (
                            <button
                              onClick={() => onSelectMethod && onSelectMethod(m.id)}
                              className="text-[10px] text-outline hover:text-primary transition-colors cursor-pointer"
                            >
                              Select Method
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 2: Visual Comparison Bar Chart (1 Group per Metric, 4 Bars per Group) */}
          <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-outline-variant/30 pb-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-primary-container text-[18px]">
                  bar_chart
                </span>
                <h3 className="font-headline-sm text-sm font-bold text-primary uppercase tracking-wide">
                  Visual Metric Sensitivity Charts
                </h3>
              </div>
              <div className="flex flex-wrap items-center gap-4 font-mono text-xs">
                {methods.map((m) => {
                  const colors = METHOD_COLORS[m.id] || METHOD_COLORS.linear;
                  const isSelected = activeMethod === m.id;
                  return (
                    <div key={m.id} className="flex items-center gap-1.5">
                      <span className={`w-2.5 h-2.5 rounded-sm ${colors.bg}`} />
                      <span
                        className={`text-[11px] ${
                          isSelected
                            ? "text-primary font-bold underline"
                            : "text-on-surface-variant"
                        }`}
                      >
                        {m.short_name}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Metric Groups Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {metricGroups.map((group) => {
                // Calculate max value in this group for proportional scaling
                const values = methods.map((m) => m[group.key]);
                const maxVal = Math.max(...values, 0.001);
                const minVal = Math.min(...values);

                return (
                  <div
                    key={group.key}
                    className="bg-surface-container-lowest p-4 rounded-lg border border-outline-variant/30 flex flex-col gap-3"
                  >
                    <div className="flex items-center justify-between font-mono">
                      <span className="text-xs font-bold text-primary">
                        {group.title}
                      </span>
                      <span className="text-[10px] text-outline">
                        {group.direction}
                      </span>
                    </div>

                    {/* Bars Container */}
                    <div className="flex items-end justify-between gap-3 h-44 pt-6 pb-2 px-3 border-b border-outline-variant/30 relative">
                      {methods.map((m) => {
                        const val = m[group.key];
                        const isSelected = activeMethod === m.id;
                        const colors =
                          METHOD_COLORS[m.id] || METHOD_COLORS.linear;

                        // Height percentage calculated relative to max value
                        // With min height of 12% so small bars remain visible
                        const heightPct = Math.max(
                          12,
                          Math.min(100, (val / maxVal) * 100)
                        );

                        return (
                          <div
                            key={m.id}
                            className="flex-1 flex flex-col items-center justify-end h-full group relative"
                          >
                            {/* Numerical Value Label */}
                            <span
                              className={`font-mono text-[10px] mb-1.5 whitespace-nowrap transition-transform group-hover:scale-110 ${
                                isSelected
                                  ? "text-primary font-bold"
                                  : "text-on-surface-variant"
                              }`}
                            >
                              {group.format(val)}
                            </span>

                            {/* Bar element */}
                            <div
                              className="w-full max-w-[48px] rounded-t transition-all duration-500 relative"
                              style={{
                                height: `${heightPct}%`,
                                backgroundColor: colors.hex,
                                opacity: isSelected ? 1.0 : 0.75,
                                boxShadow: isSelected
                                  ? `0 0 12px ${colors.hex}88`
                                  : "none",
                              }}
                            >
                              {isSelected && (
                                <div className="absolute -top-3 left-1/2 -translate-x-1/2 text-primary text-[10px] font-bold">
                                  ★
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Method Labels Under Bars */}
                    <div className="flex justify-between gap-3 font-mono text-[10px] text-center px-1">
                      {methods.map((m) => {
                        const isSelected = activeMethod === m.id;
                        return (
                          <span
                            key={m.id}
                            className={`flex-1 truncate ${
                              isSelected
                                ? "text-primary font-bold"
                                : "text-outline"
                            }`}
                          >
                            {m.short_name}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
