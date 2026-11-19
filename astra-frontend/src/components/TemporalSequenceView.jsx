import React, { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:8000";

export default function TemporalSequenceView({
  activeTripletId = "2024-01-01_18",
  availableTriplets = [],
  onSelectTriplet = () => {},
}) {
  const [method, setMethod] = useState("farneback");
  const [numSteps, setNumSteps] = useState(5);
  const [sequenceData, setSequenceData] = useState(null);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1); // 1 = ~1 frame per second
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState(null);
  const [showGroundTruthCompare, setShowGroundTruthCompare] = useState(false);

  const playTimerRef = useRef(null);

  const methods = [
    { id: "farneback", label: "Farneback Flow" },
    { id: "raft", label: "RAFT Deep Flow" },
    { id: "raft_refined", label: "RAFT + Refinement" },
    { id: "neural_synthesis", label: "Neural Synthesis (POC)" },
    { id: "linear", label: "Linear Cross-Fade" },
  ];

  // Fetch temporal sequence from backend
  const fetchSequence = (tripId, met, steps) => {
    setIsLoading(true);
    setErrorMessage(null);

    const payload = {
      triplet_id: tripId,
      method: met,
      num_steps: steps,
      preview_size: 512,
    };

    fetch(`${API_BASE}/sequence`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to generate sequence`);
        return res.json();
      })
      .then((data) => {
        setSequenceData(data);
        setCurrentFrameIdx(0);
        setIsLoading(false);
      })
      .catch((err) => {
        setErrorMessage(err.message);
        setIsLoading(false);
      });
  };

  useEffect(() => {
    fetchSequence(activeTripletId, method, numSteps);
  }, [activeTripletId, method, numSteps]);

  // Video playback loop
  useEffect(() => {
    if (isPlaying && sequenceData?.frames?.length > 0) {
      const intervalMs = Math.max(200, 1000 / (playbackSpeed * 1.5));
      playTimerRef.current = setInterval(() => {
        setCurrentFrameIdx((prev) => (prev + 1) % sequenceData.frames.length);
      }, intervalMs);
    } else {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    }

    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, playbackSpeed, sequenceData]);

  const currentFrame = sequenceData?.frames?.[currentFrameIdx];
  const groundTruthImage = sequenceData?.ground_truth_image;
  const isMidpoint = currentFrame && Math.abs(currentFrame.time_fraction - 0.5) < 1e-3;

  return (
    <div className="flex flex-col gap-6 animate-fadeIn pb-12">
      {/* Top Banner */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-surface-container-high flex items-center justify-center border border-primary-container/30 shadow-[0_0_15px_rgba(99,102,241,0.15)]">
            <span className="material-symbols-outlined text-primary-container text-[24px]">
              timeline
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="font-headline-sm text-base font-bold text-primary tracking-wide uppercase">
                Continuous Temporal Sequence Reconstruction
              </h2>
              <span className="px-2 py-0.5 rounded-full bg-primary-container/20 text-primary-container font-mono text-[10px] font-bold border border-primary-container/40">
                Multi-Cadence Playback
              </span>
            </div>
            <p className="text-on-surface-variant font-mono text-xs mt-1">
              Super-resolving 15-minute GOES-16 intervals into smooth intermediate atmospheric trajectories
            </p>
          </div>
        </div>

        {/* Global Controls: Triplet & Cadence */}
        <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
          {availableTriplets.length > 0 && (
            <div className="flex items-center gap-1.5 bg-surface-container-lowest px-3 py-1.5 rounded-lg border border-outline-variant/40">
              <span className="text-outline uppercase text-[10px]">Triplet:</span>
              <select
                value={activeTripletId}
                onChange={(e) => onSelectTriplet(e.target.value)}
                className="bg-transparent text-primary font-bold cursor-pointer focus:outline-none"
              >
                {availableTriplets.map((t) => (
                  <option key={t.id} value={t.id} className="bg-surface-container text-on-surface">
                    {t.id}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="flex items-center gap-1.5 bg-surface-container-lowest px-3 py-1.5 rounded-lg border border-outline-variant/40">
            <span className="text-outline uppercase text-[10px]">Cadence:</span>
            <select
              value={numSteps}
              onChange={(e) => setNumSteps(Number(e.target.value))}
              className="bg-transparent text-primary font-bold cursor-pointer focus:outline-none"
            >
              <option value={5} className="bg-surface-container text-on-surface">5 Frames (~3.7m)</option>
              <option value={7} className="bg-surface-container text-on-surface">7 Frames (~2.5m)</option>
              <option value={9} className="bg-surface-container text-on-surface">9 Frames (~1.9m)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Method Selection Bar */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-3 shadow-sm flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-xs text-outline ml-2 mr-1">Interpolation Method:</span>
          {methods.map((m) => (
            <button
              key={m.id}
              onClick={() => setMethod(m.id)}
              className={`px-3 py-1.5 rounded-lg font-mono text-xs font-medium transition-all cursor-pointer ${
                method === m.id
                  ? "bg-primary-container text-on-primary-container font-bold shadow-sm border border-primary-container"
                  : "bg-surface-container-high text-on-surface-variant hover:bg-surface-container-highest border border-outline-variant/30"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        {sequenceData?.motion_summary && (
          <div className="flex items-center gap-4 font-mono text-xs text-outline pr-3">
            <span>Mean Motion: <strong className="text-secondary">{sequenceData.motion_summary.mean_displacement_px} px</strong></span>
            <span>Peak Flow: <strong className="text-primary-fixed">{sequenceData.motion_summary.max_displacement_px} px</strong></span>
          </div>
        )}
      </div>

      {/* Main Player & Inspection Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Main Canvas Viewport (8 Columns) */}
        <div className="lg:col-span-8 bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-outline-variant/30 pb-3">
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
              <span className="font-bold text-on-surface">{currentFrame?.time_label || "Loading..."}</span>
              {currentFrame?.is_boundary && (
                <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-outline text-[10px]">
                  Boundary Scan
                </span>
              )}
              {isMidpoint && (
                <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-bold border border-emerald-500/40">
                  Ground Truth Midpoint (t=0.50)
                </span>
              )}
            </div>

            {isMidpoint && groundTruthImage && (
              <button
                onClick={() => setShowGroundTruthCompare(!showGroundTruthCompare)}
                className={`px-2.5 py-1 rounded text-xs font-mono transition-colors cursor-pointer border ${
                  showGroundTruthCompare
                    ? "bg-amber-500/20 text-amber-300 border-amber-500/50"
                    : "bg-surface-container-high text-outline border-outline-variant/40 hover:text-on-surface"
                }`}
              >
                {showGroundTruthCompare ? "Hide Ground Truth" : "Compare Ground Truth"}
              </button>
            )}
          </div>

          {/* Viewport Frame */}
          <div className="relative aspect-video w-full bg-black rounded-xl overflow-hidden border border-outline-variant/30 flex items-center justify-center">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center gap-3">
                <div className="w-8 h-8 border-2 border-primary-container border-t-transparent rounded-full animate-spin" />
                <span className="font-mono text-xs text-outline">Synthesizing temporal sequence frames...</span>
              </div>
            ) : errorMessage ? (
              <div className="p-6 text-center font-mono text-xs text-error">
                {errorMessage}
              </div>
            ) : showGroundTruthCompare && isMidpoint && groundTruthImage ? (
              <div className="grid grid-cols-2 w-full h-full">
                <div className="relative border-r border-outline-variant/30 overflow-hidden flex items-center justify-center">
                  <img
                    src={currentFrame?.image}
                    alt="Synthesized"
                    className="w-full h-full object-contain"
                  />
                  <span className="absolute bottom-2 left-2 px-2 py-0.5 bg-black/70 text-primary-container font-mono text-[10px] rounded">
                    Synthesized ({method})
                  </span>
                </div>
                <div className="relative overflow-hidden flex items-center justify-center">
                  <img
                    src={groundTruthImage}
                    alt="Ground Truth"
                    className="w-full h-full object-contain"
                  />
                  <span className="absolute bottom-2 left-2 px-2 py-0.5 bg-black/70 text-emerald-400 font-mono text-[10px] rounded">
                    Authentic Ground Truth
                  </span>
                </div>
              </div>
            ) : (
              <img
                src={currentFrame?.image}
                alt="Temporal Frame"
                className="w-full h-full object-contain"
              />
            )}
          </div>

          {/* Player Controls Bar */}
          <div className="flex flex-col gap-3 bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/30">
            {/* Scrubber Range */}
            <div className="flex items-center gap-3">
              <span className="font-mono text-[10px] text-outline w-12 text-left">
                T₀ (0.0)
              </span>
              <input
                type="range"
                min={0}
                max={sequenceData?.frames ? sequenceData.frames.length - 1 : 4}
                value={currentFrameIdx}
                onChange={(e) => {
                  setIsPlaying(false);
                  setCurrentFrameIdx(Number(e.target.value));
                }}
                className="flex-1 accent-primary-container cursor-pointer h-1.5 bg-surface-container-high rounded-lg"
              />
              <span className="font-mono text-[10px] text-outline w-12 text-right">
                T₁ (1.0)
              </span>
            </div>

            {/* Playback Buttons & Speed Selector */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    setIsPlaying(false);
                    setCurrentFrameIdx(0);
                  }}
                  className="p-1.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-on-surface cursor-pointer"
                  title="First Frame"
                >
                  <span className="material-symbols-outlined text-[18px]">skip_previous</span>
                </button>

                <button
                  onClick={() => {
                    setIsPlaying(false);
                    setCurrentFrameIdx((prev) => Math.max(0, prev - 1));
                  }}
                  className="p-1.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-on-surface cursor-pointer"
                  title="Previous Frame"
                >
                  <span className="material-symbols-outlined text-[18px]">fast_rewind</span>
                </button>

                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className={`px-4 py-1.5 rounded font-mono text-xs font-bold flex items-center gap-1.5 transition-colors cursor-pointer ${
                    isPlaying
                      ? "bg-amber-500 text-black shadow-md"
                      : "bg-primary-container text-on-primary-container shadow-md"
                  }`}
                >
                  <span className="material-symbols-outlined text-[16px]">
                    {isPlaying ? "pause" : "play_arrow"}
                  </span>
                  <span>{isPlaying ? "Pause" : "Play Sequence"}</span>
                </button>

                <button
                  onClick={() => {
                    setIsPlaying(false);
                    setCurrentFrameIdx((prev) =>
                      sequenceData?.frames ? Math.min(sequenceData.frames.length - 1, prev + 1) : prev
                    );
                  }}
                  className="p-1.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-on-surface cursor-pointer"
                  title="Next Frame"
                >
                  <span className="material-symbols-outlined text-[18px]">fast_forward</span>
                </button>

                <button
                  onClick={() => {
                    setIsPlaying(false);
                    if (sequenceData?.frames) setCurrentFrameIdx(sequenceData.frames.length - 1);
                  }}
                  className="p-1.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-on-surface cursor-pointer"
                  title="Last Frame"
                >
                  <span className="material-symbols-outlined text-[18px]">skip_next</span>
                </button>
              </div>

              {/* Playback Speed */}
              <div className="flex items-center gap-1 font-mono text-xs">
                <span className="text-outline text-[11px] mr-1">Speed:</span>
                {[0.5, 1, 2, 4].map((spd) => (
                  <button
                    key={spd}
                    onClick={() => setPlaybackSpeed(spd)}
                    className={`px-2 py-0.5 rounded text-[11px] cursor-pointer ${
                      playbackSpeed === spd
                        ? "bg-primary-container text-on-primary-container font-bold"
                        : "text-on-surface-variant hover:bg-surface-container-high"
                    }`}
                  >
                    {spd}×
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Diagnostic & Frame Details Sidebar (4 Columns) */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          {/* Active Frame Info Card */}
          <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-outline-variant/30 pb-2">
              <span className="font-mono text-xs font-bold text-primary uppercase">
                Active Timestamp State
              </span>
              <span className="font-mono text-[10px] text-outline">
                Step {currentFrameIdx + 1} of {sequenceData?.num_frames || numSteps}
              </span>
            </div>

            <div className="space-y-2 font-mono text-xs">
              <div className="flex justify-between py-1 border-b border-outline-variant/10">
                <span className="text-outline">Fraction t*:</span>
                <span className="font-bold text-on-surface">
                  {currentFrame?.time_fraction !== undefined ? currentFrame.time_fraction.toFixed(2) : "0.00"}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-outline-variant/10">
                <span className="text-outline">Effective Cadence:</span>
                <span className="font-bold text-secondary">
                  +{((currentFrame?.time_fraction || 0) * 15).toFixed(1)} mins
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-outline-variant/10">
                <span className="text-outline">Algorithm:</span>
                <span className="font-bold text-primary-fixed uppercase">{method}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-outline">Status:</span>
                <span className="font-bold text-emerald-400">
                  {currentFrame?.is_boundary ? "Observed Frame" : "Model Synthesized"}
                </span>
              </div>
            </div>

            {/* Validation Metrics (Only at t=0.50) */}
            {currentFrame?.metrics ? (
              <div className="mt-2 p-3 bg-surface-container-lowest rounded-lg border border-emerald-500/30 flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] text-emerald-400 font-bold uppercase">
                    Ground Truth Validation (t=0.50)
                  </span>
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                </div>
                <div className="grid grid-cols-2 gap-2 font-mono text-xs">
                  <div>
                    <span className="text-[10px] text-outline block">PSNR:</span>
                    <strong className="text-emerald-400">{currentFrame.metrics.psnr_db.toFixed(2)} dB</strong>
                  </div>
                  <div>
                    <span className="text-[10px] text-outline block">SSIM:</span>
                    <strong className="text-cyan-400">{currentFrame.metrics.ssim.toFixed(4)}</strong>
                  </div>
                  <div>
                    <span className="text-[10px] text-outline block">MSE:</span>
                    <strong className="text-on-surface">{currentFrame.metrics.mse.toFixed(2)}</strong>
                  </div>
                  <div>
                    <span className="text-[10px] text-outline block">MAE:</span>
                    <strong className="text-amber-400">{currentFrame.metrics.mae.toFixed(2)}</strong>
                  </div>
                </div>
              </div>
            ) : (
              <div className="mt-2 p-3 bg-surface-container-lowest rounded-lg border border-outline-variant/20 font-mono text-[11px] text-outline text-center">
                Quantitative metrics computed exclusively at held-out ground truth scan (t=0.50).
              </div>
            )}
          </div>

          {/* Temporal Super-Resolution Significance */}
          <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-2.5 font-mono text-xs">
            <span className="font-bold text-primary uppercase text-[11px] tracking-wide border-b border-outline-variant/30 pb-2">
              Operational Super-Resolution
            </span>
            <p className="text-[11px] text-on-surface-variant leading-relaxed">
              Standard geostationary scanning operates on a 15-minute Full-Disk cycle. By synthesizing intermediate states, ASTRA delivers virtual 3-minute rapid scan fidelity without increasing downlink bandwidth or sensor wear.
            </p>
          </div>
        </div>
      </div>

      {/* Filmstrip Reel Section */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-outline-variant/30 pb-3">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary-fixed text-[20px]">
              movie
            </span>
            <h3 className="font-headline-sm text-sm font-bold text-primary uppercase tracking-wide">
              Sequence Filmstrip Reel
            </h3>
          </div>
          <span className="font-mono text-[11px] text-outline">
            Click any thumbnail to inspect state
          </span>
        </div>

        {/* Horizontal Filmstrip Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-9 gap-3">
          {sequenceData?.frames?.map((f, idx) => (
            <button
              key={idx}
              onClick={() => {
                setIsPlaying(false);
                setCurrentFrameIdx(idx);
              }}
              className={`p-2 rounded-xl flex flex-col gap-2 text-left transition-all cursor-pointer ${
                currentFrameIdx === idx
                  ? "bg-primary-container/20 border-2 border-primary-container shadow-md"
                  : "bg-surface-container-lowest border border-outline-variant/30 hover:border-outline"
              }`}
            >
              <div className="aspect-square w-full bg-black rounded-lg overflow-hidden flex items-center justify-center">
                <img src={f.image} alt={f.time_label} className="w-full h-full object-cover" />
              </div>
              <div className="flex flex-col font-mono text-[10px]">
                <span className={`font-bold ${currentFrameIdx === idx ? "text-primary-container" : "text-on-surface"}`}>
                  {f.time_label}
                </span>
                <span className="text-outline">
                  {f.is_boundary ? "Observed" : `t=${f.time_fraction}`}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
