import React, { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

export default function NeuralSynthesisView() {
  const [demoData, setDemoData] = useState(null);
  const [selectedPatchIndex, setSelectedPatchIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [fetchError, setFetchError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);

    fetch(`${API_BASE}/neural-synthesis/samples`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (isMounted) {
          setDemoData(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setFetchError(err.message);
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const checklist = [
    {
      title: "Data Pipeline",
      desc: "GOES-16 Band 13 full-disk ingestion, variance-based cloud patch extraction (256×256), and 320/80 train/val partition.",
      status: "DONE",
      statusColor: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40",
      icon: "check_circle",
    },
    {
      title: "Baseline Comparison Suite",
      desc: "End-to-end evaluation across Linear Cross-Dissolve, Farneback Flow, RAFT, and RAFT + Refinement U-Net at full native resolution (5424×5424).",
      status: "DONE",
      statusColor: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40",
      icon: "check_circle",
    },
    {
      title: "Direct-Synthesis Model",
      desc: "Compact conditional generative U-Net taking 4-channel stacked input (T0, T1, Farneback u, Farneback v) to directly predict intermediate cloud states without spatial warping.",
      status: demoData ? "POC TRAINED & RUNNING" : "IN PROGRESS / PLANNED",
      statusColor: demoData ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40" : "bg-amber-500/20 text-amber-400 border-amber-500/40",
      icon: demoData ? "check_circle" : "pending",
    },
    {
      title: "Learned Confidence Head",
      desc: "Auxiliary network branch predicting pixel-wise epistemic uncertainty and convective ambiguity, replacing heuristic flow consistency.",
      status: "PLANNED",
      statusColor: "bg-sky-500/20 text-sky-400 border-sky-500/40",
      icon: "schedule",
    },
    {
      title: "Structural Reliability Index (SRI)",
      desc: "Meteorological domain-specific metric combining cloud boundary sharpness, conservation residuals, and structural fidelity for satellite verification.",
      status: "PLANNED",
      statusColor: "bg-sky-500/20 text-sky-400 border-sky-500/40",
      icon: "schedule",
    },
  ];

  const currentSample = demoData?.samples?.[selectedPatchIndex];

  return (
    <div className="flex flex-col gap-6 animate-fadeIn pb-12">
      {/* Top Banner with High-Visibility Required Badge */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-surface-container-high flex items-center justify-center border border-amber-400/40 shadow-[0_0_15px_rgba(251,191,36,0.15)]">
            <span className="material-symbols-outlined text-amber-400 text-[24px]">
              neurology
            </span>
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2.5">
              <h2 className="font-headline-sm text-base font-bold text-primary tracking-wide uppercase">
                Neural Direct Synthesis Pipeline
              </h2>
              {/* Mandatory Requirement Badge */}
              <span className="px-2.5 py-1 rounded-full bg-amber-500/20 text-amber-300 font-mono text-[11px] font-bold border border-amber-500/50 shadow-sm animate-pulse">
                ★ PLANNED - END-SEM (not yet trained full diffusion)
              </span>
              {demoData && (
                <span className="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-mono text-[11px] font-bold border border-emerald-500/40 shadow-sm">
                  ✓ POC U-Net Demo Online
                </span>
              )}
            </div>
            <p className="text-on-surface-variant font-mono text-xs mt-1">
              End-to-end conditional generation: boundary observations + dense flow features → direct cloud frame synthesis
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs bg-surface-container-lowest px-3 py-1.5 rounded-lg border border-outline-variant/40 text-on-surface-variant">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>Proof-of-Concept Prototype Phase</span>
        </div>
      </div>

      {/* Section 1: Architecture Pipeline Diagram */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-6 shadow-lg flex flex-col gap-5">
        <div className="flex items-center justify-between border-b border-outline-variant/30 pb-3">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary-container text-[20px]">
              account_tree
            </span>
            <h3 className="font-headline-sm text-sm font-bold text-primary uppercase tracking-wide">
              Planned Neural Synthesis Architecture
            </h3>
          </div>
          <span className="font-mono text-[11px] text-outline">
            4-Channel Input • Conditional U-Net • Multi-Head Output
          </span>
        </div>

        {/* Pipeline Diagram Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
          {/* Node 1: Input Tensor */}
          <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/50 flex flex-col gap-3 relative shadow-inner">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] font-bold text-primary-fixed uppercase tracking-wider">
                1. Input Tensor
              </span>
              <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-outline font-mono text-[10px]">
                4 Channels
              </span>
            </div>
            <div className="space-y-1.5 font-mono text-xs">
              <div className="flex items-center gap-2 p-1.5 rounded bg-surface-container-low/70 border border-outline-variant/20">
                <span className="w-2 h-2 rounded-full bg-cyan-400" />
                <span className="text-on-surface text-[11px]">T₀ Observed Frame [1, H, W]</span>
              </div>
              <div className="flex items-center gap-2 p-1.5 rounded bg-surface-container-low/70 border border-outline-variant/20">
                <span className="w-2 h-2 rounded-full bg-sky-400" />
                <span className="text-on-surface text-[11px]">T₁ Observed Frame [1, H, W]</span>
              </div>
              <div className="flex items-center gap-2 p-1.5 rounded bg-surface-container-low/70 border border-outline-variant/20">
                <span className="w-2 h-2 rounded-full bg-amber-400" />
                <span className="text-on-surface text-[11px]">Farneback Flow u [1, H, W]</span>
              </div>
              <div className="flex items-center gap-2 p-1.5 rounded bg-surface-container-low/70 border border-outline-variant/20">
                <span className="w-2 h-2 rounded-full bg-amber-400" />
                <span className="text-on-surface text-[11px]">Farneback Flow v [1, H, W]</span>
              </div>
            </div>
            <div className="text-[10px] font-mono text-outline border-t border-outline-variant/20 pt-2">
              Dimensions: [B, 4, 256, 256]
            </div>
          </div>

          {/* Node 2: Generative Model */}
          <div className="bg-surface-container-lowest p-4 rounded-xl border-2 border-primary-container/40 flex flex-col gap-3 shadow-[0_0_20px_rgba(99,102,241,0.08)]">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] font-bold text-primary-container uppercase tracking-wider">
                2. Generative U-Net
              </span>
              <span className="px-1.5 py-0.5 rounded bg-primary-container/20 text-primary-container font-mono text-[10px] font-bold">
                1.93M Params
              </span>
            </div>
            <p className="font-body-md text-xs text-on-surface-variant/90 leading-relaxed">
              Encoder-decoder network with multi-scale skip connections. Direct non-warped radiance mapping conditioned on dense optical flow.
            </p>
            <div className="text-[10px] font-mono text-on-surface-variant/70 border-t border-outline-variant/20 pt-2 flex flex-col gap-1">
              <span>• Loss: 0.8 L₁ + 0.2 (1 - SSIM)</span>
              <span>• Optimizer: AdamW (lr=2e-4)</span>
            </div>
          </div>

          {/* Node 3: Synthesized Output */}
          <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/50 flex flex-col gap-3 shadow-inner">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] font-bold text-emerald-400 uppercase tracking-wider">
                3. Direct Frame Output
              </span>
              <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-outline font-mono text-[10px]">
                t = 0.50
              </span>
            </div>
            <p className="font-body-md text-xs text-on-surface-variant/90 leading-relaxed">
              Synthesized brightness temperature radiance field without spatial warping artifacts or displacement tearing.
            </p>
            <div className="text-[10px] font-mono text-outline border-t border-outline-variant/20 pt-2">
              Pixel Output: [0.0, 1.0] Normalized
            </div>
          </div>

          {/* Node 4: Quality & Confidence */}
          <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/50 flex flex-col gap-3 shadow-inner">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] font-bold text-sky-400 uppercase tracking-wider">
                4. Quality &amp; SRI
              </span>
              <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-outline font-mono text-[10px]">
                Diagnostics
              </span>
            </div>
            <p className="font-body-md text-xs text-on-surface-variant/90 leading-relaxed">
              Learned confidence distribution map and Structural Reliability Index (SRI) gating operational readiness.
            </p>
            <div className="text-[10px] font-mono text-outline border-t border-outline-variant/20 pt-2">
              Confidence &amp; Sharpness Map
            </div>
          </div>
        </div>
      </div>

      {/* Section 2: Narrative & Difference from Warping */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-3">
          <div className="flex items-center gap-2 border-b border-outline-variant/30 pb-2">
            <span className="material-symbols-outlined text-primary-fixed text-[18px]">
              auto_fix_high
            </span>
            <h4 className="font-headline-sm text-xs font-bold text-primary-fixed uppercase tracking-wide">
              What This Generative Model Does
            </h4>
          </div>
          <p className="font-body-md text-xs text-on-surface-variant leading-relaxed">
            The direct-synthesis model is a conditional convolutional network designed to directly map temporal boundary states (T₀, T₁) and dense velocity representations (Farneback flow) to an intermediate cloud radiance field at t=0.50. Rather than treating interpolation solely as a geometric repositioning problem, the network learns an end-to-end generative representation trained on photometric (L1) and perceptual structural (SSIM) losses.
          </p>
          <div className="bg-surface-container-lowest p-3 rounded-lg border border-outline-variant/30 font-mono text-[11px] text-outline mt-auto">
            <span className="text-primary-fixed font-bold block mb-1">Architecture Statement:</span>
            This is a lightweight proof-of-concept direct-synthesis model, explicitly not the full diffusion architecture described in the project proposal, trained on a small patch dataset.
          </div>
        </div>

        <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-3">
          <div className="flex items-center gap-2 border-b border-outline-variant/30 pb-2">
            <span className="material-symbols-outlined text-secondary text-[18px]">
              compare_arrows
            </span>
            <h4 className="font-headline-sm text-xs font-bold text-secondary uppercase tracking-wide">
              How It Differs From Warping-Based Methods
            </h4>
          </div>
          <div className="space-y-2.5 font-body-md text-xs text-on-surface-variant leading-relaxed">
            <div className="flex items-start gap-2">
              <span className="text-rose-400 font-mono font-bold text-sm">✕</span>
              <div>
                <strong className="text-on-surface">Warping Methods (Farneback / RAFT):</strong> Assume strict optical intensity conservation along motion paths (<code>I(x) = I₀(x - v)</code>). When clouds cool, expand, or undergo convective updrafts, spatial warping produces shearing, ghosting, and boundary tears because pixels did not merely displace—they evolved thermodynamically.
              </div>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-emerald-400 font-mono font-bold text-sm">✓</span>
              <div>
                <strong className="text-on-surface">Direct Neural Synthesis:</strong> Employs flow purely as a conditioning feature. The network directly synthesizes intermediate pixel values from learned representations, allowing it to generate non-rigid cloud growth and dissipation patterns that pure geometric warping cannot represent.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Section 3: Status Checklist */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-outline-variant/30 pb-3">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary-container text-[20px]">
              fact_check
            </span>
            <h3 className="font-headline-sm text-sm font-bold text-primary uppercase tracking-wide">
              Implementation &amp; Milestone Status Checklist
            </h3>
          </div>
          <span className="font-mono text-[11px] text-outline">
            {demoData ? "3 of 5 Completed (POC Running) • 2 Planned for End-Semester" : "2 of 5 Completed • 3 Planned for End-Semester"}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {checklist.map((item, idx) => (
            <div
              key={idx}
              className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/30 flex flex-col justify-between gap-3 shadow-sm"
            >
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-primary">
                    {item.title}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${item.statusColor}`}
                  >
                    {item.status}
                  </span>
                </div>
                <p className="font-body-md text-[11px] text-on-surface-variant/80 leading-relaxed">
                  {item.desc}
                </p>
              </div>

              <div className="flex items-center gap-1.5 font-mono text-[10px] text-outline border-t border-outline-variant/20 pt-2">
                <span className="material-symbols-outlined text-[14px]">
                  {item.icon}
                </span>
                <span>Milestone {idx + 1}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Section 4: Real Proof-of-Concept Trained Model Outputs */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-6 shadow-lg flex flex-col gap-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-outline-variant/30 pb-3 gap-2">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-emerald-400 text-[22px]">
              deployed_code
            </span>
            <h3 className="font-headline-sm text-sm font-bold text-primary uppercase tracking-wide">
              Proof-of-Concept Direct-Synthesis Demonstration (Real Trained Model)
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-300 font-mono text-xs font-bold border border-emerald-500/40">
              {demoData ? `Model: ${demoData.model_type}` : "Connecting to Checkpoint..."}
            </span>
          </div>
        </div>

        {isLoading ? (
          <div className="py-12 flex flex-col items-center justify-center gap-3">
            <div className="w-8 h-8 border-2 border-primary-container border-t-transparent rounded-full animate-spin" />
            <span className="font-mono text-xs text-outline">Loading trained neural synthesis demo patches...</span>
          </div>
        ) : fetchError ? (
          <div className="p-4 rounded-xl bg-error-container/30 border border-error/40 text-error text-xs font-mono">
            Could not fetch neural synthesis samples: {fetchError}. Ensure FastAPI backend is running on port 8000.
          </div>
        ) : demoData?.samples && demoData.samples.length > 0 ? (
          <div className="flex flex-col gap-6">
            {/* Patch Selector Tabs */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs text-outline mr-2">Select Dataset Patch:</span>
              {demoData.samples.map((s, idx) => (
                <button
                  key={s.id}
                  onClick={() => setSelectedPatchIndex(idx)}
                  className={`px-3 py-1.5 rounded-lg font-mono text-xs transition-all cursor-pointer ${
                    selectedPatchIndex === idx
                      ? "bg-primary-container text-on-primary-container font-bold shadow-md"
                      : "bg-surface-container-high text-on-surface hover:bg-surface-container-highest border border-outline-variant/30"
                  }`}
                >
                  {s.patch_name || `Patch #${idx + 1}`}
                </button>
              ))}
            </div>

            {/* Real Image 5-Way Comparison Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              {/* T0 */}
              <div className="bg-surface-container-lowest p-3 rounded-xl border border-outline-variant/30 flex flex-col gap-2 shadow-sm">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-cyan-300">Observation T₀</span>
                  <span className="font-mono text-[10px] text-outline">18:00 UTC</span>
                </div>
                <div className="aspect-square bg-black rounded-lg overflow-hidden border border-outline-variant/20 flex items-center justify-center">
                  {currentSample?.images?.t0 ? (
                    <img
                      src={currentSample.images.t0}
                      alt="T0 Observation"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-xs text-outline font-mono">No Image</span>
                  )}
                </div>
                <span className="font-mono text-[10px] text-outline text-center">GOES-16 Clean IR (Band 13)</span>
              </div>

              {/* T1 */}
              <div className="bg-surface-container-lowest p-3 rounded-xl border border-outline-variant/30 flex flex-col gap-2 shadow-sm">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-sky-300">Observation T₁</span>
                  <span className="font-mono text-[10px] text-outline">18:15 UTC</span>
                </div>
                <div className="aspect-square bg-black rounded-lg overflow-hidden border border-outline-variant/20 flex items-center justify-center">
                  {currentSample?.images?.t1 ? (
                    <img
                      src={currentSample.images.t1}
                      alt="T1 Observation"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-xs text-outline font-mono">No Image</span>
                  )}
                </div>
                <span className="font-mono text-[10px] text-outline text-center">GOES-16 Clean IR (Band 13)</span>
              </div>

              {/* Ground Truth */}
              <div className="bg-surface-container-lowest p-3 rounded-xl border border-outline-variant/30 flex flex-col gap-2 shadow-sm">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-emerald-400">Ground Truth T₀.₅</span>
                  <span className="font-mono text-[10px] text-emerald-400/80">Held-Out Scan</span>
                </div>
                <div className="aspect-square bg-black rounded-lg overflow-hidden border border-emerald-500/30 flex items-center justify-center">
                  {currentSample?.images?.gt ? (
                    <img
                      src={currentSample.images.gt}
                      alt="Ground Truth"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-xs text-outline font-mono">No Image</span>
                  )}
                </div>
                <span className="font-mono text-[10px] text-outline text-center">NOAA Authentic 18:07:30 UTC</span>
              </div>

              {/* Direct Synthesis UNet Output */}
              <div className="bg-surface-container-lowest p-3 rounded-xl border-2 border-primary-container flex flex-col gap-2 shadow-md relative">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-primary-container">Synthesized T₀.₅</span>
                  <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-primary-container/20 text-primary-container font-bold">
                    Direct U-Net
                  </span>
                </div>
                <div className="aspect-square bg-black rounded-lg overflow-hidden border border-primary-container/40 flex items-center justify-center">
                  {currentSample?.images?.synth ? (
                    <img
                      src={currentSample.images.synth}
                      alt="Synthesized Intermediate Frame"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-xs text-outline font-mono">No Image</span>
                  )}
                </div>
                <span className="font-mono text-[10px] text-primary font-bold text-center">Non-Warped Neural Generation</span>
              </div>

              {/* Residual Map */}
              <div className="bg-surface-container-lowest p-3 rounded-xl border border-outline-variant/30 flex flex-col gap-2 shadow-sm">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-amber-400">L₁ Residual Heatmap</span>
                  <span className="font-mono text-[10px] text-outline">|Synth - GT|</span>
                </div>
                <div className="aspect-square bg-black rounded-lg overflow-hidden border border-outline-variant/20 flex items-center justify-center">
                  {currentSample?.images?.residual ? (
                    <img
                      src={currentSample.images.residual}
                      alt="Residual Error Heatmap"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-xs text-outline font-mono">No Image</span>
                  )}
                </div>
                <span className="font-mono text-[10px] text-outline text-center">Turbo Colormap Error Profile</span>
              </div>
            </div>

            {/* Metrics & Training Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Exact Computed Patch Metrics */}
              <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 flex flex-col gap-3">
                <div className="flex items-center justify-between border-b border-outline-variant/20 pb-2">
                  <span className="font-mono text-xs font-bold text-primary uppercase">
                    Computed Patch Metrics ({currentSample?.patch_name})
                  </span>
                  <span className="font-mono text-[10px] text-outline">Real Computation on Real Data</span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-lg bg-surface-container-high/40 border border-outline-variant/20 flex flex-col">
                    <span className="font-mono text-[10px] text-outline">MSE</span>
                    <span className="font-mono text-base font-bold text-on-surface">
                      {currentSample?.metrics?.mse ? currentSample.metrics.mse.toFixed(2) : "N/A"}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-surface-container-high/40 border border-outline-variant/20 flex flex-col">
                    <span className="font-mono text-[10px] text-outline">PSNR</span>
                    <span className="font-mono text-base font-bold text-emerald-400">
                      {currentSample?.metrics?.psnr_db ? `${currentSample.metrics.psnr_db.toFixed(2)} dB` : "N/A"}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-surface-container-high/40 border border-outline-variant/20 flex flex-col">
                    <span className="font-mono text-[10px] text-outline">SSIM</span>
                    <span className="font-mono text-base font-bold text-cyan-400">
                      {currentSample?.metrics?.ssim ? currentSample.metrics.ssim.toFixed(4) : "N/A"}
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-surface-container-high/40 border border-outline-variant/20 flex flex-col">
                    <span className="font-mono text-[10px] text-outline">MAE</span>
                    <span className="font-mono text-base font-bold text-amber-400">
                      {currentSample?.metrics?.mae ? currentSample.metrics.mae.toFixed(2) : "N/A"}
                    </span>
                  </div>
                </div>
                <p className="font-mono text-[11px] text-on-surface-variant/80 mt-1">
                  Evaluated against held-out ground truth patch observation without temporal interpolation cheating.
                </p>
              </div>

              {/* Training Loss Curve & Convergence Details */}
              <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 flex flex-col gap-3">
                <div className="flex items-center justify-between border-b border-outline-variant/20 pb-2">
                  <span className="font-mono text-xs font-bold text-primary uppercase">
                    Model Training &amp; Validation Convergence
                  </span>
                  <span className="font-mono text-[10px] text-emerald-400">
                    Best Epoch: {demoData.best_epoch} (Val Loss: {demoData.best_val_loss?.toFixed(4)})
                  </span>
                </div>
                {demoData.loss_curve_image ? (
                  <div className="rounded-lg overflow-hidden border border-outline-variant/20 max-h-[160px] flex items-center justify-center bg-black/40">
                    <img
                      src={demoData.loss_curve_image}
                      alt="Direct Synthesis Loss Curve"
                      className="w-full h-full object-contain"
                    />
                  </div>
                ) : (
                  <div className="h-32 flex items-center justify-center text-outline font-mono text-xs">
                    Loss curve plot saved to checkpoints/
                  </div>
                )}
                <div className="flex justify-between font-mono text-[10px] text-outline border-t border-outline-variant/20 pt-2">
                  <span>Epochs Trained: {demoData.trained_epochs}</span>
                  <span>Checkpoint: checkpoints/best_direct_synthesis_model.pth</span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="py-8 text-center text-outline font-mono text-xs">
            No demonstration patches cataloged.
          </div>
        )}
      </div>
    </div>
  );
}
