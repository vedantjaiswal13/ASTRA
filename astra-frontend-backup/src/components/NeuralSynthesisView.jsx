import React from "react";

export default function NeuralSynthesisView() {
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
      status: "IN PROGRESS / PLANNED",
      statusColor: "bg-amber-500/20 text-amber-400 border-amber-500/40",
      icon: "pending",
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

  return (
    <div className="flex flex-col gap-6 animate-fadeIn">
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
                ★ PLANNED - END-SEM (not yet trained)
              </span>
            </div>
            <p className="text-on-surface-variant font-mono text-xs mt-1">
              End-to-end conditional generation: boundary observations + dense flow features → direct cloud frame synthesis
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs bg-surface-container-lowest px-3 py-1.5 rounded-lg border border-outline-variant/40 text-on-surface-variant">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
          <span>Architecture Formulation Phase</span>
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
              Concatenated along channel dimension
            </div>
          </div>

          {/* Node 2: Arrow Connector */}
          <div className="hidden md:flex flex-col items-center justify-center text-outline">
            <span className="font-mono text-[10px] text-primary-container mb-1 tracking-wider uppercase font-semibold">
              Conditioning
            </span>
            <span className="material-symbols-outlined text-3xl text-primary-container animate-pulse">
              arrow_forward
            </span>
            <span className="font-mono text-[9px] text-outline mt-1">[B, 4, 256, 256]</span>
          </div>

          {/* Node 3: Generative U-Net Core */}
          <div className="bg-surface-container-lowest p-4 rounded-xl border-2 border-primary-container/60 flex flex-col gap-3 relative shadow-[0_0_20px_rgba(0,240,255,0.08)]">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] font-bold text-primary uppercase tracking-wider">
                2. Generative U-Net
              </span>
              <span className="px-1.5 py-0.5 rounded bg-primary-container/20 text-primary-container font-mono text-[10px] font-semibold">
                Direct Synthesis
              </span>
            </div>
            <div className="space-y-2 font-mono text-[11px] text-on-surface-variant">
              <div className="p-2 rounded bg-surface-container-low/80 border border-primary-container/20">
                <div className="text-primary font-bold text-xs mb-0.5">Encoder Path</div>
                <div className="text-[10px] text-outline">4 Contracting Conv blocks with BatchNorm + LeakyReLU (32 → 64 → 128 → 256)</div>
              </div>
              <div className="p-2 rounded bg-surface-container-low/80 border border-primary-container/20">
                <div className="text-secondary font-bold text-xs mb-0.5">Skip Connections &amp; Decoder</div>
                <div className="text-[10px] text-outline">Transposed convolutions preserving fine-scale cloud boundaries</div>
              </div>
            </div>
            <div className="text-[10px] font-mono text-primary-container/90 border-t border-outline-variant/30 pt-2 font-semibold">
              Direct pixel mapping (No remap / No warping)
            </div>
          </div>

          {/* Node 4: Outputs */}
          <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/50 flex flex-col gap-3 relative shadow-inner">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] font-bold text-secondary uppercase tracking-wider">
                3. Planned Outputs
              </span>
              <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-outline font-mono text-[10px]">
                Multi-Head
              </span>
            </div>
            <div className="space-y-2 font-mono text-xs">
              <div className="p-2 rounded bg-surface-container-low/70 border border-outline-variant/20 flex flex-col">
                <span className="text-primary font-semibold text-[11px]">Synthesized Frame (t=0.5)</span>
                <span className="text-[10px] text-outline mt-0.5">Direct intermediate intensity field</span>
              </div>
              <div className="p-2 rounded bg-surface-container-low/70 border border-outline-variant/20 flex flex-col">
                <span className="text-secondary font-semibold text-[11px]">Learned Confidence Map</span>
                <span className="text-[10px] text-outline mt-0.5">Pixel-wise uncertainty prediction</span>
              </div>
              <div className="p-2 rounded bg-surface-container-low/70 border border-outline-variant/20 flex flex-col">
                <span className="text-amber-400 font-semibold text-[11px]">SRI Index Metric</span>
                <span className="text-[10px] text-outline mt-0.5">Structural Reliability Index score</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Section 2: Scientific Description & Distinction from Warping */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-5 shadow-lg flex flex-col gap-3">
          <div className="flex items-center gap-2 border-b border-outline-variant/30 pb-2">
            <span className="material-symbols-outlined text-primary text-[18px]">
              info
            </span>
            <h4 className="font-headline-sm text-xs font-bold text-primary uppercase tracking-wide">
              What This Generative Model Does
            </h4>
          </div>
          <p className="font-body-md text-xs text-on-surface-variant leading-relaxed">
            The planned direct-synthesis model is a lightweight conditional convolutional network designed to directly map temporal boundary states (T₀, T₁) and dense velocity representations (Farneback flow) to an intermediate cloud radiance field at t=0.50. Rather than treating interpolation solely as a geometric repositioning problem, the network learns an end-to-end generative representation trained on photometric (L1) and perceptual structural (SSIM) losses.
          </p>
          <div className="bg-surface-container-lowest p-3 rounded-lg border border-outline-variant/30 font-mono text-[11px] text-outline mt-auto">
            <span className="text-primary-fixed font-bold block mb-1">Architecture Scope:</span>
            Lightweight proof-of-concept direct-synthesis model, explicitly not the full latent diffusion architecture described in the project proposal, scoped for initial validation on a compact dataset.
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
            2 of 5 Completed • 3 Planned for End-Semester
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

      {/* Section 4: Model Results & Preview Area (Strictly Honest "No Results Yet") */}
      <div className="bg-surface/80 backdrop-blur-md rounded-xl border border-outline-variant/40 p-6 shadow-lg flex flex-col gap-5">
        <div className="flex items-center justify-between border-b border-outline-variant/30 pb-3">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-outline text-[20px]">
              lock_clock
            </span>
            <h3 className="font-headline-sm text-sm font-bold text-primary uppercase tracking-wide">
              Generative Model Outputs &amp; Quantitative Metrics
            </h3>
          </div>
          <span className="font-mono text-[11px] text-amber-400 font-semibold">
            Status: No results yet (Model not yet trained)
          </span>
        </div>

        {/* Triple Placeholder Panels showing honest unfilled state */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-surface-container-lowest/80 rounded-xl border-2 border-dashed border-outline-variant/40 p-8 flex flex-col items-center justify-center text-center gap-2 min-h-[220px]">
            <span className="material-symbols-outlined text-outline text-3xl">
              image_not_supported
            </span>
            <span className="font-mono text-xs font-bold text-on-surface">
              Synthesized Frame Preview
            </span>
            <span className="font-mono text-[11px] text-outline max-w-[200px]">
              No results yet — Direct-synthesis U-Net is scheduled for end-semester training.
            </span>
          </div>

          <div className="bg-surface-container-lowest/80 rounded-xl border-2 border-dashed border-outline-variant/40 p-8 flex flex-col items-center justify-center text-center gap-2 min-h-[220px]">
            <span className="material-symbols-outlined text-outline text-3xl">
              insights
            </span>
            <span className="font-mono text-xs font-bold text-on-surface">
              Validation Metrics (MSE, PSNR, SSIM)
            </span>
            <span className="font-mono text-[11px] text-outline max-w-[200px]">
              No results yet — Quantitative evaluation pending trained model checkpoint.
            </span>
          </div>

          <div className="bg-surface-container-lowest/80 rounded-xl border-2 border-dashed border-outline-variant/40 p-8 flex flex-col items-center justify-center text-center gap-2 min-h-[220px]">
            <span className="material-symbols-outlined text-outline text-3xl">
              blur_linear
            </span>
            <span className="font-mono text-xs font-bold text-on-surface">
              Learned Confidence &amp; SRI Map
            </span>
            <span className="font-mono text-[11px] text-outline max-w-[200px]">
              No results yet — Auxiliary confidence head uncalibrated prior to training.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
