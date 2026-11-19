import React from "react";

export default function SensorHeader({
  timestamp,
  method,
  onMethodChange,
  isGenerating,
  onGenerate,
  triplets = [],
  selectedTripletId = "",
  onTripletChange = () => {},
  onOpenGoesModal = () => {},
}) {
  const methods = [
    { id: "linear", label: "Linear" },
    { id: "farneback", label: "Farneback" },
    { id: "raft", label: "RAFT" },
    { id: "raft_refined", label: "RAFT + Refinement" },
    { id: "neural_synthesis", label: "Neural Synthesis (POC)" },
  ];

  const formatTimestamp = (isoStr) => {
    if (!isoStr) return "Loading...";
    try {
      const d = new Date(isoStr);
      return isNaN(d.getTime()) ? isoStr : d.toUTCString().replace("GMT", "UTC");
    } catch {
      return isoStr;
    }
  };

  return (
    <header className="fixed top-0 left-64 right-0 h-16 bg-surface-container-lowest/90 backdrop-blur-xl z-40 flex items-center justify-between px-6 border-b border-outline-variant/40">
      <div className="flex items-center gap-4">
        {/* Sensor info & NOAA live connector */}
        <button
          onClick={onOpenGoesModal}
          className="flex items-center gap-2 bg-surface-container/70 border border-emerald-500/40 hover:bg-emerald-500/10 px-3 py-1.5 rounded text-left transition-colors cursor-pointer group"
          title="Click to view NOAA GOES-16 S3 Connection & Ingestion Center"
        >
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-primary font-label-md text-xs font-mono font-semibold group-hover:text-emerald-300 transition-colors">
            NOAA GOES-16 ABI / Band 13
          </span>
          <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            S3 LIVE
          </span>
        </button>

        <div className="flex items-center gap-2 font-label-md text-xs bg-surface-container/70 border border-outline-variant/40 px-3 py-1.5 rounded text-on-surface font-mono">
          <span className="material-symbols-outlined text-secondary text-[15px]">
            schedule
          </span>
          <span className="text-on-surface-variant">SCAN UTC:</span>
          <span id="top-timestamp" className="font-semibold text-primary-fixed">
            {formatTimestamp(timestamp)}
          </span>
        </div>

        {triplets && triplets.length > 0 && (
          <div className="flex items-center gap-2 font-label-md text-xs bg-surface-container/70 border border-outline-variant/40 px-3 py-1.5 rounded text-on-surface font-mono">
            <span className="material-symbols-outlined text-amber-400 text-[15px]">
              dataset
            </span>
            <span className="text-on-surface-variant text-[11px] uppercase">TRIPLET:</span>
            <select
              id="top-triplet-selector"
              value={selectedTripletId}
              onChange={(e) => onTripletChange(e.target.value)}
              className="bg-transparent text-primary-fixed font-bold focus:outline-none cursor-pointer text-xs max-w-[200px] truncate"
            >
              {triplets.map((t) => {
                const isFullDisk = t.type?.includes("Full Disk");
                return (
                  <option key={t.id} value={t.id} className="bg-surface-container-lowest text-on-surface">
                    {t.date} {t.hour !== undefined ? `${String(t.hour).padStart(2, '0')}:00 UTC` : t.id} ({isFullDisk ? "Full Disk" : "ROI"})
                  </option>
                );
              })}
            </select>
            <button
              onClick={onOpenGoesModal}
              className="px-1.5 py-0.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-[10px] text-outline hover:text-primary transition-colors cursor-pointer"
              title="Browse all 51 GOES-16 triplets"
            >
              All ({triplets.length})
            </button>
          </div>
        )}
      </div>

      {/* Mode & Workbench Settings Controls */}
      <div className="flex items-center gap-4">
        {/* Interpolation Algorithm Mode Selector */}
        <div className="flex items-center gap-1 bg-surface-container-lowest border border-outline-variant/50 p-1 rounded">
          <span className="font-label-sm text-[10px] uppercase text-outline px-2">
            METHOD:
          </span>
          {methods.map((m) => {
            const isActive = method === m.id;
            return (
              <button
                key={m.id}
                id={`btn-method-${m.id}`}
                onClick={() => onMethodChange(m.id)}
                className={`px-2.5 py-1 rounded font-label-md text-xs transition-all ${
                  isActive
                    ? "bg-primary-container text-on-primary-container font-semibold shadow-sm"
                    : "text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high"
                }`}
              >
                {m.label}
              </button>
            );
          })}
        </div>

        {/* Action Button: Generate */}
        <button
          id="btn-generate"
          onClick={onGenerate}
          disabled={isGenerating}
          className="flex items-center gap-2 bg-primary-container text-on-primary-container hover:brightness-110 disabled:opacity-60 px-4 py-1.5 rounded font-label-md text-xs font-bold shadow-md transition-all cursor-pointer"
        >
          <span
            id="generate-icon"
            className={`material-symbols-outlined text-[16px] ${
              isGenerating ? "animate-spin" : ""
            }`}
          >
            {isGenerating ? "sync" : "play_arrow"}
          </span>
          <span id="generate-text">
            {isGenerating ? "Processing..." : "Generate Frame"}
          </span>
        </button>
      </div>
    </header>
  );
}
