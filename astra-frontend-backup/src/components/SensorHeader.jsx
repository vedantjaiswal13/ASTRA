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
}) {
  const methods = [
    { id: "linear", label: "Linear" },
    { id: "farneback", label: "Farneback" },
    { id: "raft", label: "RAFT" },
    { id: "raft_refined", label: "RAFT + Refinement" },
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
      <div className="flex items-center gap-5">
        <div className="flex items-center gap-2 bg-surface-container/70 border border-outline-variant/40 px-3 py-1.5 rounded">
          <span className="font-label-sm text-[11px] text-on-surface-variant uppercase tracking-wide">
            SENSOR:
          </span>
          <span className="text-primary font-label-md text-xs font-mono font-semibold">
            NOAA GOES-16 ABI / Band 13 (10.35 µm)
          </span>
        </div>
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
              className="bg-transparent text-primary-fixed font-bold focus:outline-none cursor-pointer text-xs"
            >
              {triplets.map((t) => (
                <option key={t.id} value={t.id} className="bg-surface-container-lowest text-on-surface">
                  {t.date} {t.hour !== undefined ? `${String(t.hour).padStart(2, '0')}:00 UTC` : t.id}
                </option>
              ))}
            </select>
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
