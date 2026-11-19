import React, { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

export default function NoaaGoesModal({
  isOpen,
  onClose,
  triplets = [],
  selectedTripletId = "",
  onSelectTriplet = () => {},
}) {
  const [s3Status, setS3Status] = useState(null);
  const [isPinging, setIsPinging] = useState(false);
  const [queryDate, setQueryDate] = useState("2024-01-15");
  const [queryHour, setQueryHour] = useState(0);
  const [isQuerying, setIsQuerying] = useState(false);
  const [s3Files, setS3Files] = useState(null);
  const [queryError, setQueryError] = useState(null);
  const [isFetching, setIsFetching] = useState(false);
  const [fetchSuccessMsg, setFetchSuccessMsg] = useState(null);
  const [filterType, setFilterType] = useState("all");

  const pingS3 = async () => {
    setIsPinging(true);
    try {
      const res = await fetch(`${API_BASE}/goes/status`);
      if (res.ok) {
        const data = await res.json();
        setS3Status(data);
      }
    } catch (e) {
      console.warn("Failed to ping NOAA S3:", e);
    } finally {
      setIsPinging(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      pingS3();
    }
  }, [isOpen]);

  const handleQueryS3 = async () => {
    setIsQuerying(true);
    setQueryError(null);
    setS3Files(null);
    setFetchSuccessMsg(null);
    try {
      const res = await fetch(
        `${API_BASE}/goes/query-scans?date=${encodeURIComponent(queryDate)}&hour=${queryHour}`
      );
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setS3Files(data);
    } catch (err) {
      setQueryError(`Query failed: ${err.message}`);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleFetchTriplet = async () => {
    setIsFetching(true);
    setFetchSuccessMsg(null);
    setQueryError(null);
    try {
      const res = await fetch(`${API_BASE}/goes/fetch-triplet`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          date: queryDate,
          hour: parseInt(queryHour, 10),
          channel: 13,
        }),
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setFetchSuccessMsg(data.message || "Ingestion complete!");
      // Re-ping to refresh triplet count
      await pingS3();
    } catch (err) {
      setQueryError(`Fetch failed: ${err.message}`);
    } finally {
      setIsFetching(false);
    }
  };

  if (!isOpen) return null;

  const filteredTriplets = triplets.filter((t) => {
    if (filterType === "all") return true;
    if (filterType === "full_disk") return t.type?.includes("Full Disk");
    if (filterType === "roi") return t.type?.includes("Cloud Dynamics");
    return true;
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-surface border border-outline-variant/60 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-outline-variant/30 flex items-center justify-between bg-surface-container-lowest/80">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-surface-container-high flex items-center justify-center border border-cyan-500/40 text-cyan-400">
              <span className="material-symbols-outlined text-[24px]">satellite_alt</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="font-headline-sm text-base font-bold text-primary uppercase tracking-wide">
                  NOAA GOES-16 Dataset Connection
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                  PUBLIC S3 ARCHIVE
                </span>
              </div>
              <p className="text-on-surface-variant font-mono text-xs mt-0.5">
                Direct integration with NOAA ABI-L2-CMIPF Clean IR (Band 13 • 10.35 µm) Open Data
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface-variant hover:text-on-surface flex items-center justify-center transition-colors"
          >
            <span className="material-symbols-outlined text-[18px]">close</span>
          </button>
        </div>

        {/* Modal Content Scroll Area */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 custom-scrollbar">
          {/* S3 Live Connectivity Card */}
          <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="font-mono text-xs font-bold text-on-surface">
                  AWS S3: {s3Status?.bucket || "noaa-goes16"} (us-east-1)
                </span>
                <span className="text-[11px] font-mono text-emerald-400 font-semibold">
                  {s3Status?.status === "connected" ? "ONLINE • 200 OK" : "CONNECTING..."}
                </span>
              </div>
              <div className="text-[11px] font-mono text-on-surface-variant flex flex-wrap gap-x-4 gap-y-1">
                <span>Product: <strong className="text-primary-fixed">ABI-L2-CMIPF</strong></span>
                <span>Band: <strong className="text-primary-fixed">13 (10.35 µm Clean IR)</strong></span>
                <span>Native Resolution: <strong className="text-primary-fixed">5424 × 5424 px (2 km)</strong></span>
                {s3Status?.latency_ms && (
                  <span>Round-trip Latency: <strong className="text-emerald-400">{s3Status.latency_ms} ms</strong></span>
                )}
              </div>
            </div>

            <button
              onClick={pingS3}
              disabled={isPinging}
              className="px-3 py-1.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-primary font-mono text-xs border border-outline-variant/40 flex items-center gap-1.5 transition-all self-start md:self-auto cursor-pointer"
            >
              <span className={`material-symbols-outlined text-[15px] ${isPinging ? "animate-spin" : ""}`}>
                refresh
              </span>
              <span>{isPinging ? "Testing..." : "Test S3 Latency"}</span>
            </button>
          </div>

          {/* Section 1: Active Ingested Catalog Browser (51 Triplets) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-headline-sm text-xs font-bold text-primary uppercase tracking-wider">
                  Available GOES-16 Observation Triplets ({filteredTriplets.length} of {triplets.length} loaded)
                </h3>
                <p className="text-[11px] font-mono text-on-surface-variant">
                  Select any observation date to immediately load authentic GOES-16 radiance scans into workbench
                </p>
              </div>

              {/* Filter Tabs */}
              <div className="flex items-center gap-1 bg-surface-container-lowest p-1 rounded-lg border border-outline-variant/40 text-[11px] font-mono">
                <button
                  onClick={() => setFilterType("all")}
                  className={`px-2.5 py-1 rounded transition-colors ${
                    filterType === "all" ? "bg-primary-container text-on-primary-container font-bold" : "text-on-surface-variant"
                  }`}
                >
                  All ({triplets.length})
                </button>
                <button
                  onClick={() => setFilterType("full_disk")}
                  className={`px-2.5 py-1 rounded transition-colors ${
                    filterType === "full_disk" ? "bg-primary-container text-on-primary-container font-bold" : "text-on-surface-variant"
                  }`}
                >
                  Full Disk (2)
                </button>
                <button
                  onClick={() => setFilterType("roi")}
                  className={`px-2.5 py-1 rounded transition-colors ${
                    filterType === "roi" ? "bg-primary-container text-on-primary-container font-bold" : "text-on-surface-variant"
                  }`}
                >
                  Cloud ROIs (49)
                </button>
              </div>
            </div>

            {/* Triplet Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 max-h-56 overflow-y-auto p-1 bg-surface-container-lowest/50 rounded-xl border border-outline-variant/30 custom-scrollbar">
              {filteredTriplets.map((t) => {
                const isSelected = selectedTripletId === t.id;
                const isFullDisk = t.type?.includes("Full Disk");
                return (
                  <button
                    key={t.id}
                    onClick={() => {
                      onSelectTriplet(t.id);
                      onClose();
                    }}
                    className={`flex items-center gap-2.5 p-2 rounded-lg text-left transition-all border cursor-pointer ${
                      isSelected
                        ? "bg-primary-container/20 border-primary text-primary shadow-sm"
                        : "bg-surface-container-lowest hover:bg-surface-container border-outline-variant/30 text-on-surface"
                    }`}
                  >
                    {t.thumbnail ? (
                      <img
                        src={t.thumbnail}
                        alt={t.id}
                        className="w-10 h-10 rounded object-cover border border-outline-variant/40 bg-black flex-shrink-0"
                      />
                    ) : (
                      <div className="w-10 h-10 rounded bg-surface-container-high flex items-center justify-center flex-shrink-0 text-outline">
                        <span className="material-symbols-outlined text-[18px]">satellite</span>
                      </div>
                    )}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-xs font-bold truncate">
                          {t.date}
                        </span>
                        <span className="font-mono text-[10px] text-primary-fixed font-semibold">
                          {String(t.hour).padStart(2, "0")}:00 UTC
                        </span>
                      </div>
                      <div className="flex items-center gap-1 text-[10px] font-mono text-on-surface-variant truncate mt-0.5">
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            isFullDisk ? "bg-amber-400" : "bg-cyan-400"
                          }`}
                        />
                        <span>{isFullDisk ? "5424×5424 Full Disk" : "256×256 Cloud ROI"}</span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 2: Live NOAA S3 Scan Ingestion Tool */}
          <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/40 space-y-4">
            <div className="flex items-center justify-between border-b border-outline-variant/30 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-amber-400 text-[18px]">
                  cloud_download
                </span>
                <h3 className="font-headline-sm text-xs font-bold text-primary uppercase tracking-wider">
                  Live NOAA S3 Query & Ingestion
                </h3>
              </div>
              <span className="text-[11px] font-mono text-outline">
                Pull new observation scans directly from AWS
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
              <div>
                <label className="block text-[11px] font-mono text-on-surface-variant uppercase mb-1">
                  Observation Date
                </label>
                <input
                  type="date"
                  value={queryDate}
                  onChange={(e) => setQueryDate(e.target.value)}
                  className="w-full bg-surface-container border border-outline-variant/50 rounded px-2.5 py-1.5 text-xs font-mono text-on-surface focus:outline-none focus:border-primary"
                />
              </div>

              <div>
                <label className="block text-[11px] font-mono text-on-surface-variant uppercase mb-1">
                  UTC Hour (0-23)
                </label>
                <select
                  value={queryHour}
                  onChange={(e) => setQueryHour(parseInt(e.target.value, 10))}
                  className="w-full bg-surface-container border border-outline-variant/50 rounded px-2.5 py-1.5 text-xs font-mono text-on-surface focus:outline-none focus:border-primary"
                >
                  {Array.from({ length: 24 }).map((_, h) => (
                    <option key={h} value={h}>
                      {String(h).padStart(2, "0")}:00 UTC
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleQueryS3}
                  disabled={isQuerying}
                  className="flex-1 px-3 py-1.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-primary font-mono text-xs border border-outline-variant/40 flex items-center justify-center gap-1.5 transition-all cursor-pointer"
                >
                  <span className={`material-symbols-outlined text-[15px] ${isQuerying ? "animate-spin" : ""}`}>
                    search
                  </span>
                  <span>{isQuerying ? "Querying..." : "Query S3"}</span>
                </button>

                <button
                  onClick={handleFetchTriplet}
                  disabled={isFetching}
                  className="flex-1 px-3 py-1.5 rounded bg-primary text-on-primary font-mono text-xs font-bold hover:brightness-110 flex items-center justify-center gap-1.5 transition-all cursor-pointer shadow-sm"
                >
                  <span className={`material-symbols-outlined text-[15px] ${isFetching ? "animate-spin" : ""}`}>
                    download
                  </span>
                  <span>{isFetching ? "Ingesting..." : "Ingest Triplet"}</span>
                </button>
              </div>
            </div>

            {queryError && (
              <div className="p-2.5 rounded bg-rose-500/10 border border-rose-500/30 text-rose-300 font-mono text-xs">
                {queryError}
              </div>
            )}

            {fetchSuccessMsg && (
              <div className="p-2.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-mono text-xs flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px]">check_circle</span>
                <span>{fetchSuccessMsg}</span>
              </div>
            )}

            {s3Files && (
              <div className="p-3 bg-surface-container/60 rounded border border-outline-variant/30 space-y-1.5 font-mono text-xs">
                <div className="flex items-center justify-between text-on-surface font-semibold">
                  <span>Found {s3Files.total_files} Channel 13 NetCDF files in S3</span>
                  <span className="text-emerald-400">DOY {s3Files.day_of_year} • {String(s3Files.hour).padStart(2, "0")}:00 UTC</span>
                </div>
                <div className="max-h-24 overflow-y-auto space-y-1 text-[11px] text-on-surface-variant">
                  {s3Files.files.map((fname, idx) => (
                    <div key={idx} className="truncate">
                      • {fname}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-outline-variant/30 bg-surface-container-lowest/80 flex items-center justify-between">
          <span className="text-[11px] font-mono text-on-surface-variant">
            Source: NOAA Open Data Dissemination (NODD) • NOAA-GOES16 AWS Public Dataset
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded bg-surface-container-high hover:bg-surface-container-highest text-on-surface text-xs font-mono font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
