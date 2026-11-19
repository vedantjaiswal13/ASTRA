import React, { useState, useEffect, useRef, useCallback } from "react";
import SensorHeader from "./components/SensorHeader";
import FrameTrio from "./components/FrameTrio";
import GroundTruthPanel from "./components/GroundTruthPanel";
import MetricsPanel from "./components/MetricsPanel";
import ResidualAnalysisView from "./components/ResidualAnalysisView";
import BenchmarkLogsView from "./components/BenchmarkLogsView";
import NeuralSynthesisView from "./components/NeuralSynthesisView";

const API_BASE = "http://localhost:8000";

export default function App() {
  // Navigation tab state: "reconstruction" | "residual"
  const [activeTab, setActiveTab] = useState("reconstruction");

  // Method state: "linear" | "farneback" | "raft" | "raft_refined"
  const [method, setMethod] = useState("farneback");
  const [timeFraction, setTimeFraction] = useState(0.5);
  
  // Triplets catalog state
  const [triplets, setTriplets] = useState([]);
  const [selectedTripletId, setSelectedTripletId] = useState("2024-01-01_18");

  // Data states from real API responses (never hardcoded)
  const [sampleData, setSampleData] = useState(null);
  const [generatedData, setGeneratedData] = useState(null);

  // Loading & error states
  const [isLoadingSample, setIsLoadingSample] = useState(true);
  const [sampleError, setSampleError] = useState(null);

  const [isGenerating, setIsGenerating] = useState(false);
  const [generationError, setGenerationError] = useState(null);

  const [errorMessage, setErrorMessage] = useState(null);
  const [backendStatus, setBackendStatus] = useState({
    type: "loading",
    message: "Connecting to API...",
  });

  const [isPlaying, setIsPlaying] = useState(false);
  const playIntervalRef = useRef(null);
  const playDirectionRef = useRef(1);

  // 1. Fetch Sample Pair for a specific triplet_id
  const fetchSamplePair = useCallback(async (targetTripletId = selectedTripletId) => {
    setIsLoadingSample(true);
    setSampleError(null);
    setGeneratedData(null);
    setBackendStatus({ type: "loading", message: `Fetching scan pair (${targetTripletId})...` });

    try {
      const res = await fetch(`${API_BASE}/sample-pair?triplet_id=${encodeURIComponent(targetTripletId)}&preview_size=1024`);
      if (!res.ok) {
        const errDetail = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status}: ${errDetail || res.statusText}`);
      }

      const data = await res.json();
      setSampleData(data);
      setBackendStatus({ type: "online", message: `Connected (GOES-16 Ch. 13 • ${targetTripletId})` });
      setErrorMessage(null);

      // Automatically synthesize initial frame once sample data loads
      await generateFrame(timeFraction, method, targetTripletId);
    } catch (err) {
      console.error("Failed to fetch sample pair:", err);
      const msg = `Unable to connect to ASTRA Backend at ${API_BASE}: ${err.message}`;
      setSampleError(msg);
      setErrorMessage(msg);
      setBackendStatus({
        type: "error",
        message: "API Offline or Triplet Error",
      });
    } finally {
      setIsLoadingSample(false);
    }
  }, [timeFraction, method, selectedTripletId]);

  // 2. Generate Frame via POST /generate with triplet_id
  const generateFrame = async (
    targetFraction = timeFraction,
    targetMethod = method,
    targetTripletId = selectedTripletId
  ) => {
    if (isGenerating) return;
    setIsGenerating(true);
    setGenerationError(null);

    try {
      const payload = {
        triplet_id: targetTripletId,
        time_fraction: parseFloat(targetFraction),
        method: targetMethod,
        preview_size: 1024,
        pyr_scale: 0.5,
        levels: 3,
        winsize: 15,
        iterations: 3,
      };

      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errDetail = await res.text().catch(() => "");
        throw new Error(`Generation failed (HTTP ${res.status}): ${errDetail || res.statusText}`);
      }

      const data = await res.json();
      setGeneratedData(data);
      setGenerationError(null);
      setErrorMessage(null);
    } catch (err) {
      console.error("Frame generation failed:", err);
      const msg = err.message || "Failed to generate synthetic frame";
      setGenerationError(msg);
      setErrorMessage(msg);
    } finally {
      setIsGenerating(false);
    }
  };

  // Initial mount: load available triplets catalog then fetch initial sample scans
  useEffect(() => {
    const initData = async () => {
      try {
        const res = await fetch(`${API_BASE}/triplets`);
        if (res.ok) {
          const tData = await res.json();
          if (tData.triplets && tData.triplets.length > 0) {
            setTriplets(tData.triplets);
            const firstId = tData.triplets[0].id;
            setSelectedTripletId(firstId);
            await fetchSamplePair(firstId);
            return;
          }
        }
      } catch (err) {
        console.warn("Could not fetch triplets catalog:", err);
      }
      await fetchSamplePair("2024-01-01_18");
    };
    initData();
  }, []);

  // Triplet switch handler
  const handleTripletChange = async (newId) => {
    setSelectedTripletId(newId);
    setSampleData(null);
    setGeneratedData(null);
    await fetchSamplePair(newId);
  };

  // Method switch handler
  const handleMethodChange = (newMethod) => {
    setMethod(newMethod);
    generateFrame(timeFraction, newMethod, selectedTripletId);
  };

  // Stepper handler (-0.25 / +0.25)
  const handleStep = (delta) => {
    setTimeFraction((prev) => {
      const next = Math.max(0, Math.min(1, Math.round((prev + delta) * 100) / 100));
      generateFrame(next, method, selectedTripletId);
      return next;
    });
  };

  // Auto-play animation
  const handleTogglePlay = () => {
    if (isPlaying) {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
      playIntervalRef.current = null;
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      playIntervalRef.current = setInterval(() => {
        setTimeFraction((curr) => {
          let next = curr + 0.25 * playDirectionRef.current;
          if (next >= 1.0) {
            next = 1.0;
            playDirectionRef.current = -1;
          } else if (next <= 0.0) {
            next = 0.0;
            playDirectionRef.current = 1;
          }
          next = Math.round(next * 100) / 100;
          generateFrame(next, method);
          return next;
        });
      }, 1600);
    }
  };

  useEffect(() => {
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, []);

  // Active generated image from API response (never hardcoded)
  const activeImage =
    method === "linear"
      ? generatedData?.linear_image
      : generatedData?.warped_image;

  // Ground truth & confidence from API response
  const groundTruthImage = sampleData?.ground_truth_t05?.image;
  const confidenceImage = generatedData?.confidence_map_colored;

  return (
    <div className="bg-background font-body-md text-on-surface antialiased selection:bg-primary-container selection:text-on-primary-container min-h-screen">
      {/* Left Navigation Bar */}
      <aside className="fixed left-0 top-0 h-full w-64 bg-surface-container-lowest/95 backdrop-blur-xl z-50 flex flex-col border-r border-outline-variant/40">
        <div className="h-16 px-5 flex items-center justify-between border-b border-outline-variant/30">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded bg-surface-container-high flex items-center justify-center border border-primary-container/20">
              <span className="material-symbols-outlined text-primary-container text-[18px]">
                satellite_alt
              </span>
            </div>
            <div className="flex flex-col">
              <span className="font-headline-sm text-sm tracking-wider uppercase text-primary font-bold leading-none">
                ASTRA
              </span>
              <span className="font-label-sm text-[10px] tracking-widest text-on-surface-variant/70 leading-tight">
                RESEARCH WORKBENCH
              </span>
            </div>
          </div>
        </div>

        <div className="px-4 py-3 border-b border-outline-variant/20">
          <span className="font-label-sm text-[10px] uppercase tracking-wider text-outline block">
            Active Sensor Stream
          </span>
          <div className="mt-1.5 font-label-sm text-xs text-primary font-mono flex items-center gap-2">
            <span
              id="backend-status-dot"
              className={`w-2 h-2 rounded-full ${
                backendStatus.type === "online"
                  ? "bg-emerald-400"
                  : backendStatus.type === "loading"
                  ? "bg-amber-400 animate-pulse"
                  : "bg-rose-500"
              }`}
            />
            <span id="backend-status-text">{backendStatus.message}</span>
          </div>
        </div>

        {/* Dataset Triplet Selector */}
        {triplets && triplets.length > 0 && (
          <div className="px-4 py-3 border-b border-outline-variant/20">
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-label-sm text-[10px] uppercase tracking-wider text-outline block">
                Dataset Triplet
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-container-high text-primary font-mono">
                {triplets.length} loaded
              </span>
            </div>
            <select
              id="sidebar-triplet-selector"
              value={selectedTripletId}
              onChange={(e) => handleTripletChange(e.target.value)}
              className="w-full bg-surface-container-high border border-outline-variant/50 text-on-surface text-xs rounded px-2.5 py-1.5 font-mono focus:outline-none focus:border-primary-container cursor-pointer transition-colors"
            >
              {triplets.map((t) => (
                <option key={t.id} value={t.id} className="bg-surface-container-lowest text-on-surface">
                  {t.date} {t.hour !== undefined ? `${String(t.hour).padStart(2, '0')}:00 UTC` : t.id}
                </option>
              ))}
            </select>
          </div>
        )}

        <nav className="flex-1 px-3 py-4 space-y-1">
          <button
            onClick={() => setActiveTab("reconstruction")}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded font-label-md text-xs font-medium transition-colors cursor-pointer text-left ${
              activeTab === "reconstruction"
                ? "bg-primary-container/15 text-primary-container border border-primary-container/30"
                : "text-on-surface-variant hover:bg-surface-container-high/50 hover:text-on-surface border border-transparent"
            }`}
          >
            <span className="material-symbols-outlined text-[18px]">compare</span>
            Frame Reconstruction
          </button>

          <button
            onClick={() => setActiveTab("residual")}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded font-label-md text-xs font-medium transition-colors cursor-pointer text-left ${
              activeTab === "residual"
                ? "bg-primary-container/15 text-primary-container border border-primary-container/30"
                : "text-on-surface-variant hover:bg-surface-container-high/50 hover:text-on-surface border border-transparent"
            }`}
          >
            <span className="material-symbols-outlined text-[18px]">difference</span>
            Residual Analysis
          </button>

          <div
            className="w-full flex items-center justify-between px-3 py-2 rounded text-on-surface-variant/40 font-label-md text-xs select-none"
            title="Temporal sequence forecasting is not yet implemented in this release."
          >
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-[18px]">timeline</span>
              <span>Temporal Sequences</span>
            </div>
            <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-surface-container-high/60 text-outline">
              Not Implemented
            </span>
          </div>

          <button
            onClick={() => setActiveTab("benchmark")}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded font-label-md text-xs font-medium transition-colors cursor-pointer text-left ${
              activeTab === "benchmark"
                ? "bg-primary-container/15 text-primary-container border border-primary-container/30"
                : "text-on-surface-variant hover:bg-surface-container-high/50 hover:text-on-surface border border-transparent"
            }`}
          >
            <span className="material-symbols-outlined text-[18px]">query_stats</span>
            Benchmark Logs
          </button>

          <button
            onClick={() => setActiveTab("neural-synthesis")}
            className={`w-full flex items-center justify-between px-3 py-2 rounded font-label-md text-xs font-medium transition-colors cursor-pointer text-left ${
              activeTab === "neural-synthesis"
                ? "bg-amber-500/15 text-amber-300 border border-amber-500/40"
                : "text-on-surface-variant hover:bg-surface-container-high/50 hover:text-on-surface border border-transparent"
            }`}
          >
            <div className="flex items-center gap-2.5 truncate">
              <span className="material-symbols-outlined text-[18px] text-amber-400">neurology</span>
              <span className="truncate">Neural Synthesis</span>
            </div>
            <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono font-bold whitespace-nowrap ml-1">
              PLANNED - END-SEM
            </span>
          </button>
        </nav>

        <div className="p-4 border-t border-outline-variant/30 font-mono text-[11px] text-on-surface-variant/70 space-y-1">
          <div className="flex justify-between">
            <span>API Host:</span>
            <span className="text-primary-fixed">localhost:8000</span>
          </div>
          <div className="flex justify-between">
            <span>Channel:</span>
            <span className="text-secondary">Band 13 (Clean IR)</span>
          </div>
        </div>
      </aside>

      {/* Main Content Shell */}
      <div className="pl-64">
        {/* Top Navigation Bar with Method Selector & Generate Button */}
        <SensorHeader
          timestamp={sampleData?.frame_t0?.timestamp}
          method={method}
          onMethodChange={handleMethodChange}
          isGenerating={isGenerating}
          onGenerate={() => generateFrame(timeFraction, method, selectedTripletId)}
          triplets={triplets}
          selectedTripletId={selectedTripletId}
          onTripletChange={handleTripletChange}
        />

        <main className="relative pt-20 bg-background w-full min-h-screen p-6 flex flex-col gap-6">
          {/* Global Visible Error Banner */}
          {errorMessage && (
            <div className="flex items-center justify-between p-4 rounded-xl bg-error-container/40 border border-error/50 text-error backdrop-blur-md shadow-lg transition-all animate-fadeIn">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-error text-[22px]">
                  warning
                </span>
                <div className="font-mono text-xs">
                  <span className="font-bold uppercase tracking-wider block">API Error</span>
                  <span className="text-on-error-container">{errorMessage}</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    setErrorMessage(null);
                    fetchSamplePair();
                  }}
                  className="px-3 py-1 rounded bg-error/20 hover:bg-error/30 text-error border border-error/40 font-mono text-xs font-semibold cursor-pointer transition-colors"
                >
                  Retry
                </button>
                <button
                  onClick={() => setErrorMessage(null)}
                  className="p-1 rounded text-on-surface-variant hover:text-on-surface transition-colors cursor-pointer"
                  title="Dismiss"
                >
                  <span className="material-symbols-outlined text-[18px]">close</span>
                </button>
              </div>
            </div>
          )}

          {activeTab === "reconstruction" && (
            <>
              {/* Section 1: Main Three-Panel Temporal Trio Comparison */}
              <FrameTrio
                sampleData={sampleData}
                generatedData={generatedData}
                activeImage={activeImage}
                timeFraction={timeFraction}
                onTimeFractionChange={(val) => setTimeFraction(val)}
                onTimeFractionCommit={() => generateFrame(timeFraction, method)}
                method={method}
                isLoadingSample={isLoadingSample}
                sampleError={sampleError}
                isGenerating={isGenerating}
                generationError={generationError}
                isPlaying={isPlaying}
                onTogglePlay={handleTogglePlay}
                onStep={handleStep}
                onRefresh={fetchSamplePair}
              />

              {/* Section 2: Ground Truth Direct Verification & Error Residual Heatmap */}
              <GroundTruthPanel
                activeImage={activeImage}
                groundTruthImage={groundTruthImage}
                confidenceImage={confidenceImage}
                method={method}
                metrics={generatedData?.metrics}
              />

              {/* Section 3: Scientific Quantitative Metrics Panel */}
              <MetricsPanel
                metrics={generatedData?.metrics}
                dimensions={generatedData?.dimensions}
                method={method}
              />
            </>
          )}

          {activeTab === "residual" && (
            <ResidualAnalysisView
              activeImage={activeImage}
              groundTruthImage={groundTruthImage}
              method={method}
              onMethodChange={handleMethodChange}
              timeFraction={timeFraction}
              onGenerate={() => generateFrame(timeFraction, method, selectedTripletId)}
              isGenerating={isGenerating}
            />
          )}

          {activeTab === "benchmark" && (
            <BenchmarkLogsView
              activeMethod={method}
              onSelectMethod={(newMethod) => {
                setMethod(newMethod);
                generateFrame(timeFraction, newMethod, selectedTripletId);
              }}
              activeTripletId={selectedTripletId}
              availableTriplets={triplets}
              onSelectTriplet={handleTripletChange}
            />
          )}

          {activeTab === "neural-synthesis" && (
            <NeuralSynthesisView />
          )}
        </main>
      </div>
    </div>
  );
}
