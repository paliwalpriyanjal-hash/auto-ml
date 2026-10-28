import { useState, useEffect, useMemo } from "react";
import { useSearchParams, Link } from "react-router-dom";
import {
  HelpCircle,
  BarChart2,
  Layers,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Download,
  Info,
  Sliders,
  ArrowRight,
  TrendingUp,
  FileText,
  Search,
  ArrowDownUp,
  ShieldCheck,
  Cpu
} from "lucide-react";
import { getApiBase, formatMetric } from "../utils/api";

export default function Explainability() {
  const [searchParams] = useSearchParams();
  const preselectedModelId = searchParams.get("model");

  // Models state
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(preselectedModelId || "");
  const [modelDetails, setModelDetails] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Method state
  const [activeMethod, setActiveMethod] = useState("permutation"); // "permutation" | "shap"

  // Evaluation dataset state
  const [uploadedFile, setUploadedFile] = useState(null);
  const [parsedPreview, setParsedPreview] = useState(null);
  const [useSampleData, setUseSampleData] = useState(false);
  const [targetColumnOverride, setTargetColumnOverride] = useState("");

  // Permutation settings
  const [repeats, setRepeats] = useState(5);
  const [scoringMetric, setScoringMetric] = useState("");

  // SHAP settings
  const [shapMaxRows, setShapMaxRows] = useState(30);
  const [selectedClass, setSelectedClass] = useState("");
  const [selectedRowIndex, setSelectedRowIndex] = useState(0);

  // Execution & results
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [explanationResult, setExplanationResult] = useState(null);

  // Filter & sort for global view
  const [featureSearch, setFeatureSearch] = useState("");
  const [sortAscending, setSortAscending] = useState(false);

  // 1. Fetch available registered models
  useEffect(() => {
    let isMounted = true;
    const fetchModels = async () => {
      setModelsLoading(true);
      try {
        const base = getApiBase();
        const res = await fetch(`${base}/models?limit=50`);
        if (!res.ok) throw new Error("Failed to retrieve models");
        const data = await res.json();
        if (isMounted) {
          const list = data.models || [];
          setModels(list);
          if (!selectedModelId && list.length > 0) {
            setSelectedModelId(String(list[0].id));
          }
        }
      } catch (err) {
        if (isMounted) setError("Could not connect to model registry.");
      } finally {
        if (isMounted) setModelsLoading(false);
      }
    };
    fetchModels();
    return () => { isMounted = false; };
  }, []);

  // 2. Fetch selected model details and capabilities
  useEffect(() => {
    if (!selectedModelId) return;
    let isMounted = true;

    const loadCapabilities = async () => {
      setError(null);
      setExplanationResult(null);
      try {
        const base = getApiBase();
        // Model record
        const resModel = await fetch(`${base}/models/${selectedModelId}`);
        if (!resModel.ok) throw new Error("Could not load model details");
        const dataModel = await resModel.json();

        // Capabilities
        const resCap = await fetch(`${base}/models/${selectedModelId}/explainability/capabilities`);
        if (!resCap.ok) throw new Error("Could not load explainability capabilities");
        const dataCap = await resCap.json();

        if (isMounted) {
          setModelDetails(dataModel);
          setCapabilities(dataCap);
          setTargetColumnOverride(dataModel.target_column || "");
          setScoringMetric(dataCap.default_scoring || "");

          // Check if SHAP is supported; if not, force permutation
          const shapCap = dataCap.supported_methods?.find((m) => m.method === "shap");
          if (!shapCap?.supported && activeMethod === "shap") {
            setActiveMethod("permutation");
          }
        }
      } catch (err) {
        if (isMounted) setError(err.message);
      }
    };

    loadCapabilities();
    return () => { isMounted = false; };
  }, [selectedModelId]);

  // 3. Handle CSV file upload
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedFile(file);
    setUseSampleData(false);
    setError(null);

    // Parse header preview
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result;
      if (typeof text === "string") {
        const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
        if (lines.length > 0) {
          const headers = lines[0].split(",").map((h) => h.trim());
          const rowCount = Math.max(0, lines.length - 1);
          setParsedPreview({ headers, rowCount, filename: file.name });
        }
      }
    };
    reader.readAsText(file);
  };

  // 4. Run explanation
  const handleRunExplanation = async (e) => {
    e.preventDefault();
    if (!selectedModelId) return;

    setRunning(true);
    setError(null);
    setExplanationResult(null);

    try {
      const base = getApiBase();
      const endpoint = `${base}/models/${selectedModelId}/explainability/${activeMethod}`;

      let res;
      if (uploadedFile && !useSampleData) {
        // Multipart Form Data
        const formData = new FormData();
        formData.append("file", uploadedFile);
        if (targetColumnOverride) formData.append("target_column", targetColumnOverride);
        formData.append("dataset_source", "evaluation_data");

        if (activeMethod === "permutation") {
          formData.append("n_repeats", String(repeats));
          if (scoringMetric) formData.append("scoring", scoringMetric);
        } else {
          formData.append("max_rows", String(shapMaxRows));
        }

        res = await fetch(endpoint, {
          method: "POST",
          body: formData,
        });
      } else {
        // Quick synthetic evaluation sample matching model feature contract
        const expectedCols = modelDetails?.feature_columns || [];
        const sampleRows = [];
        for (let i = 0; i < 15; i++) {
          const row = {};
          expectedCols.forEach((col) => {
            row[col] = (i + 1) * 10;
          });
          if (targetColumnOverride) {
            row[targetColumnOverride] = modelDetails?.task_type === "classification" ? (i % 2 === 0 ? "0" : "1") : i * 50;
          }
          sampleRows.push(row);
        }

        const payload = {
          rows: sampleRows,
          target_column: targetColumnOverride,
          dataset_source: "training_data",
        };

        if (activeMethod === "permutation") {
          payload.n_repeats = repeats;
          if (scoringMetric) payload.scoring = scoringMetric;
        } else {
          payload.max_rows = shapMaxRows;
          if (selectedClass) payload.selected_class = selectedClass;
        }

        res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail?.message || data.detail || "Explainability execution failed.");
      }

      setExplanationResult(data);
      setSelectedRowIndex(0);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  // Export JSON
  const handleExportJSON = () => {
    if (!explanationResult) return;
    const blob = new Blob([JSON.stringify(explanationResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `autoveyra_explanation_${activeMethod}_model${selectedModelId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Export CSV
  const handleExportCSV = () => {
    if (!explanationResult) return;
    let csvContent = "";

    if (activeMethod === "permutation") {
      csvContent = "Rank,Feature,Importance_Mean,Importance_Std\n";
      (explanationResult.features || []).forEach((f) => {
        csvContent += `${f.rank},"${f.feature}",${f.importance_mean},${f.importance_std}\n`;
      });
    } else {
      csvContent = "Rank,Feature,Mean_Absolute_SHAP\n";
      (explanationResult.global_importance || []).forEach((f) => {
        csvContent += `${f.rank},"${f.feature}",${f.mean_abs_shap}\n`;
      });
    }

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `autoveyra_importance_${activeMethod}_model${selectedModelId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Processed global features for display
  const displayedFeatures = useMemo(() => {
    if (!explanationResult) return [];

    let list = [];
    if (activeMethod === "permutation") {
      list = [...(explanationResult.features || [])];
      if (featureSearch.trim()) {
        list = list.filter((item) => item.feature.toLowerCase().includes(featureSearch.toLowerCase()));
      }
      list.sort((a, b) => sortAscending ? a.importance_mean - b.importance_mean : b.importance_mean - a.importance_mean);
    } else {
      list = [...(explanationResult.global_importance || [])];
      if (featureSearch.trim()) {
        list = list.filter((item) => item.feature.toLowerCase().includes(featureSearch.toLowerCase()));
      }
      list.sort((a, b) => sortAscending ? a.mean_abs_shap - b.mean_abs_shap : b.mean_abs_shap - a.mean_abs_shap);
    }
    return list;
  }, [explanationResult, activeMethod, featureSearch, sortAscending]);

  // Max value for relative bar widths
  const maxImportanceVal = useMemo(() => {
    if (displayedFeatures.length === 0) return 1;
    if (activeMethod === "permutation") {
      const vals = displayedFeatures.map((f) => Math.abs(f.importance_mean || 0));
      return Math.max(...vals, 0.0001);
    } else {
      const vals = displayedFeatures.map((f) => f.mean_abs_shap || 0);
      return Math.max(...vals, 0.0001);
    }
  }, [displayedFeatures, activeMethod]);

  const shapCap = capabilities?.supported_methods?.find((m) => m.method === "shap");
  const isShapSupported = Boolean(shapCap?.supported);

  return (
    <div className="min-h-screen pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.08] pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[#8265FF] bg-[#8265FF]/10 px-2 py-0.5 rounded-full border border-[#8265FF]/20 flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5" />
              Explainable AI Workspace
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Model Explainability & Auditing</h1>
          <p className="text-sm text-[#A4A4B8] mt-1">
            Permutation importance, game-theoretic SHAP attributions, and local prediction waterfalls.
          </p>
        </div>

        {explanationResult && (
          <div className="flex items-center gap-2">
            <button
              onClick={handleExportJSON}
              className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] flex items-center gap-1.5 transition-all shadow-sm"
            >
              <Download className="w-3.5 h-3.5 text-[#8265FF]" />
              Export JSON
            </button>
            <button
              onClick={handleExportCSV}
              className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] flex items-center gap-1.5 transition-all shadow-sm"
            >
              <FileText className="w-3.5 h-3.5 text-emerald-400" />
              Export CSV
            </button>
          </div>
        )}
      </div>

      {/* Model Selection & Methodology Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Card 1: Target Model */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-5 glow-purple space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold text-sm border-b border-white/[0.06] pb-3">
            <Cpu className="w-4 h-4 text-[#8265FF]" />
            <h3>1. Select Model</h3>
          </div>

          {modelsLoading ? (
            <div className="flex items-center gap-2 text-xs text-[#A4A4B8] py-4">
              <RefreshCw className="w-4 h-4 animate-spin text-[#8265FF]" />
              Loading registered models...
            </div>
          ) : models.length === 0 ? (
            <div className="text-center py-6 space-y-3">
              <p className="text-xs text-[#A4A4B8]">No trained models found in registry.</p>
              <Link
                to="/automl"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-[#8265FF] text-white hover:bg-[#7152f8]"
              >
                Train a Model First <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <label className="text-xs text-[#A4A4B8] font-medium block mb-1.5">Registered Model</label>
                <select
                  value={selectedModelId}
                  onChange={(e) => setSelectedModelId(e.target.value)}
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      #{m.id} — {m.model_name} ({m.model_type})
                    </option>
                  ))}
                </select>
              </div>

              {modelDetails && (
                <div className="p-3 rounded-xl bg-[#171630]/60 border border-white/[0.04] space-y-1.5 text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-[#A4A4B8]">Task Type:</span>
                    <span className="text-white font-semibold uppercase tracking-wider">{modelDetails.task_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A4A4B8]">Target Column:</span>
                    <span className="text-[#8265FF] font-semibold">{modelDetails.target_column || "—"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#A4A4B8]">Feature Columns:</span>
                    <span className="text-white font-mono">{modelDetails.feature_columns?.length || 0} features</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Card 2: Method Selector */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-5 glow-purple space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold text-sm border-b border-white/[0.06] pb-3">
            <Sliders className="w-4 h-4 text-[#8265FF]" />
            <h3>2. Explainability Method</h3>
          </div>

          <div className="space-y-2.5">
            <button
              type="button"
              onClick={() => setActiveMethod("permutation")}
              className={`w-full p-3 rounded-xl border text-left transition-all flex items-start gap-3 ${
                activeMethod === "permutation"
                  ? "bg-[#8265FF]/15 border-[#8265FF] text-white"
                  : "bg-[#171630]/60 border-white/[0.06] text-[#A4A4B8] hover:text-white"
              }`}
            >
              <div className={`p-1.5 rounded-lg mt-0.5 ${activeMethod === "permutation" ? "bg-[#8265FF] text-white" : "bg-white/[0.05]"}`}>
                <BarChart2 className="w-4 h-4" />
              </div>
              <div>
                <span className="text-xs font-bold block">Permutation Feature Importance</span>
                <span className="text-[11px] text-[#A4A4B8] leading-tight block mt-0.5">
                  Model-agnostic baseline measuring score degradation when features are shuffled.
                </span>
              </div>
            </button>

            <button
              type="button"
              onClick={() => isShapSupported && setActiveMethod("shap")}
              disabled={!isShapSupported}
              className={`w-full p-3 rounded-xl border text-left transition-all flex items-start gap-3 ${
                !isShapSupported
                  ? "opacity-40 cursor-not-allowed bg-[#171630]/30 border-white/[0.04] text-[#A4A4B8]"
                  : activeMethod === "shap"
                  ? "bg-[#8265FF]/15 border-[#8265FF] text-white"
                  : "bg-[#171630]/60 border-white/[0.06] text-[#A4A4B8] hover:text-white"
              }`}
            >
              <div className={`p-1.5 rounded-lg mt-0.5 ${activeMethod === "shap" ? "bg-[#8265FF] text-white" : "bg-white/[0.05]"}`}>
                <TrendingUp className="w-4 h-4" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold">SHAP Attributions</span>
                  {isShapSupported ? (
                    <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-semibold">
                      Supported
                    </span>
                  ) : (
                    <span className="text-[9px] uppercase px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 font-semibold">
                      Unsupported
                    </span>
                  )}
                </div>
                <span className="text-[11px] text-[#A4A4B8] leading-tight block mt-0.5">
                  Game-theoretic local feature contributions and waterfall explanations.
                </span>
              </div>
            </button>
          </div>
        </div>

        {/* Card 3: Evaluation Dataset & Parameters */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-5 glow-purple space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold text-sm border-b border-white/[0.06] pb-3">
            <UploadCloud className="w-4 h-4 text-[#8265FF]" />
            <h3>3. Evaluation Dataset</h3>
          </div>

          <div className="space-y-3">
            {/* Upload Area */}
            <div>
              <label className="text-xs text-[#A4A4B8] font-medium block mb-1">
                Evaluation CSV (Holdout Recommended)
              </label>
              <label className="flex flex-col items-center justify-center p-3.5 border-2 border-dashed border-white/[0.12] hover:border-[#8265FF]/60 rounded-xl cursor-pointer bg-[#171630]/40 transition-all text-center">
                <UploadCloud className="w-5 h-5 text-[#8265FF] mb-1" />
                <span className="text-xs text-white font-medium">
                  {uploadedFile ? uploadedFile.name : "Select or Drop CSV"}
                </span>
                <span className="text-[10px] text-[#A4A4B8]">
                  {parsedPreview ? `${parsedPreview.rowCount} rows detected` : "Matches model feature contract"}
                </span>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>
            </div>

            {/* Quick Sample Fallback Option */}
            <div className="flex items-center justify-between text-xs pt-1">
              <span className="text-[#A4A4B8]">Or run on sample data:</span>
              <button
                type="button"
                onClick={() => {
                  setUseSampleData(!useSampleData);
                  setUploadedFile(null);
                  setParsedPreview(null);
                }}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold border transition-all ${
                  useSampleData
                    ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
                    : "bg-[#171630] text-[#A4A4B8] border-white/[0.06] hover:text-white"
                }`}
              >
                {useSampleData ? "Using Model Samples" : "Use Sample Rows"}
              </button>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunExplanation}
              disabled={running || (!uploadedFile && !useSampleData)}
              className="w-full mt-2 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {running ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Computing {activeMethod === "permutation" ? "Permutation" : "SHAP"} Attributions...
                </>
              ) : (
                <>
                  <BarChart2 className="w-3.5 h-3.5" />
                  Run {activeMethod === "permutation" ? "Permutation Importance" : "SHAP Analysis"}
                </>
              )}
            </button>
          </div>
        </div>

      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-start gap-3 text-rose-300 text-xs">
          <AlertCircle className="w-4 h-4 mt-0.5 text-rose-400 shrink-0" />
          <div className="space-y-1">
            <span className="font-semibold block">Explainability Request Failed</span>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Results Section */}
      {explanationResult && (
        <div className="space-y-8 animate-fadeIn">
          
          {/* Audit & Limitations Banner */}
          <div className="p-4 rounded-2xl bg-[#111126] border border-white/[0.08] flex flex-col md:flex-row md:items-center justify-between gap-4 text-xs">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded font-mono text-[10px] uppercase font-bold bg-[#8265FF]/20 text-[#8265FF] border border-[#8265FF]/30">
                  {explanationResult.method} Analysis
                </span>
                <span className="text-white font-semibold">
                  Evaluated {explanationResult.rows_evaluated} Rows
                </span>
                <span className="text-[#A4A4B8]">
                  • Source: <span className="text-white font-medium capitalize">{explanationResult.dataset_source?.replace("_", " ")}</span>
                </span>
              </div>
              <p className="text-[11px] text-[#A4A4B8]">
                {explanationResult.warnings?.[0]}
              </p>
            </div>

            <div className="flex items-center gap-4 text-[#A4A4B8] font-mono text-[11px] shrink-0">
              <div>
                <span>Duration: </span>
                <span className="text-emerald-400 font-bold">{explanationResult.duration_seconds}s</span>
              </div>
              {explanationResult.scoring_metric && (
                <div>
                  <span>Metric: </span>
                  <span className="text-white font-bold">{explanationResult.scoring_metric}</span>
                </div>
              )}
            </div>
          </div>

          {/* Global Feature Importance Chart */}
          <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 glow-purple space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
              <div>
                <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
                  <BarChart2 className="w-5 h-5 text-[#8265FF]" />
                  Global Feature Importance Ranking
                </h2>
                <p className="text-xs text-[#A4A4B8] mt-0.5">
                  {activeMethod === "permutation"
                    ? "Mean metric decrease when each feature column is randomly shuffled across repeats."
                    : "Mean absolute SHAP value (|SHAP|) quantifying overall contribution across all evaluated samples."}
                </p>
              </div>

              {/* Filter & Sort Controls */}
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 text-[#A4A4B8] absolute left-3 top-2.5" />
                  <input
                    type="text"
                    placeholder="Search features..."
                    value={featureSearch}
                    onChange={(e) => setFeatureSearch(e.target.value)}
                    className="pl-8 pr-3 py-1.5 rounded-xl bg-[#171630] border border-white/[0.08] text-xs text-white focus:outline-none focus:border-[#8265FF] w-40"
                  />
                </div>
                <button
                  onClick={() => setSortAscending(!sortAscending)}
                  className="px-2.5 py-1.5 rounded-xl bg-[#171630] border border-white/[0.08] text-xs text-[#A4A4B8] hover:text-white flex items-center gap-1.5"
                >
                  <ArrowDownUp className="w-3.5 h-3.5" />
                  {sortAscending ? "Lowest First" : "Top Features"}
                </button>
              </div>
            </div>

            {/* Feature Bars */}
            <div className="space-y-3.5">
              {displayedFeatures.map((f, i) => {
                const val = activeMethod === "permutation" ? f.importance_mean : f.mean_abs_shap;
                const isNegative = val < 0;
                const barWidthPct = Math.min(100, Math.max(3, (Math.abs(val) / maxImportanceVal) * 100));

                return (
                  <div key={f.feature} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-6 text-[10px] font-mono text-[#A4A4B8]">#{f.rank || i + 1}</span>
                        <span className="font-semibold text-white">{f.feature}</span>
                      </div>
                      <div className="flex items-center gap-2 font-mono">
                        <span className={isNegative ? "text-amber-400 font-bold" : "text-emerald-400 font-bold"}>
                          {isNegative ? "" : "+"}{formatMetric(val, 5)}
                        </span>
                        {f.importance_std !== undefined && (
                          <span className="text-[10px] text-[#A4A4B8] px-1.5 py-0.2 rounded bg-white/[0.04]">
                            ±{formatMetric(f.importance_std, 4)}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Progress Bar Container */}
                    <div className="h-3 w-full bg-[#171630] rounded-full overflow-hidden p-0.5 border border-white/[0.04]">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isNegative
                            ? "bg-gradient-to-r from-amber-500 to-amber-400"
                            : "bg-gradient-to-r from-[#8265FF] via-[#9B7DFF] to-[#A855F7]"
                        }`}
                        style={{ width: `${barWidthPct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Explanatory Callout */}
            <div className="p-3.5 rounded-xl bg-[#171630]/40 border border-white/[0.06] flex items-start gap-3 text-[11px] text-[#A4A4B8]">
              <Info className="w-4 h-4 text-[#8265FF] shrink-0 mt-0.5" />
              <span>
                <strong>Interpretation Guide:</strong> Features with the highest positive importance represent the primary predictive drivers. Negative importance values indicate features whose randomization slightly improved performance, typically identifying uninformative noise.
              </span>
            </div>
          </div>

          {/* Local Row Waterfall (SHAP only) */}
          {activeMethod === "shap" && explanationResult.local_explanations?.length > 0 && (
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 glow-purple space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-emerald-400" />
                    Local Prediction Explanation (SHAP Waterfall)
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Deconstructs an individual prediction row into positive and negative feature pushes.
                  </p>
                </div>

                {/* Row Selector Tabs */}
                <div className="flex items-center gap-1.5 bg-[#171630] p-1 rounded-xl border border-white/[0.08]">
                  {explanationResult.local_explanations.map((loc, idx) => (
                    <button
                      key={loc.row_index}
                      onClick={() => setSelectedRowIndex(idx)}
                      className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                        selectedRowIndex === idx
                          ? "bg-[#8265FF] text-white shadow"
                          : "text-[#A4A4B8] hover:text-white"
                      }`}
                    >
                      Row #{loc.row_index}
                    </button>
                  ))}
                </div>
              </div>

              {/* Selected Row Detail Banner */}
              {(() => {
                const row = explanationResult.local_explanations[selectedRowIndex] || explanationResult.local_explanations[0];
                if (!row) return null;

                return (
                  <div className="space-y-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4 rounded-xl bg-[#171630]/60 border border-white/[0.04]">
                      <div>
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Model Prediction</span>
                        <span className="text-base font-bold font-mono text-emerald-400">
                          {String(row.prediction)}
                          {row.prediction_probability !== null && (
                            <span className="text-xs text-[#A4A4B8] font-normal ml-1.5">
                              ({(row.prediction_probability * 100).toFixed(1)}%)
                            </span>
                          )}
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Actual Target</span>
                        <span className="text-base font-bold font-mono text-white">
                          {row.actual_target !== null ? String(row.actual_target) : "Not Supplied"}
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Base Value E[f(x)]</span>
                        <span className="text-base font-bold font-mono text-[#8265FF]">
                          {formatMetric(row.base_value, 4)}
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Net SHAP Shift</span>
                        <span className="text-base font-bold font-mono text-white">
                          {formatMetric(row.sum_shap_values, 4)}
                        </span>
                      </div>
                    </div>

                    {/* Waterfall Contribution Rows */}
                    <div className="space-y-3">
                      <h4 className="text-xs font-semibold text-white uppercase tracking-wider">Feature Attributions</h4>
                      <div className="space-y-2">
                        {row.contributions?.map((c) => {
                          const isPos = c.shap_value > 0;
                          return (
                            <div
                              key={c.feature}
                              className="p-3 rounded-xl bg-[#171630]/40 border border-white/[0.04] flex items-center justify-between text-xs"
                            >
                              <div className="space-y-0.5">
                                <div className="flex items-center gap-2">
                                  <span className="font-bold text-white">{c.feature}</span>
                                  <span className="text-[11px] text-[#A4A4B8] font-mono">
                                    = {String(c.original_value)}
                                  </span>
                                </div>
                              </div>

                              <div className="flex items-center gap-3">
                                <span className={`text-[10px] px-2 py-0.5 rounded font-semibold ${
                                  isPos ? "bg-emerald-500/15 text-emerald-400" : "bg-rose-500/15 text-rose-400"
                                }`}>
                                  {isPos ? "Pushes Up (+)" : "Pulls Down (-)"}
                                </span>
                                <span className={`font-mono font-bold ${isPos ? "text-emerald-400" : "text-rose-400"}`}>
                                  {isPos ? "+" : ""}{formatMetric(c.shap_value, 5)}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

        </div>
      )}

    </div>
  );
}
