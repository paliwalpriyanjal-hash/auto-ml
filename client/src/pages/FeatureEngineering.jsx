import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Wand2,
  Sparkles,
  Cpu,
  Layers,
  PlayCircle,
  ShieldCheck,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  ArrowRight,
  RefreshCw,
  Upload,
  FileText,
  AlertTriangle,
  Clock,
  Database,
  ChevronRight,
  XCircle,
  BarChart3,
  Filter,
  Trash2,
  Sliders,
  Calendar,
  Zap,
  Scissors,
  Check,
  AlertCircle,
  GitCompare,
  Search,
  Code,
  HelpCircle,
  Activity,
  Info
} from "lucide-react";
import { getApiBase } from "../utils/api";
import PageHeader from "../components/ui/PageHeader";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Button from "../components/ui/Button";

export default function FeatureEngineering() {
  const navigate = useNavigate();

  // Mode: 'auto' (Beginner-friendly) vs 'advanced'
  const [activeMode, setActiveMode] = useState("auto"); // 'auto' | 'advanced'
  const [activeStep, setActiveStep] = useState(1);

  // Dataset & Profiling State
  const [file, setFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [taskType, setTaskType] = useState("classification");
  const [modelType, setModelType] = useState("RandomForestClassifier");
  const [rankingMetric, setRankingMetric] = useState("f1_macro");
  const [profiling, setProfiling] = useState(false);
  const [profileData, setProfileData] = useState(null);
  const [edaHistory, setEdaHistory] = useState([]);
  const [selectedEdaId, setSelectedEdaId] = useState("");

  // Strategy Selector & Recommendations State
  const [recommendations, setRecommendations] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState("balanced"); // 'conservative' | 'balanced' | 'advanced' | 'custom'

  // Transformation Configuration State
  const [datetimeCols, setDatetimeCols] = useState([]);
  const [cyclicalDatetime, setCyclicalDatetime] = useState(true);
  const [logCols, setLogCols] = useState([]);
  const [yeoJohnsonCols, setYeoJohnsonCols] = useState([]);
  const [sqrtCols, setSqrtCols] = useState([]);
  const [clipOutliers, setClipOutliers] = useState(true);
  const [groupRare, setGroupRare] = useState(true);
  const [freqCols, setFreqCols] = useState([]);
  const [targetEncodeCols, setTargetEncodeCols] = useState([]);
  const [interactionPairs, setInteractionPairs] = useState([]);
  const [polyCols, setPolyCols] = useState([]);
  const [textCols, setTextCols] = useState([]);

  // Feature Selection Configuration State
  const [selectionMethod, setSelectionMethod] = useState("mutual_info");
  const [varianceThreshold, setVarianceThreshold] = useState(0.01);
  const [correlationThreshold, setCorrelationThreshold] = useState(0.85);
  const [kBestCount, setKBestCount] = useState(10);
  const [nSplits, setNSplits] = useState(3);
  const [scaler, setScaler] = useState("robust");
  const [missing, setMissing] = useState("median");

  // Preview State
  const [previewing, setPreviewing] = useState(false);
  const [previewResult, setPreviewResult] = useState(null);

  // Strategy Comparison & Benchmark State
  const [comparing, setComparing] = useState(false);
  const [compareResult, setCompareResult] = useState(null);

  // Evaluation & Single Run State
  const [evaluating, setEvaluating] = useState(false);
  const [evalError, setEvalError] = useState(null);
  const [evalResult, setEvalResult] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  // Lineage & Search Filter State
  const [lineageSearch, setLineageSearch] = useState("");
  const [lineageStatusFilter, setLineageStatusFilter] = useState("all");

  // History State
  const [historyRuns, setHistoryRuns] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  // Classification & Regression Model Options
  const classificationModels = [
    { id: "RandomForestClassifier", name: "Random Forest Classifier" },
    { id: "DecisionTreeClassifier", name: "Decision Tree Classifier" },
    { id: "ExtraTreesClassifier", name: "Extra Trees Classifier" },
    { id: "LogisticRegression", name: "Logistic Regression" },
  ];

  const regressionModels = [
    { id: "RandomForest", name: "Random Forest Regressor" },
    { id: "DecisionTree", name: "Decision Tree Regressor" },
    { id: "ExtraTreesRegressor", name: "Extra Trees Regressor" },
  ];

  // Timer for execution feedback
  useEffect(() => {
    let interval;
    if (evaluating || comparing) {
      const startTime = Date.now();
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
      }, 500);
    } else {
      setElapsedTime(0);
    }
    return () => clearInterval(interval);
  }, [evaluating, comparing]);

  // Load EDA and Feature History on Mount
  useEffect(() => {
    fetchHistory();
    fetchEdaHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${getApiBase()}/features/history?limit=20`);
      if (res.ok) {
        const data = await res.json();
        setHistoryRuns(data.items || []);
      }
    } catch (err) {
      console.debug("Failed to fetch feature history:", err);
    }
  };

  const fetchEdaHistory = async () => {
    try {
      const res = await fetch(`${getApiBase()}/eda/history`);
      if (res.ok) {
        const data = await res.json();
        setEdaHistory(data || []);
      }
    } catch (err) {
      console.debug("Failed to fetch feature history:", err);
    }
  };

  // CSV File Handler & Auto-Profile
  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    if (!selectedFile.name.endsWith(".csv")) {
      setEvalError("Please upload a valid CSV file (.csv)");
      return;
    }

    setFile(selectedFile);
    setEvalError(null);
    setProfileData(null);
    setEvalResult(null);
    setCompareResult(null);
    setPreviewResult(null);
    setRecommendations(null);

    setProfiling(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("correlation_threshold", correlationThreshold);
    if (selectedEdaId) {
      formData.append("eda_analysis_id", selectedEdaId);
    }

    try {
      const res = await fetch(`${getApiBase()}/features/profile`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Profiling failed");
      setProfileData(data);

      // Auto-suggest target
      let targetCandidate = "";
      if (data.columns && data.columns.length > 0) {
        targetCandidate = data.columns[data.columns.length - 1].name;
        setTargetColumn(targetCandidate);
      }

      // Fetch EDA-derived recommendations
      fetchRecommendations(selectedFile, targetCandidate, selectedEdaId);

      setActiveStep(2);
    } catch (err) {
      setEvalError(err.message);
    } finally {
      setProfiling(false);
    }
  };

  // Fetch Dataset-Adaptive Recommendations
  const fetchRecommendations = async (targetFile, targetCol, edaId) => {
    if (!targetFile) return;
    try {
      const formData = new FormData();
      formData.append("file", targetFile);
      if (targetCol) formData.append("target_column", targetCol);
      if (edaId) formData.append("eda_analysis_id", edaId);

      const res = await fetch(`${getApiBase()}/features/recommendations`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setRecommendations(data);
        applyPreset("balanced", data);
      }
    } catch (err) {
      console.debug("Failed to fetch feature history:", err);
    }
  };

  // Apply a Strategy Preset (Conservative, Balanced, Advanced)
  const applyPreset = (presetKey, recsObj) => {
    setSelectedPreset(presetKey);
    const recs = recsObj || recommendations;
    if (!recs || !recs.presets || !recs.presets[presetKey]) return;

    const p = recs.presets[presetKey];
    setScaler(p.scaler || "standard");
    setMissing(p.imputation || "median");
    setSelectionMethod(p.selection_method || "none");
    setVarianceThreshold(p.variance_threshold ?? 0.01);
    setCorrelationThreshold(p.correlation_threshold ?? 0.85);
    setKBestCount(p.k_best_count ?? 10);
    setClipOutliers(p.clip_outliers ?? false);
    setGroupRare(p.group_rare_categories ?? false);
    setDatetimeCols(p.datetime_cols || []);
    setCyclicalDatetime(p.cyclical_datetime ?? false);
    setLogCols(p.log_cols || []);
    setYeoJohnsonCols(p.yeo_johnson_cols || []);
    setInteractionPairs(p.interaction_pairs || []);
    setPolyCols(p.polynomial_cols || []);
    setTargetEncodeCols(p.target_encode_cols || []);
    setTextCols(p.text_cols || []);
  };

  // Transformation Preview
  const handlePreviewTransformations = async () => {
    if (!file) return;
    setPreviewing(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("datetime_cols", JSON.stringify(datetimeCols));
    formData.append("cyclical_datetime", cyclicalDatetime);
    formData.append("log_cols", JSON.stringify(logCols));
    formData.append("yeo_johnson_cols", JSON.stringify(yeoJohnsonCols));
    formData.append("sqrt_cols", JSON.stringify(sqrtCols));
    formData.append("clip_outliers", clipOutliers);
    formData.append("group_rare_categories", groupRare);
    formData.append("frequency_encode_cols", JSON.stringify(freqCols));
    formData.append("interaction_pairs", JSON.stringify(interactionPairs));
    formData.append("polynomial_cols", JSON.stringify(polyCols));
    formData.append("text_cols", JSON.stringify(textCols));

    try {
      const res = await fetch(`${getApiBase()}/features/preview`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Preview failed");
      setPreviewResult(data);
      setActiveStep(6);
    } catch (err) {
      setEvalError(err.message);
    } finally {
      setPreviewing(false);
    }
  };

  // Compare Strategies Benchmark (Baseline vs Conservative vs Balanced vs Advanced)
  const handleCompareStrategies = async () => {
    if (!file || !targetColumn) {
      setEvalError("Please select a dataset and target column.");
      return;
    }

    setComparing(true);
    setEvalError(null);
    setCompareResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("task_type", taskType);
    formData.append("target_column", targetColumn);
    formData.append("model_type", modelType);
    formData.append("ranking_metric", rankingMetric);
    formData.append("n_splits", nSplits);
    if (selectedEdaId) {
      formData.append("eda_analysis_id", selectedEdaId);
    }

    try {
      const res = await fetch(`${getApiBase()}/features/compare-strategies`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Strategy comparison failed");
      setCompareResult(data);
      fetchHistory();
      setActiveStep(8);
    } catch (err) {
      setEvalError(err.message);
    } finally {
      setComparing(false);
    }
  };

  // Evaluate Custom or Selected Single Strategy
  const handleEvaluateCustom = async (e) => {
    if (e) e.preventDefault();
    if (!file || !targetColumn) {
      setEvalError("Please select a dataset and target column.");
      return;
    }

    setEvaluating(true);
    setEvalError(null);
    setEvalResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("task_type", taskType);
    formData.append("target_column", targetColumn);
    formData.append("model_type", modelType);
    formData.append("ranking_metric", rankingMetric);
    formData.append("n_splits", nSplits);
    formData.append("scaler", scaler);
    formData.append("missing", missing);
    formData.append("datetime_cols", JSON.stringify(datetimeCols));
    formData.append("cyclical_datetime", cyclicalDatetime);
    formData.append("log_cols", JSON.stringify(logCols));
    formData.append("yeo_johnson_cols", JSON.stringify(yeoJohnsonCols));
    formData.append("sqrt_cols", JSON.stringify(sqrtCols));
    formData.append("clip_outliers", clipOutliers);
    formData.append("group_rare_categories", groupRare);
    formData.append("frequency_encode_cols", JSON.stringify(freqCols));
    formData.append("interaction_pairs", JSON.stringify(interactionPairs));
    formData.append("polynomial_cols", JSON.stringify(polyCols));
    formData.append("target_encode_cols", JSON.stringify(targetEncodeCols));
    formData.append("text_cols", JSON.stringify(textCols));
    formData.append("selection_method", selectionMethod);
    formData.append("variance_threshold", varianceThreshold);
    formData.append("correlation_threshold", correlationThreshold);
    formData.append("k_best_count", kBestCount);
    if (selectedEdaId) {
      formData.append("eda_analysis_id", selectedEdaId);
    }

    try {
      const res = await fetch(`${getApiBase()}/features/evaluate`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Evaluation failed");
      setEvalResult(data);
      fetchHistory();
      setActiveStep(9);
    } catch (err) {
      setEvalError(err.message);
    } finally {
      setEvaluating(false);
    }
  };

  const handleTaskChange = (newTask) => {
    setTaskType(newTask);
    if (newTask === "classification") {
      setModelType("RandomForestClassifier");
      setRankingMetric("f1_macro");
    } else {
      setModelType("RandomForest");
      setRankingMetric("rmse");
    }
  };

  // Feature Lineage records combined from compareResult or evalResult or previewResult
  const currentLineage = (compareResult?.feature_lineage) || (evalResult?.feature_lineage) || (previewResult?.feature_lineage) || [];
  const filteredLineage = currentLineage.filter((item) => {
    const matchesSearch = item.feature_name.toLowerCase().includes(lineageSearch.toLowerCase()) ||
      (item.parent_columns && item.parent_columns.some((p) => p.toLowerCase().includes(lineageSearch.toLowerCase())));
    if (!matchesSearch) return false;
    if (lineageStatusFilter === "engineered") return item.is_engineered;
    if (lineageStatusFilter === "original") return !item.is_engineered;
    return true;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-20 space-y-8">
      {/* Universal Page Header */}
      <PageHeader
        category="Data Studio"
        title="Feature Engineering & Selection Studio"
        subtitle="Intelligently transform raw tabular data into highly predictive features, evaluate multi-strategy benchmarks, and prune redundancy without lookahead leakage."
        badge={
          <div className="flex items-center gap-2">
            <Badge variant="violet" size="sm">
              <Sparkles className="w-3 h-3 text-[#A855F7] mr-1 inline" /> Dataset-Adaptive
            </Badge>
            <Badge variant="success" size="sm">
              <ShieldCheck className="w-3.5 h-3.5 mr-1 inline" /> Leak-Free CV
            </Badge>
          </div>
        }
        actions={
          <div className="flex items-center gap-2.5">
            <div className="bg-white/[0.04] border border-white/[0.08] p-1 rounded-xl flex items-center">
              <button
                type="button"
                onClick={() => setActiveMode("auto")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeMode === "auto"
                    ? "bg-[#8265FF] text-white shadow-sm"
                    : "text-[#9FA2B6] hover:text-white"
                }`}
              >
                Auto Mode (Guided)
              </button>
              <button
                type="button"
                onClick={() => setActiveMode("advanced")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeMode === "advanced"
                    ? "bg-[#8265FF] text-white shadow-sm"
                    : "text-[#9FA2B6] hover:text-white"
                }`}
              >
                Advanced Mode
              </button>
            </div>

            <Button
              variant="secondary"
              size="sm"
              icon={Clock}
              onClick={() => setShowHistory(!showHistory)}
            >
              History ({historyRuns.length})
            </Button>
          </div>
        }
      />

      {/* Guided Workflow Progress Stepper */}
      <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-2xl overflow-x-auto shadow-lg">
        <div className="flex items-center justify-between min-w-[700px] text-xs">
          {[
            { num: 1, label: "Dataset & EDA" },
            { num: 2, label: "Target & Task" },
            { num: 3, label: "Recommendations" },
            { num: 4, label: "Strategy Presets" },
            { num: 5, label: "Transformations" },
            { num: 6, label: "Preview" },
            { num: 7, label: "Selection" },
            { num: 8, label: "Strategy Benchmark" },
            { num: 9, label: "Validation Results" },
            { num: 10, label: "Lineage Graph" },
          ].map((st) => (
            <div
              key={st.num}
              onClick={() => setActiveStep(st.num)}
              className={`flex items-center gap-2 cursor-pointer transition-all px-2 py-1 rounded-lg ${
                activeStep === st.num
                  ? "text-white font-bold bg-[#8265FF]/20 border border-[#8265FF]/40"
                  : activeStep > st.num
                  ? "text-emerald-400 font-medium"
                  : "text-[#A4A4B8] hover:text-white"
              }`}
            >
              <span
                className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                  activeStep === st.num
                    ? "bg-[#8265FF] text-white"
                    : activeStep > st.num
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                    : "bg-white/[0.06] text-[#A4A4B8]"
                }`}
              >
                {activeStep > st.num ? <Check className="w-3 h-3" /> : st.num}
              </span>
              <span>{st.label}</span>
              {st.num < 10 && <ChevronRight className="w-3 h-3 text-white/[0.2] ml-1" />}
            </div>
          ))}
        </div>
      </div>

      {/* Error Alert Banner */}
      {evalError && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center justify-between gap-3 text-red-400 text-xs">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{evalError}</span>
          </div>
          <button onClick={() => setEvalError(null)} className="text-white hover:underline text-xs">Dismiss</button>
        </div>
      )}

      {/* STEP 1: Dataset & Phase 2 EDA Cross-Reference */}
      <section className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl space-y-6 shadow-xl">
        <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
          <div className="flex items-center gap-3">
            <span className="w-6 h-6 rounded-full bg-[#8265FF]/20 text-[#8265FF] flex items-center justify-center text-xs font-bold border border-[#8265FF]/30">1</span>
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">Dataset Upload & Phase 2 EDA Handoff</h2>
          </div>
          {file && (
            <span className="text-xs text-emerald-400 font-mono flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" /> {file.name} ({(file.size / 1024).toFixed(1)} KB)
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* File Upload Zone */}
          <div>
            <label className="text-xs font-semibold text-[#A4A4B8] block mb-2">Upload Tabular CSV Dataset</label>
            <div className="relative border-2 border-dashed border-white/[0.12] hover:border-[#8265FF]/50 transition-all rounded-xl p-6 text-center bg-[#070711]/50 cursor-pointer">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <Upload className="w-8 h-8 text-[#8265FF] mx-auto mb-2" />
              <p className="text-xs font-bold text-white">Click to browse or drop CSV dataset here</p>
              <p className="text-[11px] text-[#A4A4B8] mt-1">UTF-8 tabular data with headers supported</p>
            </div>
          </div>

          {/* Phase 2 EDA Cross-Reference Selector */}
          <div className="space-y-3">
            <label className="text-xs font-semibold text-[#A4A4B8] block flex items-center gap-1.5">
              <BarChart3 className="w-3.5 h-3.5 text-[#8265FF]" /> Link Existing Phase 2 EDA Profile (Optional)
            </label>
            <p className="text-xs text-[#A4A4B8]">
              Connect an existing exploratory analysis from the EDA module to instantly reuse skewness, outlier, and correlation diagnostics without recomputing them.
            </p>
            <select
              value={selectedEdaId}
              onChange={(e) => {
                setSelectedEdaId(e.target.value);
                if (file) fetchRecommendations(file, targetColumn, e.target.value);
              }}
              className="w-full bg-[#070711] border border-white/[0.12] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
            >
              <option value="">-- No EDA Profile Linked (Run Fresh Profile) --</option>
              {edaHistory.map((item) => (
                <option key={item.analysis_id} value={item.analysis_id}>
                  {item.filename} ({item.row_count} rows, {item.column_count} cols) - {new Date(item.created_at).toLocaleDateString()}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Profiling Loading State */}
        {profiling && (
          <div className="flex items-center gap-3 p-4 bg-[#8265FF]/10 border border-[#8265FF]/20 rounded-xl text-xs text-[#8265FF]">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span>Analyzing dataset dimensions, column distributions, and semantic data types...</span>
          </div>
        )}
      </section>

      {/* STEP 2 & 3: Target, Task & EDA-Derived Recommendations */}
      {profileData && (
        <section className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl space-y-6 shadow-xl">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
            <div className="flex items-center gap-3">
              <span className="w-6 h-6 rounded-full bg-[#8265FF]/20 text-[#8265FF] flex items-center justify-center text-xs font-bold border border-[#8265FF]/30">2</span>
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">Target Specification & EDA Recommendations</h2>
            </div>
            {recommendations && (
              <div className="flex flex-wrap gap-1.5">
                {recommendations.active_profiles.map((p) => (
                  <span key={p} className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-[#8265FF]/20 text-[#8265FF] border border-[#8265FF]/30">
                    {p.replace("_", " ")}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Target Column Selector */}
            <div>
              <label className="text-xs font-semibold text-[#A4A4B8] block mb-2">Target Feature</label>
              <select
                value={targetColumn}
                onChange={(e) => {
                  setTargetColumn(e.target.value);
                  fetchRecommendations(file, e.target.value, selectedEdaId);
                }}
                className="w-full bg-[#070711] border border-white/[0.12] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                {profileData.columns.map((c) => (
                  <option key={c.name} value={c.name}>{c.name} ({c.inferred_type})</option>
                ))}
              </select>
            </div>

            {/* Task Type Selector */}
            <div>
              <label className="text-xs font-semibold text-[#A4A4B8] block mb-2">ML Task Type</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => handleTaskChange("classification")}
                  className={`py-2 text-xs font-semibold rounded-xl border transition-all ${
                    taskType === "classification"
                      ? "bg-[#8265FF] text-white border-[#8265FF]"
                      : "bg-[#070711] border-white/[0.08] text-[#A4A4B8] hover:text-white"
                  }`}
                >
                  Classification
                </button>
                <button
                  type="button"
                  onClick={() => handleTaskChange("regression")}
                  className={`py-2 text-xs font-semibold rounded-xl border transition-all ${
                    taskType === "regression"
                      ? "bg-[#8265FF] text-white border-[#8265FF]"
                      : "bg-[#070711] border-white/[0.08] text-[#A4A4B8] hover:text-white"
                  }`}
                >
                  Regression
                </button>
              </div>
            </div>

            {/* Benchmark Estimator */}
            <div>
              <label className="text-xs font-semibold text-[#A4A4B8] block mb-2">Evaluation Estimator</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full bg-[#070711] border border-white/[0.12] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                {(taskType === "classification" ? classificationModels : regressionModels).map((m) => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>
          </div>

          {/* EDA Recommendations Card Banner */}
          {recommendations && (
            <div className="bg-[#070711] border border-[#8265FF]/30 p-4 rounded-xl space-y-2">
              <div className="flex items-center gap-2 text-xs font-bold text-white uppercase tracking-wider">
                <Sparkles className="w-4 h-4 text-[#8265FF]" /> Empirical EDA Intelligence Recommendations
              </div>
              <ul className="text-xs text-[#A4A4B8] space-y-1 list-disc list-inside">
                {recommendations.recommendation_explanations.map((exp, idx) => (
                  <li key={idx}>{exp}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {/* STEP 4: Strategy Preset Selector */}
      {profileData && (
        <section className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl space-y-6 shadow-xl">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
            <div className="flex items-center gap-3">
              <span className="w-6 h-6 rounded-full bg-[#8265FF]/20 text-[#8265FF] flex items-center justify-center text-xs font-bold border border-[#8265FF]/30">3</span>
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">Select Feature Engineering Strategy Preset</h2>
            </div>
            <span className="text-xs text-[#A4A4B8]">Deterministic, non-leaking scikit-learn pipelines</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Conservative Preset */}
            <div
              onClick={() => applyPreset("conservative")}
              className={`p-5 rounded-2xl border cursor-pointer transition-all space-y-3 ${
                selectedPreset === "conservative"
                  ? "bg-[#171630] border-[#8265FF] shadow-lg shadow-[#8265FF]/10 ring-1 ring-[#8265FF]"
                  : "bg-[#070711] border-white/[0.08] hover:border-white/[0.2]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" /> Conservative
                </span>
                <span className="text-[10px] font-mono text-[#A4A4B8] bg-white/[0.04] px-2 py-0.5 rounded-full">Low Risk</span>
              </div>
              <p className="text-xs text-[#A4A4B8]">
                Standard scaling, calendar extraction, rare category consolidation, and collinearity pruning. Zero combinatorial explosion.
              </p>
              <div className="text-[11px] font-mono text-[#8265FF] pt-2 border-t border-white/[0.06] flex items-center justify-between">
                <span>Speed: &lt; 2s</span>
                <span>Variance Filter</span>
              </div>
            </div>

            {/* Balanced Preset */}
            <div
              onClick={() => applyPreset("balanced")}
              className={`p-5 rounded-2xl border cursor-pointer transition-all space-y-3 relative ${
                selectedPreset === "balanced"
                  ? "bg-[#171630] border-[#8265FF] shadow-lg shadow-[#8265FF]/20 ring-2 ring-[#8265FF]"
                  : "bg-[#070711] border-white/[0.08] hover:border-white/[0.2]"
              }`}
            >
              <div className="absolute -top-2.5 right-4 bg-gradient-to-r from-[#8265FF] to-[#A855F7] text-white text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full shadow-md">
                Recommended
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <Zap className="w-4 h-4 text-amber-400" /> Balanced
                </span>
                <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full">Optimal Lift</span>
              </div>
              <p className="text-xs text-[#A4A4B8]">
                Yeo-Johnson distribution transforms, cyclical sine/cosine datetime features, top 5 interactions, and mutual information selection.
              </p>
              <div className="text-[11px] font-mono text-[#8265FF] pt-2 border-t border-white/[0.06] flex items-center justify-between">
                <span>Speed: 2 - 5s</span>
                <span>Mutual Information</span>
              </div>
            </div>

            {/* Advanced Preset */}
            <div
              onClick={() => applyPreset("advanced")}
              className={`p-5 rounded-2xl border cursor-pointer transition-all space-y-3 ${
                selectedPreset === "advanced"
                  ? "bg-[#171630] border-[#8265FF] shadow-lg shadow-[#8265FF]/10 ring-1 ring-[#8265FF]"
                  : "bg-[#070711] border-white/[0.08] hover:border-white/[0.2]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-purple-400" /> Advanced
                </span>
                <span className="text-[10px] font-mono text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded-full">High Capacity</span>
              </div>
              <p className="text-xs text-[#A4A4B8]">
                Polynomial expansions, safe-ratio interactions, out-of-fold target encoding, text metrics, and tree-based model feature selection.
              </p>
              <div className="text-[11px] font-mono text-[#8265FF] pt-2 border-t border-white/[0.06] flex items-center justify-between">
                <span>Speed: 5 - 15s</span>
                <span>Model-Based (RF)</span>
              </div>
            </div>
          </div>

          {/* Preset Benchmark Button */}
          <div className="flex items-center justify-between pt-2">
            <p className="text-xs text-[#A4A4B8]">
              Want to see exactly how these strategies compare against the unengineered baseline? Run a strict cross-validated benchmark:
            </p>
            <button
              type="button"
              onClick={handleCompareStrategies}
              disabled={comparing}
              className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7154eb] hover:to-[#9645e3] text-white text-xs font-bold rounded-xl shadow-lg shadow-[#8265FF]/25 disabled:opacity-50 transition-all cursor-pointer"
            >
              {comparing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" /> Benchmarking ({elapsedTime}s)...
                </>
              ) : (
                <>
                  <GitCompare className="w-4 h-4" /> Compare 4 Feature Strategies
                </>
              )}
            </button>
          </div>
        </section>
      )}

      {/* STEP 5: Granular Transformation Controls (Advanced Mode) */}
      {profileData && activeMode === "advanced" && (
        <section className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl space-y-6 shadow-xl">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
            <div className="flex items-center gap-3">
              <span className="w-6 h-6 rounded-full bg-[#8265FF]/20 text-[#8265FF] flex items-center justify-center text-xs font-bold border border-[#8265FF]/30">4</span>
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">Fine-Grained Transformation Controls</h2>
            </div>
            <button
              type="button"
              onClick={handlePreviewTransformations}
              disabled={previewing}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-[#171630] hover:bg-[#1f1d40] border border-white/[0.08] text-white text-xs font-semibold rounded-xl transition-all"
            >
              {previewing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <PlayCircle className="w-3.5 h-3.5 text-[#8265FF]" />}
              Live Transformation Preview
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Numerical Transformations */}
            <div className="space-y-4 bg-[#070711] p-4 rounded-xl border border-white/[0.06]">
              <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-[#8265FF]" /> Numerical Transforms
              </h4>
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-xs text-[#A4A4B8] cursor-pointer">
                  <input
                    type="checkbox"
                    checked={clipOutliers}
                    onChange={(e) => setClipOutliers(e.target.checked)}
                    className="accent-[#8265FF] rounded"
                  />
                  <span>Tukey Outlier Percentile Clipping (1% - 99%)</span>
                </label>
                <div className="text-[11px] text-[#A4A4B8]">
                  <span className="block mb-1 font-semibold text-white">Yeo-Johnson Power Transform:</span>
                  <div className="flex flex-wrap gap-1">
                    {profileData.columns.filter((c) => c.inferred_type === "numeric" && !c.is_target).map((c) => (
                      <button
                        key={c.name}
                        type="button"
                        onClick={() => {
                          setYeoJohnsonCols(
                            yeoJohnsonCols.includes(c.name)
                              ? yeoJohnsonCols.filter((x) => x !== c.name)
                              : [...yeoJohnsonCols, c.name]
                          );
                        }}
                        className={`text-[10px] px-2 py-0.5 rounded border transition-all ${
                          yeoJohnsonCols.includes(c.name)
                            ? "bg-[#8265FF]/20 text-[#8265FF] border-[#8265FF]"
                            : "bg-white/[0.04] text-[#A4A4B8] border-white/[0.08]"
                        }`}
                      >
                        {c.name}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Datetime & Cyclical */}
            <div className="space-y-4 bg-[#070711] p-4 rounded-xl border border-white/[0.06]">
              <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Calendar className="w-3.5 h-3.5 text-[#8265FF]" /> Temporal & Cyclical
              </h4>
              <label className="flex items-center gap-2 text-xs text-[#A4A4B8] cursor-pointer">
                <input
                  type="checkbox"
                  checked={cyclicalDatetime}
                  onChange={(e) => setCyclicalDatetime(e.target.checked)}
                  className="accent-[#8265FF] rounded"
                />
                <span>Cyclical Sine/Cosine Encoding (Preserves Month/Day Wrap)</span>
              </label>
              <div className="text-[11px] text-[#A4A4B8]">
                <span className="block mb-1 font-semibold text-white">Target Datetime Columns:</span>
                <div className="flex flex-wrap gap-1">
                  {profileData.datetime_candidates.length > 0 ? (
                    profileData.datetime_candidates.map((c) => (
                      <span key={c} className="text-[10px] px-2 py-0.5 rounded bg-[#8265FF]/20 text-[#8265FF] border border-[#8265FF]">
                        {c}
                      </span>
                    ))
                  ) : (
                    <span className="text-[11px] text-[#A4A4B8] italic">No datetime columns detected</span>
                  )}
                </div>
              </div>
            </div>

            {/* Categorical & Feature Selection */}
            <div className="space-y-4 bg-[#070711] p-4 rounded-xl border border-white/[0.06]">
              <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Filter className="w-3.5 h-3.5 text-[#8265FF]" /> Feature Selection
              </h4>
              <div>
                <label className="text-[11px] font-semibold text-white block mb-1">Selection Algorithm</label>
                <select
                  value={selectionMethod}
                  onChange={(e) => setSelectionMethod(e.target.value)}
                  className="w-full bg-[#111126] border border-white/[0.12] rounded-lg px-2.5 py-1.5 text-xs text-white"
                >
                  <option value="none">Passthrough (Keep All Features)</option>
                  <option value="mutual_info">Mutual Information (Non-Linear Dependencies)</option>
                  <option value="model_based">Model-Based Importance (Random Forest)</option>
                  <option value="l1_regularized">L1 Sparsity Regularization (Lasso/Logistic)</option>
                  <option value="rfe">Recursive Feature Elimination (RFE)</option>
                  <option value="correlation">Correlation Filter (Remove Multicollinear |r| &gt; 0.85)</option>
                  <option value="variance">VarianceThreshold (Drop Zero-Variance)</option>
                </select>
              </div>
              <div>
                <label className="text-[11px] font-semibold text-white block mb-1">Max Retained Features (K): {kBestCount}</label>
                <input
                  type="range"
                  min="3"
                  max="40"
                  value={kBestCount}
                  onChange={(e) => setKBestCount(parseInt(e.target.value))}
                  className="w-full accent-[#8265FF]"
                />
              </div>
            </div>
          </div>
        </section>
      )}

      {/* STEP 6: Live Transformation Preview Table */}
      {previewResult && (
        <section className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl space-y-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <PlayCircle className="text-[#8265FF]" /> Transformation Live Preview (Top Sample Rows)
            </h3>
            <div className="text-xs text-[#A4A4B8] flex items-center gap-3">
              <span>Original: <strong className="text-white">{previewResult.original_columns.length}</strong> cols</span>
              <span>Transformed: <strong className="text-emerald-400">{previewResult.transformed_columns.length}</strong> cols</span>
              <span>Created: <strong className="text-[#8265FF]">+{previewResult.created_columns.length}</strong> cols</span>
            </div>
          </div>

          <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#070711] text-[#A4A4B8] border-b border-white/[0.08]">
                <tr>
                  {previewResult.transformed_columns.slice(0, 10).map((col) => (
                    <th key={col} className="p-3 font-mono font-semibold whitespace-nowrap">
                      {col}
                      {previewResult.created_columns.includes(col) && (
                        <span className="ml-1 text-[9px] text-[#8265FF] bg-[#8265FF]/20 px-1 py-0.5 rounded">new</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {previewResult.sample_preview.map((row, idx) => (
                  <tr key={idx} className="hover:bg-white/[0.02]">
                    {previewResult.transformed_columns.slice(0, 10).map((col) => (
                      <td key={col} className="p-3 font-mono text-white whitespace-nowrap">
                        {typeof row[col] === "number" ? row[col].toFixed(4) : String(row[col] ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* STEP 8: Multi-Strategy Benchmark Comparison Dashboard */}
      {compareResult && (
        <section className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl space-y-6 shadow-2xl">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  CV Benchmark Completed
                </span>
                <span className="text-xs text-[#A4A4B8]">Run ID: {compareResult.run_id.slice(0, 8)}</span>
              </div>
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <GitCompare className="text-[#8265FF]" /> Multi-Strategy Performance Comparison
              </h3>
            </div>

            {compareResult.registered_model_id && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-[#A4A4B8]">Winning Model Registered:</span>
                <span className="px-2.5 py-1 rounded-lg bg-[#8265FF]/20 text-[#8265FF] font-mono text-xs font-bold border border-[#8265FF]/30">
                  ID #{compareResult.registered_model_id}
                </span>
                <Link
                  to={`/models`}
                  className="text-xs text-white hover:text-[#8265FF] flex items-center gap-1 font-semibold underline ml-2"
                >
                  View in Registry <ArrowRight className="w-3 h-3" />
                </Link>
              </div>
            )}
          </div>

          {/* Statistical Verdict Banner */}
          <div className="p-4 bg-gradient-to-r from-[#8265FF]/10 to-transparent border-l-4 border-[#8265FF] rounded-r-xl space-y-1">
            <span className="text-[10px] font-bold text-[#8265FF] uppercase tracking-wider">Statistical Evaluation Verdict</span>
            <p className="text-sm font-semibold text-white">{compareResult.statistical_verdict}</p>
          </div>

          {/* 4 Strategy Comparison Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {compareResult.comparison_matrix.map((strat) => {
              const isWinner = strat.strategy === compareResult.winning_strategy;
              return (
                <div
                  key={strat.strategy}
                  className={`p-4 rounded-xl border space-y-3 relative ${
                    isWinner
                      ? "bg-[#171630] border-[#8265FF] shadow-lg shadow-[#8265FF]/20 ring-1 ring-[#8265FF]"
                      : "bg-[#070711] border-white/[0.08]"
                  }`}
                >
                  {isWinner && (
                    <span className="absolute -top-2.5 right-3 bg-[#8265FF] text-white text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full shadow-md">
                      Winning Strategy
                    </span>
                  )}
                  <div>
                    <span className="text-[10px] font-mono uppercase text-[#A4A4B8] block">{strat.strategy}</span>
                    <h4 className="text-sm font-bold text-white">{strat.strategy_name}</h4>
                  </div>

                  <div className="pt-2 border-t border-white/[0.06] space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-[#A4A4B8]">CV Score ({compareResult.ranking_metric}):</span>
                      <strong className="text-white font-mono">{strat.cv_mean ?? "N/A"}</strong>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-[#A4A4B8]">Stability (± Std):</span>
                      <span className="text-[#A4A4B8] font-mono">±{strat.cv_std ?? 0}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-[#A4A4B8]">Active Features:</span>
                      <span className="text-emerald-400 font-mono font-semibold">{strat.active_features_count}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-[#A4A4B8]">Duration:</span>
                      <span className="text-[#A4A4B8] font-mono">{strat.duration_seconds}s</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Holdout Delta Metric Cards */}
          {compareResult.delta_metrics && (
            <div className="space-y-3 pt-2 border-t border-white/[0.06]">
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                Holdout Validation Lift (Baseline vs Winning Strategy)
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(compareResult.delta_metrics).map(([metricKey, d]) => (
                  <div key={metricKey} className="bg-[#070711] border border-white/[0.08] p-3 rounded-xl">
                    <span className="text-[10px] font-semibold text-[#A4A4B8] uppercase block mb-1">
                      {metricKey.replace("_", " ")}
                    </span>
                    <div className="flex items-baseline justify-between">
                      <span className="text-lg font-bold font-mono text-white">{d.winning}</span>
                      <span
                        className={`text-xs font-mono font-bold flex items-center ${
                          d.improved ? "text-emerald-400" : "text-red-400"
                        }`}
                      >
                        {d.improved ? <TrendingUp className="w-3 h-3 mr-0.5" /> : <TrendingDown className="w-3 h-3 mr-0.5" />}
                        {d.percentage_change !== null ? `${d.percentage_change > 0 ? "+" : ""}${d.percentage_change}%` : d.delta}
                      </span>
                    </div>
                    <span className="text-[10px] text-[#A4A4B8] block mt-1">Baseline: {d.baseline}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {/* STEP 10: Interactive Feature Lineage Table */}
      {currentLineage.length > 0 && (
        <section className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl space-y-4 shadow-xl">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
            <div>
              <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Code className="text-[#8265FF]" /> Feature Lineage & Explainability Registry
              </h3>
              <p className="text-[11px] text-[#A4A4B8] mt-0.5">
                Inspect how original raw columns were engineered into new features with full mathematical parameters.
              </p>
            </div>

            {/* Lineage Filter & Search Controls */}
            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-[#A4A4B8] absolute left-2.5 top-2.5" />
                <input
                  type="text"
                  placeholder="Search features..."
                  value={lineageSearch}
                  onChange={(e) => setLineageSearch(e.target.value)}
                  className="bg-[#070711] border border-white/[0.12] rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-[#A4A4B8] focus:outline-none focus:border-[#8265FF]"
                />
              </div>
              <select
                value={lineageStatusFilter}
                onChange={(e) => setLineageStatusFilter(e.target.value)}
                className="bg-[#070711] border border-white/[0.12] rounded-lg px-2.5 py-1.5 text-xs text-white"
              >
                <option value="all">All Features</option>
                <option value="engineered">Engineered Only</option>
                <option value="original">Original Only</option>
              </select>
            </div>
          </div>

          <div className="overflow-x-auto rounded-xl border border-white/[0.06] max-h-96">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#070711] text-[#A4A4B8] border-b border-white/[0.08] sticky top-0 z-10">
                <tr>
                  <th className="p-3 font-semibold">Engineered Feature</th>
                  <th className="p-3 font-semibold">Parent Columns</th>
                  <th className="p-3 font-semibold">Operation Applied</th>
                  <th className="p-3 font-semibold">Parameters</th>
                  <th className="p-3 font-semibold text-center">Type</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {filteredLineage.map((item, idx) => (
                  <tr key={idx} className="hover:bg-white/[0.02]">
                    <td className="p-3 font-mono font-bold text-white whitespace-nowrap">
                      {item.feature_name}
                    </td>
                    <td className="p-3 font-mono text-[#A4A4B8]">
                      {item.parent_columns.join(", ")}
                    </td>
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#8265FF]/10 text-[#8265FF] border border-[#8265FF]/20">
                        {item.operation}
                      </span>
                    </td>
                    <td className="p-3 font-mono text-[10px] text-[#A4A4B8]">
                      {Object.keys(item.parameters || {}).length > 0 ? JSON.stringify(item.parameters) : "--"}
                    </td>
                    <td className="p-3 text-center">
                      <span
                        className={`text-[9px] uppercase font-semibold px-2 py-0.5 rounded-full border ${
                          item.is_engineered
                            ? "bg-purple-500/20 text-purple-300 border-purple-500/30"
                            : "bg-white/[0.04] text-[#A4A4B8] border-white/[0.08]"
                        }`}
                      >
                        {item.is_engineered ? "Engineered" : "Original"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* STEP 11: Continuous Workflow Handoff Bar */}
      {(compareResult || evalResult) && (
        <div className="bg-[#111126] border border-[#8265FF]/40 p-6 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-4 shadow-xl">
          <div>
            <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              Pipeline Fitted & Ready
            </span>
            <h4 className="text-base font-bold text-white mt-1">Ready for AutoML Model Training or Predictions?</h4>
            <p className="text-xs text-[#A4A4B8]">
              Your engineered features and transformation pipeline are permanently saved in the model registry.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link
              to="/models"
              className="px-4 py-2 bg-[#171630] hover:bg-[#1f1d40] border border-white/[0.08] text-white text-xs font-semibold rounded-xl transition-all"
            >
              Saved Models
            </Link>
            <Link
              to="/automl"
              className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7154eb] hover:to-[#9645e3] text-white text-xs font-bold rounded-xl shadow-lg shadow-[#8265FF]/25 transition-all"
            >
              <Cpu className="w-4 h-4" /> Train Full AutoML Models
            </Link>
          </div>
        </div>
      )}

      {/* Run History Drawer */}
      {showHistory && (
        <section className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl space-y-4 shadow-xl">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Clock className="text-[#8265FF]" /> Historical Feature Engineering Runs
            </h3>
            <span className="text-xs text-[#A4A4B8]">{historyRuns.length} total runs recorded</span>
          </div>

          <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#070711] text-[#A4A4B8] border-b border-white/[0.08]">
                <tr>
                  <th className="p-3 font-semibold">Run ID</th>
                  <th className="p-3 font-semibold">Dataset</th>
                  <th className="p-3 font-semibold">Target / Task</th>
                  <th className="p-3 font-semibold">Winning Strategy</th>
                  <th className="p-3 font-semibold">Model Artifact</th>
                  <th className="p-3 font-semibold">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {historyRuns.map((r) => (
                  <tr key={r.run_id} className="hover:bg-white/[0.02]">
                    <td className="p-3 font-mono font-bold text-[#8265FF]">{r.run_id.slice(0, 8)}</td>
                    <td className="p-3 font-mono text-white">{r.dataset_name}</td>
                    <td className="p-3 text-[#A4A4B8]">
                      {r.target_column} <span className="text-[10px] uppercase font-mono">({r.task_type})</span>
                    </td>
                    <td className="p-3 font-semibold text-white capitalize">{r.winning_strategy}</td>
                    <td className="p-3 font-mono text-xs text-emerald-400">
                      {r.registered_model_id ? `#${r.registered_model_id}` : "--"}
                    </td>
                    <td className="p-3 text-[#A4A4B8]">{new Date(r.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

    </div>
  );
}
