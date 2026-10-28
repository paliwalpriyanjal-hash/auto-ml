import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
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
  AlertCircle
} from "lucide-react";
import { getApiBase } from "../utils/api";

export default function FeatureEngineering() {
  // Dataset & Profiling State
  const [file, setFile] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [taskType, setTaskType] = useState("classification");
  const [modelType, setModelType] = useState("RandomForestClassifier");
  const [rankingMetric, setRankingMetric] = useState("f1_macro");
  const [profiling, setProfiling] = useState(false);
  const [profileData, setProfileData] = useState(null);

  // Transformation Builder State
  const [datetimeCols, setDatetimeCols] = useState([]);
  const [logCols, setLogCols] = useState([]);
  const [clipOutliers, setClipOutliers] = useState(false);
  const [groupRare, setGroupRare] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [previewResult, setPreviewResult] = useState(null);

  // Feature Selection State
  const [selectionMethod, setSelectionMethod] = useState("model_based");
  const [varianceThreshold, setVarianceThreshold] = useState(0.01);
  const [correlationThreshold, setCorrelationThreshold] = useState(0.85);
  const [kBestCount, setKBestCount] = useState(8);
  const [nSplits, setNSplits] = useState(3);
  const [scaler, setScaler] = useState("standard");
  const [missing, setMissing] = useState("median");

  // Evaluation & Results State
  const [evaluating, setEvaluating] = useState(false);
  const [evalError, setEvalError] = useState(null);
  const [evalResult, setEvalResult] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  // History State
  const [historyRuns, setHistoryRuns] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  // Model Options
  const classificationModels = [
    { id: "RandomForestClassifier", name: "Random Forest Classifier" },
    { id: "DecisionTreeClassifier", name: "Decision Tree Classifier" },
    { id: "ExtraTreesClassifier", name: "Extra Trees Classifier" },
    { id: "LogisticRegression", name: "Logistic Regression" },
  ];

  const regressionModels = [
    { id: "RandomForest", name: "Random Forest Regressor" },
    { id: "DecisionTree", name: "Decision Tree Regressor" },
  ];

  // Live Timer
  useEffect(() => {
    let interval;
    if (evaluating) {
      const startTime = Date.now();
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
      }, 500);
    } else {
      setElapsedTime(0);
    }
    return () => clearInterval(interval);
  }, [evaluating]);

  // Load history on mount
  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${getApiBase()}/features/history?limit=20`);
      if (res.ok) {
        const data = await res.json();
        setHistoryRuns(data.items || []);
      }
    } catch {}
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
    setPreviewResult(null);

    // Trigger auto-profiling
    setProfiling(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("correlation_threshold", correlationThreshold);

    try {
      const res = await fetch(`${getApiBase()}/features/profile`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Profiling failed");
      setProfileData(data);

      // Auto-suggest target
      if (data.columns && data.columns.length > 0) {
        const lastCol = data.columns[data.columns.length - 1].name;
        setTargetColumn(lastCol);
      }

      // Auto-select detected datetime candidates
      if (data.datetime_candidates && data.datetime_candidates.length > 0) {
        setDatetimeCols(data.datetime_candidates);
      }
    } catch (err) {
      setEvalError(err.message);
    } finally {
      setProfiling(false);
    }
  };

  // Transformation Preview
  const handlePreviewTransformations = async () => {
    if (!file) return;
    setPreviewing(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("datetime_cols", JSON.stringify(datetimeCols));
    formData.append("log_cols", JSON.stringify(logCols));
    formData.append("clip_outliers", clipOutliers);
    formData.append("group_rare_categories", groupRare);

    try {
      const res = await fetch(`${getApiBase()}/features/preview`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Preview failed");
      setPreviewResult(data);
    } catch (err) {
      setEvalError(err.message);
    } finally {
      setPreviewing(false);
    }
  };

  // Run Strategy Evaluation
  const handleEvaluateStrategies = async (e) => {
    e.preventDefault();
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
    formData.append("log_cols", JSON.stringify(logCols));
    formData.append("clip_outliers", clipOutliers);
    formData.append("group_rare_categories", groupRare);
    formData.append("selection_method", selectionMethod);
    formData.append("variance_threshold", varianceThreshold);
    formData.append("correlation_threshold", correlationThreshold);
    formData.append("k_best_count", kBestCount);

    try {
      const res = await fetch(`${getApiBase()}/features/evaluate`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Evaluation failed");
      setEvalResult(data);
      fetchHistory();
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

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-20 space-y-10">
      
      {/* Workspace Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 border-b border-white/[0.08] pb-8">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[#8265FF] bg-[#8265FF]/10 px-2.5 py-0.5 rounded-full border border-[#8265FF]/20 flex items-center gap-1.5">
              <Sparkles className="w-3 h-3 text-[#A855F7]" />
              Feature Engineering & Selection Studio
            </span>
            <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-400 bg-emerald-500/10 px-2.5 py-0.5 rounded-full border border-emerald-500/20">
              Leakage-Free Pipelines
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Automated Feature Engineering & Selection
          </h1>
          <p className="text-sm text-[#A4A4B8] mt-1 max-w-2xl">
            Detect collinearity, extract calendar features, apply safe log/clipping transforms, and evaluate model-based or variance feature selection across training CV folds.
          </p>
        </div>

        <button
          onClick={() => setShowHistory(!showHistory)}
          className="self-start md:self-auto px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] transition-all flex items-center gap-2"
        >
          <Clock className="w-3.5 h-3.5 text-[#8265FF]" />
          Feature Runs ({historyRuns.length})
        </button>
      </div>

      {/* History Modal / Drawer */}
      {showHistory && (
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-[#8265FF]" />
              <h3 className="text-sm font-bold text-white">Historical Feature Strategy Runs</h3>
            </div>
            <button onClick={() => setShowHistory(false)} className="text-[#A4A4B8] hover:text-white p-1">
              <XCircle className="w-4 h-4" />
            </button>
          </div>

          {historyRuns.length === 0 ? (
            <div className="py-6 text-center text-xs text-[#A4A4B8]">No previous feature runs recorded.</div>
          ) : (
            <div className="divide-y divide-white/[0.04] max-h-60 overflow-y-auto">
              {historyRuns.map((r) => (
                <div key={r.run_id} className="py-2.5 px-3 flex items-center justify-between hover:bg-[#171630] rounded-xl text-xs">
                  <div>
                    <span className="font-semibold text-white">{r.model_type}</span> • Winner:{" "}
                    <span className="text-emerald-400 font-mono font-bold capitalize">{r.winning_strategy}</span>
                    <span className="text-[#A4A4B8] block text-[11px]">
                      Target: {r.target_column} • {new Date(r.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    {r.registered_model_id && (
                      <Link
                        to={`/predict?model=${r.registered_model_id}`}
                        className="px-2.5 py-1 bg-[#8265FF] hover:bg-[#7152f8] text-white rounded text-[11px] font-semibold"
                      >
                        Predict
                      </Link>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Top Section: Dataset Upload & Profiling Summary */}
      <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#8265FF]/10 border border-[#8265FF]/20 flex items-center justify-center text-[#8265FF]">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white">Dataset Ingestion & Auto-Profiling</h2>
              <p className="text-xs text-[#A4A4B8]">
                Upload CSV to detect column types, missing values, duplicates, and collinearity alerts.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="file"
              id="csv-upload"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />
            <label
              htmlFor="csv-upload"
              className="px-4 py-2 bg-[#8265FF] hover:bg-[#7152f8] text-white rounded-xl text-xs font-semibold cursor-pointer shadow-md shadow-[#8265FF]/25 flex items-center gap-2 transition-all"
            >
              <Upload className="w-3.5 h-3.5" />
              {file ? "Change CSV Dataset" : "Upload Tabular CSV"}
            </label>
          </div>
        </div>

        {/* Profiling Summary Chips */}
        {profiling ? (
          <div className="py-6 text-center text-xs text-[#A4A4B8] flex items-center justify-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin text-[#8265FF]" />
            Running statistical profiling and collinearity analysis...
          </div>
        ) : profileData ? (
          <div className="space-y-6">
            
            {/* Quick Metrics Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
              <div className="p-3 bg-[#0D0D1B] rounded-xl border border-white/[0.04]">
                <span className="text-[10px] uppercase text-[#A4A4B8] block">Total Rows</span>
                <span className="text-base font-bold font-mono text-white">
                  {profileData.dataset_summary.total_rows}
                </span>
              </div>
              <div className="p-3 bg-[#0D0D1B] rounded-xl border border-white/[0.04]">
                <span className="text-[10px] uppercase text-[#A4A4B8] block">Total Columns</span>
                <span className="text-base font-bold font-mono text-white">
                  {profileData.dataset_summary.total_columns}
                </span>
              </div>
              <div className="p-3 bg-[#0D0D1B] rounded-xl border border-white/[0.04]">
                <span className="text-[10px] uppercase text-[#A4A4B8] block">Numeric Cols</span>
                <span className="text-base font-bold font-mono text-emerald-400">
                  {profileData.dataset_summary.numeric_columns_count}
                </span>
              </div>
              <div className="p-3 bg-[#0D0D1B] rounded-xl border border-white/[0.04]">
                <span className="text-[10px] uppercase text-[#A4A4B8] block">Categorical Cols</span>
                <span className="text-base font-bold font-mono text-[#8265FF]">
                  {profileData.dataset_summary.categorical_columns_count}
                </span>
              </div>
              <div className="p-3 bg-[#0D0D1B] rounded-xl border border-white/[0.04]">
                <span className="text-[10px] uppercase text-[#A4A4B8] block">Datetime Candidates</span>
                <span className="text-base font-bold font-mono text-amber-400">
                  {profileData.dataset_summary.datetime_columns_count}
                </span>
              </div>
              <div className="p-3 bg-[#0D0D1B] rounded-xl border border-white/[0.04]">
                <span className="text-[10px] uppercase text-[#A4A4B8] block">Duplicate Rows</span>
                <span className="text-base font-bold font-mono text-gray-300">
                  {profileData.dataset_summary.duplicate_rows}
                </span>
              </div>
            </div>

            {/* Quality Alerts & Collinearity Banner */}
            {profileData.quality_alerts && profileData.quality_alerts.length > 0 && (
              <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl space-y-2">
                <div className="flex items-center gap-2 text-xs font-bold text-amber-400">
                  <AlertTriangle className="w-4 h-4" />
                  Feature Quality Diagnostics ({profileData.quality_alerts.length} alerts)
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-[#A4A4B8]">
                  {profileData.quality_alerts.slice(0, 4).map((alert, idx) => (
                    <div key={idx} className="flex items-start gap-1.5">
                      <span className="text-amber-400">•</span>
                      <span>{alert.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Collinear Redundant Pairs */}
            {profileData.redundant_pairs && profileData.redundant_pairs.length > 0 && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl space-y-2">
                <div className="flex items-center gap-2 text-xs font-bold text-rose-400">
                  <Scissors className="w-4 h-4" />
                  High Collinearity Warning ({profileData.redundant_pairs.length} pairs with |r| ≥ {correlationThreshold})
                </div>
                <div className="flex flex-wrap gap-2 pt-1">
                  {profileData.redundant_pairs.map((p, idx) => (
                    <span key={idx} className="px-2.5 py-1 bg-[#0D0D1B] border border-rose-500/30 rounded text-[11px] font-mono text-rose-300">
                      {p.feature_a} ↔ {p.feature_b} (r = {p.correlation})
                    </span>
                  ))}
                </div>
              </div>
            )}

          </div>
        ) : (
          <div className="py-8 text-center text-xs text-[#A4A4B8]">
            Upload a CSV dataset above to start feature profiling and automated diagnostic inspection.
          </div>
        )}
      </div>

      {/* Main Grid: Transformation Builder vs Strategy Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: Transformation & Selector Configuration (5 cols) */}
        <div className="lg:col-span-5 bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center gap-2 border-b border-white/[0.06] pb-4">
            <Wand2 className="w-5 h-5 text-[#8265FF]" />
            <h2 className="text-base font-bold text-white tracking-tight">Pipeline Configuration</h2>
          </div>

          <form onSubmit={handleEvaluateStrategies} className="space-y-5">
            
            {/* Target Column & Task */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider block">
                  Target Column
                </label>
                {profileData?.columns ? (
                  <select
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
                  >
                    {profileData.columns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    placeholder="e.g. target"
                    className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
                  />
                )}
              </div>

              <div className="space-y-1.5">
                <label className="text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider block">
                  Task Type
                </label>
                <div className="grid grid-cols-2 gap-1.5">
                  <button
                    type="button"
                    onClick={() => handleTaskChange("classification")}
                    className={`py-2 rounded-xl text-xs font-semibold ${
                      taskType === "classification" ? "bg-[#8265FF] text-white" : "bg-[#171630] text-[#A4A4B8]"
                    }`}
                  >
                    Classify
                  </button>
                  <button
                    type="button"
                    onClick={() => handleTaskChange("regression")}
                    className={`py-2 rounded-xl text-xs font-semibold ${
                      taskType === "regression" ? "bg-[#8265FF] text-white" : "bg-[#171630] text-[#A4A4B8]"
                    }`}
                  >
                    Regress
                  </button>
                </div>
              </div>
            </div>

            {/* Estimator Architecture */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider block">
                Base Estimator
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
              >
                {(taskType === "classification" ? classificationModels : regressionModels).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Feature Engineering Options */}
            <div className="p-4 bg-[#0D0D1B] rounded-xl border border-white/[0.04] space-y-3">
              <span className="text-xs font-bold text-white flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-[#8265FF]" />
                Transformations (In-Pipeline)
              </span>

              {/* Datetime Selection */}
              {profileData?.datetime_candidates && profileData.datetime_candidates.length > 0 && (
                <div className="space-y-1">
                  <label className="text-[10px] uppercase text-[#A4A4B8] block">
                    Calendar Feature Extraction (Year, Month, Day, DOW)
                  </label>
                  <div className="flex flex-wrap gap-1.5">
                    {profileData.datetime_candidates.map((col) => (
                      <button
                        key={col}
                        type="button"
                        onClick={() => {
                          setDatetimeCols((prev) =>
                            prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
                          );
                        }}
                        className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-all ${
                          datetimeCols.includes(col)
                            ? "bg-[#8265FF] text-white font-bold"
                            : "bg-[#171630] text-[#A4A4B8]"
                        }`}
                      >
                        {col}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Log1p Transformation */}
              {profileData?.columns && (
                <div className="space-y-1">
                  <label className="text-[10px] uppercase text-[#A4A4B8] block">
                    Logarithmic Transformation (log1p on non-negative)
                  </label>
                  <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto pr-1">
                    {profileData.columns
                      .filter((c) => c.inferred_type === "numeric" && c.is_non_negative && !c.is_target)
                      .map((c) => (
                        <button
                          key={c.name}
                          type="button"
                          onClick={() => {
                            setLogCols((prev) =>
                              prev.includes(c.name) ? prev.filter((x) => x !== c.name) : [...prev, c.name]
                            );
                          }}
                          className={`px-2 py-0.5 rounded text-[11px] font-mono transition-all ${
                            logCols.includes(c.name)
                              ? "bg-emerald-600 text-white font-bold"
                              : "bg-[#171630] text-[#A4A4B8]"
                          }`}
                        >
                          {c.name}
                        </button>
                      ))}
                  </div>
                </div>
              )}

              {/* Toggles */}
              <div className="flex items-center justify-between pt-1 text-xs text-[#A4A4B8]">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={clipOutliers}
                    onChange={(e) => setClipOutliers(e.target.checked)}
                    className="accent-[#8265FF]"
                  />
                  <span>Clip 1st/99th Outliers</span>
                </label>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={groupRare}
                    onChange={(e) => setGroupRare(e.target.checked)}
                    className="accent-[#8265FF]"
                  />
                  <span>Group Rare Categories (&lt;2%)</span>
                </label>
              </div>

              {/* Preview Button */}
              <button
                type="button"
                onClick={handlePreviewTransformations}
                disabled={previewing || !file}
                className="w-full py-1.5 rounded-lg text-xs bg-[#171630] hover:bg-[#1f1e40] text-gray-300 border border-white/[0.08] transition-all flex items-center justify-center gap-1.5"
              >
                {previewing ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3 h-3 text-[#A855F7]" />}
                Preview Sample Transformations
              </button>
            </div>

            {/* Transformation Preview Modal / Box */}
            {previewResult && (
              <div className="p-3 bg-[#0D0D1B] border border-white/[0.06] rounded-xl text-xs space-y-2">
                <span className="font-bold text-white block">Preview Results:</span>
                <div className="text-[11px] text-[#A4A4B8]">
                  Created Columns: <span className="text-emerald-400 font-mono">{previewResult.created_columns?.join(", ") || "None"}</span>
                </div>
                <div className="text-[11px] text-[#A4A4B8]">
                  Dropped Original: <span className="text-rose-400 font-mono">{previewResult.dropped_columns?.join(", ") || "None"}</span>
                </div>
              </div>
            )}

            {/* Feature Selection Strategy */}
            <div className="space-y-3 p-4 bg-[#0D0D1B] rounded-xl border border-white/[0.04]">
              <span className="text-xs font-bold text-white flex items-center gap-1.5">
                <Filter className="w-3.5 h-3.5 text-[#8265FF]" />
                Feature Selection Method
              </span>

              <select
                value={selectionMethod}
                onChange={(e) => setSelectionMethod(e.target.value)}
                className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
              >
                <option value="none">None (Preserve All Features)</option>
                <option value="model_based">Model-Based Importance (Random Forest)</option>
                <option value="variance">Variance Threshold (Drop Low Variance)</option>
                <option value="correlation">Correlation Filter (Drop Collinear Pairs)</option>
                <option value="k_best">SelectKBest (ANOVA / F-Test)</option>
              </select>

              {selectionMethod === "variance" && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-[#A4A4B8]">
                    <span>Variance Threshold</span>
                    <span className="font-mono text-white">{varianceThreshold}</span>
                  </div>
                  <input
                    type="range"
                    min="0.001"
                    max="0.05"
                    step="0.005"
                    value={varianceThreshold}
                    onChange={(e) => setVarianceThreshold(parseFloat(e.target.value))}
                    className="w-full accent-[#8265FF]"
                  />
                </div>
              )}

              {selectionMethod === "correlation" && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-[#A4A4B8]">
                    <span>Redundancy |r| Threshold</span>
                    <span className="font-mono text-white">{correlationThreshold}</span>
                  </div>
                  <input
                    type="range"
                    min="0.75"
                    max="0.95"
                    step="0.05"
                    value={correlationThreshold}
                    onChange={(e) => setCorrelationThreshold(parseFloat(e.target.value))}
                    className="w-full accent-[#8265FF]"
                  />
                </div>
              )}

              {(selectionMethod === "model_based" || selectionMethod === "k_best") && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-[#A4A4B8]">
                    <span>Top Features Count (K)</span>
                    <span className="font-mono text-white">{kBestCount}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="20"
                    step="1"
                    value={kBestCount}
                    onChange={(e) => setKBestCount(parseInt(e.target.value))}
                    className="w-full accent-[#8265FF]"
                  />
                </div>
              )}
            </div>

            {/* Error Display */}
            {evalError && (
              <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{evalError}</span>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={evaluating || !file}
              className="w-full py-3 px-4 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7354f8] hover:to-[#9945e8] text-white shadow-lg shadow-[#8265FF]/25 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
            >
              {evaluating ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-white" />
                  <span>Evaluating 4 Strategies ({elapsedTime}s)...</span>
                </>
              ) : (
                <>
                  <PlayCircle className="w-4 h-4" />
                  <span>Evaluate Feature Strategies</span>
                </>
              )}
            </button>

          </form>
        </div>

        {/* Right Column: Multi-Strategy Leaderboard & Results (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          
          {evalResult ? (
            <div className="space-y-6">
              
              {/* Winning Strategy Spotlight Card */}
              <div className="bg-gradient-to-br from-[#111126] to-[#171630] border border-[#8265FF]/30 rounded-2xl p-6 shadow-xl space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.08] pb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                      <CheckCircle2 className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] uppercase font-bold tracking-wider text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                          Winning Strategy: {evalResult.winning_strategy}
                        </span>
                        <span className="text-xs text-[#A4A4B8] font-mono">
                          {evalResult.duration_seconds}s
                        </span>
                      </div>
                      <h3 className="text-lg font-extrabold text-white mt-1">
                        Cross-Validation Selected Strategy
                      </h3>
                    </div>
                  </div>

                  <div className="text-right sm:self-center">
                    <div className="text-[10px] uppercase text-[#A4A4B8]">Features Kept</div>
                    <div className="text-2xl font-black font-mono text-emerald-400">
                      {evalResult.data_quality?.selected_feature_count} / {evalResult.data_quality?.original_feature_count}
                    </div>
                  </div>
                </div>

                {/* Downstream Actions */}
                {evalResult.registered_model_id && (
                  <div className="pt-2 flex flex-wrap items-center justify-between gap-3">
                    <span className="text-xs text-[#A4A4B8]">
                      Registered Pipeline: <span className="text-white font-mono">ID #{evalResult.registered_model_id}</span>
                    </span>
                    <div className="flex items-center gap-2">
                      <Link
                        to={`/predict?model=${evalResult.registered_model_id}`}
                        className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white shadow-sm flex items-center gap-1.5"
                      >
                        <PlayCircle className="w-3.5 h-3.5" />
                        Inference
                      </Link>
                      <Link
                        to={`/explain?model=${evalResult.registered_model_id}`}
                        className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] flex items-center gap-1.5"
                      >
                        <ShieldCheck className="w-3.5 h-3.5 text-[#8265FF]" />
                        Explain
                      </Link>
                      <Link
                        to={`/tune?model_type=${evalResult.model_type}&task=${evalResult.task_type}`}
                        className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] flex items-center gap-1.5"
                      >
                        <Sliders className="w-3.5 h-3.5 text-[#A855F7]" />
                        Tune
                      </Link>
                    </div>
                  </div>
                )}
              </div>

              {/* Multi-Strategy Comparison Table */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-[#8265FF]" />
                    <h3 className="text-sm font-bold text-white">Strategy Cross-Validation Leaderboard</h3>
                  </div>
                  <span className="text-[11px] text-[#A4A4B8] font-mono">
                    Scored on Training Partition
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-white/[0.06] text-[#A4A4B8] text-[10px] uppercase">
                        <th className="py-2.5 px-3">Strategy</th>
                        <th className="py-2.5 px-3">CV Mean Score</th>
                        <th className="py-2.5 px-3">CV Std Dev</th>
                        <th className="py-2.5 px-3">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {evalResult.strategy_results?.map((s) => {
                        const isWinner = s.strategy === evalResult.winning_strategy;
                        return (
                          <tr key={s.strategy} className={`hover:bg-white/[0.02] ${isWinner ? "bg-[#8265FF]/10 font-bold" : ""}`}>
                            <td className="py-2.5 px-3 capitalize flex items-center gap-1.5">
                              {s.strategy.replace("_", " ")}
                              {isWinner && (
                                <span className="text-[9px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.2 rounded border border-emerald-500/30">
                                  WINNER
                                </span>
                              )}
                            </td>
                            <td className="py-2.5 px-3 font-mono">
                              <span className={isWinner ? "text-emerald-400" : "text-white"}>{s.cv_mean}</span>
                            </td>
                            <td className="py-2.5 px-3 font-mono text-[#A4A4B8]">±{s.cv_std}</td>
                            <td className="py-2.5 px-3 uppercase text-[10px] text-emerald-400">{s.status}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Holdout Test Performance (Baseline vs Winner) */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-[#8265FF]" />
                    Holdout Test Evaluation (Baseline vs Winning Pipeline)
                  </h3>
                  <span className="text-[11px] text-[#A4A4B8]">
                    {evalResult.data_quality?.test_samples} test samples
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {Object.entries(evalResult.delta_metrics || {}).map(([metricKey, d]) => (
                    <div key={metricKey} className="p-3 bg-[#0D0D1B] rounded-xl border border-white/[0.04] space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] uppercase text-[#A4A4B8] font-semibold">{metricKey}</span>
                        {d.percentage_change !== null && (
                          <span className={`text-[10px] font-bold px-1.5 py-0.2 rounded ${d.improved ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"}`}>
                            {d.percentage_change > 0 ? `+${d.percentage_change}%` : `${d.percentage_change}%`}
                          </span>
                        )}
                      </div>
                      <div className="grid grid-cols-2 text-xs">
                        <div>
                          <span className="text-[10px] text-[#A4A4B8] block">Baseline</span>
                          <span className="font-mono text-gray-300">{d.baseline}</span>
                        </div>
                        <div>
                          <span className="text-[10px] text-[#A4A4B8] block">Winner</span>
                          <span className="font-mono text-emerald-400 font-bold">{d.winning}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Selected Features Chips */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-3">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Check className="w-4 h-4 text-emerald-400" />
                  Selected Final Features ({evalResult.selected_features?.length || 0})
                </h3>
                <div className="flex flex-wrap gap-2 max-h-36 overflow-y-auto">
                  {evalResult.selected_features?.map((f, i) => (
                    <span key={i} className="px-2.5 py-1 bg-[#0D0D1B] border border-white/[0.08] rounded-lg text-xs font-mono text-emerald-300">
                      {f}
                    </span>
                  ))}
                </div>
              </div>

            </div>
          ) : (
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-12 text-center space-y-4">
              <div className="w-16 h-16 rounded-3xl bg-[#171630] border border-white/[0.08] flex items-center justify-center mx-auto text-[#8265FF]">
                <Wand2 className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-bold text-white">No Active Feature Evaluation</h3>
              <p className="text-xs text-[#A4A4B8] max-w-md mx-auto leading-relaxed">
                Configure your transformation pipeline and feature selection method on the left, then run multi-strategy cross-validation.
              </p>
            </div>
          )}

        </div>

      </div>

    </div>
  );
}
