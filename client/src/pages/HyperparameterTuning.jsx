import { useState, useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  Sliders,
  Sparkles,
  Cpu,
  PlayCircle,
  ShieldCheck,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  ArrowRight,
  RefreshCw,
  Upload,
  FileText,
  AlertCircle,
  Clock,
  Database,
  ChevronRight,
  XCircle,
  BarChart3,
  Filter,
  Trash2,
  ChevronDown,
  Info
} from "lucide-react";
import { getApiBase } from "../utils/api";

export default function HyperparameterTuning() {
  const [searchParams] = useSearchParams();
  const prefillModelType = searchParams.get("model_type") || "";
  const prefillTask = searchParams.get("task") || "";

  // Dataset State
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [rowCount, setRowCount] = useState(null);

  // Configuration State
  const [taskType, setTaskType] = useState(prefillTask === "regression" ? "regression" : "classification");
  const [targetColumn, setTargetColumn] = useState("");
  const [modelType, setModelType] = useState(
    prefillModelType || (prefillTask === "regression" ? "RandomForest" : "RandomForestClassifier")
  );
  const [metric, setMetric] = useState(prefillTask === "regression" ? "rmse" : "f1_macro");
  const [nTrials, setNTrials] = useState(15);
  const [nSplits, setNSplits] = useState(3);
  const [scaler, setScaler] = useState("standard");
  const [missing, setMissing] = useState("median");
  const [splitRatio, setSplitRatio] = useState(0.8);

  // Run & Execution State
  const [tuningActive, setTuningActive] = useState(false);
  const [tuningError, setTuningError] = useState(null);
  const [result, setResult] = useState(null);
  const [activeRunId, setActiveRunId] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  // History State
  const [historyRuns, setHistoryRuns] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  // Model and Metric Options
  const classificationModels = [
    { id: "RandomForestClassifier", name: "Random Forest Classifier", desc: "Ensemble of balanced decision trees" },
    { id: "DecisionTreeClassifier", name: "Decision Tree Classifier", desc: "Single interpretable tree model" },
    { id: "ExtraTreesClassifier", name: "Extra Trees Classifier", desc: "Extremely randomized ensemble trees" },
    { id: "LogisticRegression", name: "Logistic Regression", desc: "Linear classification with L-BFGS" },
  ];

  const regressionModels = [
    { id: "RandomForest", name: "Random Forest Regressor", desc: "Ensemble of regression trees" },
    { id: "DecisionTree", name: "Decision Tree Regressor", desc: "Single decision tree regressor" },
  ];

  const classificationMetrics = [
    { id: "f1_macro", name: "F1 Macro (Default)", dir: "maximize" },
    { id: "accuracy", name: "Accuracy", dir: "maximize" },
    { id: "balanced_accuracy", name: "Balanced Accuracy", dir: "maximize" },
    { id: "f1_weighted", name: "F1 Weighted", dir: "maximize" },
    { id: "roc_auc", name: "ROC-AUC Score", dir: "maximize" },
    { id: "log_loss", name: "Log Loss", dir: "minimize" },
  ];

  const regressionMetrics = [
    { id: "rmse", name: "RMSE (Root Mean Squared Error)", dir: "minimize" },
    { id: "mae", name: "MAE (Mean Absolute Error)", dir: "minimize" },
    { id: "r2", name: "R² Coefficient of Determination", dir: "maximize" },
  ];

  // Update default model/metric when taskType changes
  const handleTaskTypeChange = (newTask) => {
    setTaskType(newTask);
    if (newTask === "classification") {
      setModelType("RandomForestClassifier");
      setMetric("f1_macro");
    } else {
      setModelType("RandomForest");
      setMetric("rmse");
    }
  };

  // Timer while tuning is active
  useEffect(() => {
    let interval;
    if (tuningActive) {
      const startTime = Date.now();
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
      }, 500);
    } else {
      setElapsedTime(0);
    }
    return () => clearInterval(interval);
  }, [tuningActive]);

  // Load history on mount
  useEffect(() => {
    fetchTuningHistory();
  }, []);

  const fetchTuningHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/tuning?limit=20`);
      if (res.ok) {
        const data = await res.json();
        setHistoryRuns(data.items || []);
      }
    } catch {
      // ignore network errors on init
    } finally {
      setHistoryLoading(false);
    }
  };

  // CSV File Handler
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    if (!selectedFile.name.endsWith(".csv")) {
      setTuningError("Please upload a valid CSV file (.csv)");
      return;
    }

    setFile(selectedFile);
    setTuningError(null);

    // Quick preview parse for columns
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const lines = text.split("\n").filter((l) => l.trim().length > 0);
        if (lines.length > 0) {
          const headers = lines[0].split(",").map((h) => h.trim().replace(/^["']|["']$/g, ""));
          setColumns(headers);
          setRowCount(lines.length - 1);
          if (headers.length > 0 && !targetColumn) {
            setTargetColumn(headers[headers.length - 1]);
          }
        }
      } catch (err) {
        console.error("Preview parse failed", err);
      }
    };
    reader.readAsText(selectedFile.slice(0, 65536));
  };

  // Start Tuning Run
  const handleStartTuning = async (e) => {
    e.preventDefault();
    if (!file) {
      setTuningError("Please select a CSV dataset to optimize.");
      return;
    }
    if (!targetColumn) {
      setTuningError("Please select a target column.");
      return;
    }

    setTuningActive(true);
    setTuningError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("task_type", taskType);
    formData.append("target_column", targetColumn);
    formData.append("model_type", modelType);
    formData.append("metric", metric);
    formData.append("n_trials", nTrials);
    formData.append("n_splits", nSplits);
    formData.append("scaler", scaler);
    formData.append("missing", missing);
    formData.append("splitRatio", splitRatio);

    try {
      const res = await fetch(`${getApiBase()}/tuning/start`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Hyperparameter optimization failed.");
      }

      setResult(data);
      setActiveRunId(data.run_id);
      fetchTuningHistory();
    } catch (err) {
      setTuningError(err.message);
    } finally {
      setTuningActive(false);
    }
  };

  // Cancel Run
  const handleCancelRun = async () => {
    if (!activeRunId) return;
    try {
      await fetch(`${getApiBase()}/tuning/${activeRunId}/cancel`, { method: "POST" });
    } catch (err) {
      console.error("Failed to signal cancellation", err);
    }
  };

  // Load a historical run
  const handleSelectHistoryRun = async (runId) => {
    setHistoryLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/tuning/${runId}`);
      if (res.ok) {
        const data = await res.json();
        setResult(data);
        setActiveRunId(data.run_id);
        setShowHistory(false);
      }
    } catch (err) {
      setTuningError("Failed to load historical tuning run.");
    } finally {
      setHistoryLoading(false);
    }
  };

  // Delete a historical run
  const handleDeleteHistoryRun = async (runId, e) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${getApiBase()}/tuning/${runId}`, { method: "DELETE" });
      if (res.ok) {
        setHistoryRuns((prev) => prev.filter((r) => r.run_id !== runId));
        if (result?.run_id === runId) {
          setResult(null);
        }
      }
    } catch (err) {
      console.error("Failed to delete run", err);
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
              Optuna Optimization Engine
            </span>
            <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-400 bg-emerald-500/10 px-2.5 py-0.5 rounded-full border border-emerald-500/20">
              Zero-Leakage Folds
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Hyperparameter Tuning Workspace</h1>
          <p className="text-sm text-[#A4A4B8] mt-1 max-w-2xl">
            Bayesian Tree-structured Parzen Estimator (TPE) hyperparameter search with fold-isolated preprocessing, cross-validation scoring, and unbiased holdout evaluation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] transition-all flex items-center gap-2"
          >
            <Clock className="w-3.5 h-3.5 text-[#8265FF]" />
            Tuning History ({historyRuns.length})
          </button>
        </div>
      </div>

      {/* History Drawer Modal */}
      {showHistory && (
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-[#8265FF]" />
              <h3 className="text-sm font-bold text-white">Historical Tuning Runs</h3>
            </div>
            <button
              onClick={() => setShowHistory(false)}
              className="text-[#A4A4B8] hover:text-white p-1"
            >
              <XCircle className="w-4 h-4" />
            </button>
          </div>

          {historyLoading ? (
            <div className="py-8 text-center text-xs text-[#A4A4B8]">Loading tuning history...</div>
          ) : historyRuns.length === 0 ? (
            <div className="py-8 text-center text-xs text-[#A4A4B8]">
              No previous tuning runs recorded yet. Start a new run below!
            </div>
          ) : (
            <div className="divide-y divide-white/[0.04] max-h-72 overflow-y-auto pr-1">
              {historyRuns.map((run) => (
                <div
                  key={run.run_id}
                  onClick={() => handleSelectHistoryRun(run.run_id)}
                  className="py-3 px-3 rounded-xl hover:bg-[#171630] cursor-pointer flex items-center justify-between transition-colors group"
                >
                  <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-xs text-white group-hover:text-[#8265FF] transition-colors">
                        {run.model_type}
                      </span>
                      <span className="text-[10px] uppercase px-1.5 py-0.2 rounded bg-white/[0.06] text-[#A4A4B8]">
                        {run.task_type}
                      </span>
                    </div>
                    <div className="text-[11px] text-[#A4A4B8] mt-0.5">
                      Target: <span className="text-white">{run.target_column}</span> • {run.n_trials} trials • Best Score:{" "}
                      <span className="text-emerald-400 font-mono font-semibold">{run.best_score}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <span className="text-[10px] text-[#A4A4B8]">
                      {new Date(run.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <button
                      onClick={(e) => handleDeleteHistoryRun(run.run_id, e)}
                      className="p-1 rounded text-[#A4A4B8] hover:text-rose-400 opacity-60 hover:opacity-100"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                    <ChevronRight className="w-4 h-4 text-[#A4A4B8] group-hover:text-white" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Main Grid: Config vs Progress/Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: Configuration Form (5 cols) */}
        <div className="lg:col-span-5 bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-6">
          <div className="flex items-center gap-2 border-b border-white/[0.06] pb-4">
            <Sliders className="w-5 h-5 text-[#8265FF]" />
            <h2 className="text-base font-bold text-white tracking-tight">Optimization Parameters</h2>
          </div>

          <form onSubmit={handleStartTuning} className="space-y-5">
            
            {/* Dataset Upload Area */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#A4A4B8] uppercase tracking-wider flex items-center justify-between">
                <span>Tabular CSV Dataset</span>
                {rowCount !== null && (
                  <span className="text-emerald-400 text-[10px] font-mono">
                    {rowCount} rows • {columns.length} cols
                  </span>
                )}
              </label>
              
              <div className="relative border-2 border-dashed border-white/[0.1] hover:border-[#8265FF]/40 rounded-xl p-4 text-center cursor-pointer transition-all bg-[#0D0D1B]/50 group">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  disabled={tuningActive}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                />
                <div className="flex flex-col items-center justify-center space-y-2">
                  <div className="w-9 h-9 rounded-lg bg-[#171630] flex items-center justify-center text-[#8265FF] group-hover:scale-105 transition-transform">
                    <Upload className="w-4 h-4" />
                  </div>
                  <div className="text-xs text-[#A4A4B8]">
                    {file ? (
                      <span className="font-semibold text-white break-all">{file.name}</span>
                    ) : (
                      <span>Drop CSV file here or <span className="text-[#8265FF] underline">browse</span></span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Task Type Switcher */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#A4A4B8] uppercase tracking-wider">
                Supervised Task Type
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => handleTaskTypeChange("classification")}
                  className={`py-2 px-3 rounded-xl text-xs font-semibold transition-all ${
                    taskType === "classification"
                      ? "bg-[#8265FF] text-white shadow-lg shadow-[#8265FF]/25"
                      : "bg-[#171630] text-[#A4A4B8] hover:text-white"
                  }`}
                >
                  Classification
                </button>
                <button
                  type="button"
                  onClick={() => handleTaskTypeChange("regression")}
                  className={`py-2 px-3 rounded-xl text-xs font-semibold transition-all ${
                    taskType === "regression"
                      ? "bg-[#8265FF] text-white shadow-lg shadow-[#8265FF]/25"
                      : "bg-[#171630] text-[#A4A4B8] hover:text-white"
                  }`}
                >
                  Regression
                </button>
              </div>
            </div>

            {/* Target Column Selector */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#A4A4B8] uppercase tracking-wider">
                Target Column
              </label>
              {columns.length > 0 ? (
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  disabled={tuningActive}
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                >
                  <option value="">Select target column...</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  placeholder="e.g. target or label"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  disabled={tuningActive}
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2.5 text-xs text-white placeholder-[#A4A4B8]/50 focus:outline-none focus:border-[#8265FF]"
                />
              )}
            </div>

            {/* Model Architecture Selector */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#A4A4B8] uppercase tracking-wider">
                Candidate Estimator
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                disabled={tuningActive}
                className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                {(taskType === "classification" ? classificationModels : regressionModels).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-[#A4A4B8]">
                {(taskType === "classification" ? classificationModels : regressionModels).find((m) => m.id === modelType)?.desc}
              </p>
            </div>

            {/* Optimization Objective Metric */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[#A4A4B8] uppercase tracking-wider">
                Optimization Objective Metric
              </label>
              <select
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                disabled={tuningActive}
                className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                {(taskType === "classification" ? classificationMetrics : regressionMetrics).map((met) => (
                  <option key={met.id} value={met.id}>
                    {met.name} ({met.dir === "maximize" ? "Maximize ▲" : "Minimize ▼"})
                  </option>
                ))}
              </select>
            </div>

            {/* Sliders for Trial Budget and CV Folds */}
            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-[#A4A4B8]">
                  <span>Trials Budget</span>
                  <span className="font-mono text-white font-semibold">{nTrials}</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="40"
                  step="5"
                  value={nTrials}
                  onChange={(e) => setNTrials(parseInt(e.target.value))}
                  disabled={tuningActive}
                  className="w-full accent-[#8265FF] cursor-pointer"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-[#A4A4B8]">
                  <span>CV Folds</span>
                  <span className="font-mono text-white font-semibold">{nSplits}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="5"
                  step="1"
                  value={nSplits}
                  onChange={(e) => setNSplits(parseInt(e.target.value))}
                  disabled={tuningActive}
                  className="w-full accent-[#8265FF] cursor-pointer"
                />
              </div>
            </div>

            {/* Preprocessing and Split Controls */}
            <div className="grid grid-cols-3 gap-3 pt-2 text-xs">
              <div>
                <label className="text-[10px] uppercase text-[#A4A4B8] font-semibold block mb-1">Scaler</label>
                <select
                  value={scaler}
                  onChange={(e) => setScaler(e.target.value)}
                  disabled={tuningActive}
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-lg p-2 text-white"
                >
                  <option value="standard">Standard</option>
                  <option value="minmax">MinMax</option>
                  <option value="robust">Robust</option>
                </select>
              </div>

              <div>
                <label className="text-[10px] uppercase text-[#A4A4B8] font-semibold block mb-1">Imputer</label>
                <select
                  value={missing}
                  onChange={(e) => setMissing(e.target.value)}
                  disabled={tuningActive}
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-lg p-2 text-white"
                >
                  <option value="median">Median</option>
                  <option value="mean">Mean</option>
                </select>
              </div>

              <div>
                <label className="text-[10px] uppercase text-[#A4A4B8] font-semibold block mb-1">Train Split</label>
                <select
                  value={splitRatio}
                  onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
                  disabled={tuningActive}
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-lg p-2 text-white"
                >
                  <option value="0.8">80 / 20</option>
                  <option value="0.75">75 / 25</option>
                  <option value="0.7">70 / 30</option>
                </select>
              </div>
            </div>

            {/* Error Message */}
            {tuningError && (
              <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{tuningError}</span>
              </div>
            )}

            {/* Submit CTA & Cancel Button */}
            <div className="pt-2 flex items-center gap-3">
              <button
                type="submit"
                disabled={tuningActive || !file}
                className="flex-1 py-3 px-4 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7354f8] hover:to-[#9945e8] text-white shadow-lg shadow-[#8265FF]/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
              >
                {tuningActive ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-white" />
                    <span>Optimizing ({elapsedTime}s)...</span>
                  </>
                ) : (
                  <>
                    <PlayCircle className="w-4 h-4" />
                    <span>Run Optuna Optimization</span>
                  </>
                )}
              </button>

              {tuningActive && (
                <button
                  type="button"
                  onClick={handleCancelRun}
                  className="px-4 py-3 rounded-xl font-semibold text-xs bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 transition-all"
                >
                  Stop
                </button>
              )}
            </div>

          </form>
        </div>

        {/* Right Column: Execution Status & Leaderboard (7 cols) */}
        <div className="lg:col-span-7 space-y-6">

          {/* Running State Banner */}
          {tuningActive && (
            <div className="bg-[#111126] border border-[#8265FF]/30 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-[#8265FF] via-[#A855F7] to-[#8265FF] animate-pulse" />
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-2xl bg-[#8265FF]/10 border border-[#8265FF]/20 flex items-center justify-center text-[#8265FF]">
                  <Cpu className="w-6 h-6 animate-pulse" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    Bayesian Optimization in Progress
                    <span className="text-xs font-mono text-[#8265FF] font-normal">[{elapsedTime}s]</span>
                  </h3>
                  <p className="text-xs text-[#A4A4B8] mt-1">
                    Sampling hyperparameter trials with Optuna TPESampler across {nSplits} cross-validation folds. Preserving holdout partition for unbiased validation.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Result View */}
          {result && !tuningActive ? (
            <div className="space-y-6">
              
              {/* Best Trial Spotlight Card */}
              <div className="bg-gradient-to-br from-[#111126] to-[#171630] border border-[#8265FF]/30 rounded-2xl p-6 shadow-xl relative overflow-hidden">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.08] pb-5">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                      <CheckCircle2 className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] uppercase font-bold tracking-wider text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                          Best Trial #{result.best_trial_number}
                        </span>
                        <span className="text-xs text-[#A4A4B8] font-mono">
                          Completed in {result.duration_seconds}s
                        </span>
                      </div>
                      <h3 className="text-lg font-extrabold text-white mt-1">
                        {result.model_type} Optimization Succeeded
                      </h3>
                    </div>
                  </div>

                  <div className="text-right sm:self-center">
                    <div className="text-[10px] uppercase text-[#A4A4B8] tracking-wider">CV Objective Score</div>
                    <div className="text-2xl font-black font-mono text-emerald-400">
                      {result.best_score}
                    </div>
                    <div className="text-[10px] text-[#A4A4B8] capitalize">
                      {result.metric} ({result.direction})
                    </div>
                  </div>
                </div>

                {/* Best Hyperparameters Chips */}
                <div className="pt-5 space-y-2">
                  <span className="text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider block">
                    Winning Hyperparameter Configuration:
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(result.best_params || {}).map(([paramKey, paramVal]) => (
                      <div
                        key={paramKey}
                        className="px-3 py-1.5 rounded-xl bg-[#0D0D1B] border border-white/[0.08] flex items-center gap-1.5 text-xs"
                      >
                        <span className="text-[#A4A4B8] font-mono">{paramKey}:</span>
                        <span className="text-[#8265FF] font-mono font-bold">
                          {typeof paramVal === "number" ? paramVal.toFixed ? paramVal.toFixed(4) : paramVal : String(paramVal)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Action Links for downstream registry, predictions, and explainability */}
                {result.registered_model_id && (
                  <div className="mt-6 pt-5 border-t border-white/[0.08] flex flex-wrap items-center justify-between gap-3">
                    <div className="text-xs text-[#A4A4B8]">
                      Registered Artifact: <span className="text-white font-mono">ID #{result.registered_model_id}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Link
                        to={`/predict?model=${result.registered_model_id}`}
                        className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white shadow-md shadow-[#8265FF]/20 flex items-center gap-1.5 transition-all"
                      >
                        <PlayCircle className="w-3.5 h-3.5" />
                        Test Predictions
                      </Link>
                      <Link
                        to={`/explain?model=${result.registered_model_id}`}
                        className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] flex items-center gap-1.5 transition-all"
                      >
                        <ShieldCheck className="w-3.5 h-3.5 text-[#8265FF]" />
                        Explain Model
                      </Link>
                    </div>
                  </div>
                )}
              </div>

              {/* Baseline vs Tuned Holdout Comparison */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-[#8265FF]" />
                      Holdout Test Performance (Baseline vs Tuned)
                    </h3>
                    <p className="text-[11px] text-[#A4A4B8] mt-0.5">
                      Evaluated on untouched holdout split ({result.data_quality?.testing_samples} samples)
                    </p>
                  </div>
                </div>

                {/* Comparison Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {Object.entries(result.delta_metrics || {}).slice(0, 6).map(([mKey, deltaInfo]) => {
                    const isImproved = deltaInfo.improved;
                    return (
                      <div
                        key={mKey}
                        className="p-3.5 rounded-xl bg-[#0D0D1B] border border-white/[0.04] space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold uppercase tracking-wider text-[#A4A4B8]">
                            {mKey}
                          </span>
                          {deltaInfo.percentage_change !== null && (
                            <span
                              className={`text-[10px] font-bold px-1.5 py-0.5 rounded flex items-center gap-0.5 ${
                                isImproved
                                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                  : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                              }`}
                            >
                              {isImproved ? <TrendingUp className="w-2.5 h-2.5" /> : <TrendingDown className="w-2.5 h-2.5" />}
                              {deltaInfo.percentage_change > 0 ? `+${deltaInfo.percentage_change}%` : `${deltaInfo.percentage_change}%`}
                            </span>
                          )}
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-[10px] text-[#A4A4B8] block">Baseline</span>
                            <span className="font-mono text-gray-300 font-semibold">
                              {deltaInfo.baseline !== null ? deltaInfo.baseline : "—"}
                            </span>
                          </div>
                          <div>
                            <span className="text-[10px] text-[#A4A4B8] block">Tuned</span>
                            <span className="font-mono text-emerald-400 font-bold">
                              {deltaInfo.tuned !== null ? deltaInfo.tuned : "—"}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Optuna Trials Leaderboard Table */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <div className="flex items-center gap-2">
                    <Filter className="w-4 h-4 text-[#8265FF]" />
                    <h3 className="text-sm font-bold text-white">Optuna Trial Exploration Leaderboard</h3>
                  </div>
                  <span className="text-xs text-[#A4A4B8] font-mono">
                    {result.trials?.length || 0} Trials Total
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-white/[0.06] text-[#A4A4B8] text-[10px] uppercase tracking-wider">
                        <th className="py-2.5 px-3">Trial #</th>
                        <th className="py-2.5 px-3">CV Score ({result.metric})</th>
                        <th className="py-2.5 px-3">Duration</th>
                        <th className="py-2.5 px-3">Parameters Sampled</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {(result.trials || []).map((t) => {
                        const isWinningTrial = t.trial_number === result.best_trial_number;
                        return (
                          <tr
                            key={t.trial_number}
                            className={`hover:bg-white/[0.02] transition-colors ${
                              isWinningTrial ? "bg-[#8265FF]/10 font-semibold" : ""
                            }`}
                          >
                            <td className="py-2.5 px-3 font-mono">
                              <span className="flex items-center gap-1.5">
                                #{t.trial_number}
                                {isWinningTrial && (
                                  <span className="text-[9px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.2 rounded border border-emerald-500/30">
                                    BEST
                                  </span>
                                )}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 font-mono">
                              <span className={isWinningTrial ? "text-emerald-400 font-bold" : "text-white"}>
                                {t.cv_score !== null ? t.cv_score : "Pruned"}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 font-mono text-[#A4A4B8]">
                              {t.duration_seconds}s
                            </td>
                            <td className="py-2.5 px-3 font-mono text-[11px] text-[#A4A4B8] max-w-xs truncate">
                              {JSON.stringify(t.params)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

            </div>
          ) : !tuningActive ? (
            /* Empty State Guide */
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-12 text-center space-y-4">
              <div className="w-16 h-16 rounded-3xl bg-[#171630] border border-white/[0.08] flex items-center justify-center mx-auto text-[#8265FF]">
                <Sliders className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-bold text-white">No Active Tuning Run</h3>
              <p className="text-xs text-[#A4A4B8] max-w-md mx-auto leading-relaxed">
                Upload your structured dataset on the left, pick an estimator architecture, define your trial budget, and run automated Optuna optimization.
              </p>
              <div className="pt-2 flex justify-center gap-4 text-[11px] text-[#A4A4B8]">
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Leakage-free CV
                </span>
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Baseline Comparison
                </span>
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Direct Registry
                </span>
              </div>
            </div>
          ) : null}

        </div>

      </div>

    </div>
  );
}
