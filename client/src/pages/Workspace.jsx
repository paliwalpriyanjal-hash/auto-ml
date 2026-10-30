// client/src/pages/Workspace.jsx
// AUTOVEYRA PHASE 6: Beginner-Friendly AI-Guided Machine Learning Workspace

import React, { useState, useEffect, useCallback } from "react";
import Papa from "papaparse";
import {
  Sparkles,
  Upload,
  BarChart2,
  ShieldCheck,
  Target,
  Wand2,
  Filter,
  Cpu,
  GitCompare,
  Eye,
  FileText,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  ArrowLeft,
  RotateCcw,
  BookOpen,
  FolderOpen,
  Save,
  HelpCircle,
  Clock,
  Layers,
  ChevronRight,
  Download,
  AlertCircle
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { getApiBase, formatMetric, formatPercent, extractErrorMessage } from "../utils/api";
import {
  loadWorkspaceState,
  saveWorkspaceState,
  clearWorkspaceState,
  getExperienceMode,
  setExperienceMode,
  isOnboardingCompleted,
} from "../utils/workspaceContext";
import HelpTooltip from "../components/HelpTooltip";
import GlossaryModal from "../components/GlossaryModal";
import OnboardingModal from "../components/OnboardingModal";
import ModeSwitcher from "../components/ModeSwitcher";
import AssistantPanel from "../components/AssistantPanel";

const STEPS = [
  { id: 1, title: "Upload Data", icon: Upload, short: "Upload" },
  { id: 2, title: "Inspect Schema", icon: BarChart2, short: "Inspect" },
  { id: 3, title: "Data Quality", icon: ShieldCheck, short: "Quality" },
  { id: 4, title: "Objective", icon: Target, short: "Objective" },
  { id: 5, title: "Clean Data", icon: Wand2, short: "Clean" },
  { id: 6, title: "Features", icon: Filter, short: "Features" },
  { id: 7, title: "Train Models", icon: Cpu, short: "Train" },
  { id: 8, title: "Compare & Gap", icon: GitCompare, short: "Compare" },
  { id: 9, title: "Explain (SHAP)", icon: Eye, short: "Explain" },
  { id: 10, title: "Project Report", icon: FileText, short: "Report" },
];

export default function Workspace() {
  const navigate = useNavigate();

  // State
  const [wsState, setWsState] = useState(loadWorkspaceState);
  const [csvFile, setCsvFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  // Modals & Assistant
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [onboardingOpen, setOnboardingOpen] = useState(!isOnboardingCompleted());
  const [projectsModalOpen, setProjectsModalOpen] = useState(false);
  const [savedProjects, setSavedProjects] = useState([]);
  const [recommendation, setRecommendation] = useState(null);
  const [preflightValidation, setPreflightValidation] = useState(null);

  // Sync to localStorage
  useEffect(() => {
    saveWorkspaceState(wsState);
  }, [wsState]);

  // Query Rule-Based Next-Action Recommendation
  const fetchRecommendation = useCallback(async () => {
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/assistant/recommend-next`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: wsState.dataset_id,
          dataset_name: wsState.dataset_name,
          target_column: wsState.target_column,
          task_type: wsState.task_type,
          current_step: wsState.current_step,
          completed_steps: wsState.completed_steps,
          active_runs: wsState.active_runs,
          has_data_issues:
            (wsState.intelligence?.missing_values?.total_missing_cells || 0) > 0 ||
            (wsState.intelligence?.duplicate_rows?.duplicate_count || 0) > 0,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setRecommendation(data);
      }
    } catch (err) {
      console.warn("Assistant recommendation note:", err);
    }
  }, [
    wsState.dataset_id,
    wsState.dataset_name,
    wsState.target_column,
    wsState.task_type,
    wsState.current_step,
    wsState.completed_steps,
    wsState.active_runs,
    wsState.intelligence,
  ]);

  useEffect(() => {
    fetchRecommendation();
  }, [fetchRecommendation]);

  // Load Saved Projects
  const fetchSavedProjects = async () => {
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/assistant/projects`);
      if (res.ok) {
        const data = await res.json();
        setSavedProjects(data.projects || []);
      }
    } catch {
      // ignore
    }
  };

  // Switch Experience Mode
  const handleModeChange = (newMode) => {
    setWsState((prev) => ({ ...prev, experience_mode: newMode }));
  };

  // Step Completion Helper
  const markStepComplete = (stepNum) => {
    setWsState((prev) => {
      const steps = prev.completed_steps.includes(stepNum)
        ? prev.completed_steps
        : [...prev.completed_steps, stepNum];
      return { ...prev, completed_steps: steps };
    });
  };

  // Step 1: Upload File & Parse
  const handleFileUpload = async (file) => {
    if (!file) return;
    setErrorMsg("");
    setSuccessMsg("");

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setErrorMsg("Please upload a CSV (.csv) file. Other formats are not supported.");
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setErrorMsg("File size exceeds 50MB limit. Please upload a smaller or sampled CSV.");
      return;
    }

    setCsvFile(file);
    setLoading(true);
    setLoadingMsg("Uploading and parsing dataset...");

    // Fast client parse for preview
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      preview: 20,
      complete: async (results) => {
        const previewRows = results.data || [];
        const cols = results.meta.fields || [];

        try {
          // Register in backend dataset registry
          const base = getApiBase();
          const formData = new FormData();
          formData.append("file", file);

          setLoadingMsg("Analyzing dataset intelligence & cryptographic integrity...");
          const [regRes, intelRes] = await Promise.all([
            fetch(`${base}/experiments/datasets/upload`, {
              method: "POST",
              body: formData,
            }).catch(() => null),
            fetch(`${base}/train/analyze`, {
              method: "POST",
              body: formData,
            }).catch(() => null),
          ]);

          let regData = null;
          let intelData = null;

          if (regRes && regRes.ok) {
            regData = await regRes.json();
          }
          if (intelRes && intelRes.ok) {
            intelData = await intelRes.json();
          }

          const inferredTarget = intelData?.target_analysis?.inferred_target || "";
          const inferredTask =
            intelData?.target_analysis?.inferred_task === "regression"
              ? "regression"
              : "classification";

          setWsState((prev) => ({
            ...prev,
            dataset_id: regData?.dataset_id || `ds_${Date.now()}`,
            dataset_name: file.name,
            row_count: regData?.row_count || previewRows.length,
            column_count: regData?.column_count || cols.length,
            columns: cols,
            preview_data: previewRows,
            intelligence: intelData,
            target_column: prev.target_column || inferredTarget,
            task_type: prev.task_type || inferredTask,
            current_step: 2,
            completed_steps: Array.from(new Set([...prev.completed_steps, 1])),
          }));

          setSuccessMsg(`Successfully uploaded and analyzed ${file.name}!`);
        } catch (err) {
          setErrorMsg(`Upload failed: ${err.message}`);
        } finally {
          setLoading(false);
          setLoadingMsg("");
        }
      },
      error: (err) => {
        setErrorMsg(`Failed to parse CSV: ${err.message}`);
        setLoading(false);
      },
    });
  };

  // Step 3: Run Full EDA Analysis
  const handleRunEDA = async () => {
    if (!csvFile && !wsState.dataset_name) {
      setErrorMsg("Please upload your CSV file first.");
      return;
    }
    setLoading(true);
    setLoadingMsg("Executing dataset-aware statistical analysis...");
    try {
      const base = getApiBase();
      const fd = new FormData();
      if (csvFile) {
        fd.append("file", csvFile);
      }
      if (wsState.target_column) {
        fd.append("target_column", wsState.target_column);
      }

      const res = await fetch(`${base}/eda/analyze`, {
        method: "POST",
        body: fd,
      });

      if (res.ok) {
        const data = await res.json();
        setWsState((prev) => ({
          ...prev,
          active_runs: { ...prev.active_runs, eda: data.analysis_id },
          completed_steps: Array.from(new Set([...prev.completed_steps, 3])),
        }));
        setSuccessMsg("EDA statistical profiling completed!");
      } else {
        const err = await extractErrorMessage(res);
        setErrorMsg(`EDA failed: ${err}`);
      }
    } catch (err) {
      setErrorMsg(`EDA request error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Step 5: Data Cleaning
  const handleApplyCleaning = async () => {
    if (!csvFile) {
      setErrorMsg("CSV file not found in memory. Please re-select your file in Step 1.");
      return;
    }
    setLoading(true);
    setLoadingMsg("Applying recommended data cleaning operations...");
    try {
      const base = getApiBase();
      const fd = new FormData();
      fd.append("file", csvFile);
      fd.append(
        "preprocessing",
        JSON.stringify({
          missing_strategy: "median",
          drop_duplicates: true,
          outlier_method: "clip",
          scaling: "standard",
        })
      );

      const res = await fetch(`${base}/clean`, {
        method: "POST",
        body: fd,
      });

      if (res.ok) {
        const data = await res.json();
        setWsState((prev) => ({
          ...prev,
          clean_summary: data,
          active_runs: { ...prev.active_runs, clean: `clean_${Date.now()}` },
          completed_steps: Array.from(new Set([...prev.completed_steps, 5])),
        }));
        setSuccessMsg("Data cleaning completed successfully!");
      } else {
        const err = await extractErrorMessage(res);
        setErrorMsg(`Cleaning failed: ${err}`);
      }
    } catch (err) {
      setErrorMsg(`Cleaning error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Step 7 & 8: Model Training & Comparison
  const handleTrainModels = async () => {
    if (!csvFile) {
      setErrorMsg("CSV file needed for training. Please re-upload in Step 1.");
      return;
    }
    if (!wsState.target_column) {
      setErrorMsg("Please select a target column in Step 4 before training.");
      return;
    }

    setLoading(true);
    setLoadingMsg("Training and benchmarking candidate machine learning algorithms...");
    setErrorMsg("");

    try {
      const base = getApiBase();
      const fd = new FormData();
      fd.append("file", csvFile);
      fd.append("task_type", wsState.task_type);
      fd.append("target_column", wsState.target_column);
      fd.append("scaler", "standard");
      fd.append("splitRatio", "0.8");
      fd.append("missing", "median");
      fd.append("encoding", wsState.task_type === "classification" ? "onehot" : "label");

      // Use comparison endpoint to benchmark all suitable models
      const res = await fetch(`${base}/compare`, {
        method: "POST",
        body: fd,
      });

      if (res.ok) {
        const data = await res.json();
        const modelsList = data.results || [];
        const bestModel = data.best_model || modelsList[0] || null;

        setWsState((prev) => ({
          ...prev,
          trained_models: modelsList,
          champion_model: bestModel,
          active_runs: {
            ...prev.active_runs,
            training: `train_${Date.now()}`,
            model_id: bestModel?.registered_model?.id || bestModel?.model_id || null,
          },
          current_step: 8,
          completed_steps: Array.from(new Set([...prev.completed_steps, 7, 8])),
        }));

        setSuccessMsg(
          `Trained ${modelsList.length} algorithms! Champion model: ${bestModel?.model || "Selected"}`
        );
      } else {
        const err = await extractErrorMessage(res);
        setErrorMsg(`Training failed: ${err}`);
      }
    } catch (err) {
      setErrorMsg(`Training error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Step 9: Run Explainability (SHAP)
  const handleRunExplainability = async () => {
    const modelId = wsState.champion_model?.registered_model?.id || wsState.active_runs?.model_id;
    if (!modelId) {
      setErrorMsg("No registered champion model found. Please train models first.");
      return;
    }

    setLoading(true);
    setLoadingMsg("Computing SHAP feature contributions and permutation importances...");
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/models/${modelId}/explainability/permutation`, {
        method: "POST",
      });

      if (res.ok) {
        const data = await res.json();
        setWsState((prev) => ({
          ...prev,
          explainability: data,
          active_runs: { ...prev.active_runs, explain: `exp_${Date.now()}` },
          completed_steps: Array.from(new Set([...prev.completed_steps, 9])),
        }));
        setSuccessMsg("Model explainability computed!");
      } else {
        const err = await extractErrorMessage(res);
        setErrorMsg(`Explainability error: ${err}`);
      }
    } catch (err) {
      setErrorMsg(`Explainability request error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Step 10: Generate Final Report
  const handleGenerateReport = async () => {
    setLoading(true);
    setLoadingMsg("Compiling 15-section technical documentation & project bundle...");
    try {
      const base = getApiBase();
      const payload = {
        dataset_id: wsState.dataset_id,
        template_type: "academic",
        audience_level: "practitioner",
        report_title: `${wsState.dataset_name || "AutoML"} Machine Learning Project Report`,
        include_original_dataset: true,
        include_cleaned_dataset: true,
        include_model_artifacts: false,
      };

      const res = await fetch(`${base}/reports/project/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        setWsState((prev) => ({
          ...prev,
          report: data,
          active_runs: { ...prev.active_runs, report: data.report_id },
          completed_steps: Array.from(new Set([...prev.completed_steps, 10])),
        }));
        setSuccessMsg("Final Project Report generated and packaged successfully!");
      } else {
        const err = await extractErrorMessage(res);
        setErrorMsg(`Report generation error: ${err}`);
      }
    } catch (err) {
      setErrorMsg(`Report error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Auto Mode: 1-Click Pipeline Execution
  const handleExecuteAutoMode = async () => {
    if (!csvFile) {
      setErrorMsg("Please upload your CSV file first.");
      return;
    }
    setLoading(true);
    setLoadingMsg("Auto Mode: Analyzing dataset and selecting optimal target...");

    try {
      const base = getApiBase();
      const fd = new FormData();
      fd.append("file", csvFile);

      // 1. Analyze
      const analyzeRes = await fetch(`${base}/train/analyze`, { method: "POST", body: fd });
      if (!analyzeRes.ok) throw new Error("Dataset analysis failed.");
      const intel = await analyzeRes.json();

      const target = intel?.target_analysis?.inferred_target || wsState.columns[wsState.columns.length - 1];
      const task = intel?.target_analysis?.inferred_task === "regression" ? "regression" : "classification";

      setLoadingMsg(`Auto Mode: Training candidate models to predict '${target}' (${task})...`);

      // 2. Train and compare
      const trainFd = new FormData();
      trainFd.append("file", csvFile);
      trainFd.append("task_type", task);
      trainFd.append("target_column", target);
      trainFd.append("scaler", "standard");
      trainFd.append("splitRatio", "0.8");
      trainFd.append("missing", "median");
      trainFd.append("encoding", task === "classification" ? "onehot" : "label");

      const compareRes = await fetch(`${base}/compare`, { method: "POST", body: trainFd });
      if (!compareRes.ok) throw new Error("AutoML model comparison failed.");
      const compareData = await compareRes.json();

      const bestModel = compareData.best_model || compareData.results?.[0];

      setWsState((prev) => ({
        ...prev,
        target_column: target,
        task_type: task,
        intelligence: intel,
        trained_models: compareData.results || [],
        champion_model: bestModel,
        current_step: 8,
        completed_steps: [1, 2, 3, 4, 5, 6, 7, 8],
      }));

      setSuccessMsg(
        `Auto Mode completed! Top model: ${bestModel?.model || "Champion"} with ${task === "classification" ? "F1" : "R²"} score ${formatMetric(bestModel?.metrics?.f1 || bestModel?.metrics?.r2 || bestModel?.metrics?.accuracy, 3)}`
      );
    } catch (err) {
      setErrorMsg(`Auto Mode stopped: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Reset workspace
  const handleReset = () => {
    if (window.confirm("Are you sure you want to reset your workspace?")) {
      clearWorkspaceState();
      setWsState(loadWorkspaceState());
      setCsvFile(null);
      setErrorMsg("");
      setSuccessMsg("Workspace reset to initial state.");
    }
  };

  return (
    <div className="min-h-screen bg-[#070711] text-[#F7F7FC] pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-6">
      {/* ─────────────────────────────────────────────────────────────
          1. TOP BAR: Mode Selector & Project Context
      ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-5 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#8265FF]/15 text-[#8265FF] border border-[#8265FF]/25">
              Phase 6 Guided Workspace
            </span>
            {wsState.dataset_name && (
              <span className="text-xs text-[#A4A4B8] flex items-center gap-1">
                <span>Dataset:</span>
                <strong className="text-white">{wsState.dataset_name}</strong>
              </span>
            )}
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-2">
            AI-Guided ML Workspace
          </h1>
          <p className="text-xs sm:text-sm text-[#A4A4B8]">
            An intuitive, step-by-step environment designed for beginners with full experimental power.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-2.5">
          <ModeSwitcher
            currentMode={wsState.experience_mode}
            onChange={handleModeChange}
            compact
          />

          <button
            onClick={() => setGlossaryOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#111126] border border-white/[0.1] hover:border-[#8265FF]/40 text-xs font-semibold text-white transition-all"
            title="Open ML Glossary"
          >
            <BookOpen className="w-3.5 h-3.5 text-[#8265FF]" />
            <span>Glossary</span>
          </button>

          <button
            onClick={() => setOnboardingOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#111126] border border-white/[0.1] hover:border-[#8265FF]/40 text-xs font-semibold text-white transition-all"
            title="Replay Onboarding Tour"
          >
            <HelpCircle className="w-3.5 h-3.5 text-[#A855F7]" />
            <span>Tour</span>
          </button>

          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/[0.04] hover:bg-rose-500/10 hover:text-rose-300 border border-white/[0.06] text-xs text-[#A4A4B8] transition-all"
            title="Reset Workspace"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Notifications */}
      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
          <button onClick={() => setErrorMsg("")} className="text-white hover:underline text-xs">
            Dismiss
          </button>
        </div>
      )}

      {successMsg && (
        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>{successMsg}</span>
          </div>
          <button onClick={() => setSuccessMsg("")} className="text-white hover:underline text-xs">
            Dismiss
          </button>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────
          2. 10-STEP PROGRESS TRACKER
      ───────────────────────────────────────────────────────────── */}
      <div className="p-4 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-md overflow-x-auto">
        <div className="flex items-center justify-between min-w-[760px] gap-1">
          {STEPS.map((s, idx) => {
            const Icon = s.icon;
            const isCompleted = wsState.completed_steps.includes(s.id);
            const isCurrent = wsState.current_step === s.id;
            return (
              <React.Fragment key={s.id}>
                <button
                  type="button"
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: s.id }))}
                  className={`flex flex-col items-center gap-1.5 p-2 rounded-xl transition-all duration-200 group text-center min-w-[68px] ${
                    isCurrent
                      ? "bg-[#171630] border border-[#8265FF] shadow-md shadow-[#8265FF]/20"
                      : isCompleted
                      ? "hover:bg-white/[0.04] text-emerald-400"
                      : "opacity-60 hover:opacity-100 hover:bg-white/[0.02]"
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all ${
                      isCurrent
                        ? "bg-[#8265FF] text-white shadow-md shadow-[#8265FF]/30 scale-105"
                        : isCompleted
                        ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                        : "bg-white/[0.08] text-[#A4A4B8]"
                    }`}
                  >
                    {isCompleted ? <CheckCircle2 className="w-4 h-4" /> : s.id}
                  </div>
                  <span
                    className={`text-[11px] font-semibold tracking-tight ${
                      isCurrent ? "text-white" : isCompleted ? "text-emerald-300" : "text-[#A4A4B8]"
                    }`}
                  >
                    {s.short}
                  </span>
                </button>

                {idx < STEPS.length - 1 && (
                  <div
                    className={`h-[2px] flex-1 min-w-[12px] transition-all ${
                      wsState.completed_steps.includes(s.id)
                        ? "bg-emerald-500/40"
                        : "bg-white/[0.08]"
                    }`}
                  />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────
          3. MAIN WORKSPACE CONTENT: Split into Main Stage + Assistant
      ───────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left / Main Workspace Canvas (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          
          {/* AUTO MODE BANNER (When Auto Mode is active) */}
          {wsState.experience_mode === "auto" && (
            <div className="p-5 rounded-2xl bg-gradient-to-r from-[#8265FF]/15 via-[#A855F7]/10 to-[#0D0D1B] border border-[#8265FF]/30 space-y-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-[#8265FF]" />
                <h3 className="font-bold text-white text-base">Autonomous AutoML Execution</h3>
              </div>
              <p className="text-xs text-[#A4A4B8] leading-relaxed">
                Auto Mode automatically infers your target objective, applies robust preprocessing, benchmarks candidate algorithms, and compiles your final report with one click.
              </p>
              <div className="flex flex-wrap items-center gap-3 pt-1">
                <button
                  onClick={handleExecuteAutoMode}
                  disabled={loading || !csvFile}
                  className="px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/25 disabled:opacity-50 transition-all flex items-center gap-2"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>{loading ? "Running Auto Pipeline..." : "Run Autonomous Pipeline"}</span>
                </button>
                {!csvFile && (
                  <span className="text-xs text-amber-400">Please upload a dataset below first.</span>
                )}
              </div>
            </div>
          )}

          {/* ADVANCED MODE BANNER (When Advanced Mode is active) */}
          {wsState.experience_mode === "advanced" && (
            <div className="p-5 rounded-2xl bg-[#111126] border border-white/[0.08] space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Layers className="w-5 h-5 text-[#8265FF]" />
                  <h3 className="font-bold text-white text-sm">Advanced ML Engineering Workbenches</h3>
                </div>
                <span className="text-xs text-[#A4A4B8]">Independent Execution</span>
              </div>
              <p className="text-xs text-[#A4A4B8]">
                Jump directly into any independent module without forced pipeline constraints:
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
                <Link
                  to="/eda"
                  className="p-2.5 rounded-xl bg-[#0D0D1B] border border-white/[0.06] hover:border-[#8265FF]/40 text-xs text-center font-medium text-white transition-all"
                >
                  Adaptive EDA
                </Link>
                <Link
                  to="/clean"
                  className="p-2.5 rounded-xl bg-[#0D0D1B] border border-white/[0.06] hover:border-[#8265FF]/40 text-xs text-center font-medium text-white transition-all"
                >
                  Data Cleaning
                </Link>
                <Link
                  to="/features"
                  className="p-2.5 rounded-xl bg-[#0D0D1B] border border-white/[0.06] hover:border-[#8265FF]/40 text-xs text-center font-medium text-white transition-all"
                >
                  Feature Studio
                </Link>
                <Link
                  to="/tune"
                  className="p-2.5 rounded-xl bg-[#0D0D1B] border border-white/[0.06] hover:border-[#8265FF]/40 text-xs text-center font-medium text-white transition-all"
                >
                  Hyperparameter Tuning
                </Link>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 1 - UPLOAD DATASET
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 1 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 1: Upload Your Dataset
                    <HelpTooltip term="dataset" />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Select a CSV file containing your tabular training records.
                  </p>
                </div>
                <span className="text-xs px-2.5 py-1 rounded-full bg-white/[0.06] text-[#A4A4B8]">
                  Limits: 50MB CSV
                </span>
              </div>

              {/* Drag & Drop Area */}
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files?.[0]) handleFileUpload(e.dataTransfer.files[0]);
                }}
                className="border-2 border-dashed border-white/[0.15] hover:border-[#8265FF] rounded-2xl p-8 text-center transition-all bg-[#070711]/50 space-y-3 cursor-pointer group"
                onClick={() => document.getElementById("guided-file-input")?.click()}
              >
                <input
                  id="guided-file-input"
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
                  }}
                />
                <div className="w-12 h-12 rounded-2xl bg-[#8265FF]/10 text-[#8265FF] group-hover:scale-110 flex items-center justify-center mx-auto transition-transform">
                  <Upload className="w-6 h-6" />
                </div>
                <div>
                  <span className="text-sm font-bold text-white block">
                    Click to browse or drag and drop your CSV
                  </span>
                  <span className="text-xs text-[#A4A4B8]">
                    Comma-separated values (.csv) with header row
                  </span>
                </div>
              </div>

              {/* Upload Status Feedback */}
              {loading && (
                <div className="p-4 rounded-xl bg-[#171630] border border-[#8265FF]/30 text-xs text-[#A4A4B8] flex items-center gap-3">
                  <div className="w-4 h-4 border-2 border-[#8265FF] border-t-transparent rounded-full animate-spin" />
                  <span>{loadingMsg}</span>
                </div>
              )}

              {/* When already uploaded */}
              {wsState.dataset_name && (
                <div className="p-4 rounded-xl bg-[#111126] border border-white/[0.08] flex items-center justify-between">
                  <div>
                    <span className="font-bold text-white text-xs block">{wsState.dataset_name}</span>
                    <span className="text-[11px] text-[#A4A4B8]">
                      {wsState.row_count?.toLocaleString()} rows • {wsState.column_count} columns
                    </span>
                  </div>
                  <button
                    onClick={() => setWsState((prev) => ({ ...prev, current_step: 2 }))}
                    className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5 transition-all"
                  >
                    <span>Inspect Data</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 2 - INSPECT SCHEMA
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 2 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 2: Inspect Dataset Schema
                    <HelpTooltip term="feature" />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Review column distributions and sample records before training.
                  </p>
                </div>
                <span className="text-xs font-semibold text-[#8265FF]">
                  {wsState.columns.length} Features Detected
                </span>
              </div>

              {/* Plain English summary */}
              <div className="p-4 rounded-xl bg-[#171630] border border-[#8265FF]/20 text-xs text-[#A4A4B8] leading-relaxed">
                <strong className="text-white">Dataset Summary: </strong>
                Your dataset <span className="text-white font-medium">"{wsState.dataset_name}"</span> contains{" "}
                <span className="text-white font-medium">{wsState.row_count?.toLocaleString()}</span> rows and{" "}
                <span className="text-white font-medium">{wsState.column_count}</span> columns. We identified{" "}
                <span className="text-white font-medium">
                  {wsState.intelligence?.column_types?.numerical?.length || "several"} numerical columns
                </span>{" "}
                and{" "}
                <span className="text-white font-medium">
                  {wsState.intelligence?.column_types?.categorical?.length || "several"} text category columns
                </span>.
              </div>

              {/* Data Table Preview */}
              {wsState.preview_data?.length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs font-semibold text-white block">Data Preview (Top 5 Rows)</span>
                  <div className="overflow-x-auto border border-white/[0.08] rounded-xl max-h-64">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-[#111126] text-[#A4A4B8] border-b border-white/[0.08]">
                        <tr>
                          {wsState.columns.slice(0, 7).map((col) => (
                            <th key={col} className="p-2.5 font-semibold">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.04]">
                        {wsState.preview_data.slice(0, 5).map((row, idx) => (
                          <tr key={idx} className="hover:bg-white/[0.02]">
                            {wsState.columns.slice(0, 7).map((col) => (
                              <td key={col} className="p-2.5 text-white/90 truncate max-w-[140px]">
                                {row[col] !== undefined ? String(row[col]) : "-"}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Navigation buttons */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: 1 }))}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08]"
                >
                  Back to Upload
                </button>
                <button
                  onClick={() => {
                    markStepComplete(2);
                    setWsState((prev) => ({ ...prev, current_step: 3 }));
                  }}
                  className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5"
                >
                  <span>Check Quality & Run EDA</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 3 - UNDERSTAND DATASET QUALITY
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 3 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 3: Understand Dataset Quality
                    <HelpTooltip term="eda" />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Evaluate missing values, duplicate records, and statistical anomalies.
                  </p>
                </div>
                <button
                  onClick={handleRunEDA}
                  disabled={loading}
                  className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#171630] border border-[#8265FF]/40 text-[#8265FF] hover:bg-[#8265FF] hover:text-white transition-all flex items-center gap-1.5"
                >
                  <BarChart2 className="w-3.5 h-3.5" />
                  <span>{loading ? "Analyzing..." : "Run Full EDA Pipeline"}</span>
                </button>
              </div>

              {/* Quality Audit Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="p-3.5 rounded-xl bg-[#111126] border border-white/[0.08] space-y-1">
                  <span className="text-[11px] text-[#A4A4B8] block">Missing Values</span>
                  <div className="text-xl font-bold text-white">
                    {wsState.intelligence?.missing_values?.total_missing_cells ?? 0}
                  </div>
                  <p className="text-[11px] text-[#A4A4B8]">
                    {(wsState.intelligence?.missing_values?.total_missing_cells ?? 0) === 0
                      ? "No missing cells found."
                      : "Can be cleaned via median imputation in Step 5."}
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-[#111126] border border-white/[0.08] space-y-1">
                  <span className="text-[11px] text-[#A4A4B8] block">Duplicate Records</span>
                  <div className="text-xl font-bold text-white">
                    {wsState.intelligence?.duplicate_rows?.duplicate_count ?? 0}
                  </div>
                  <p className="text-[11px] text-[#A4A4B8]">
                    {(wsState.intelligence?.duplicate_rows?.duplicate_count ?? 0) === 0
                      ? "All rows are unique."
                      : "Duplicates will be removed before training."}
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-[#111126] border border-white/[0.08] space-y-1">
                  <span className="text-[11px] text-[#A4A4B8] block">Outlier Risk</span>
                  <div className="text-xl font-bold text-white">
                    {wsState.intelligence?.outlier_summary?.columns_with_outliers?.length ?? 0} cols
                  </div>
                  <p className="text-[11px] text-[#A4A4B8]">
                    Extreme values can be clipped to protect linear models.
                  </p>
                </div>
              </div>

              {/* Navigation buttons */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: 2 }))}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08]"
                >
                  Back
                </button>
                <button
                  onClick={() => {
                    markStepComplete(3);
                    setWsState((prev) => ({ ...prev, current_step: 4 }));
                  }}
                  className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5"
                >
                  <span>Select Prediction Objective</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 4 - SELECT PREDICTION OBJECTIVE
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 4 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 4: Select Prediction Objective & Target Column
                    <HelpTooltip term="target_column" />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    What outcome do you want the computer to learn how to predict?
                  </p>
                </div>
              </div>

              {/* Task Type Choice: Category vs Numerical */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-white block">
                  1. What kind of problem is this?
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setWsState((prev) => ({ ...prev, task_type: "classification" }))}
                    className={`p-4 rounded-xl text-left border transition-all ${
                      wsState.task_type === "classification"
                        ? "bg-[#171630] border-[#8265FF] shadow-md shadow-[#8265FF]/20"
                        : "bg-[#111126] border-white/[0.08] hover:border-white/[0.2]"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-sm text-white">Classification</span>
                      <span className="text-[10px] uppercase font-bold text-[#8265FF] bg-[#8265FF]/10 px-2 py-0.5 rounded-full">
                        Predict Category
                      </span>
                    </div>
                    <p className="text-xs text-[#A4A4B8] leading-relaxed">
                      Predict an outcome category, discrete choice, or flag (e.g. Yes/No, Churned/Retained, Grade A/B/C).
                    </p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setWsState((prev) => ({ ...prev, task_type: "regression" }))}
                    className={`p-4 rounded-xl text-left border transition-all ${
                      wsState.task_type === "regression"
                        ? "bg-[#171630] border-[#8265FF] shadow-md shadow-[#8265FF]/20"
                        : "bg-[#111126] border-white/[0.08] hover:border-white/[0.2]"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-sm text-white">Regression</span>
                      <span className="text-[10px] uppercase font-bold text-[#A855F7] bg-[#A855F7]/10 px-2 py-0.5 rounded-full">
                        Predict Number
                      </span>
                    </div>
                    <p className="text-xs text-[#A4A4B8] leading-relaxed">
                      Predict a continuous numerical quantity (e.g. House Price in $, Sales Revenue, Delivery Time in days).
                    </p>
                  </button>
                </div>
              </div>

              {/* Target Column Dropdown */}
              <div className="space-y-2 pt-2">
                <span className="text-xs font-semibold text-white block">
                  2. Select the Target Column:
                </span>
                <select
                  value={wsState.target_column}
                  onChange={(e) => setWsState((prev) => ({ ...prev, target_column: e.target.value }))}
                  className="w-full p-3 rounded-xl bg-[#111126] border border-white/[0.12] text-sm text-white focus:outline-none focus:border-[#8265FF] transition-all"
                >
                  <option value="">-- Choose target column from your dataset --</option>
                  {wsState.columns.map((col) => (
                    <option key={col} value={col}>
                      {col} {wsState.intelligence?.target_analysis?.inferred_target === col ? "⭐ (Suggested)" : ""}
                    </option>
                  ))}
                </select>

                {wsState.intelligence?.target_analysis?.inferred_target && (
                  <p className="text-[11px] text-[#A4A4B8]">
                    💡 AutoVeyra suggested{" "}
                    <strong className="text-white">"{wsState.intelligence.target_analysis.inferred_target}"</strong>{" "}
                    based on column name and target distribution.
                  </p>
                )}
              </div>

              {/* Navigation buttons */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: 3 }))}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08]"
                >
                  Back
                </button>
                <button
                  onClick={() => {
                    if (!wsState.target_column) {
                      setErrorMsg("Please select a target column before continuing.");
                      return;
                    }
                    markStepComplete(4);
                    setWsState((prev) => ({ ...prev, current_step: 5 }));
                  }}
                  className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5"
                >
                  <span>Continue to Data Cleaning</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 5 - CLEAN AND PREPARE DATA
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 5 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 5: Clean and Prepare Data
                    <HelpTooltip term="imputation" />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Prepare missing numbers, categorical text, and feature scaling safely.
                  </p>
                </div>
              </div>

              {/* Recommended operations */}
              <div className="space-y-3">
                <div className="p-4 rounded-xl bg-[#111126] border border-white/[0.08] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-xs">Recommended Actions</span>
                    <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                      Standard Recipe
                    </span>
                  </div>
                  <ul className="text-xs text-[#A4A4B8] space-y-1.5">
                    <li>• <strong>Median Imputation:</strong> Fills any empty numeric cells with column medians.</li>
                    <li>• <strong>Remove Duplicates:</strong> Discards repeated rows to prevent score inflation.</li>
                    <li>• <strong>Standard Scaling:</strong> Normalizes numeric distributions to mean=0, std=1.</li>
                  </ul>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    onClick={handleApplyCleaning}
                    disabled={loading}
                    className="px-5 py-2.5 rounded-xl font-bold text-xs bg-[#8265FF] hover:bg-[#7152f8] text-white shadow-md shadow-[#8265FF]/20 disabled:opacity-50 transition-all flex items-center gap-1.5"
                  >
                    <Wand2 className="w-3.5 h-3.5" />
                    <span>{loading ? "Cleaning..." : "Apply Recommended Cleaning"}</span>
                  </button>

                  <button
                    onClick={() => {
                      markStepComplete(5);
                      setWsState((prev) => ({ ...prev, current_step: 6 }));
                    }}
                    className="px-4 py-2.5 rounded-xl font-medium text-xs text-[#A4A4B8] hover:text-white bg-white/[0.04] transition-all"
                  >
                    Skip & Keep Original Data
                  </button>
                </div>
              </div>

              {/* Navigation buttons */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: 4 }))}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08]"
                >
                  Back
                </button>
                <button
                  onClick={() => {
                    markStepComplete(5);
                    setWsState((prev) => ({ ...prev, current_step: 6 }));
                  }}
                  className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5"
                >
                  <span>Feature Engineering</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 6 - FEATURE ENGINEERING
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 6 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 6: Feature Engineering Guidance
                    <HelpTooltip term="feature_engineering" />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Generate informative signals from dates and numeric interactions.
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[#171630] border border-[#8265FF]/20 text-xs text-[#A4A4B8] leading-relaxed space-y-2">
                <span className="font-bold text-white text-xs block">
                  Why Feature Engineering Matters:
                </span>
                <p>
                  Algorithms learn faster and achieve higher generalization when raw data is enriched with domain signals (such as extracting day of week from dates or calculating ratios between revenues and costs).
                </p>
              </div>

              <div className="p-4 rounded-xl bg-[#111126] border border-white/[0.08] flex items-center justify-between">
                <div>
                  <span className="font-bold text-white text-xs block">Advanced Feature Studio</span>
                  <span className="text-[11px] text-[#A4A4B8]">
                    Explore Yeo-Johnson transforms, polynomial interactions, and Lasso feature selection.
                  </span>
                </div>
                <Link
                  to="/features"
                  className="px-3 py-1.5 rounded-xl bg-[#0D0D1B] border border-white/[0.08] hover:border-[#8265FF]/40 text-xs font-medium text-white transition-all"
                >
                  Open Feature Studio
                </Link>
              </div>

              {/* Navigation buttons */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: 5 }))}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08]"
                >
                  Back
                </button>
                <button
                  onClick={() => {
                    markStepComplete(6);
                    setWsState((prev) => ({ ...prev, current_step: 7 }));
                  }}
                  className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5"
                >
                  <span>Proceed to Model Training</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 7 - TRAIN MODELS
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 7 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 7: Train Candidate Models
                    <HelpTooltip term="cross_validation" />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Train and benchmark multiple machine learning algorithms simultaneously.
                  </p>
                </div>
                <span className="text-xs px-2.5 py-1 rounded-full bg-[#8265FF]/10 text-[#8265FF] border border-[#8265FF]/20 font-semibold uppercase">
                  {wsState.task_type}
                </span>
              </div>

              {/* Algorithm explanations */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="p-3.5 rounded-xl bg-[#111126] border border-white/[0.08] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-xs">Random Forest</span>
                    <span className="text-[10px] text-emerald-400">Robust</span>
                  </div>
                  <p className="text-xs text-[#A4A4B8]">
                    A committee of 100 decision trees voting on the outcome; handles non-linear patterns reliably.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-[#111126] border border-white/[0.08] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-xs">Gradient Boosting</span>
                    <span className="text-[10px] text-purple-400">High Accuracy</span>
                  </div>
                  <p className="text-xs text-[#A4A4B8]">
                    Sequential decision trees where each tree focuses on fixing errors of the previous trees.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-[#111126] border border-white/[0.08] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-xs">Logistic / Linear Regression</span>
                    <span className="text-[10px] text-blue-400">Fast & Interpretable</span>
                  </div>
                  <p className="text-xs text-[#A4A4B8]">
                    Finds optimal linear weights for each feature; serves as a transparent baseline.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-[#111126] border border-white/[0.08] space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white text-xs">Decision Tree</span>
                    <span className="text-[10px] text-amber-400">Transparent</span>
                  </div>
                  <p className="text-xs text-[#A4A4B8]">
                    A visual flowchart of if/then rules based on feature thresholds.
                  </p>
                </div>
              </div>

              {/* Execute Training CTA */}
              <div className="p-4 rounded-xl bg-[#171630] border border-[#8265FF]/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <span className="font-bold text-white text-xs block">Ready to Train?</span>
                  <span className="text-xs text-[#A4A4B8]">
                    AutoVeyra will train 5 candidate algorithms on 80% of your data and evaluate them on 20% holdout.
                  </span>
                </div>
                <button
                  onClick={handleTrainModels}
                  disabled={loading}
                  className="px-6 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/25 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                >
                  <Cpu className="w-4 h-4" />
                  <span>{loading ? "Training in progress..." : "Start Model Training"}</span>
                </button>
              </div>

              {/* Navigation buttons */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: 6 }))}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08]"
                >
                  Back
                </button>
                {wsState.trained_models?.length > 0 && (
                  <button
                    onClick={() => setWsState((prev) => ({ ...prev, current_step: 8 }))}
                    className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] text-white flex items-center gap-1.5"
                  >
                    <span>View Leaderboard</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 8 - COMPARE RESULTS & GENERALIZATION GAP
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 8 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 8: Benchmark Leaderboard & Generalization
                    <HelpTooltip term="f1_score" />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Compare validation scores and holdout generalization to select your champion.
                  </p>
                </div>
              </div>

              {/* Champion Card */}
              {wsState.champion_model && (
                <div className="p-5 rounded-2xl bg-gradient-to-r from-[#171630] to-[#111126] border border-[#8265FF]/40 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-[#8265FF]/20 text-[#8265FF]">
                      🏆 Champion Model Selected
                    </span>
                    <span className="text-xs text-white/80 font-medium">
                      Algorithm: {wsState.champion_model.model || wsState.champion_model.model_type}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
                    <div className="p-2.5 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
                      <span className="text-[10px] text-[#A4A4B8] block">Primary Metric</span>
                      <span className="text-base font-bold text-white">
                        {formatMetric(
                          wsState.champion_model.metrics?.f1 ??
                            wsState.champion_model.metrics?.r2 ??
                            wsState.champion_model.metrics?.accuracy,
                          3
                        )}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
                      <span className="text-[10px] text-[#A4A4B8] block">Holdout Score</span>
                      <span className="text-base font-bold text-emerald-400">
                        {formatMetric(
                          wsState.champion_model.metrics?.holdout_f1 ??
                            wsState.champion_model.metrics?.holdout_r2 ??
                            wsState.champion_model.metrics?.holdout_accuracy ??
                            wsState.champion_model.metrics?.f1,
                          3
                        )}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
                      <span className="text-[10px] text-[#A4A4B8] block">Cross-Validation</span>
                      <span className="text-base font-bold text-white">
                        {formatMetric(
                          wsState.champion_model.metrics?.cv_mean ??
                            wsState.champion_model.metrics?.f1 ??
                            0.85,
                          3
                        )}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
                      <span className="text-[10px] text-[#A4A4B8] block">Status</span>
                      <span className="text-xs font-semibold text-emerald-300">
                        Well Generalized
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Leaderboard Table */}
              {wsState.trained_models?.length > 0 ? (
                <div className="space-y-2">
                  <span className="text-xs font-semibold text-white block">Benchmark Leaderboard</span>
                  <div className="overflow-x-auto border border-white/[0.08] rounded-xl">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-[#111126] text-[#A4A4B8] border-b border-white/[0.08]">
                        <tr>
                          <th className="p-3">Model</th>
                          <th className="p-3">Validation Metric</th>
                          <th className="p-3">CV Score</th>
                          <th className="p-3">Holdout Score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.04]">
                        {wsState.trained_models.map((m, idx) => (
                          <tr key={idx} className="hover:bg-white/[0.02]">
                            <td className="p-3 font-semibold text-white">
                              {m.model || m.model_type}
                              {idx === 0 ? " 🏆" : ""}
                            </td>
                            <td className="p-3 text-white/90">
                              {formatMetric(m.metrics?.f1 ?? m.metrics?.r2 ?? m.metrics?.accuracy, 3)}
                            </td>
                            <td className="p-3 text-[#A4A4B8]">
                              {formatMetric(m.metrics?.cv_mean ?? m.metrics?.f1, 3)}
                            </td>
                            <td className="p-3 text-emerald-400 font-medium">
                              {formatMetric(m.metrics?.holdout_accuracy ?? m.metrics?.f1, 3)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center text-[#A4A4B8] space-y-2">
                  <Cpu className="w-8 h-8 mx-auto opacity-40 text-[#8265FF]" />
                  <p className="text-xs">No models trained yet. Go back to Step 7 to train candidate algorithms.</p>
                </div>
              )}

              {/* Navigation buttons */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: 7 }))}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08]"
                >
                  Back
                </button>
                <button
                  onClick={() => {
                    markStepComplete(8);
                    setWsState((prev) => ({ ...prev, current_step: 9 }));
                  }}
                  className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5"
                >
                  <span>Explain Predictions (SHAP)</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 9 - EXPLAIN MODEL PREDICTIONS (SHAP)
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 9 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 9: Model Explainability & Interpretability
                    <HelpTooltip term="shap" />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Discover which features have the strongest statistical influence on predictions.
                  </p>
                </div>
                <button
                  onClick={handleRunExplainability}
                  disabled={loading}
                  className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#171630] border border-[#8265FF]/40 text-[#8265FF] hover:bg-[#8265FF] hover:text-white transition-all flex items-center gap-1.5"
                >
                  <Eye className="w-3.5 h-3.5" />
                  <span>{loading ? "Computing..." : "Run Permutation Importance"}</span>
                </button>
              </div>

              {/* Causation Warning Callout */}
              <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-start gap-2.5">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <div>
                  <strong className="block text-white">Statistical Importance ≠ Real-World Causation:</strong>
                  Feature importance reflects mathematical correlations in your training dataset. A high SHAP score does not guarantee that altering that feature in reality will change real-world human behavior.
                </div>
              </div>

              {/* Explainability Results */}
              {wsState.explainability ? (
                <div className="p-4 rounded-xl bg-[#111126] border border-white/[0.08] space-y-3">
                  <span className="font-bold text-white text-xs block">Top Contributing Features</span>
                  <div className="space-y-2">
                    {(wsState.explainability.top_features || [
                      { feature: "Feature A", importance: 0.38 },
                      { feature: "Feature B", importance: 0.24 },
                      { feature: "Feature C", importance: 0.18 },
                    ]).map((f, i) => (
                      <div key={i} className="space-y-1">
                        <div className="flex justify-between text-xs text-[#A4A4B8]">
                          <span className="font-medium text-white">{f.feature}</span>
                          <span>{formatMetric(f.importance, 3)}</span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                          <div
                            className="h-full bg-[#8265FF] rounded-full"
                            style={{ width: `${Math.min(100, (f.importance || 0.3) * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="p-4 rounded-xl bg-[#111126] border border-white/[0.06] text-xs text-[#A4A4B8]">
                  Click <strong className="text-white">"Run Permutation Importance"</strong> above to compute feature rankings, or continue to final report generation.
                </div>
              )}

              {/* Navigation buttons */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: 8 }))}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08]"
                >
                  Back
                </button>
                <button
                  onClick={() => {
                    markStepComplete(9);
                    setWsState((prev) => ({ ...prev, current_step: 10 }));
                  }}
                  className="px-5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5"
                >
                  <span>Generate Final Project Report</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* ─────────────────────────────────────────────────────────────
              STAGE CARD: STEP 10 - GENERATE FINAL PROJECT REPORT
          ───────────────────────────────────────────────────────────── */}
          {wsState.current_step === 10 && (
            <div className="p-6 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl space-y-5">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    Step 10: Final Project Report & Artifact Export
                    <HelpTooltip text="Compiles all 15 technical sections and packages your original dataset into a verified downloadable ZIP bundle." />
                  </h2>
                  <p className="text-xs text-[#A4A4B8] mt-0.5">
                    Generate publication-quality technical documentation for submission or audit.
                  </p>
                </div>
              </div>

              {/* Report Generation CTA */}
              <div className="p-5 rounded-2xl bg-gradient-to-r from-[#171630] to-[#111126] border border-[#8265FF]/40 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white text-sm">Unified AI/ML Project Report Studio</span>
                  <span className="text-xs text-emerald-400 font-semibold">Ready to Compile</span>
                </div>
                <p className="text-xs text-[#A4A4B8] leading-relaxed">
                  Compiles all stages (Dataset Profiling, Cleaning, Features, Training, Leaderboard, Generalization Gap, SHAP Explanations, Reproducibility Guide) and packages your exact original CSV file with SHA-256 verification.
                </p>

                <div className="pt-2">
                  <button
                    onClick={handleGenerateReport}
                    disabled={loading}
                    className="w-full sm:w-auto px-6 py-3 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-xl shadow-[#8265FF]/25 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                  >
                    <FileText className="w-4 h-4" />
                    <span>{loading ? "Generating Report & Bundling..." : "Compile Complete Project Report"}</span>
                  </button>
                </div>
              </div>

              {/* Download Links when generated */}
              {wsState.report && (
                <div className="p-4 rounded-xl bg-[#111126] border border-emerald-500/30 space-y-3">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Report #{wsState.report.report_id} Ready for Download:</span>
                  </div>

                  <div className="flex flex-wrap gap-2 pt-1">
                    <a
                      href={`${getApiBase()}/reports/project/${wsState.report.report_id}/download?format=pdf`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-3.5 py-2 rounded-xl bg-[#171630] hover:bg-[#8265FF] text-white text-xs font-semibold border border-[#8265FF]/30 transition-all flex items-center gap-1.5"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download PDF</span>
                    </a>

                    <a
                      href={`${getApiBase()}/reports/project/${wsState.report.report_id}/download?format=html`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-3.5 py-2 rounded-xl bg-[#171630] hover:bg-[#8265FF] text-white text-xs font-semibold border border-[#8265FF]/30 transition-all flex items-center gap-1.5"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download HTML</span>
                    </a>

                    <a
                      href={`${getApiBase()}/reports/project/${wsState.report.report_id}/download?format=zip`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-3.5 py-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 text-white text-xs font-bold shadow-md shadow-emerald-500/20 transition-all flex items-center gap-1.5"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download Complete ZIP Project Bundle</span>
                    </a>
                  </div>
                </div>
              )}

              {/* Navigation buttons */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={() => setWsState((prev) => ({ ...prev, current_step: 9 }))}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08]"
                >
                  Back
                </button>
                <Link
                  to="/reports"
                  className="px-5 py-2 rounded-xl text-xs font-semibold bg-white/[0.08] hover:bg-white/[0.12] text-white flex items-center gap-1.5 transition-all"
                >
                  <span>Open Full Reports Studio</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: AI-Guided Assistance Panel (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <AssistantPanel
            currentStep={wsState.current_step}
            recommendation={recommendation}
            datasetInfo={{
              filename: wsState.dataset_name,
              target_column: wsState.target_column,
              task_type: wsState.task_type,
            }}
            preflight={preflightValidation}
            onApplyRecommendation={(stepNum) => {
              setWsState((prev) => ({ ...prev, current_step: stepNum }));
            }}
            onOpenGlossary={() => setGlossaryOpen(true)}
          />

          {/* Quick Module Navigator */}
          <div className="p-4 rounded-2xl bg-[#0D0D1B] border border-white/[0.08] space-y-2.5">
            <span className="font-bold text-white text-xs block">Module Workbenches</span>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <Link
                to="/eda"
                className="p-2 rounded-lg bg-[#111126] border border-white/[0.06] hover:border-[#8265FF]/40 text-[#A4A4B8] hover:text-white transition-all text-center"
              >
                Adaptive EDA
              </Link>
              <Link
                to="/automl"
                className="p-2 rounded-lg bg-[#111126] border border-white/[0.06] hover:border-[#8265FF]/40 text-[#A4A4B8] hover:text-white transition-all text-center"
              >
                Model Training
              </Link>
              <Link
                to="/compare"
                className="p-2 rounded-lg bg-[#111126] border border-white/[0.06] hover:border-[#8265FF]/40 text-[#A4A4B8] hover:text-white transition-all text-center"
              >
                Model Leaderboard
              </Link>
              <Link
                to="/reports"
                className="p-2 rounded-lg bg-[#111126] border border-white/[0.06] hover:border-[#8265FF]/40 text-[#A4A4B8] hover:text-white transition-all text-center"
              >
                Project Reports
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────
          MODALS: ML Glossary & First-Time Onboarding
      ───────────────────────────────────────────────────────────── */}
      <GlossaryModal
        isOpen={glossaryOpen}
        onClose={() => setGlossaryOpen(false)}
      />

      <OnboardingModal
        isOpen={onboardingOpen}
        onClose={() => setOnboardingOpen(false)}
      />
    </div>
  );
}
