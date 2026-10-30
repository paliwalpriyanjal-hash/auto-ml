import { useState, useEffect, useMemo } from "react";
import { useSearchParams, Link } from "react-router-dom";
import {
  FileText,
  Download,
  Layers,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Sliders,
  Filter,
  PlayCircle,
  Trash2,
  Edit3,
  Save,
  X,
  Calendar,
  Cpu,
  Database,
  Sparkles,
  Search,
  ExternalLink,
  ChevronRight,
  ChevronLeft,
  ArrowRight,
  Check,
  Package,
  FileCode,
  BarChart2,
  Workflow,
  FolderGit2,
  Info,
  Eye,
  AlertTriangle,
  FileSpreadsheet,
  CheckSquare,
  Square,
} from "lucide-react";
import { getApiBase, formatMetric } from "../utils/api";
import PageHeader from "../components/ui/PageHeader";
import Badge from "../components/ui/Badge";
import Button from "../components/ui/Button";

const REPORT_TEMPLATES = [
  { id: "academic", name: "Academic / University Project Report", desc: "Structured for final-year projects, theses, research methodology, and academic defense." },
  { id: "professional", name: "Professional Data Science Report", desc: "Executive summary, analytical findings, data quality audits, and business recommendations." },
  { id: "technical", name: "Technical ML Model Engineering Report", desc: "Deep dive into model architecture, hyperparameters, cross-validation, and deployment constraints." },
  { id: "complete", name: "Complete AutoVeyra Project Report", desc: "End-to-end comprehensive documentation integrating all 15 stages from raw dataset to final registry." },
];

const STANDARD_SECTIONS = [
  { id: "cover_page", name: "1. Cover Page & Metadata", desc: "Branded title, author, institution, dataset, and timestamp." },
  { id: "executive_summary", name: "2. Executive Summary", desc: "Problem statement, dataset volume, winning model, and primary holdout metrics." },
  { id: "project_overview", name: "3. Project Scope & Architecture", desc: "Task definitions, technology stack, and workflow execution strategy." },
  { id: "dataset_information", name: "4. Original Dataset Integrity", desc: "Exact filename, dimensions, SHA-256 cryptographic checksum, and column schema." },
  { id: "exploratory_data_analysis", name: "5. Exploratory Data Analysis", desc: "Distributions, missingness audits, and correlation heatmaps." },
  { id: "data_cleaning", name: "6. Data Cleaning & Preprocessing", desc: "Missing value imputation, outlier pruning, and data retention percentages." },
  { id: "feature_engineering", name: "7. Feature Engineering & Selection", desc: "Polynomial features, interactions, and lineage transformation audit." },
  { id: "model_training", name: "8. Model Training & Holdout Split", desc: "Candidate algorithms, train/holdout partition ratios, and latency." },
  { id: "model_comparison", name: "9. Model Comparison Matrix", desc: "Side-by-side benchmark rankings and winning model selection justification." },
  { id: "hyperparameter_tuning", name: "10. Hyperparameter Optimization", desc: "Optuna Bayesian trials, search spaces, and score delta improvements." },
  { id: "final_model_evaluation", name: "11. Holdout Model Evaluation", desc: "Classification (Accuracy, F1, ROC-AUC) or Regression (R², RMSE, MAE)." },
  { id: "model_explainability", name: "12. Model Explainability", desc: "Global feature importance weights and attribution methodology." },
  { id: "workflow_flowchart", name: "13. Workflow Flowchart & Lineage", desc: "End-to-end directed execution path and stage status." },
  { id: "experiment_history", name: "14. Experiment History & Reproducibility", desc: "Run IDs, parameter payloads, and reproduction commands." },
  { id: "conclusion_limitations", name: "15. Governance Conclusions & Limitations", desc: "Honest evaluation boundaries, risk notes, and deployment guidelines." },
];

export default function Reports() {
  const [searchParams, setSearchParams] = useSearchParams();
  const preselectedModelId = searchParams.get("model");
  const preselectedReportId = searchParams.get("report");

  // Mode: 'project_studio' (Phase 5) vs 'model_cards' (Existing Single-Model)
  const [activeMode, setActiveMode] = useState("project_studio");

  // Project Studio State
  const [studioTab, setStudioTab] = useState("history"); // 'history', 'builder', 'inspector'
  const [projectReports, setProjectReports] = useState([]);
  const [projectReportsLoading, setProjectReportsLoading] = useState(false);
  const [activeProjectReport, setActiveProjectReport] = useState(null);

  // Workbench Data for Builder
  const [datasets, setDatasets] = useState([]);
  const [runs, setRuns] = useState([]);
  const [workflows, setWorkflows] = useState([]);

  // Wizard Steps (1 to 9)
  const [wizardStep, setWizardStep] = useState(1);
  const [builderData, setBuilderData] = useState({
    project_name: "",
    project_description: "",
    author_name: "",
    institution: "",
    report_template: "complete",
    original_dataset_id: "",
    selected_scope_type: "custom", // 'workflow' or 'custom'
    selected_workflow_id: "",
    selected_runs: {
      eda: "",
      cleaning: "",
      features: "",
      training: "",
      tuning: "",
      evaluation: "",
      explainability: "",
    },
    included_sections: STANDARD_SECTIONS.map((s) => s.id),
    include_original_dataset: true,
    include_cleaned_dataset: false,
    include_model_artifact: false,
  });

  const [validationResult, setValidationResult] = useState(null);
  const [validating, setValidating] = useState(false);
  const [generatingProject, setGeneratingProject] = useState(false);

  // ------------------------------------------------------------------
  // Existing Model Governance Cards State (Phase 5 Baseline)
  // ------------------------------------------------------------------
  const [models, setModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState(preselectedModelId || "");
  const [reportTitleInput, setReportTitleInput] = useState("");
  const [reportsList, setReportsList] = useState([]);
  const [reportsLoading, setReportsLoading] = useState(true);
  const [activeReportId, setActiveReportId] = useState(preselectedReportId || null);
  const [currentReport, setCurrentReport] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState(null);
  const [isEditCardOpen, setIsEditCardOpen] = useState(false);
  const [editCardData, setEditCardData] = useState({
    intended_use: "",
    out_of_scope_use: "",
    known_limitations: "",
    ethical_considerations: "",
    author_notes: "",
  });
  const [savingCard, setSavingCard] = useState(false);
  const [cardSaveSuccess, setCardSaveSuccess] = useState(false);
  const [featureSearch, setFeatureSearch] = useState("");

  // ------------------------------------------------------------------
  // Data Fetching
  // ------------------------------------------------------------------
  const fetchProjectReports = async () => {
    setProjectReportsLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/reports/project?limit=50`);
      if (res.ok) {
        const data = await res.json();
        setProjectReports(data.reports || []);
      }
    } catch (err) {
      console.error("Failed to load project reports", err);
    } finally {
      setProjectReportsLoading(false);
    }
  };

  const fetchWorkbenchEntities = async () => {
    const base = getApiBase();
    try {
      const [resDatasets, resRuns, resWf] = await Promise.all([
        fetch(`${base}/experiments/datasets?limit=100`).then((r) => r.ok ? r.json() : { datasets: [] }),
        fetch(`${base}/experiments/runs?limit=100`).then((r) => r.ok ? r.json() : { runs: [] }),
        fetch(`${base}/experiments/workflows?limit=100`).then((r) => r.ok ? r.json() : { workflows: [] }),
      ]);
      setDatasets(resDatasets.datasets || []);
      setRuns(resRuns.runs || []);
      setWorkflows(resWf.workflows || []);

      if (resDatasets.datasets && resDatasets.datasets.length > 0 && !builderData.original_dataset_id) {
        setBuilderData((prev) => ({ ...prev, original_dataset_id: resDatasets.datasets[0].dataset_id }));
      }
    } catch (err) {
      console.error("Failed to load workbench entities", err);
    }
  };

  const fetchModels = async () => {
    setModelsLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/models?limit=100`);
      if (res.ok) {
        const data = await res.json();
        const items = data.models || data || [];
        setModels(items);
        if (!selectedModelId && items.length > 0 && !preselectedReportId) {
          setSelectedModelId(items[0].id.toString());
        }
      }
    } catch (err) {
      console.error("Failed to load registered models", err);
    } finally {
      setModelsLoading(false);
    }
  };

  const fetchModelReports = async () => {
    setReportsLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/reports?limit=50`);
      if (res.ok) {
        const data = await res.json();
        const items = data.items || [];
        setReportsList(items);
        if (!activeReportId && items.length > 0 && !preselectedModelId) {
          setActiveReportId(items[0].report_id);
        }
      }
    } catch (err) {
      console.error("Failed to load model reports", err);
    } finally {
      setReportsLoading(false);
    }
  };

  useEffect(() => {
    fetchProjectReports();
    fetchWorkbenchEntities();
    fetchModels();
    fetchModelReports();
  }, []);

  // Single Model Report Detail Loader
  useEffect(() => {
    if (!activeReportId) {
      setCurrentReport(null);
      return;
    }
    const fetchReportDetail = async () => {
      setDetailLoading(true);
      try {
        const res = await fetch(`${getApiBase()}/reports/${activeReportId}`);
        if (res.ok) {
          const data = await res.json();
          setCurrentReport(data);
          const card = data.model_card || {};
          setEditCardData({
            intended_use: card.intended_use || "",
            out_of_scope_use: card.out_of_scope_use || "",
            known_limitations: card.known_limitations || "",
            ethical_considerations: card.ethical_considerations || "",
            author_notes: card.author_notes || "",
          });
        }
      } catch (err) {
        console.error("Error loading single model report", err);
      } finally {
        setDetailLoading(false);
      }
    };
    fetchReportDetail();
  }, [activeReportId]);

  // Compatibility validator
  const handleValidateCompatibility = async () => {
    if (!builderData.original_dataset_id) {
      alert("Please select a target dataset first.");
      return;
    }
    setValidating(true);
    try {
      const res = await fetch(`${getApiBase()}/reports/project/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: builderData.original_dataset_id,
          selected_runs: builderData.selected_runs,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setValidationResult(data);
      }
    } catch (err) {
      console.error("Validation error", err);
    } finally {
      setValidating(false);
    }
  };

  // Compile & Generate Project Report
  const handleGenerateProjectReport = async () => {
    if (!builderData.project_name.trim()) {
      alert("Please provide a project name.");
      return;
    }
    setGeneratingProject(true);
    try {
      const res = await fetch(`${getApiBase()}/reports/project/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(builderData),
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Project report compilation failed.");
      }
      const data = await res.json();
      setActiveProjectReport(data);
      setWizardStep(9); // Jump to download step
      fetchProjectReports();
    } catch (err) {
      alert(`Generation failed: ${err.message}`);
    } finally {
      setGeneratingProject(false);
    }
  };

  // Open Project Report Inspector
  const handleInspectProjectReport = async (reportId) => {
    setProjectReportsLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/reports/project/${reportId}`);
      if (res.ok) {
        const data = await res.json();
        setActiveProjectReport(data);
        setStudioTab("inspector");
      }
    } catch (err) {
      alert("Could not load report details.");
    } finally {
      setProjectReportsLoading(false);
    }
  };

  // Delete Project Report
  const handleDeleteProjectReport = async (reportId) => {
    if (!window.confirm(`Delete project report ${reportId}?`)) return;
    try {
      const res = await fetch(`${getApiBase()}/reports/project/${reportId}`, { method: "DELETE" });
      if (res.ok) {
        setProjectReports((prev) => prev.filter((r) => r.report_id !== reportId));
        if (activeProjectReport && activeProjectReport.report_id === reportId) {
          setActiveProjectReport(null);
          setStudioTab("history");
        }
      }
    } catch (err) {
      alert("Failed to delete project report.");
    }
  };

  // Download Trigger Helper
  const triggerProjectDownload = (reportId, format) => {
    const url = `${getApiBase()}/reports/project/${reportId}/download?format=${format}`;
    window.open(url, "_blank");
  };

  // Toggle Section Inclusion
  const toggleSection = (secId) => {
    setBuilderData((prev) => {
      const exists = prev.included_sections.includes(secId);
      return {
        ...prev,
        included_sections: exists
          ? prev.included_sections.filter((id) => id !== secId)
          : [...prev.included_sections, secId],
      };
    });
  };

  return (
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-6">
      {/* Universal Page Header */}
      <PageHeader
        category="Intelligence"
        title="Project Report Studio"
        subtitle="Compile end-to-end multi-module project documentation with original dataset preservation & cryptographic manifests."
        badge={
          <Badge variant="violet" size="sm">
            <Package className="w-3.5 h-3.5 mr-1 inline" /> Technical Governance
          </Badge>
        }
        actions={
          <div className="flex items-center gap-2 bg-white/[0.04] border border-white/[0.08] p-1 rounded-xl">
            <button
              onClick={() => setActiveMode("project_studio")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-2 ${
                activeMode === "project_studio"
                  ? "bg-[#8265FF] text-white shadow-sm"
                  : "text-[#9FA2B6] hover:text-white"
              }`}
            >
              <FolderGit2 className="w-3.5 h-3.5" /> Project Report Studio
            </button>
            <button
              onClick={() => setActiveMode("model_cards")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-2 ${
                activeMode === "model_cards"
                  ? "bg-[#8265FF] text-white shadow-sm"
                  : "text-[#9FA2B6] hover:text-white"
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5" /> Model Governance Cards
            </button>
          </div>
        }
      />

      {/* ===================================================================== */}
      {/* MODE 1: UNIFIED PROJECT REPORT STUDIO (PHASE 5) */}
      {/* ===================================================================== */}
      {activeMode === "project_studio" && (
        <div className="space-y-6">
          {/* Sub Navigation */}
          <div className="flex border-b border-white/[0.08] gap-4 pb-2">
            <button
              onClick={() => setStudioTab("history")}
              className={`text-xs font-semibold pb-2 border-b-2 transition-all flex items-center gap-2 ${
                studioTab === "history"
                  ? "text-purple-400 border-purple-500"
                  : "text-[#A4A4B8] border-transparent hover:text-white"
              }`}
            >
              <FileText className="w-3.5 h-3.5" /> Project Reports History ({projectReports.length})
            </button>
            <button
              onClick={() => {
                setStudioTab("builder");
                setWizardStep(1);
              }}
              className={`text-xs font-semibold pb-2 border-b-2 transition-all flex items-center gap-2 ${
                studioTab === "builder"
                  ? "text-purple-400 border-purple-500"
                  : "text-[#A4A4B8] border-transparent hover:text-white"
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" /> 9-Step Report Builder Wizard
            </button>
            {activeProjectReport && (
              <button
                onClick={() => setStudioTab("inspector")}
                className={`text-xs font-semibold pb-2 border-b-2 transition-all flex items-center gap-2 ${
                  studioTab === "inspector"
                    ? "text-purple-400 border-purple-500"
                    : "text-[#A4A4B8] border-transparent hover:text-white"
                }`}
              >
                <Eye className="w-3.5 h-3.5" /> Active Report Inspector
              </button>
            )}
          </div>

          {/* SUB-VIEW A: HISTORY DASHBOARD */}
          {studioTab === "history" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                  Compiled Project Documentation
                </h3>
                <button
                  onClick={() => {
                    setStudioTab("builder");
                    setWizardStep(1);
                  }}
                  className="px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-md flex items-center gap-2"
                >
                  <Sparkles className="w-3.5 h-3.5" /> Create Project Report
                </button>
              </div>

              {projectReports.length === 0 ? (
                <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-12 text-center text-[#A4A4B8]">
                  <Package className="w-10 h-10 mx-auto mb-3 opacity-40 text-purple-400" />
                  <h4 className="text-base font-bold text-white">No Project Reports Compiled Yet</h4>
                  <p className="text-xs text-[#A4A4B8] mt-1 max-w-md mx-auto">
                    Select an uploaded dataset and combine your completed EDA, cleaning, feature engineering, and model training runs into a full technical project report.
                  </p>
                  <button
                    onClick={() => {
                      setStudioTab("builder");
                      setWizardStep(1);
                    }}
                    className="mt-4 px-4 py-2 rounded-xl text-xs font-semibold bg-purple-600 text-white"
                  >
                    Launch Report Wizard
                  </button>
                </div>
              ) : (
                <div className="bg-[#111126] border border-white/[0.08] rounded-2xl overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-white/[0.08] bg-[#171630]/70 text-[#A4A4B8] uppercase text-[10px] tracking-wider">
                          <th className="py-3.5 px-4">Project & Template</th>
                          <th className="py-3.5 px-4">Original Dataset</th>
                          <th className="py-3.5 px-4">Dimensions</th>
                          <th className="py-3.5 px-4">Author / Org</th>
                          <th className="py-3.5 px-4">Compiled Date</th>
                          <th className="py-3.5 px-4 text-right">Downloads & Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.06] text-white">
                        {projectReports.map((r) => (
                          <tr key={r.report_id} className="hover:bg-white/[0.02] transition-colors">
                            <td className="py-3.5 px-4">
                              <div className="font-semibold text-white">{r.project_name}</div>
                              <div className="text-[10px] text-purple-300 font-mono capitalize">
                                {r.report_template} Template • {r.report_id}
                              </div>
                            </td>
                            <td className="py-3.5 px-4">
                              <div className="flex items-center gap-1.5 text-white">
                                <Database className="w-3.5 h-3.5 text-amber-400" />
                                <span>{r.original_filename}</span>
                              </div>
                              <div className="text-[10px] font-mono text-[#A4A4B8] truncate max-w-[120px]">
                                SHA: {r.dataset_sha256?.substring(0, 10)}...
                              </div>
                            </td>
                            <td className="py-3.5 px-4 text-[#A4A4B8]">{r.dataset_dimensions}</td>
                            <td className="py-3.5 px-4 text-[#A4A4B8]">
                              {r.author_name || "AutoVeyra"}
                              {r.institution ? ` (${r.institution})` : ""}
                            </td>
                            <td className="py-3.5 px-4 text-[#A4A4B8] text-[11px]">
                              {new Date(r.created_at).toLocaleDateString()}
                            </td>
                            <td className="py-3.5 px-4 text-right">
                              <div className="flex items-center justify-end gap-1.5">
                                <button
                                  onClick={() => triggerProjectDownload(r.report_id, "pdf")}
                                  title="Download PDF Report"
                                  className="p-1.5 rounded-lg bg-purple-500/10 text-purple-300 hover:bg-purple-500/20 border border-purple-500/20"
                                >
                                  <FileText className="w-3.5 h-3.5" />
                                </button>
                                <button
                                  onClick={() => triggerProjectDownload(r.report_id, "html")}
                                  title="Download HTML Report"
                                  className="p-1.5 rounded-lg bg-blue-500/10 text-blue-300 hover:bg-blue-500/20 border border-blue-500/20"
                                >
                                  <FileCode className="w-3.5 h-3.5" />
                                </button>
                                <button
                                  onClick={() => triggerProjectDownload(r.report_id, "zip")}
                                  title="Download Complete Project ZIP Bundle (with original dataset)"
                                  className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 border border-emerald-500/20"
                                >
                                  <Package className="w-3.5 h-3.5" />
                                </button>
                                <button
                                  onClick={() => handleInspectProjectReport(r.report_id)}
                                  title="Inspect in UI"
                                  className="p-1.5 rounded-lg bg-white/[0.05] text-white hover:bg-white/[0.08] border border-white/[0.08]"
                                >
                                  <Eye className="w-3.5 h-3.5" />
                                </button>
                                <button
                                  onClick={() => handleDeleteProjectReport(r.report_id)}
                                  title="Delete Report"
                                  className="p-1.5 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* SUB-VIEW B: 9-STEP REPORT BUILDER WIZARD */}
          {studioTab === "builder" && (
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-6">
              {/* Stepper Progress Bar */}
              <div className="border-b border-white/[0.08] pb-4">
                <div className="flex items-center justify-between text-xs font-semibold text-purple-400 mb-2">
                  <span>STEP {wizardStep} OF 9: {
                    wizardStep === 1 ? "Select Project Dataset" :
                    wizardStep === 2 ? "Select Workflow or Experiment Runs" :
                    wizardStep === 3 ? "Validate Compatibility" :
                    wizardStep === 4 ? "Select Report Sections" :
                    wizardStep === 5 ? "Configure Template & Audience" :
                    wizardStep === 6 ? "Configure Dataset & Artifact Inclusion" :
                    wizardStep === 7 ? "Preview Report Structure" :
                    wizardStep === 8 ? "Generate Report" :
                    "Download & Export Package"
                  }</span>
                  <span className="text-[#A4A4B8]">Step {wizardStep}/9</span>
                </div>
                <div className="w-full bg-[#171630] rounded-full h-1.5">
                  <div
                    className="bg-gradient-to-r from-purple-500 to-indigo-500 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${(wizardStep / 9) * 100}%` }}
                  />
                </div>
              </div>

              {/* STEP 1: SELECT DATASET */}
              {wizardStep === 1 && (
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Step 1: Choose Original Target Dataset
                  </h3>
                  <p className="text-xs text-[#A4A4B8]">
                    Select the exact dataset from your immutable library. AutoVeyra guarantees this file will never be modified or overwritten.
                  </p>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-96 overflow-y-auto">
                    {datasets.map((ds) => {
                      const isSelected = builderData.original_dataset_id === ds.dataset_id;
                      return (
                        <div
                          key={ds.dataset_id}
                          onClick={() => setBuilderData({ ...builderData, original_dataset_id: ds.dataset_id })}
                          className={`p-4 rounded-xl border cursor-pointer transition-all ${
                            isSelected
                              ? "bg-purple-600/20 border-purple-500 ring-1 ring-purple-500"
                              : "bg-[#171630] border-white/[0.06] hover:border-white/[0.15]"
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-bold text-white flex items-center gap-2">
                              <Database className="w-4 h-4 text-amber-400" />
                              {ds.filename}
                            </span>
                            {isSelected && <Check className="w-4 h-4 text-purple-400" />}
                          </div>
                          <div className="mt-2 text-[11px] text-[#A4A4B8] space-y-0.5">
                            <div>Dimensions: {ds.row_count?.toLocaleString()} rows × {ds.column_count} columns</div>
                            <div className="font-mono text-[10px] truncate">SHA: {ds.file_hash}</div>
                            <div>Size: {(ds.file_size / 1024).toFixed(1)} KB</div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* STEP 2: SELECT WORKFLOW OR RUNS */}
              {wizardStep === 2 && (
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Step 2: Select Execution Scope & Source Runs
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* EDA Run */}
                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1">EDA Profiling Run</label>
                      <select
                        value={builderData.selected_runs.eda}
                        onChange={(e) =>
                          setBuilderData({
                            ...builderData,
                            selected_runs: { ...builderData.selected_runs, eda: e.target.value },
                          })
                        }
                        className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
                      >
                        <option value="">-- Auto In-Memory Profiling --</option>
                        {runs
                          .filter((r) => r.module === "eda")
                          .map((r) => (
                            <option key={r.run_id} value={r.run_id}>
                              {r.run_name} ({r.run_id})
                            </option>
                          ))}
                      </select>
                    </div>

                    {/* Cleaning Run */}
                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1">Data Cleaning Run</label>
                      <select
                        value={builderData.selected_runs.cleaning}
                        onChange={(e) =>
                          setBuilderData({
                            ...builderData,
                            selected_runs: { ...builderData.selected_runs, cleaning: e.target.value },
                          })
                        }
                        className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
                      >
                        <option value="">-- None (Skip standalone cleaning) --</option>
                        {runs
                          .filter((r) => r.module === "cleaning")
                          .map((r) => (
                            <option key={r.run_id} value={r.run_id}>
                              {r.run_name} ({r.run_id})
                            </option>
                          ))}
                      </select>
                    </div>

                    {/* Features Run */}
                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1">Feature Engineering Run</label>
                      <select
                        value={builderData.selected_runs.features}
                        onChange={(e) =>
                          setBuilderData({
                            ...builderData,
                            selected_runs: { ...builderData.selected_runs, features: e.target.value },
                          })
                        }
                        className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
                      >
                        <option value="">-- None (Raw features) --</option>
                        {runs
                          .filter((r) => r.module === "features")
                          .map((r) => (
                            <option key={r.run_id} value={r.run_id}>
                              {r.run_name} ({r.run_id})
                            </option>
                          ))}
                      </select>
                    </div>

                    {/* Training Run */}
                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1">Model Training Run</label>
                      <select
                        value={builderData.selected_runs.training}
                        onChange={(e) =>
                          setBuilderData({
                            ...builderData,
                            selected_runs: { ...builderData.selected_runs, training: e.target.value },
                          })
                        }
                        className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
                      >
                        <option value="">-- Select Candidate Training Run --</option>
                        {runs
                          .filter((r) => r.module === "training")
                          .map((r) => (
                            <option key={r.run_id} value={r.run_id}>
                              {r.run_name} ({r.run_id})
                            </option>
                          ))}
                      </select>
                    </div>

                    {/* Tuning Run */}
                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1">Hyperparameter Tuning Run</label>
                      <select
                        value={builderData.selected_runs.tuning}
                        onChange={(e) =>
                          setBuilderData({
                            ...builderData,
                            selected_runs: { ...builderData.selected_runs, tuning: e.target.value },
                          })
                        }
                        className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
                      >
                        <option value="">-- None (Untuned Baseline) --</option>
                        {runs
                          .filter((r) => r.module === "tuning")
                          .map((r) => (
                            <option key={r.run_id} value={r.run_id}>
                              {r.run_name} ({r.run_id})
                            </option>
                          ))}
                      </select>
                    </div>
                  </div>
                </div>
              )}

              {/* STEP 3: COMPATIBILITY VALIDATION */}
              {wizardStep === 3 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                        Step 3: Experiment Compatibility & Lineage Validation
                      </h3>
                      <p className="text-xs text-[#A4A4B8] mt-0.5">
                        Verifying that all chosen runs trace to the same dataset and do not mix conflicting targets.
                      </p>
                    </div>
                    <button
                      onClick={handleValidateCompatibility}
                      disabled={validating}
                      className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-purple-600 hover:bg-purple-500 text-white flex items-center gap-1.5"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${validating ? "animate-spin" : ""}`} />
                      Run Verification
                    </button>
                  </div>

                  {!validationResult ? (
                    <div className="p-6 rounded-xl bg-[#171630] border border-white/[0.06] text-center text-xs text-[#A4A4B8]">
                      Click "Run Verification" to check compatibility between the selected runs and dataset.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div
                        className={`p-4 rounded-xl border flex items-center gap-3 ${
                          validationResult.is_compatible
                            ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                            : "bg-red-500/10 border-red-500/30 text-red-300"
                        }`}
                      >
                        {validationResult.is_compatible ? (
                          <CheckCircle2 className="w-5 h-5 flex-shrink-0" />
                        ) : (
                          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                        )}
                        <div className="text-xs">
                          <span className="font-bold">
                            {validationResult.is_compatible ? "All Selected Runs Are Compatible!" : "Compatibility Conflicts Found:"}
                          </span>
                          <div className="mt-0.5 text-[11px] opacity-90">
                            Target Column: {validationResult.resolved_target || "Auto-detected"} • Task: {validationResult.resolved_task || "General"}
                          </div>
                        </div>
                      </div>

                      {validationResult.conflicts.map((c, i) => (
                        <div key={i} className="p-3 rounded-lg bg-red-950/40 border border-red-500/30 text-xs text-red-200">
                          {c.reason}
                        </div>
                      ))}

                      {validationResult.warnings.map((w, i) => (
                        <div key={i} className="p-3 rounded-lg bg-amber-950/40 border border-amber-500/30 text-xs text-amber-200">
                          Note: {w.reason}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* STEP 4: SELECT REPORT SECTIONS */}
              {wizardStep === 4 && (
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Step 4: Select Included Report Sections
                  </h3>
                  <p className="text-xs text-[#A4A4B8]">
                    Choose which sections to include in the generated PDF, HTML, and JSON documentation.
                  </p>

                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    {STANDARD_SECTIONS.map((sec) => {
                      const isIncluded = builderData.included_sections.includes(sec.id);
                      return (
                        <div
                          key={sec.id}
                          onClick={() => toggleSection(sec.id)}
                          className={`p-3 rounded-xl border cursor-pointer transition-all ${
                            isIncluded
                              ? "bg-purple-600/15 border-purple-500/50"
                              : "bg-[#171630] border-white/[0.06] opacity-60"
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            {isIncluded ? (
                              <CheckSquare className="w-4 h-4 text-purple-400" />
                            ) : (
                              <Square className="w-4 h-4 text-gray-500" />
                            )}
                            <span className="text-xs font-semibold text-white">{sec.name}</span>
                          </div>
                          <p className="text-[10px] text-[#A4A4B8] mt-1 pl-6">{sec.desc}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* STEP 5: CONFIGURE TEMPLATE & AUDIENCE */}
              {wizardStep === 5 && (
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Step 5: Configure Report Template & Project Metadata
                  </h3>

                  <div>
                    <label className="block text-xs text-[#A4A4B8] mb-2 font-medium">Select Report Template</label>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {REPORT_TEMPLATES.map((tpl) => {
                        const isSelected = builderData.report_template === tpl.id;
                        return (
                          <div
                            key={tpl.id}
                            onClick={() => setBuilderData({ ...builderData, report_template: tpl.id })}
                            className={`p-4 rounded-xl border cursor-pointer transition-all ${
                              isSelected
                                ? "bg-purple-600/20 border-purple-500 ring-1 ring-purple-500"
                                : "bg-[#171630] border-white/[0.06]"
                            }`}
                          >
                            <div className="text-xs font-bold text-white">{tpl.name}</div>
                            <div className="text-[11px] text-[#A4A4B8] mt-1">{tpl.desc}</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1 font-medium">Project Title</label>
                      <input
                        type="text"
                        value={builderData.project_name}
                        onChange={(e) => setBuilderData({ ...builderData, project_name: e.target.value })}
                        placeholder="e.g. Credit Risk Scoring Model"
                        className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white"
                      />
                    </div>

                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1 font-medium">Author / Practitioner</label>
                      <input
                        type="text"
                        value={builderData.author_name}
                        onChange={(e) => setBuilderData({ ...builderData, author_name: e.target.value })}
                        placeholder="e.g. Lead ML Researcher"
                        className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white"
                      />
                    </div>

                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1 font-medium">Institution / Organization</label>
                      <input
                        type="text"
                        value={builderData.institution}
                        onChange={(e) => setBuilderData({ ...builderData, institution: e.target.value })}
                        placeholder="e.g. Department of Computer Science"
                        className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white"
                      />
                    </div>

                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1 font-medium">Project Notes</label>
                      <input
                        type="text"
                        value={builderData.project_description}
                        onChange={(e) => setBuilderData({ ...builderData, project_description: e.target.value })}
                        placeholder="Brief summary of research or deployment objective"
                        className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* STEP 6: DATASET & ARTIFACT INCLUSION */}
              {wizardStep === 6 && (
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Step 6: Configure Dataset & Artifact Inclusion (ZIP Export)
                  </h3>
                  <p className="text-xs text-[#A4A4B8]">
                    Control which raw files and derivatives are bundled into the project ZIP export.
                  </p>

                  <div className="space-y-3">
                    <label className="flex items-start gap-3 p-4 rounded-xl bg-[#171630] border border-white/[0.06] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={builderData.include_original_dataset}
                        onChange={(e) => setBuilderData({ ...builderData, include_original_dataset: e.target.checked })}
                        className="mt-0.5 rounded bg-[#111126] border-white/20 text-purple-600"
                      />
                      <div>
                        <div className="text-xs font-bold text-white flex items-center gap-2">
                          <Database className="w-3.5 h-3.5 text-amber-400" />
                          Include Exact Original Uploaded Dataset in ZIP
                        </div>
                        <p className="text-[11px] text-[#A4A4B8] mt-0.5">
                          Preserves the original CSV with SHA-256 integrity verification inside `Dataset/`.
                        </p>
                      </div>
                    </label>

                    <label className="flex items-start gap-3 p-4 rounded-xl bg-[#171630] border border-white/[0.06] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={builderData.include_cleaned_dataset}
                        onChange={(e) => setBuilderData({ ...builderData, include_cleaned_dataset: e.target.checked })}
                        className="mt-0.5 rounded bg-[#111126] border-white/20 text-purple-600"
                      />
                      <div>
                        <div className="text-xs font-bold text-white flex items-center gap-2">
                          <FileSpreadsheet className="w-3.5 h-3.5 text-blue-400" />
                          Include Derivative Cleaned Dataset (Optional)
                        </div>
                        <p className="text-[11px] text-[#A4A4B8] mt-0.5">
                          Bundles the intermediate cleaned CSV if a data cleaning run was selected.
                        </p>
                      </div>
                    </label>

                    <label className="flex items-start gap-3 p-4 rounded-xl bg-[#171630] border border-white/[0.06] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={builderData.include_model_artifact}
                        onChange={(e) => setBuilderData({ ...builderData, include_model_artifact: e.target.checked })}
                        className="mt-0.5 rounded bg-[#111126] border-white/20 text-purple-600"
                      />
                      <div>
                        <div className="text-xs font-bold text-white flex items-center gap-2">
                          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                          Include Trained Model Joblib Artifact (Optional)
                        </div>
                        <p className="text-[11px] text-[#A4A4B8] mt-0.5">
                          Note: Model weights may increase the ZIP download size significantly.
                        </p>
                      </div>
                    </label>
                  </div>
                </div>
              )}

              {/* STEP 7: PREVIEW SUMMARY */}
              {wizardStep === 7 && (
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Step 7: Preview Report Configuration
                  </h3>

                  <div className="p-5 rounded-xl bg-[#171630] border border-white/[0.06] space-y-3 text-xs">
                    <div className="flex justify-between border-b border-white/[0.06] pb-2">
                      <span className="text-[#A4A4B8]">Project Title:</span>
                      <span className="text-white font-bold">{builderData.project_name || "Untitled Project"}</span>
                    </div>
                    <div className="flex justify-between border-b border-white/[0.06] pb-2">
                      <span className="text-[#A4A4B8]">Template:</span>
                      <span className="text-purple-300 font-semibold uppercase">{builderData.report_template}</span>
                    </div>
                    <div className="flex justify-between border-b border-white/[0.06] pb-2">
                      <span className="text-[#A4A4B8]">Target Dataset ID:</span>
                      <span className="font-mono text-white">{builderData.original_dataset_id}</span>
                    </div>
                    <div className="flex justify-between border-b border-white/[0.06] pb-2">
                      <span className="text-[#A4A4B8]">Included Sections:</span>
                      <span className="text-emerald-400 font-bold">{builderData.included_sections.length} / 15 Sections</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#A4A4B8]">Original Dataset Inclusion:</span>
                      <span className="text-white">{builderData.include_original_dataset ? "Yes (Preserved in ZIP)" : "Excluded"}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* STEP 8: GENERATE REPORT */}
              {wizardStep === 8 && (
                <div className="space-y-4 text-center py-8">
                  <h3 className="text-base font-bold text-white">Ready to Compile Project Documentation</h3>
                  <p className="text-xs text-[#A4A4B8] max-w-md mx-auto">
                    Clicking compile will assemble data from all selected modules, render publication charts, generate the PDF & HTML documents, and prepare the project ZIP bundle.
                  </p>

                  <button
                    onClick={handleGenerateProjectReport}
                    disabled={generatingProject}
                    className="mt-4 px-6 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-lg shadow-purple-900/30 flex items-center gap-2 mx-auto"
                  >
                    <Sparkles className={`w-4 h-4 ${generatingProject ? "animate-spin" : ""}`} />
                    {generatingProject ? "Compiling Document..." : "Compile & Finalize Report"}
                  </button>
                </div>
              )}

              {/* STEP 9: DOWNLOAD & EXPORT */}
              {wizardStep === 9 && activeProjectReport && (
                <div className="space-y-5 text-center py-6">
                  <div className="w-12 h-12 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center mx-auto mb-2">
                    <CheckCircle2 className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-white">Project Report Successfully Generated!</h3>
                  <p className="text-xs text-[#A4A4B8] max-w-md mx-auto">
                    Your complete machine learning documentation is ready for download in your preferred format.
                  </p>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-2xl mx-auto pt-4">
                    <button
                      onClick={() => triggerProjectDownload(activeProjectReport.report_id, "pdf")}
                      className="p-4 rounded-xl bg-[#171630] hover:bg-[#1f1e40] border border-white/[0.08] text-white flex flex-col items-center gap-2 transition-all"
                    >
                      <FileText className="w-6 h-6 text-purple-400" />
                      <span className="text-xs font-bold">Download PDF</span>
                      <span className="text-[10px] text-[#A4A4B8]">ReportLab Document</span>
                    </button>

                    <button
                      onClick={() => triggerProjectDownload(activeProjectReport.report_id, "html")}
                      className="p-4 rounded-xl bg-[#171630] hover:bg-[#1f1e40] border border-white/[0.08] text-white flex flex-col items-center gap-2 transition-all"
                    >
                      <FileCode className="w-6 h-6 text-blue-400" />
                      <span className="text-xs font-bold">Download HTML</span>
                      <span className="text-[10px] text-[#A4A4B8]">Interactive Report</span>
                    </button>

                    <button
                      onClick={() => triggerProjectDownload(activeProjectReport.report_id, "json")}
                      className="p-4 rounded-xl bg-[#171630] hover:bg-[#1f1e40] border border-white/[0.08] text-white flex flex-col items-center gap-2 transition-all"
                    >
                      <FileSpreadsheet className="w-6 h-6 text-cyan-400" />
                      <span className="text-xs font-bold">Download JSON</span>
                      <span className="text-[10px] text-[#A4A4B8]">15-Section Schema</span>
                    </button>

                    <button
                      onClick={() => triggerProjectDownload(activeProjectReport.report_id, "zip")}
                      className="p-4 rounded-xl bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/30 text-white flex flex-col items-center gap-2 transition-all"
                    >
                      <Package className="w-6 h-6 text-emerald-400" />
                      <span className="text-xs font-bold">Download ZIP</span>
                      <span className="text-[10px] text-purple-200">Full Project Bundle</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Wizard Navigation Buttons */}
              <div className="flex items-center justify-between border-t border-white/[0.08] pt-4">
                <button
                  onClick={() => setWizardStep((prev) => Math.max(1, prev - 1))}
                  disabled={wizardStep === 1}
                  className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] disabled:opacity-40 flex items-center gap-1.5"
                >
                  <ChevronLeft className="w-3.5 h-3.5" /> Previous
                </button>

                {wizardStep < 8 ? (
                  <button
                    onClick={() => {
                      if (wizardStep === 1 && !builderData.original_dataset_id) {
                        alert("Select a dataset first.");
                        return;
                      }
                      setWizardStep((prev) => prev + 1);
                    }}
                    className="px-4 py-2 rounded-xl text-xs font-semibold bg-purple-600 hover:bg-purple-500 text-white flex items-center gap-1.5"
                  >
                    Next Step <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                ) : wizardStep === 9 ? (
                  <button
                    onClick={() => setStudioTab("history")}
                    className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#171630] text-white border border-white/[0.08]"
                  >
                    Back to History
                  </button>
                ) : null}
              </div>
            </div>
          )}

          {/* SUB-VIEW C: ACTIVE REPORT INSPECTOR */}
          {studioTab === "inspector" && activeProjectReport && (
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.08] pb-4">
                <div>
                  <span className="text-[10px] uppercase font-bold text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">
                    {activeProjectReport.report_template} Template
                  </span>
                  <h2 className="text-xl font-bold text-white mt-1">{activeProjectReport.project_name}</h2>
                  <div className="text-xs text-[#A4A4B8] font-mono mt-0.5">ID: {activeProjectReport.report_id}</div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => triggerProjectDownload(activeProjectReport.report_id, "pdf")}
                    className="px-3 py-1.5 rounded-xl text-xs bg-purple-600 hover:bg-purple-500 text-white flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" /> PDF
                  </button>
                  <button
                    onClick={() => triggerProjectDownload(activeProjectReport.report_id, "zip")}
                    className="px-3 py-1.5 rounded-xl text-xs bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5"
                  >
                    <Package className="w-3.5 h-3.5" /> ZIP Bundle
                  </button>
                </div>
              </div>

              {/* Report Preview Sections */}
              <div className="space-y-4 text-xs text-[#A4A4B8]">
                <div className="p-4 rounded-xl bg-[#171630] border border-white/[0.06]">
                  <h4 className="font-bold text-white uppercase tracking-wider mb-2">Executive Summary</h4>
                  <p>{activeProjectReport.report_data?.executive_summary?.summary_text}</p>
                </div>

                <div className="p-4 rounded-xl bg-[#171630] border border-white/[0.06]">
                  <h4 className="font-bold text-white uppercase tracking-wider mb-2">Original Dataset Specifications</h4>
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div>Filename: <span className="text-white font-bold">{activeProjectReport.original_filename}</span></div>
                    <div>Dimensions: <span className="text-white">{activeProjectReport.dataset_dimensions}</span></div>
                    <div className="col-span-2 font-mono">SHA-256: <span className="text-emerald-400">{activeProjectReport.dataset_sha256}</span></div>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-[#171630] border border-white/[0.06]">
                  <h4 className="font-bold text-white uppercase tracking-wider mb-2">Selected Module Executions</h4>
                  <pre className="text-[10px] font-mono text-purple-300 overflow-x-auto">
                    {JSON.stringify(activeProjectReport.selected_runs, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ===================================================================== */}
      {/* MODE 2: MODEL GOVERNANCE CARDS (PHASE 5 BASELINE PRESERVED) */}
      {/* ===================================================================== */}
      {activeMode === "model_cards" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left: Model Selector & Generate */}
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-4 h-fit">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-purple-400" /> Model Card Generator
              </h3>
              <p className="text-xs text-[#A4A4B8]">
                Generate an immutable evaluation snapshot and Model Card for a registered model artifact.
              </p>

              <div>
                <label className="block text-xs text-[#A4A4B8] mb-1 font-medium">Select Registered Model</label>
                <select
                  value={selectedModelId}
                  onChange={(e) => setSelectedModelId(e.target.value)}
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      #{m.id} - {m.name || m.model_name} ({m.task_type})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs text-[#A4A4B8] mb-1 font-medium">Custom Report Title (Optional)</label>
                <input
                  type="text"
                  value={reportTitleInput}
                  onChange={(e) => setReportTitleInput(e.target.value)}
                  placeholder="e.g. Production Risk Assessment Report"
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white"
                />
              </div>

              <button
                onClick={async () => {
                  if (!selectedModelId) return;
                  setGenerating(true);
                  try {
                    const res = await fetch(`${getApiBase()}/reports/generate`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        model_id: parseInt(selectedModelId, 10),
                        report_title: reportTitleInput || undefined,
                      }),
                    });
                    if (res.ok) {
                      const data = await res.json();
                      setActiveReportId(data.report_id);
                      fetchModelReports();
                    }
                  } catch (err) {
                    alert("Report generation failed.");
                  } finally {
                    setGenerating(false);
                  }
                }}
                disabled={generating || !selectedModelId}
                className="w-full py-2.5 rounded-xl text-xs font-semibold bg-purple-600 hover:bg-purple-500 text-white flex items-center justify-center gap-2"
              >
                <PlayCircle className={`w-4 h-4 ${generating ? "animate-spin" : ""}`} />
                {generating ? "Generating..." : "Generate Model Card"}
              </button>
            </div>

            {/* Right: Reports History & Viewer */}
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4">
                  Historical Model Cards ({reportsList.length})
                </h3>

                {reportsList.length === 0 ? (
                  <div className="text-center py-8 text-xs text-[#A4A4B8]">No model cards generated yet.</div>
                ) : (
                  <div className="space-y-3">
                    {reportsList.map((rep) => (
                      <div
                        key={rep.report_id}
                        className="p-4 rounded-xl bg-[#171630]/60 border border-white/[0.06] flex items-center justify-between"
                      >
                        <div>
                          <div className="text-xs font-bold text-white">{rep.report_title}</div>
                          <div className="text-[10px] text-[#A4A4B8] font-mono mt-0.5">
                            Model #{rep.model_id} • {rep.task_type} • {new Date(rep.created_at).toLocaleDateString()}
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => window.open(`${getApiBase()}/reports/${rep.report_id}/download?format=pdf`, "_blank")}
                            className="p-1.5 rounded-lg bg-purple-500/10 text-purple-300 hover:bg-purple-500/20 border border-purple-500/20"
                            title="Download PDF"
                          >
                            <FileText className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => window.open(`${getApiBase()}/reports/${rep.report_id}/download?format=html`, "_blank")}
                            className="p-1.5 rounded-lg bg-blue-500/10 text-blue-300 hover:bg-blue-500/20 border border-blue-500/20"
                            title="Download HTML"
                          >
                            <FileCode className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => setActiveReportId(rep.report_id)}
                            className="p-1.5 rounded-lg bg-white/[0.05] text-white hover:bg-white/[0.08] border border-white/[0.08]"
                            title="View Detail"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
