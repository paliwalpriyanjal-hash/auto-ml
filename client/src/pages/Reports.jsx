import { useState, useEffect } from "react";
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
  ChevronRight
} from "lucide-react";
import { getApiBase, formatMetric } from "../utils/api";

export default function Reports() {
  const [searchParams, setSearchParams] = useSearchParams();
  const preselectedModelId = searchParams.get("model");
  const preselectedReportId = searchParams.get("report");

  // Registry Models & Selection
  const [models, setModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState(preselectedModelId || "");
  const [reportTitleInput, setReportTitleInput] = useState("");

  // Reports List (History)
  const [reportsList, setReportsList] = useState([]);
  const [reportsLoading, setReportsLoading] = useState(true);
  const [activeReportId, setActiveReportId] = useState(preselectedReportId || null);
  const [currentReport, setCurrentReport] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Generation state
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState(null);

  // Model Card Modal
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

  // Feature filter in detail view
  const [featureSearch, setFeatureSearch] = useState("");

  // 1. Fetch Registered Models
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

  // 2. Fetch Reports History
  const fetchReports = async () => {
    setReportsLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/reports?limit=50`);
      if (res.ok) {
        const data = await res.json();
        const items = data.items || [];
        setReportsList(items);
        // If no active report but we have items, and no preselection
        if (!activeReportId && items.length > 0 && !preselectedModelId) {
          setActiveReportId(items[0].report_id);
        }
      }
    } catch (err) {
      console.error("Failed to load reports history", err);
    } finally {
      setReportsLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
    fetchReports();
  }, []);

  // 3. Fetch Full Report Detail when activeReportId changes
  useEffect(() => {
    if (!activeReportId) {
      setCurrentReport(null);
      return;
    }

    let isMounted = true;
    const fetchReportDetail = async () => {
      setDetailLoading(true);
      try {
        const res = await fetch(`${getApiBase()}/reports/${activeReportId}`);
        if (res.ok) {
          const data = await res.json();
          if (isMounted) {
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
        } else {
          if (isMounted) setCurrentReport(null);
        }
      } catch (err) {
        console.error("Error loading report detail", err);
      } finally {
        if (isMounted) setDetailLoading(false);
      }
    };

    fetchReportDetail();
    return () => {
      isMounted = false;
    };
  }, [activeReportId]);

  // Handle Model Generation
  const handleGenerateReport = async () => {
    if (!selectedModelId) return;
    setGenerating(true);
    setGenError(null);

    try {
      const payload = {
        model_id: parseInt(selectedModelId, 10),
      };
      if (reportTitleInput.trim()) {
        payload.report_title = reportTitleInput.trim();
      }

      const res = await fetch(`${getApiBase()}/reports/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to generate evaluation report");
      }

      const generated = await res.json();
      setReportTitleInput("");
      await fetchReports();
      setActiveReportId(generated.report_id);
      setCurrentReport(generated);
    } catch (err) {
      setGenError(err.message);
    } finally {
      setGenerating(false);
    }
  };

  // Handle Delete Report
  const handleDeleteReport = async (repId, e) => {
    e.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this report? The registered model and pipeline will remain intact.")) {
      return;
    }

    try {
      const res = await fetch(`${getApiBase()}/reports/${repId}`, { method: "DELETE" });
      if (res.ok) {
        await fetchReports();
        if (activeReportId === repId) {
          setActiveReportId(null);
          setCurrentReport(null);
        }
      }
    } catch (err) {
      console.error("Failed to delete report", err);
    }
  };

  // Handle Save Model Card Updates
  const handleSaveModelCard = async () => {
    if (!activeReportId) return;
    setSavingCard(true);
    setCardSaveSuccess(false);

    try {
      const res = await fetch(`${getApiBase()}/reports/${activeReportId}/model-card`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editCardData),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to update model card");
      }

      const updated = await res.json();
      setCurrentReport(updated);
      setCardSaveSuccess(true);
      setTimeout(() => {
        setIsEditCardOpen(false);
        setCardSaveSuccess(false);
      }, 900);
    } catch (err) {
      alert("Error saving model card: " + err.message);
    } finally {
      setSavingCard(false);
    }
  };

  // Handle Download File (PDF, HTML, JSON)
  const handleDownload = (fmt) => {
    if (!activeReportId) return;
    const url = `${getApiBase()}/reports/${activeReportId}/download?format=${fmt}`;
    const a = document.createElement("a");
    a.href = url;
    a.download = `autoveyra_report_${activeReportId.slice(0, 8)}.${fmt}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const selectedModelObj = models.find((m) => m.id.toString() === selectedModelId);

  // Filter feature chips in detail view
  const allFeatures = currentReport?.dataset_profile?.feature_columns || [];
  const filteredFeatures = allFeatures.filter((f) =>
    f.toLowerCase().includes(featureSearch.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-[#070711] text-[#F7F7FC] pt-20 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="mb-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.08] pb-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wider bg-[#8265FF]/10 text-[#8265FF] border border-[#8265FF]/20 flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5" />
                Audit & Compliance
              </span>
              <span className="text-xs text-[#A4A4B8]">• Immutable Snapshot Engine</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
              Model Evaluation Reports & Model Cards
            </h1>
            <p className="mt-1 text-xs sm:text-sm text-[#A4A4B8] max-w-2xl">
              Generate reproducible audit-ready evaluation reports, standardized model cards, and exportable artifacts (PDF, HTML, JSON) grounded in validated test evidence.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Link
              to="/models"
              className="px-3 py-2 rounded-xl text-xs font-semibold bg-[#111126] hover:bg-[#171630] text-[#A4A4B8] hover:text-white border border-white/[0.08] flex items-center gap-2 transition-all"
            >
              <Layers className="w-3.5 h-3.5" />
              Model Registry
            </Link>
          </div>
        </div>
      </div>

      {/* Model Selection & Generation Bar */}
      <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-5 mb-8 shadow-xl">
        <h2 className="text-xs font-bold uppercase tracking-wider text-[#A4A4B8] mb-3 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-[#8265FF]" />
          Generate New Evaluation Report
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
          {/* Model Selector */}
          <div className="md:col-span-5">
            <label className="block text-xs font-medium text-[#A4A4B8] mb-1">
              Select Registered Model
            </label>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              disabled={modelsLoading || generating}
              className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2.5 text-xs text-white focus:outline-none focus:border-[#8265FF] transition-colors"
            >
              {models.length === 0 ? (
                <option value="">No models registered in registry</option>
              ) : (
                models.map((m) => (
                  <option key={m.id} value={m.id}>
                    #{m.id} — {m.model_name} ({m.model_type} • {m.task_type})
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Optional Title */}
          <div className="md:col-span-4">
            <label className="block text-xs font-medium text-[#A4A4B8] mb-1">
              Custom Report Title <span className="text-[#A4A4B8]/60">(Optional)</span>
            </label>
            <input
              type="text"
              placeholder={
                selectedModelObj
                  ? `Model Evaluation Report — ${selectedModelObj.model_name}`
                  : "Model Evaluation Report..."
              }
              value={reportTitleInput}
              onChange={(e) => setReportTitleInput(e.target.value)}
              disabled={generating}
              className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2.5 text-xs text-white placeholder-[#A4A4B8]/50 focus:outline-none focus:border-[#8265FF] transition-colors"
            />
          </div>

          {/* Generate Button */}
          <div className="md:col-span-3">
            <button
              onClick={handleGenerateReport}
              disabled={generating || !selectedModelId || models.length === 0}
              className="w-full py-2.5 px-4 rounded-xl text-xs font-bold bg-[#8265FF] hover:bg-[#7152f8] disabled:opacity-50 text-white shadow-lg shadow-[#8265FF]/20 flex items-center justify-center gap-2 transition-all cursor-pointer"
            >
              {generating ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Generating Snapshot...
                </>
              ) : (
                <>
                  <FileText className="w-4 h-4" />
                  Generate Report
                </>
              )}
            </button>
          </div>
        </div>

        {genError && (
          <div className="mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{genError}</span>
          </div>
        )}
      </div>

      {/* Main Workspace Layout: History Sidebar + Active Report Details */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Reports History Drawer / Sidebar */}
        <div className="lg:col-span-4 space-y-3">
          <div className="flex items-center justify-between pb-1">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[#A4A4B8] flex items-center gap-2">
              <Calendar className="w-4 h-4 text-[#8265FF]" />
              Generated Reports ({reportsList.length})
            </h3>
            <button
              onClick={fetchReports}
              disabled={reportsLoading}
              className="p-1 rounded-lg text-[#A4A4B8] hover:text-white transition-colors"
              title="Refresh reports"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${reportsLoading ? "animate-spin" : ""}`} />
            </button>
          </div>

          {reportsLoading && reportsList.length === 0 ? (
            <div className="p-8 text-center bg-[#111126] border border-white/[0.06] rounded-2xl">
              <RefreshCw className="w-5 h-5 animate-spin mx-auto text-[#8265FF] mb-2" />
              <p className="text-xs text-[#A4A4B8]">Loading audit records...</p>
            </div>
          ) : reportsList.length === 0 ? (
            <div className="p-8 text-center bg-[#111126] border border-white/[0.06] rounded-2xl">
              <FileText className="w-8 h-8 mx-auto text-[#A4A4B8]/40 mb-2" />
              <p className="text-xs text-[#A4A4B8]">No reports generated yet.</p>
              <p className="text-[11px] text-[#A4A4B8]/60 mt-1">
                Select a model above and click "Generate Report" to create an audit-ready evaluation snapshot.
              </p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[720px] overflow-y-auto pr-1 custom-scrollbar">
              {reportsList.map((rep) => {
                const isActive = rep.report_id === activeReportId;
                return (
                  <div
                    key={rep.report_id}
                    onClick={() => setActiveReportId(rep.report_id)}
                    className={`p-3.5 rounded-xl border transition-all cursor-pointer relative group ${
                      isActive
                        ? "bg-[#171630] border-[#8265FF] shadow-md shadow-[#8265FF]/10"
                        : "bg-[#111126] border-white/[0.06] hover:border-white/[0.15]"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                              rep.task_type === "classification"
                                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                : "bg-purple-500/10 text-purple-400 border border-purple-500/20"
                            }`}
                          >
                            {rep.task_type}
                          </span>
                          <span className="text-[11px] font-mono text-[#A4A4B8]">
                            Model #{rep.model_id}
                          </span>
                        </div>
                        <h4 className="text-xs font-bold text-white truncate">
                          {rep.report_title || rep.model_name}
                        </h4>
                        <p className="text-[11px] text-[#A4A4B8] truncate mt-0.5">
                          {rep.model_name}
                        </p>
                        <span className="text-[10px] text-[#A4A4B8]/60 block mt-2">
                          {new Date(rep.created_at).toLocaleString()}
                        </span>
                      </div>

                      <button
                        onClick={(e) => handleDeleteReport(rep.report_id, e)}
                        className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-[#A4A4B8] hover:text-rose-400 hover:bg-rose-500/10 transition-all"
                        title="Delete report snapshot"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Active Report Details & Dashboard */}
        <div className="lg:col-span-8">
          {detailLoading ? (
            <div className="p-16 text-center bg-[#111126] border border-white/[0.08] rounded-2xl">
              <RefreshCw className="w-8 h-8 animate-spin mx-auto text-[#8265FF] mb-3" />
              <p className="text-sm text-[#A4A4B8]">Loading immutable report snapshot...</p>
            </div>
          ) : !currentReport ? (
            <div className="p-16 text-center bg-[#111126] border border-white/[0.08] rounded-2xl">
              <FileText className="w-12 h-12 mx-auto text-[#A4A4B8]/30 mb-3" />
              <h3 className="text-base font-bold text-white">No Report Selected</h3>
              <p className="text-xs text-[#A4A4B8] max-w-md mx-auto mt-1">
                Select an existing evaluation report from the left sidebar or generate a new one from your registered models.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Report Header & Action Bar */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.06] pb-5">
                  <div>
                    <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                      <span className="px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider bg-[#8265FF]/15 text-[#A855F7] border border-[#8265FF]/30">
                        {currentReport.task_type} Evaluation
                      </span>
                      <span className="text-xs text-[#A4A4B8] font-mono">
                        ID: {currentReport.report_id.slice(0, 10)}...
                      </span>
                      <span className="text-xs text-[#A4A4B8]">
                        • Registered Model #{currentReport.model_id}
                      </span>
                    </div>
                    <h2 className="text-xl font-extrabold text-white">
                      {currentReport.report_title}
                    </h2>
                    <p className="text-xs text-[#A4A4B8] mt-1">
                      Evaluated Algorithm: <strong className="text-white">{currentReport.model_name}</strong> • Created {new Date(currentReport.created_at).toLocaleString()}
                    </p>
                  </div>

                  {/* Multi-Format Export Buttons */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <button
                      onClick={() => handleDownload("pdf")}
                      className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#201f44] text-white border border-white/[0.08] flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
                      title="Download printable PDF Report"
                    >
                      <Download className="w-3.5 h-3.5 text-rose-400" />
                      PDF
                    </button>
                    <button
                      onClick={() => handleDownload("html")}
                      className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#201f44] text-white border border-white/[0.08] flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
                      title="Download self-contained HTML Report"
                    >
                      <Download className="w-3.5 h-3.5 text-emerald-400" />
                      HTML
                    </button>
                    <button
                      onClick={() => handleDownload("json")}
                      className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#201f44] text-white border border-white/[0.08] flex items-center gap-1.5 transition-all shadow-sm cursor-pointer"
                      title="Download machine-readable JSON Snapshot"
                    >
                      <Download className="w-3.5 h-3.5 text-[#8265FF]" />
                      JSON
                    </button>
                    <button
                      onClick={() => setIsEditCardOpen(true)}
                      className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5 transition-all shadow-sm shadow-[#8265FF]/20 cursor-pointer"
                    >
                      <Edit3 className="w-3.5 h-3.5" />
                      Edit Model Card
                    </button>
                  </div>
                </div>

                {/* Quick Highlight Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
                  <div className="p-3 rounded-xl bg-[#171630]/60 border border-white/[0.04]">
                    <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block">Primary Metric</span>
                    <span className="text-base font-extrabold text-emerald-400 font-mono mt-0.5 block">
                      {currentReport.task_type === "classification"
                        ? `${formatMetric(currentReport.performance_metrics?.metrics?.f1_macro, 4)} (F1)`
                        : `${formatMetric(currentReport.performance_metrics?.metrics?.rmse, 3)} (RMSE)`}
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-[#171630]/60 border border-white/[0.04]">
                    <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block">Input Features</span>
                    <span className="text-base font-extrabold text-white font-mono mt-0.5 block">
                      {currentReport.dataset_profile?.feature_columns?.length || 0} columns
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-[#171630]/60 border border-white/[0.04]">
                    <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block">CV Protocol</span>
                    <span className="text-base font-extrabold text-white font-mono mt-0.5 block">
                      {currentReport.evaluation_methodology?.cv_folds || 3}-Fold CV
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-[#171630]/60 border border-white/[0.04]">
                    <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block">Data Leakage</span>
                    <span className="text-xs font-bold text-emerald-400 flex items-center gap-1 mt-1">
                      <ShieldCheck className="w-3.5 h-3.5" />
                      Zero Leakage
                    </span>
                  </div>
                </div>
              </div>

              {/* 1. Executive Summary */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl">
                <h3 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] mb-3 flex items-center gap-2">
                  <Sparkles className="w-4 h-4" />
                  1. Executive Summary
                </h3>
                <div className="space-y-2">
                  {(currentReport.executive_summary?.key_findings || []).map((finding, idx) => (
                    <div key={idx} className="flex items-start gap-2.5 text-xs text-[#D5D5E6]">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <span>{finding}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* 2. Standardized Model Card */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4" />
                    2. Standardized Model Card (Mitchell et al. Schema)
                  </h3>
                  <button
                    onClick={() => setIsEditCardOpen(true)}
                    className="text-xs text-[#8265FF] hover:text-[#A855F7] font-semibold flex items-center gap-1 transition-colors"
                  >
                    <Edit3 className="w-3 h-3" />
                    Edit Narrative
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-[#171630]/70 border border-white/[0.04]">
                    <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block mb-1">
                      Intended Operational Use
                    </span>
                    <p className="text-xs text-white leading-relaxed">
                      {currentReport.model_card?.intended_use || "Not specified."}
                    </p>
                  </div>
                  <div className="p-4 rounded-xl bg-[#171630]/70 border border-white/[0.04]">
                    <span className="text-[10px] uppercase font-bold text-rose-300 block mb-1">
                      Out-of-Scope & Prohibited Use
                    </span>
                    <p className="text-xs text-[#F0F0FF] leading-relaxed">
                      {currentReport.model_card?.out_of_scope_use || "Not specified."}
                    </p>
                  </div>
                </div>

                <div className="mt-4 p-4 rounded-xl bg-[#171630]/70 border border-white/[0.04]">
                  <span className="text-[10px] uppercase font-bold text-amber-300 block mb-1">
                    Known Limitations & Constraints
                  </span>
                  <p className="text-xs text-[#F0F0FF] leading-relaxed">
                    {currentReport.model_card?.known_limitations || "Not specified."}
                  </p>
                </div>

                <div className="mt-4 p-4 rounded-xl bg-[#171630]/70 border border-white/[0.04]">
                  <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block mb-1">
                    Ethical, Subgroup Disparities & Monitoring
                  </span>
                  <p className="text-xs text-[#F0F0FF] leading-relaxed">
                    {currentReport.model_card?.ethical_considerations || "Not specified."}
                  </p>
                </div>

                {currentReport.model_card?.author_notes && (
                  <div className="mt-4 p-3 rounded-xl bg-[#0D0D1B] border border-white/[0.03]">
                    <span className="text-[10px] uppercase font-mono text-[#A4A4B8] block mb-0.5">
                      Author / Reviewer Notes:
                    </span>
                    <p className="text-xs text-[#C5C5D8] italic">
                      "{currentReport.model_card.author_notes}"
                    </p>
                  </div>
                )}
              </div>

              {/* 3. Performance Metrics Table */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl">
                <h3 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] mb-3 flex items-center gap-2">
                  <Database className="w-4 h-4" />
                  3. Performance Metrics (Holdout Partition)
                </h3>
                <p className="text-xs text-[#A4A4B8] mb-4">
                  Scores measured strictly on the untouched test partition evaluated at registration time:
                </p>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-white/[0.08] bg-[#171630]">
                        <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#A4A4B8]">
                          Metric
                        </th>
                        <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#A4A4B8] text-right">
                          Evaluated Score
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {Object.entries(currentReport.performance_metrics?.metrics || {})
                        .filter(([k, v]) => k !== "confusion_matrix" && typeof v === "number")
                        .map(([k, v]) => (
                          <tr key={k} className="hover:bg-white/[0.02]">
                            <td className="py-2.5 px-4 font-mono text-[#C5C5D8]">{k}</td>
                            <td className="py-2.5 px-4 font-mono font-bold text-emerald-400 text-right">
                              {typeof v === "number" ? formatMetric(v, 4) : String(v)}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>

                {/* Confusion Matrix if available */}
                {currentReport.performance_metrics?.metrics?.confusion_matrix && (
                  <div className="mt-5 p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.04]">
                    <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block mb-2">
                      Holdout Confusion Matrix
                    </span>
                    <div className="overflow-x-auto">
                      <table className="text-xs font-mono text-center">
                        <tbody>
                          {currentReport.performance_metrics.metrics.confusion_matrix.map(
                            (row, rIdx) => (
                              <tr key={rIdx}>
                                {row.map((val, cIdx) => (
                                  <td
                                    key={cIdx}
                                    className="p-2 border border-white/[0.08] bg-[#171630] font-bold text-white min-w-[50px]"
                                  >
                                    {val}
                                  </td>
                                ))}
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>

              {/* 4. Dataset Profile & Features */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] flex items-center gap-2">
                      <Cpu className="w-4 h-4" />
                      4. Dataset Profile & Expected Features
                    </h3>
                    <p className="text-xs text-[#A4A4B8] mt-1">
                      Target Column: <strong className="text-[#8265FF]">{currentReport.dataset_profile?.target_column}</strong> • Source: {currentReport.dataset_profile?.dataset_name}
                    </p>
                  </div>

                  <div className="relative w-full sm:w-56">
                    <Search className="w-3.5 h-3.5 text-[#A4A4B8] absolute left-2.5 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder="Filter features..."
                      value={featureSearch}
                      onChange={(e) => setFeatureSearch(e.target.value)}
                      className="w-full bg-[#171630] border border-white/[0.08] rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-[#A4A4B8]/60 focus:outline-none focus:border-[#8265FF]"
                    />
                  </div>
                </div>

                <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto pr-1">
                  {filteredFeatures.map((feat, idx) => (
                    <span
                      key={idx}
                      className="px-2.5 py-1 rounded-lg bg-[#171630] border border-white/[0.06] text-[11px] font-mono text-[#8265FF]"
                    >
                      {feat}
                    </span>
                  ))}
                  {filteredFeatures.length === 0 && (
                    <p className="text-xs text-[#A4A4B8]">No matching feature columns found.</p>
                  )}
                </div>
              </div>

              {/* 5. Hyperparameter Tuning & Feature Selection Evidence */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Tuning Evidence */}
                <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] mb-3 flex items-center gap-2">
                    <Sliders className="w-4 h-4" />
                    5A. Optuna Tuning Evidence
                  </h3>
                  {currentReport.tuning_history?.status === "available" ? (
                    <div className="space-y-3">
                      <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs">
                        Best Trial #{currentReport.tuning_history.best_trial_number} scored{" "}
                        <strong>{currentReport.tuning_history.best_score}</strong> ({currentReport.tuning_history.metric})
                      </div>
                      <div className="p-3 rounded-xl bg-[#0D0D1B] border border-white/[0.04]">
                        <span className="text-[10px] uppercase font-mono text-[#A4A4B8] block mb-1">
                          Optimal Hyperparameters:
                        </span>
                        <pre className="text-[11px] font-mono text-purple-300 overflow-x-auto whitespace-pre-wrap">
                          {JSON.stringify(currentReport.tuning_history.best_params, null, 2)}
                        </pre>
                      </div>
                    </div>
                  ) : (
                    <div className="p-4 rounded-xl bg-[#171630]/60 border border-white/[0.04] text-xs text-[#A4A4B8]">
                      <p>No tuning run linked directly to this model registration.</p>
                      <Link
                        to="/tune"
                        className="inline-flex items-center gap-1 text-[#8265FF] hover:underline mt-2 text-xs font-semibold"
                      >
                        Launch Optuna Tuning <ChevronRight className="w-3 h-3" />
                      </Link>
                    </div>
                  )}
                </div>

                {/* Feature Engineering Evidence */}
                <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] mb-3 flex items-center gap-2">
                    <Filter className="w-4 h-4" />
                    5B. Feature Engineering Evidence
                  </h3>
                  {currentReport.feature_engineering?.status === "available" ? (
                    <div className="space-y-3">
                      <div className="p-3 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-300 text-xs">
                        Winning Strategy: <strong>{currentReport.feature_engineering.winning_strategy}</strong>
                      </div>
                      <div className="text-xs text-[#A4A4B8]">
                        Selected {currentReport.feature_engineering.selected_features?.length || 0} features, pruned {currentReport.feature_engineering.removed_features?.length || 0}.
                      </div>
                    </div>
                  ) : (
                    <div className="p-4 rounded-xl bg-[#171630]/60 border border-white/[0.04] text-xs text-[#A4A4B8]">
                      <p>Trained on raw or standard pipeline features.</p>
                      <Link
                        to="/features"
                        className="inline-flex items-center gap-1 text-[#8265FF] hover:underline mt-2 text-xs font-semibold"
                      >
                        Explore Feature Selection <ChevronRight className="w-3 h-3" />
                      </Link>
                    </div>
                  )}
                </div>
              </div>

              {/* 6. Explainability Capabilities & Actions */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] mb-1 flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4" />
                      6. Explainability & Attribution
                    </h3>
                    <p className="text-xs text-[#A4A4B8]">
                      Supported Methods:{" "}
                      <strong className="text-white">
                        {(currentReport.explainability_summary?.supported_methods || ["Permutation Importance"]).join(", ")}
                      </strong>
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Link
                      to={`/explain?model=${currentReport.model_id}`}
                      className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#201f44] text-white border border-white/[0.08] flex items-center gap-1.5 transition-all shadow-sm"
                    >
                      <ShieldCheck className="w-3.5 h-3.5 text-[#8265FF]" />
                      Open Explainability
                    </Link>
                    <Link
                      to={`/predict?model=${currentReport.model_id}`}
                      className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white flex items-center gap-1.5 transition-all shadow-sm shadow-[#8265FF]/20"
                    >
                      <PlayCircle className="w-3.5 h-3.5" />
                      Test Inference
                    </Link>
                  </div>
                </div>
              </div>

              {/* 7. Provenance & Audit Trail Footer */}
              <div className="p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.04] text-[11px] text-[#6E6E88] flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <span>
                  AutoVeyra Platform • Report Schema v{currentReport.provenance?.report_schema_version || "1.0.0"} • Python {currentReport.provenance?.python_version}
                </span>
                <span className="font-mono text-[#8E8EA8]">
                  Checksum ID: {currentReport.report_id}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Model Card Edit Modal */}
      {isEditCardOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-[#111126] border border-white/[0.1] rounded-2xl p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Edit3 className="w-4 h-4 text-[#8265FF]" />
                Edit Standardized Model Card Narrative
              </h3>
              <button
                onClick={() => setIsEditCardOpen(false)}
                className="p-1 rounded-lg text-[#A4A4B8] hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-[#A4A4B8]">
              Update operational intent, limitations, and governance context. Measured metrics and baseline pipeline facts remain tamper-proof.
            </p>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-[#A4A4B8] mb-1">
                  Intended Operational Use
                </label>
                <textarea
                  rows={3}
                  value={editCardData.intended_use}
                  onChange={(e) =>
                    setEditCardData({ ...editCardData, intended_use: e.target.value })
                  }
                  className="w-full bg-[#171630] border border-white/[0.1] rounded-xl p-3 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                  placeholder="Describe target deployment context, operational environment, and intended users..."
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-rose-300 mb-1">
                  Out-of-Scope & Prohibited Use
                </label>
                <textarea
                  rows={2}
                  value={editCardData.out_of_scope_use}
                  onChange={(e) =>
                    setEditCardData({ ...editCardData, out_of_scope_use: e.target.value })
                  }
                  className="w-full bg-[#171630] border border-white/[0.1] rounded-xl p-3 text-xs text-white focus:outline-none focus:border-rose-500"
                  placeholder="Describe domains or high-stakes scenarios where this model should NOT be used..."
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-amber-300 mb-1">
                  Known Limitations & Constraints
                </label>
                <textarea
                  rows={2}
                  value={editCardData.known_limitations}
                  onChange={(e) =>
                    setEditCardData({ ...editCardData, known_limitations: e.target.value })
                  }
                  className="w-full bg-[#171630] border border-white/[0.1] rounded-xl p-3 text-xs text-white focus:outline-none focus:border-amber-500"
                  placeholder="e.g. Covariate shift sensitivity, data range boundaries, missing categories..."
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#A4A4B8] mb-1">
                  Ethical Considerations & Monitoring
                </label>
                <textarea
                  rows={2}
                  value={editCardData.ethical_considerations}
                  onChange={(e) =>
                    setEditCardData({ ...editCardData, ethical_considerations: e.target.value })
                  }
                  className="w-full bg-[#171630] border border-white/[0.1] rounded-xl p-3 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                  placeholder="Subgroup disparity audits, human-in-the-loop validation, retraining triggers..."
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#A4A4B8] mb-1">
                  Author / Reviewer Notes
                </label>
                <input
                  type="text"
                  value={editCardData.author_notes}
                  onChange={(e) =>
                    setEditCardData({ ...editCardData, author_notes: e.target.value })
                  }
                  className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                  placeholder="Review sign-off, auditor name, or compliance notes..."
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 pt-4 border-t border-white/[0.06]">
              {cardSaveSuccess && (
                <span className="text-xs text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Saved successfully!
                </span>
              )}
              <button
                onClick={() => setIsEditCardOpen(false)}
                className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171630] text-gray-300 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveModelCard}
                disabled={savingCard}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] disabled:opacity-50 text-white flex items-center gap-1.5 transition-all shadow-md shadow-[#8265FF]/20 cursor-pointer"
              >
                {savingCard ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-3.5 h-3.5" />
                    Save Model Card
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
