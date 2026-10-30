import React, { useState, useEffect } from "react";
import Papa from "papaparse";
import { Link, useNavigate } from "react-router-dom";
import {
  FiUpload,
  FiActivity,
  FiBarChart2,
  FiPieChart,
  FiAlertTriangle,
  FiCheckCircle,
  FiInfo,
  FiDownload,
  FiArrowRight,
  FiSliders,
  FiZap,
  FiTrendingUp,
  FiLayers,
  FiFileText,
  FiLoader,
  FiRefreshCw,
  FiGitCommit,
  FiCheck,
  FiShield
} from "react-icons/fi";
import { getApiBase } from "../utils/api";
import PageHeader from "../components/ui/PageHeader";
import Badge from "../components/ui/Badge";
import Button from "../components/ui/Button";

function formatNumber(val, digits = 4) {
  if (val === null || val === undefined || val === "") return "N/A";
  const num = Number(val);
  return Number.isFinite(num) ? num.toFixed(digits) : "N/A";
}

function formatPercent(val, digits = 2) {
  if (val === null || val === undefined || val === "") return "N/A";
  const num = Number(val);
  return Number.isFinite(num) ? `${(num * 100).toFixed(digits)}%` : "N/A";
}

const EDA = () => {
  const navigate = useNavigate();
  const [csvFile, setCsvFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [previewData, setPreviewData] = useState([]);
  const [targetColumn, setTargetColumn] = useState("");
  
  const [autoMode, setAutoMode] = useState(true);
  const [activeTab, setActiveTab] = useState("overview");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysis, setAnalysis] = useState(null);

  const [correlationMethod, setCorrelationMethod] = useState("pearson");
  const [insightFilter, setInsightFilter] = useState("all");

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError("");
    setAnalysis(null);
    setCsvFile(null);
    setPreviewData([]);
    setColumns([]);
    setTargetColumn("");

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Please upload a valid CSV file.");
      return;
    }

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (h) => h.trim(),
      complete: (result) => {
        const fields = (result.meta.fields || []).filter(Boolean);
        const data = result.data || [];

        if (fields.length < 2) {
          setError("CSV must contain a header row and at least 2 columns.");
          return;
        }

        setCsvFile(file);
        setColumns(fields);
        setPreviewData(data.slice(0, 5));
        setTargetColumn(fields[fields.length - 1]);
      },
    });
  };

  const runAnalysis = async () => {
    if (!csvFile) {
      setError("Please choose a CSV file first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const apiBase = getApiBase();
      const formData = new FormData();
      formData.append("file", csvFile);
      if (targetColumn) {
        formData.append("target_column", targetColumn);
      }

      const res = await fetch(`${apiBase}/eda/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Analysis request failed with status ${res.status}`);
      }

      const data = await res.json();
      setAnalysis(data);
      setActiveTab("overview");
    } catch (err) {
      console.error("EDA analysis execution error:", err);
      setError(err.message || "Failed to execute EDA analysis.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (format) => {
    if (!analysis?.analysis_id) return;
    const apiBase = getApiBase();
    window.open(`${apiBase}/eda/${analysis.analysis_id}/report/${format}`, "_blank");
  };

  const summary = analysis?.profile?.dataset_summary;
  const colProfiles = analysis?.profile?.column_profiles || {};
  const quality = analysis?.quality;
  const strategy = analysis?.strategy;
  const numerical = analysis?.numerical;
  const categorical = analysis?.categorical;
  const targetRel = analysis?.target_relationships;
  const insights = analysis?.insights || {};
  const flowchart = analysis?.flowchart;
  const visuals = analysis?.visualizations || {};

  const allInsightsList = [
    ...(insights.critical || []),
    ...(insights.warnings || []),
    ...(insights.info || []),
    ...(insights.recommendations || []),
  ];

  const filteredInsights = insightFilter === "all"
    ? allInsightsList
    : (insights[insightFilter] || []);

  return (
    <div className="min-h-screen pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
      {/* Universal Page Header */}
      <PageHeader
        category="Data Studio"
        title="Adaptive Exploratory Data Analysis"
        subtitle="Dataset-aware statistical profiling, correlation heatmaps, data quality diagnostics, and professional downloadable reports."
        badge={
          <div className="flex items-center gap-2">
            <Badge variant="violet" size="sm">
              Statistical Intelligence
            </Badge>
            <Badge variant="success" size="sm">
              Adaptive Profiling
            </Badge>
          </div>
        }
        actions={
          analysis && (
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="primary"
                size="sm"
                onClick={() => handleDownload("pdf")}
              >
                Download PDF Report
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => handleDownload("html")}
              >
                HTML Report
              </Button>
            </div>
          )
        }
      />

      {/* Upload and Configuration Card */}
      <section className="bg-[#111126] p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <FiUpload className="text-[#8265FF]" /> Dataset Source & Target Selection
          </h3>
          <div className="inline-flex rounded-xl bg-[#171630] p-1 border border-white/[0.08]">
            <button
              type="button"
              onClick={() => setAutoMode(true)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                autoMode
                  ? "bg-[#8265FF] text-white shadow-md shadow-[#8265FF]/30"
                  : "text-[#A4A4B8] hover:text-white"
              }`}
            >
              <FiZap /> Auto EDA (Recommended)
            </button>
            <button
              type="button"
              onClick={() => setAutoMode(false)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                !autoMode
                  ? "bg-[#8265FF] text-white shadow-md shadow-[#8265FF]/30"
                  : "text-[#A4A4B8] hover:text-white"
              }`}
            >
              <FiSliders /> Advanced EDA
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
          <div className="md:col-span-2">
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
              Upload Tabular CSV File
            </label>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="block w-full text-xs text-[#A4A4B8] file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-[#8265FF]/15 file:text-[#8265FF] hover:file:bg-[#8265FF]/25 cursor-pointer bg-[#171630] p-2 rounded-xl border border-white/[0.08]"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
              Target Column (Optional / Inferred)
            </label>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              disabled={columns.length === 0}
              className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2.5 text-xs text-white focus:outline-none focus:border-[#8265FF] disabled:opacity-40"
            >
              <option value="">Auto-infer target variable...</option>
              {columns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </div>
        </div>

        {csvFile && (
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
            <span className="text-xs text-emerald-400 flex items-center gap-1.5">
              <FiCheckCircle /> {csvFile.name} loaded ({columns.length} columns detected)
            </span>

            <button
              onClick={runAnalysis}
              disabled={loading}
              className="w-full sm:w-auto px-6 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-xl shadow-[#8265FF]/25 flex items-center justify-center gap-2 transition-all cursor-pointer disabled:opacity-50"
            >
              {loading ? (
                <>
                  <FiLoader className="animate-spin" />
                  Computing Adaptive Statistical Profiles...
                </>
              ) : (
                <>
                  <FiActivity />
                  Execute Comprehensive EDA
                </>
              )}
            </button>
          </div>
        )}

        {previewData.length > 0 && !analysis && (
          <div className="pt-3 border-t border-white/[0.06] overflow-x-auto">
            <p className="text-xs font-semibold text-[#A4A4B8] mb-2">Raw Dataset Preview (First 5 Rows):</p>
            <table className="min-w-full text-xs text-left divide-y divide-white/[0.06] border border-white/[0.06] rounded-xl overflow-hidden">
              <thead className="bg-[#0D0D1B]">
                <tr>
                  {columns.slice(0, 10).map((c) => (
                    <th key={c} className="px-3 py-2 text-[#A4A4B8] font-medium uppercase">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {previewData.map((row, i) => (
                  <tr key={i}>
                    {columns.slice(0, 10).map((c) => (
                      <td key={c} className="px-3 py-1.5 font-mono text-[#F7F7FC]">{row[c]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
          <FiAlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <section className="bg-[#111126] border border-white/[0.08] rounded-2xl p-16 flex flex-col items-center justify-center text-center space-y-4">
          <FiLoader className="h-10 w-10 animate-spin text-[#8265FF]" />
          <h3 className="text-xl font-bold text-white">Running Adaptive EDA Analysis...</h3>
          <p className="text-xs text-[#A4A4B8] max-w-md leading-relaxed">
            Profiling moments, evaluating Chi-square & Cramer's V correlations, auditing data quality, and rendering statistical visualizations.
          </p>
        </section>
      )}

      {/* Analysis Results Workspace */}
      {analysis && !loading && (
        <div className="space-y-6">
          
          {/* Navigation Tabs Bar */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-2 border-b border-white/[0.08]">
            {[
              { id: "overview", label: "Dataset Overview", icon: FiLayers },
              { id: "profiles", label: "Column Profiling", icon: FiSliders },
              { id: "quality", label: "Data Quality", icon: FiShield },
              { id: "numerical", label: "Numerical Analysis", icon: FiTrendingUp },
              { id: "categorical", label: "Categorical Analysis", icon: FiPieChart },
              { id: "target", label: "Target Relationships", icon: FiActivity },
              { id: "correlations", label: "Correlation Matrix", icon: FiBarChart2 },
              { id: "insights", label: `Insights (${insights.total_insights_count || 0})`, icon: FiAlertTriangle },
              { id: "flowchart", label: "EDA Workflow", icon: FiGitCommit },
              { id: "export", label: "Report Downloads", icon: FiDownload },
            ].map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                    active
                      ? "bg-[#8265FF] text-white shadow-lg shadow-[#8265FF]/25 border border-[#8265FF]/40"
                      : "bg-[#111126] text-[#A4A4B8] hover:text-white border border-white/[0.06] hover:bg-[#171630]"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* TAB 1: OVERVIEW */}
          {activeTab === "overview" && (
            <div className="space-y-6">
              {/* Stat Cards Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-xl shadow-md">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-[#A4A4B8] block mb-1">Row × Column Count</span>
                  <p className="text-2xl font-bold font-mono text-white">{summary.row_count} × {summary.column_count}</p>
                  <p className="text-[11px] text-[#A4A4B8] mt-1">{summary.memory_usage_str} total memory</p>
                </div>

                <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-xl shadow-md">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-[#A4A4B8] block mb-1">Target Feature</span>
                  <p className="text-xl font-bold font-mono text-[#8265FF] truncate">{strategy.target_column}</p>
                  <p className="text-[11px] text-[#A4A4B8] mt-1 capitalize">{strategy.task_type.replace('_', ' ')}</p>
                </div>

                <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-xl shadow-md">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-[#A4A4B8] block mb-1">Dataset Completeness</span>
                  <p className="text-2xl font-bold font-mono text-emerald-400">{(100 - (quality.total_missing_pct || 0)).toFixed(1)}%</p>
                  <p className="text-[11px] text-[#A4A4B8] mt-1">{quality.complete_rows_pct}% rows complete</p>
                </div>

                <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-xl shadow-md">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-[#A4A4B8] block mb-1">Duplicate Rows</span>
                  <p className="text-2xl font-bold font-mono text-white">{quality.duplicate_rows}</p>
                  <p className="text-[11px] text-[#A4A4B8] mt-1">{quality.duplicate_columns} duplicate cols</p>
                </div>
              </div>

              {/* Active Strategy Profiles Banner */}
              <div className="bg-[#111126] border border-white/[0.08] p-5 rounded-2xl space-y-3">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <FiZap className="text-[#8265FF]" /> Active Adaptive Profiles
                  </h3>
                  {strategy.is_sampled && (
                    <span className="text-[11px] font-mono text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20">
                      Sampled: {strategy.sample_size} Rows
                    </span>
                  )}
                </div>

                <div className="flex flex-wrap gap-2">
                  {strategy.active_profiles.map((p) => (
                    <span key={p} className="px-2.5 py-1 rounded-lg text-xs font-mono font-medium bg-[#8265FF]/10 text-[#8265FF] border border-[#8265FF]/20 capitalize">
                      {p.replace('_', ' ')}
                    </span>
                  ))}
                </div>

                <ul className="text-xs text-[#A4A4B8] space-y-1 pt-1 list-disc list-inside">
                  {strategy.strategy_reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>

              {/* Visualizations Preview */}
              {visuals.target_distribution_chart && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-2xl">
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Target Variable Distribution</h4>
                    <img src={visuals.target_distribution_chart} alt="Target Distribution" className="w-full rounded-xl object-contain" />
                  </div>
                  {visuals.correlation_heatmap_pearson && (
                    <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-2xl">
                      <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Pearson Correlation Heatmap</h4>
                      <img src={visuals.correlation_heatmap_pearson} alt="Correlation Heatmap" className="w-full rounded-xl object-contain" />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* TAB 2: COLUMN PROFILING */}
          {activeTab === "profiles" && (
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">Per-Column Statistical Profiles</h3>
              <div className="overflow-x-auto border border-white/[0.06] rounded-xl">
                <table className="min-w-full text-xs text-left divide-y divide-white/[0.06]">
                  <thead className="bg-[#0D0D1B]">
                    <tr>
                      <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Feature Name</th>
                      <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Semantic Type</th>
                      <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Missing %</th>
                      <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Distinct</th>
                      <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Mean / Top Value</th>
                      <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Skewness</th>
                      <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">IQR Outliers</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04]">
                    {Object.entries(colProfiles).map(([cName, prof]) => (
                      <tr key={cName} className="hover:bg-white/[0.02]">
                        <td className="px-4 py-3 font-mono font-bold text-white">{cName}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase border ${
                            prof.semantic_type === "numeric"
                              ? "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
                              : prof.semantic_type === "categorical"
                              ? "bg-purple-500/10 text-purple-400 border-purple-500/20"
                              : prof.semantic_type === "datetime"
                              ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                              : "bg-gray-500/10 text-gray-400 border-gray-500/20"
                          }`}>
                            {prof.semantic_type}
                          </span>
                        </td>
                        <td className="px-4 py-3 font-mono text-[#A4A4B8]">{prof.missing_pct}%</td>
                        <td className="px-4 py-3 font-mono text-white">{prof.unique_count}</td>
                        <td className="px-4 py-3 font-mono text-[#F7F7FC] truncate max-w-[160px]">
                          {prof.semantic_type === "numeric" ? formatNumber(prof.mean, 2) : String(prof.top_value ?? "N/A")}
                        </td>
                        <td className="px-4 py-3 font-mono text-[#A4A4B8]">
                          {prof.semantic_type === "numeric" ? formatNumber(prof.skewness, 2) : "-"}
                        </td>
                        <td className="px-4 py-3 font-mono text-amber-400">
                          {prof.semantic_type === "numeric" ? `${prof.outlier_count_iqr} (${prof.outlier_pct_iqr}%)` : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 3: DATA QUALITY */}
          {activeTab === "quality" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-xl">
                  <span className="text-[11px] font-semibold text-[#A4A4B8] block mb-1">Total Missing Cells</span>
                  <p className="text-2xl font-bold font-mono text-white">{quality.total_missing_cells} ({quality.total_missing_pct}%)</p>
                  <p className="text-xs text-[#A4A4B8] mt-1">{quality.columns_with_missing_count} columns affected</p>
                </div>
                <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-xl">
                  <span className="text-[11px] font-semibold text-[#A4A4B8] block mb-1">Completely Empty / Constant Cols</span>
                  <p className="text-2xl font-bold font-mono text-white">{summary.constant_columns_count}</p>
                  <p className="text-xs text-[#A4A4B8] mt-1">{summary.near_constant_columns_count} near-constant</p>
                </div>
                <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-xl">
                  <span className="text-[11px] font-semibold text-[#A4A4B8] block mb-1">Infinite / Invalid Values</span>
                  <p className="text-2xl font-bold font-mono text-white">{quality.infinite_values_count}</p>
                  <p className="text-xs text-[#A4A4B8] mt-1">Numerical range audit</p>
                </div>
              </div>

              {visuals.missingness_bar_chart && (
                <div className="bg-[#111126] border border-white/[0.08] p-5 rounded-2xl">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Missing Values Proportion Chart</h4>
                  <img src={visuals.missingness_bar_chart} alt="Missingness Chart" className="w-full rounded-xl object-contain max-h-80" />
                </div>
              )}

              {quality.leakage_candidates && quality.leakage_candidates.length > 0 && (
                <div className="bg-rose-950/20 border border-rose-500/30 p-5 rounded-2xl space-y-2">
                  <h4 className="text-xs font-bold text-rose-400 uppercase tracking-wider flex items-center gap-2">
                    <FiAlertTriangle /> Target Leakage Risk Candidates
                  </h4>
                  {quality.leakage_candidates.map((l) => (
                    <div key={l.column} className="text-xs text-rose-200">
                      <b>{l.column}</b>: {l.warning}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* TAB 4: NUMERICAL ANALYSIS */}
          {activeTab === "numerical" && (
            <div className="space-y-6">
              {visuals.numerical_distributions_chart && (
                <div className="bg-[#111126] border border-white/[0.08] p-5 rounded-2xl">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Top Numerical Feature Distributions</h4>
                  <img src={visuals.numerical_distributions_chart} alt="Numerical Distributions" className="w-full rounded-xl object-contain max-h-80" />
                </div>
              )}

              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-4">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">Numerical Moments & Distribution Statistics</h3>
                <div className="overflow-x-auto border border-white/[0.06] rounded-xl">
                  <table className="min-w-full text-xs text-left divide-y divide-white/[0.06]">
                    <thead className="bg-[#0D0D1B]">
                      <tr>
                        <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Feature</th>
                        <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Mean</th>
                        <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Std Dev</th>
                        <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Median</th>
                        <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Min / Max</th>
                        <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Skewness</th>
                        <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Kurtosis</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {(numerical?.summary_table || []).map((row) => (
                        <tr key={row.column} className="hover:bg-white/[0.02]">
                          <td className="px-4 py-2.5 font-mono font-bold text-white">{row.column}</td>
                          <td className="px-4 py-2.5 font-mono text-cyan-400">{formatNumber(row.mean, 2)}</td>
                          <td className="px-4 py-2.5 font-mono text-[#A4A4B8]">{formatNumber(row.std, 2)}</td>
                          <td className="px-4 py-2.5 font-mono text-white">{formatNumber(row.median, 2)}</td>
                          <td className="px-4 py-2.5 font-mono text-[#A4A4B8]">{formatNumber(row.min, 1)} / {formatNumber(row.max, 1)}</td>
                          <td className={`px-4 py-2.5 font-mono ${Math.abs(row.skewness) > 1.5 ? "text-amber-400 font-bold" : "text-[#A4A4B8]"}`}>
                            {formatNumber(row.skewness, 2)}
                          </td>
                          <td className="px-4 py-2.5 font-mono text-[#A4A4B8]">{formatNumber(row.kurtosis, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: CATEGORICAL ANALYSIS */}
          {activeTab === "categorical" && (
            <div className="space-y-6">
              {visuals.categorical_distributions_chart && (
                <div className="bg-[#111126] border border-white/[0.08] p-5 rounded-2xl">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Categorical Feature Top Categories</h4>
                  <img src={visuals.categorical_distributions_chart} alt="Categorical Distributions" className="w-full rounded-xl object-contain max-h-80" />
                </div>
              )}

              {categorical?.associations && categorical.associations.length > 0 && (
                <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-4">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <FiPieChart className="text-[#8265FF]" /> Categorical Associations (Cramer's V & Chi-Square Tests)
                  </h3>
                  <div className="overflow-x-auto border border-white/[0.06] rounded-xl">
                    <table className="min-w-full text-xs text-left divide-y divide-white/[0.06]">
                      <thead className="bg-[#0D0D1B]">
                        <tr>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Feature 1</th>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Feature 2</th>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Cramer's V</th>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">p-Value</th>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Significance</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.04]">
                        {categorical.associations.map((a, i) => (
                          <tr key={i} className="hover:bg-white/[0.02]">
                            <td className="px-4 py-2.5 font-mono text-white">{a.feature_1}</td>
                            <td className="px-4 py-2.5 font-mono text-white">{a.feature_2}</td>
                            <td className="px-4 py-2.5 font-mono font-bold text-[#8265FF]">{a.cramers_v}</td>
                            <td className="px-4 py-2.5 font-mono text-[#A4A4B8]">{formatNumber(a.p_value, 4)}</td>
                            <td className="px-4 py-2.5 font-semibold">
                              {a.significant ? (
                                <span className="text-emerald-400">p &lt; 0.05 (Significant)</span>
                              ) : (
                                <span className="text-[#A4A4B8]">p &ge; 0.05</span>
                              )}
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

          {/* TAB 6: TARGET RELATIONSHIPS */}
          {activeTab === "target" && (
            <div className="space-y-6">
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <FiActivity className="text-[#8265FF]" /> Target Variable: {targetRel.target_column}
                  </h3>
                  <span className="text-xs uppercase font-mono px-3 py-1 rounded-full bg-[#8265FF]/20 text-[#8265FF] border border-[#8265FF]/30">
                    {targetRel.task_type}
                  </span>
                </div>

                {targetRel.task_type === "regression" ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="bg-[#171630] p-3 rounded-xl border border-white/[0.06]">
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Mean</span>
                        <span className="text-lg font-bold font-mono text-white">{formatNumber(targetRel.target_statistics?.mean, 2)}</span>
                      </div>
                      <div className="bg-[#171630] p-3 rounded-xl border border-white/[0.06]">
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Std Dev</span>
                        <span className="text-lg font-bold font-mono text-white">{formatNumber(targetRel.target_statistics?.std, 2)}</span>
                      </div>
                      <div className="bg-[#171630] p-3 rounded-xl border border-white/[0.06]">
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Skewness</span>
                        <span className="text-lg font-bold font-mono text-amber-400">{formatNumber(targetRel.target_statistics?.skewness, 2)}</span>
                      </div>
                      <div className="bg-[#171630] p-3 rounded-xl border border-white/[0.06]">
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Min / Max</span>
                        <span className="text-lg font-bold font-mono text-white">{formatNumber(targetRel.target_statistics?.min, 1)} - {formatNumber(targetRel.target_statistics?.max, 1)}</span>
                      </div>
                    </div>

                    <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider pt-2">Feature-to-Target Correlation Ranking</h4>
                    <div className="overflow-x-auto border border-white/[0.06] rounded-xl">
                      <table className="min-w-full text-xs text-left divide-y divide-white/[0.06]">
                        <thead className="bg-[#0D0D1B]">
                          <tr>
                            <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Feature Name</th>
                            <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Pearson r</th>
                            <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Spearman rho</th>
                            <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Relationship Type</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/[0.04]">
                          {(targetRel.correlation_ranking || []).map((cr) => (
                            <tr key={cr.feature} className="hover:bg-white/[0.02]">
                              <td className="px-4 py-2.5 font-mono font-bold text-white">{cr.feature}</td>
                              <td className="px-4 py-2.5 font-mono font-bold text-cyan-400">{cr.pearson}</td>
                              <td className="px-4 py-2.5 font-mono text-[#A4A4B8]">{cr.spearman}</td>
                              <td className="px-4 py-2.5 font-medium">
                                {cr.non_linear_potential ? (
                                  <span className="text-amber-400">Potential Non-Linear (Spearman &gt; Pearson)</span>
                                ) : (
                                  <span className="text-emerald-400">Monotonic / Linear</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="bg-[#171630] p-3 rounded-xl border border-white/[0.06]">
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Distinct Classes</span>
                        <span className="text-lg font-bold font-mono text-white">{targetRel.classes_count}</span>
                      </div>
                      <div className="bg-[#171630] p-3 rounded-xl border border-white/[0.06]">
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Imbalance Ratio</span>
                        <span className="text-lg font-bold font-mono text-[#8265FF]">{targetRel.imbalance_ratio}:1</span>
                      </div>
                      <div className="bg-[#171630] p-3 rounded-xl border border-white/[0.06]">
                        <span className="text-[10px] text-[#A4A4B8] uppercase block">Severity</span>
                        <span className="text-lg font-bold font-mono text-amber-400 capitalize">{targetRel.imbalance_severity}</span>
                      </div>
                    </div>

                    <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider pt-2">Class Frequency Breakdown</h4>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {(targetRel.class_proportions || []).map((cp) => (
                        <div key={cp.class} className="bg-[#171630] p-3 rounded-xl border border-white/[0.06]">
                          <span className="text-xs font-bold font-mono text-white block">{cp.class}</span>
                          <span className="text-sm font-bold text-cyan-400">{cp.count}</span>
                          <span className="text-xs text-[#A4A4B8] ml-2">({cp.percentage}%)</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 7: CORRELATION MATRIX */}
          {activeTab === "correlations" && (
            <div className="space-y-6">
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <FiBarChart2 className="text-[#8265FF]" /> Numerical Feature Correlation Workbench
                  </h3>
                  <div className="inline-flex rounded-xl bg-[#171630] p-1 border border-white/[0.08]">
                    <button
                      onClick={() => setCorrelationMethod("pearson")}
                      className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                        correlationMethod === "pearson" ? "bg-[#8265FF] text-white" : "text-[#A4A4B8]"
                      }`}
                    >
                      Pearson (Linear)
                    </button>
                    <button
                      onClick={() => setCorrelationMethod("spearman")}
                      className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                        correlationMethod === "spearman" ? "bg-[#8265FF] text-white" : "text-[#A4A4B8]"
                      }`}
                    >
                      Spearman (Rank)
                    </button>
                  </div>
                </div>

                {visuals.correlation_heatmap_pearson && (
                  <div className="bg-[#171630] p-4 rounded-xl border border-white/[0.06] flex justify-center">
                    <img src={visuals.correlation_heatmap_pearson} alt="Correlation Heatmap" className="max-h-96 object-contain" />
                  </div>
                )}

                <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider pt-2">Highly Collinear Pairs (|r| &ge; 0.80)</h4>
                {numerical?.highly_correlated_pairs && numerical.highly_correlated_pairs.length > 0 ? (
                  <div className="overflow-x-auto border border-white/[0.06] rounded-xl">
                    <table className="min-w-full text-xs text-left divide-y divide-white/[0.06]">
                      <thead className="bg-[#0D0D1B]">
                        <tr>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Feature 1</th>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Feature 2</th>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Pearson r</th>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Spearman rho</th>
                          <th className="px-4 py-3 font-semibold text-[#A4A4B8] uppercase">Collinearity Level</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.04]">
                        {numerical.highly_correlated_pairs.map((pair, idx) => (
                          <tr key={idx} className="hover:bg-white/[0.02]">
                            <td className="px-4 py-2.5 font-mono text-white">{pair.feature_1}</td>
                            <td className="px-4 py-2.5 font-mono text-white">{pair.feature_2}</td>
                            <td className="px-4 py-2.5 font-mono font-bold text-cyan-400">{pair.pearson}</td>
                            <td className="px-4 py-2.5 font-mono text-[#A4A4B8]">{pair.spearman}</td>
                            <td className="px-4 py-2.5 font-semibold text-amber-400">{pair.risk}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-xs text-[#A4A4B8]">No severe collinearity detected among numerical features.</p>
                )}
              </div>
            </div>
          )}

          {/* TAB 8: AUTOMATED INSIGHTS */}
          {activeTab === "insights" && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                {[
                  { id: "all", label: `All Findings (${insights.total_insights_count || 0})` },
                  { id: "critical", label: `Critical Alerts (${insights.critical?.length || 0})` },
                  { id: "warnings", label: `Warnings (${insights.warnings?.length || 0})` },
                  { id: "recommendations", label: `Recommendations (${insights.recommendations?.length || 0})` },
                  { id: "info", label: `Dataset Info (${insights.info?.length || 0})` },
                ].map((f) => (
                  <button
                    key={f.id}
                    onClick={() => setInsightFilter(f.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                      insightFilter === f.id
                        ? "bg-[#8265FF] text-white shadow-md shadow-[#8265FF]/20"
                        : "bg-[#111126] text-[#A4A4B8] hover:text-white border border-white/[0.06]"
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>

              <div className="grid grid-cols-1 gap-3">
                {filteredInsights.map((itm, i) => (
                  <div
                    key={i}
                    className={`p-5 rounded-2xl border space-y-2 bg-[#111126] ${
                      itm.severity === "critical"
                        ? "border-rose-500/30"
                        : itm.severity === "warning"
                        ? "border-amber-500/30"
                        : itm.severity === "recommendation"
                        ? "border-[#8265FF]/30"
                        : "border-white/[0.06]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white flex items-center gap-2">
                        {itm.severity === "critical" && <FiAlertTriangle className="text-rose-400" />}
                        {itm.severity === "warning" && <FiAlertTriangle className="text-amber-400" />}
                        {itm.severity === "recommendation" && <FiCheckCircle className="text-[#8265FF]" />}
                        {itm.severity === "info" && <FiInfo className="text-cyan-400" />}
                        {itm.title}
                      </span>
                      <span className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded-full border ${
                        itm.severity === "critical"
                          ? "bg-rose-500/10 text-rose-400 border-rose-500/20"
                          : itm.severity === "warning"
                          ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                          : itm.severity === "recommendation"
                          ? "bg-[#8265FF]/10 text-[#8265FF] border-[#8265FF]/20"
                          : "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
                      }`}>
                        {itm.category}
                      </span>
                    </div>

                    <p className="text-xs text-[#A4A4B8] leading-relaxed">{itm.explanation}</p>

                    <div className="pt-1 flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-t border-white/[0.04]">
                      <span className="text-[11px] font-mono text-[#F7F7FC]">
                        <b>Supporting Statistic:</b> {itm.supporting_statistic}
                      </span>
                      <span className="text-xs font-semibold text-emerald-400">
                        Action: {itm.recommendation}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 9: FLOWCHART */}
          {activeTab === "flowchart" && (
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-6">
              <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <FiGitCommit className="text-[#8265FF]" /> Dynamic EDA Execution Flowchart
                </h3>
                <span className="text-xs font-mono text-emerald-400">
                  Execution Time: {flowchart?.execution_time_ms} ms ({flowchart?.completed_steps} Completed, {flowchart?.skipped_steps} Skipped)
                </span>
              </div>

              <div className="space-y-3">
                {(flowchart?.steps || []).map((step, idx) => (
                  <div key={step.id} className="flex items-start gap-3">
                    <div className="flex flex-col items-center">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                        step.status === "completed"
                          ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                          : "bg-gray-500/20 text-gray-400 border border-gray-500/40"
                      }`}>
                        {idx + 1}
                      </div>
                      {idx < flowchart.steps.length - 1 && (
                        <div className="w-0.5 h-6 bg-white/[0.08] my-1" />
                      )}
                    </div>

                    <div className="flex-1 bg-[#171630] p-3 rounded-xl border border-white/[0.06] flex items-center justify-between">
                      <div>
                        <span className="text-xs font-bold text-white block">{step.title}</span>
                        <span className="text-[11px] text-[#A4A4B8]">{step.details}</span>
                      </div>
                      <span className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded-full border ${
                        step.status === "completed"
                          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                          : "bg-gray-500/10 text-gray-400 border-gray-500/20"
                      }`}>
                        {step.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 10: REPORT DOWNLOADS & ACTION BUTTONS */}
          {activeTab === "export" && (
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-6">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <FiDownload className="text-[#8265FF]" /> Export Professional Reports & Artifacts
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-[#171630] border border-white/[0.06] p-5 rounded-xl space-y-3 flex flex-col justify-between">
                  <div>
                    <span className="text-xs font-bold text-white block">Multi-Page PDF Executive Report</span>
                    <p className="text-xs text-[#A4A4B8] mt-1">
                      Formal 9-page audit report with ReportLab styling, data tables, and embedded chart figures suitable for executive review.
                    </p>
                  </div>
                  <button
                    onClick={() => handleDownload("pdf")}
                    className="w-full py-2.5 rounded-xl bg-[#8265FF] hover:bg-[#7152f8] text-white font-bold text-xs flex items-center justify-center gap-2 transition"
                  >
                    <FiDownload /> Download PDF Report
                  </button>
                </div>

                <div className="bg-[#171630] border border-white/[0.06] p-5 rounded-xl space-y-3 flex flex-col justify-between">
                  <div>
                    <span className="text-xs font-bold text-white block">Standalone HTML Report</span>
                    <p className="text-xs text-[#A4A4B8] mt-1">
                      Interactive dark-themed HTML report containing embedded base64 statistical visualizations.
                    </p>
                  </div>
                  <button
                    onClick={() => handleDownload("html")}
                    className="w-full py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-white font-bold text-xs flex items-center justify-center gap-2 transition"
                  >
                    <FiDownload /> Download HTML Report
                  </button>
                </div>

                <div className="bg-[#171630] border border-white/[0.06] p-5 rounded-xl space-y-3 flex flex-col justify-between">
                  <div>
                    <span className="text-xs font-bold text-white block">Structured JSON Data Artifact</span>
                    <p className="text-xs text-[#A4A4B8] mt-1">
                      Machine-readable JSON schema export containing moments, correlations, and quality diagnostics.
                    </p>
                  </div>
                  <button
                    onClick={() => handleDownload("json")}
                    className="w-full py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-700 text-white font-bold text-xs flex items-center justify-center gap-2 transition"
                  >
                    <FiDownload /> Download JSON Export
                  </button>
                </div>
              </div>

              {/* Handoff Next Steps */}
              <div className="border-t border-white/[0.06] pt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div>
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider">Next Pipeline Steps</h4>
                  <p className="text-xs text-[#A4A4B8]">Handoff this audited dataset directly into preprocessing or model training.</p>
                </div>

                <div className="flex items-center gap-3">
                  <Link
                    to="/clean"
                    className="px-4 py-2.5 bg-[#171630] border border-white/[0.1] hover:border-white/[0.2] text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition"
                  >
                    Continue to Data Cleaning <FiArrowRight />
                  </Link>
                  <Link
                    to="/automl"
                    className="px-5 py-2.5 bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white rounded-xl text-xs font-bold flex items-center gap-2 shadow-lg shadow-[#8265FF]/25 transition"
                  >
                    Continue to Model Training <FiArrowRight />
                  </Link>
                </div>
              </div>
            </div>
          )}

        </div>
      )}

    </div>
  );
};

export default EDA;
