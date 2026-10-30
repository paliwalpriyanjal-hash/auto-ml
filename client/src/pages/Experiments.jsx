import { useState, useEffect, useMemo } from "react";
import {
  Layers,
  Database,
  Play,
  GitBranch,
  Sliders,
  CheckCircle2,
  Clock,
  AlertTriangle,
  XCircle,
  Search,
  RefreshCw,
  Eye,
  Trash2,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Plus,
  ArrowLeftRight,
  FileText,
  Filter,
  Check,
  BarChart2,
  Activity,
  Workflow,
  Download,
  Share2,
  Cpu,
  Sparkles,
  HelpCircle,
  FolderGit2
} from "lucide-react";
import { getApiBase, formatMetric } from "../utils/api";
import PageHeader from "../components/ui/PageHeader";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Button from "../components/ui/Button";

const MODULES = [
  { id: "eda", label: "EDA Profiling", icon: Eye, color: "text-blue-400 bg-blue-500/10 border-blue-500/20" },
  { id: "cleaning", label: "Data Cleaning", icon: Sparkles, color: "text-amber-400 bg-amber-500/10 border-amber-500/20" },
  { id: "features", label: "Feature Engineering", icon: Sliders, color: "text-purple-400 bg-purple-500/10 border-purple-500/20" },
  { id: "training", label: "Model Training", icon: Cpu, color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" },
  { id: "tuning", label: "Hyperparameter Tuning", icon: Activity, color: "text-pink-400 bg-pink-500/10 border-pink-500/20" },
  { id: "evaluation", label: "Model Evaluation", icon: BarChart2, color: "text-cyan-400 bg-cyan-500/10 border-cyan-500/20" },
];

export default function Experiments() {
  const [activeTab, setActiveTab] = useState("overview"); // overview, new, runs, compare, lineage, workflows
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [notification, setNotification] = useState(null);

  // Core Data Collections
  const [datasets, setDatasets] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [runs, setRuns] = useState([]);
  const [workflows, setWorkflows] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [lineageData, setLineageData] = useState(null);

  // Comparison State
  const [selectedRunIdsForCompare, setSelectedRunIdsForCompare] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [comparingLoading, setComparingLoading] = useState(false);

  // Filter States
  const [moduleFilter, setModuleFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");

  // New Experiment Form State
  const [newExpName, setNewExpName] = useState("");
  const [newExpModule, setNewExpModule] = useState("training");
  const [newExpDatasetId, setNewExpDatasetId] = useState("");
  const [newExpDesc, setNewExpDesc] = useState("");
  const [newExpParams, setNewExpParams] = useState({
    target_column: "",
    model_type: "logistic",
    task_type: "classification",
    scaler: "standard",
    split_ratio: 0.8,
    missing_strategy: "median",
    impute_missing: true,
    remove_duplicates: true,
    enable_polynomial: true,
    enable_interactions: true,
    n_trials: 10,
    n_splits: 3,
  });

  // Workflow Form State
  const [workflowName, setWorkflowName] = useState("");
  const [workflowDesc, setWorkflowDesc] = useState("");
  const [workflowStages, setWorkflowStages] = useState([
    { name: "Clean Stage", module: "cleaning", parameters: { impute_missing: true, remove_duplicates: true } },
    { name: "Feature Expansion", module: "features", parameters: { enable_polynomial: true, enable_interactions: true } },
    { name: "Candidate Training", module: "training", parameters: { model_type: "logistic", scaler: "standard" } },
  ]);
  const [wfExecDatasetId, setWfExecDatasetId] = useState("");
  const [activeWfExecution, setActiveWfExecution] = useState(null);

  const notify = (msg, type = "success") => {
    setNotification({ msg, type });
    setTimeout(() => setNotification(null), 4000);
  };

  // Fetch all initial workbench data
  const refreshAllData = async () => {
    setLoading(true);
    setError(null);
    const base = getApiBase();

    try {
      const [resExp, resRuns, resDatasets, resWf] = await Promise.all([
        fetch(`${base}/experiments/workbench?limit=100`).then((r) => r.ok ? r.json() : { experiments: [] }),
        fetch(`${base}/experiments/runs?limit=100`).then((r) => r.ok ? r.json() : { runs: [] }),
        fetch(`${base}/experiments/datasets?limit=100`).then((r) => r.ok ? r.json() : { datasets: [] }),
        fetch(`${base}/experiments/workflows?limit=100`).then((r) => r.ok ? r.json() : { workflows: [] }),
      ]);

      setExperiments(resExp.experiments || []);
      setRuns(resRuns.runs || []);
      setDatasets(resDatasets.datasets || []);
      setWorkflows(resWf.workflows || []);

      if (resDatasets.datasets && resDatasets.datasets.length > 0 && !newExpDatasetId) {
        setNewExpDatasetId(resDatasets.datasets[0].dataset_id);
        setWfExecDatasetId(resDatasets.datasets[0].dataset_id);
      }
    } catch (err) {
      setError("Failed to synchronize workbench records from backend.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshAllData();
  }, []);

  // Filtered Runs
  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      const matchesModule = moduleFilter === "all" || r.module === moduleFilter;
      const matchesStatus = statusFilter === "all" || r.status === statusFilter;
      const term = searchTerm.toLowerCase();
      const matchesSearch =
        !term ||
        r.run_name.toLowerCase().includes(term) ||
        r.run_id.toLowerCase().includes(term) ||
        r.module.toLowerCase().includes(term);
      return matchesModule && matchesStatus && matchesSearch;
    });
  }, [runs, moduleFilter, statusFilter, searchTerm]);

  // Execute an Experiment Run
  const handleExecuteRun = async (runId) => {
    const base = getApiBase();
    notify(`Starting execution for run ${runId}...`, "info");
    try {
      const res = await fetch(`${base}/experiments/runs/${runId}/execute`, { method: "POST" });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Run execution failed.");
      }
      const updated = await res.json();
      notify(`Run ${runId} completed in ${updated.duration_seconds}s!`, "success");
      refreshAllData();
      if (selectedRun && selectedRun.run_id === runId) {
        setSelectedRun(updated);
      }
    } catch (err) {
      notify(`Execution error: ${err.message}`, "error");
      refreshAllData();
    }
  };

  // Cancel an Experiment Run
  const handleCancelRun = async (runId) => {
    const base = getApiBase();
    try {
      const res = await fetch(`${base}/experiments/runs/${runId}/cancel`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to cancel run.");
      notify(`Run ${runId} cancelled.`, "info");
      refreshAllData();
    } catch (err) {
      notify(`Error: ${err.message}`, "error");
    }
  };

  // Delete an Experiment Run
  const handleDeleteRun = async (runId) => {
    if (!window.confirm(`Delete run ${runId}?`)) return;
    const base = getApiBase();
    try {
      const res = await fetch(`${base}/experiments/runs/${runId}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete run.");
      notify(`Run ${runId} deleted.`, "success");
      setSelectedRunIdsForCompare((prev) => prev.filter((id) => id !== runId));
      if (selectedRun && selectedRun.run_id === runId) setSelectedRun(null);
      refreshAllData();
    } catch (err) {
      notify(`Error: ${err.message}`, "error");
    }
  };

  // Create & Optionally Execute New Experiment
  const handleCreateNewExperiment = async (executeNow = false) => {
    if (!newExpName.trim()) {
      alert("Please provide an experiment name.");
      return;
    }
    if (!newExpDatasetId) {
      alert("Please select a dataset.");
      return;
    }

    const base = getApiBase();
    setLoading(true);
    try {
      // 1. Create Experiment container
      const expRes = await fetch(`${base}/experiments/workbench`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newExpName,
          module: newExpModule,
          description: newExpDesc,
          dataset_id: newExpDatasetId,
          tags: [newExpModule, "manual"],
        }),
      });
      if (!expRes.ok) throw new Error("Failed to create experiment.");
      const expData = await expRes.json();

      // 2. Create Run inside experiment
      const runRes = await fetch(`${base}/experiments/workbench/${expData.experiment_id}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_name: `${newExpName} - Run 1`,
          module: newExpModule,
          parameters: newExpParams,
          dataset_id: newExpDatasetId,
        }),
      });
      if (!runRes.ok) throw new Error("Failed to create initial run.");
      const runData = await runRes.json();

      notify(`Experiment created! Run ID: ${runData.run_id}`, "success");

      if (executeNow) {
        handleExecuteRun(runData.run_id);
      } else {
        refreshAllData();
      }

      setActiveTab("runs");
    } catch (err) {
      notify(`Error: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  };

  // Compare Selected Runs
  const handleTriggerCompare = async () => {
    if (selectedRunIdsForCompare.length < 2) {
      alert("Select at least 2 runs from the same module to compare.");
      return;
    }

    const base = getApiBase();
    setComparingLoading(true);
    try {
      const res = await fetch(`${base}/experiments/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: selectedRunIdsForCompare }),
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Comparison failed.");
      }
      const data = await res.json();
      setComparisonResult(data);
      setActiveTab("compare");
    } catch (err) {
      alert(err.message);
    } finally {
      setComparingLoading(false);
    }
  };

  // Toggle Run Selection for Comparison
  const toggleSelectRunForCompare = (runId) => {
    setSelectedRunIdsForCompare((prev) => {
      if (prev.includes(runId)) {
        return prev.filter((id) => id !== runId);
      } else {
        if (prev.length >= 4) {
          alert("Maximum 4 runs can be compared simultaneously.");
          return prev;
        }
        return [...prev, runId];
      }
    });
  };

  // Load Lineage for a Run
  const handleViewLineage = async (runId) => {
    const base = getApiBase();
    setLoading(true);
    try {
      const res = await fetch(`${base}/experiments/runs/${runId}/lineage`);
      if (!res.ok) throw new Error("Failed to fetch lineage DAG.");
      const data = await res.json();
      setLineageData(data);
      setActiveTab("lineage");
    } catch (err) {
      notify(`Lineage error: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  };

  // Execute Workflow
  const handleExecuteWorkflow = async (workflowId) => {
    if (!wfExecDatasetId) {
      alert("Select a dataset to run the workflow on.");
      return;
    }
    const base = getApiBase();
    notify("Initiating sequential workflow pipeline...", "info");
    try {
      const res = await fetch(`${base}/experiments/workflows/${workflowId}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: wfExecDatasetId }),
      });
      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || "Workflow execution failed.");
      }
      const execResult = await res.json();
      setActiveWfExecution(execResult);
      notify(`Workflow execution completed with status: ${execResult.status}!`, "success");
      refreshAllData();
    } catch (err) {
      notify(`Workflow error: ${err.message}`, "error");
      refreshAllData();
    }
  };

  // Save New Workflow
  const handleSaveWorkflow = async () => {
    if (!workflowName.trim()) {
      alert("Workflow name is required.");
      return;
    }
    const base = getApiBase();
    try {
      const res = await fetch(`${base}/experiments/workflows`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: workflowName,
          description: workflowDesc,
          stages: workflowStages,
        }),
      });
      if (!res.ok) throw new Error("Failed to save workflow definition.");
      notify("Workflow defined successfully!", "success");
      setWorkflowName("");
      setWorkflowDesc("");
      refreshAllData();
    } catch (err) {
      notify(`Error: ${err.message}`, "error");
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-6">
      {/* Toast Notification */}
      {notification && (
        <div
          className={`fixed bottom-6 right-6 z-50 px-4 py-3 rounded-xl border text-sm font-medium shadow-2xl transition-all ${
            notification.type === "error"
              ? "bg-red-500/10 border-red-500/30 text-red-300"
              : notification.type === "info"
              ? "bg-blue-500/10 border-blue-500/30 text-blue-300"
              : "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
          }`}
        >
          {notification.msg}
        </div>
      )}

      {/* Universal Page Header */}
      <PageHeader
        category="Experiments"
        title="Experiment Workbench"
        subtitle="Configure, run, track, compare, and chain isolated machine learning module experiments across the full modeling lifecycle."
        badge={
          <Badge variant="violet" size="sm">
            <Layers className="w-3.5 h-3.5 mr-1 inline" /> Module-Wise Execution
          </Badge>
        }
        actions={
          <>
            <Button
              variant="secondary"
              size="sm"
              icon={RefreshCw}
              loading={loading}
              onClick={refreshAllData}
            >
              Refresh
            </Button>
            <Button
              variant="primary"
              size="sm"
              icon={Plus}
              onClick={() => setActiveTab("new")}
            >
              New Experiment
            </Button>
          </>
        }
      />

      {/* Top Nav Tabs */}
      <div className="flex overflow-x-auto border-b border-white/[0.08] gap-2 pb-2">
        {[
          { id: "overview", label: "Overview", icon: Activity },
          { id: "new", label: "New Experiment", icon: Plus },
          { id: "runs", label: `All Runs (${runs.length})`, icon: Clock },
          { id: "compare", label: `Comparisons ${selectedRunIdsForCompare.length > 0 ? `(${selectedRunIdsForCompare.length})` : ""}`, icon: ArrowLeftRight },
          { id: "lineage", label: "Lineage Graph", icon: GitBranch },
          { id: "workflows", label: `Saved Workflows (${workflows.length})`, icon: Workflow },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                isActive
                  ? "bg-[#1f1e40] text-purple-300 border border-purple-500/30 shadow-sm"
                  : "text-[#A4A4B8] hover:text-white hover:bg-white/[0.04]"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* ------------------------------------------------------------- */}
      {/* 1. OVERVIEW VIEW */}
      {/* ------------------------------------------------------------- */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* Quick Stats Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-[#111126] border border-white/[0.08] p-5 rounded-2xl">
              <div className="flex items-center justify-between text-[#A4A4B8] text-xs font-medium mb-2">
                <span>Total Experiments</span>
                <FolderGit2 className="w-4 h-4 text-purple-400" />
              </div>
              <div className="text-2xl font-bold text-white">{experiments.length}</div>
              <p className="text-[11px] text-[#A4A4B8] mt-1">Logical experiment groups</p>
            </div>

            <div className="bg-[#111126] border border-white/[0.08] p-5 rounded-2xl">
              <div className="flex items-center justify-between text-[#A4A4B8] text-xs font-medium mb-2">
                <span>Total Runs Executed</span>
                <Clock className="w-4 h-4 text-blue-400" />
              </div>
              <div className="text-2xl font-bold text-white">{runs.length}</div>
              <p className="text-[11px] text-[#A4A4B8] mt-1">Across all independent modules</p>
            </div>

            <div className="bg-[#111126] border border-white/[0.08] p-5 rounded-2xl">
              <div className="flex items-center justify-between text-[#A4A4B8] text-xs font-medium mb-2">
                <span>Completed Runs</span>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-2xl font-bold text-emerald-400">
                {runs.filter((r) => r.status === "completed").length}
              </div>
              <p className="text-[11px] text-[#A4A4B8] mt-1">Ready for comparison & deployment</p>
            </div>

            <div className="bg-[#111126] border border-white/[0.08] p-5 rounded-2xl">
              <div className="flex items-center justify-between text-[#A4A4B8] text-xs font-medium mb-2">
                <span>Datasets Registered</span>
                <Database className="w-4 h-4 text-amber-400" />
              </div>
              <div className="text-2xl font-bold text-white">{datasets.length}</div>
              <p className="text-[11px] text-[#A4A4B8] mt-1">Immutable SHA-256 integrity</p>
            </div>
          </div>

          {/* Module Breakdown Grid */}
          <div className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
              <Layers className="w-4 h-4 text-purple-400" /> Independent Execution Modules
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {MODULES.map((m) => {
                const Icon = m.icon;
                const count = runs.filter((r) => r.module === m.id).length;
                return (
                  <div
                    key={m.id}
                    onClick={() => {
                      setModuleFilter(m.id);
                      setActiveTab("runs");
                    }}
                    className="p-3.5 rounded-xl border border-white/[0.06] bg-[#171630]/60 hover:bg-[#1f1e40] cursor-pointer transition-all flex flex-col justify-between"
                  >
                    <div className="flex items-center justify-between">
                      <span className={`p-2 rounded-lg border ${m.color}`}>
                        <Icon className="w-4 h-4" />
                      </span>
                      <span className="text-xs font-bold text-white">{count}</span>
                    </div>
                    <div className="mt-3">
                      <div className="text-xs font-medium text-white">{m.label}</div>
                      <div className="text-[10px] text-[#A4A4B8]">Click to filter runs</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Recent Runs Feed */}
          <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Clock className="w-4 h-4 text-blue-400" /> Recent Activity Feed
              </h3>
              <button
                onClick={() => setActiveTab("runs")}
                className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1 font-medium"
              >
                View all runs <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            {runs.length === 0 ? (
              <div className="text-center py-12 text-[#A4A4B8]">
                <Clock className="w-8 h-8 mx-auto mb-2 opacity-40" />
                <p className="text-sm">No experiment runs executed yet.</p>
                <button
                  onClick={() => setActiveTab("new")}
                  className="mt-3 text-xs text-purple-400 hover:underline"
                >
                  Create your first experiment
                </button>
              </div>
            ) : (
              <div className="divide-y divide-white/[0.06]">
                {runs.slice(0, 5).map((run) => (
                  <div key={run.run_id} className="py-3.5 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <span
                        className={`w-2.5 h-2.5 rounded-full ${
                          run.status === "completed"
                            ? "bg-emerald-400"
                            : run.status === "running"
                            ? "bg-amber-400 animate-ping"
                            : run.status === "failed"
                            ? "bg-red-400"
                            : "bg-gray-400"
                        }`}
                      />
                      <div>
                        <div className="text-xs font-semibold text-white">{run.run_name}</div>
                        <div className="text-[11px] text-[#A4A4B8] flex items-center gap-2 mt-0.5">
                          <span className="uppercase text-[9px] px-1.5 py-0.2 rounded bg-white/[0.05] border border-white/[0.08]">
                            {run.module}
                          </span>
                          <span>•</span>
                          <span>{run.duration_seconds}s</span>
                          <span>•</span>
                          <span>{new Date(run.created_at).toLocaleTimeString()}</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {run.status === "draft" && (
                        <button
                          onClick={() => handleExecuteRun(run.run_id)}
                          className="px-2.5 py-1 rounded-lg text-xs bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5"
                        >
                          <Play className="w-3 h-3" /> Run
                        </button>
                      )}
                      <button
                        onClick={() => {
                          setSelectedRun(run);
                        }}
                        className="px-2.5 py-1 rounded-lg text-xs bg-white/[0.05] hover:bg-white/[0.08] text-white border border-white/[0.08] flex items-center gap-1.5"
                      >
                        <Eye className="w-3 h-3" /> Inspect
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 2. NEW EXPERIMENT VIEW */}
      {/* ------------------------------------------------------------- */}
      {activeTab === "new" && (
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-6">
          <div>
            <h2 className="text-lg font-bold text-white">Create Independent Experiment</h2>
            <p className="text-xs text-[#A4A4B8] mt-0.5">
              Configure and isolate a specific module run without executing the entire AutoML pipeline.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left: General Config */}
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-[#A4A4B8] mb-1.5">Experiment Name</label>
                <input
                  type="text"
                  value={newExpName}
                  onChange={(e) => setNewExpName(e.target.value)}
                  placeholder="e.g. Tree Depth Exploration"
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-[#A4A4B8] mb-1.5">Target Module</label>
                <div className="grid grid-cols-2 gap-2">
                  {MODULES.map((m) => {
                    const Icon = m.icon;
                    const isSelected = newExpModule === m.id;
                    return (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => setNewExpModule(m.id)}
                        className={`p-2.5 rounded-xl border text-left flex items-center gap-2.5 transition-all ${
                          isSelected
                            ? "bg-purple-600/20 border-purple-500 text-white"
                            : "bg-[#171630] border-white/[0.06] text-[#A4A4B8] hover:text-white"
                        }`}
                      >
                        <Icon className="w-4 h-4 text-purple-400" />
                        <span className="text-xs font-medium">{m.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-[#A4A4B8] mb-1.5">Select Dataset</label>
                {datasets.length === 0 ? (
                  <div className="text-xs text-amber-400 bg-amber-500/10 p-3 rounded-xl border border-amber-500/20">
                    No datasets currently registered. Upload or train a dataset to register one.
                  </div>
                ) : (
                  <select
                    value={newExpDatasetId}
                    onChange={(e) => setNewExpDatasetId(e.target.value)}
                    className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-purple-500"
                  >
                    {datasets.map((ds) => (
                      <option key={ds.dataset_id} value={ds.dataset_id}>
                        {ds.filename} ({ds.row_count} rows × {ds.column_count} cols)
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <div>
                <label className="block text-xs font-medium text-[#A4A4B8] mb-1.5">Description (Optional)</label>
                <textarea
                  value={newExpDesc}
                  onChange={(e) => setNewExpDesc(e.target.value)}
                  rows={3}
                  placeholder="Notes on the hypothesis being tested..."
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl p-3 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>
            </div>

            {/* Right: Module-Specific Parameters */}
            <div className="bg-[#171630] border border-white/[0.06] rounded-xl p-5 space-y-4">
              <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-purple-400" />
                {newExpModule.toUpperCase()} Execution Parameters
              </h4>

              {/* Target column (common for supervised tasks) */}
              {["training", "tuning", "features", "eda"].includes(newExpModule) && (
                <div>
                  <label className="block text-xs text-[#A4A4B8] mb-1">Target Column (Optional)</label>
                  <input
                    type="text"
                    value={newExpParams.target_column}
                    onChange={(e) => setNewExpParams({ ...newExpParams, target_column: e.target.value })}
                    placeholder="Defaults to last column"
                    className="w-full bg-[#111126] border border-white/[0.08] rounded-lg px-3 py-1.5 text-xs text-white"
                  />
                </div>
              )}

              {/* Training / Tuning params */}
              {["training", "tuning"].includes(newExpModule) && (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1">Task Type</label>
                      <select
                        value={newExpParams.task_type}
                        onChange={(e) => setNewExpParams({ ...newExpParams, task_type: e.target.value })}
                        className="w-full bg-[#111126] border border-white/[0.08] rounded-lg px-2.5 py-1.5 text-xs text-white"
                      >
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1">Model Architecture</label>
                      <select
                        value={newExpParams.model_type}
                        onChange={(e) => setNewExpParams({ ...newExpParams, model_type: e.target.value })}
                        className="w-full bg-[#111126] border border-white/[0.08] rounded-lg px-2.5 py-1.5 text-xs text-white"
                      >
                        <option value="logistic">Logistic Regression / Linear</option>
                        <option value="random_forest">Random Forest</option>
                        <option value="decision_tree">Decision Tree</option>
                        <option value="extra_trees">Extra Trees</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1">Scaler</label>
                      <select
                        value={newExpParams.scaler}
                        onChange={(e) => setNewExpParams({ ...newExpParams, scaler: e.target.value })}
                        className="w-full bg-[#111126] border border-white/[0.08] rounded-lg px-2.5 py-1.5 text-xs text-white"
                      >
                        <option value="standard">StandardScaler</option>
                        <option value="minmax">MinMaxScaler</option>
                        <option value="robust">RobustScaler</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs text-[#A4A4B8] mb-1">Split Ratio</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0.5"
                        max="0.9"
                        value={newExpParams.split_ratio}
                        onChange={(e) => setNewExpParams({ ...newExpParams, split_ratio: parseFloat(e.target.value) })}
                        className="w-full bg-[#111126] border border-white/[0.08] rounded-lg px-2.5 py-1.5 text-xs text-white"
                      />
                    </div>
                  </div>
                </>
              )}

              {/* Tuning specific */}
              {newExpModule === "tuning" && (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-[#A4A4B8] mb-1">Optuna Trials</label>
                    <input
                      type="number"
                      min="2"
                      max="50"
                      value={newExpParams.n_trials}
                      onChange={(e) => setNewExpParams({ ...newExpParams, n_trials: parseInt(e.target.value) })}
                      className="w-full bg-[#111126] border border-white/[0.08] rounded-lg px-2.5 py-1.5 text-xs text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-[#A4A4B8] mb-1">CV Folds (n_splits)</label>
                    <input
                      type="number"
                      min="2"
                      max="10"
                      value={newExpParams.n_splits}
                      onChange={(e) => setNewExpParams({ ...newExpParams, n_splits: parseInt(e.target.value) })}
                      className="w-full bg-[#111126] border border-white/[0.08] rounded-lg px-2.5 py-1.5 text-xs text-white"
                    />
                  </div>
                </div>
              )}

              {/* Cleaning specific */}
              {newExpModule === "cleaning" && (
                <div className="space-y-3">
                  <label className="flex items-center gap-2 text-xs text-white cursor-pointer">
                    <input
                      type="checkbox"
                      checked={newExpParams.impute_missing}
                      onChange={(e) => setNewExpParams({ ...newExpParams, impute_missing: e.target.checked })}
                      className="rounded bg-[#111126] border-white/20 text-purple-600"
                    />
                    <span>Impute Missing Values</span>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-white cursor-pointer">
                    <input
                      type="checkbox"
                      checked={newExpParams.remove_duplicates}
                      onChange={(e) => setNewExpParams({ ...newExpParams, remove_duplicates: e.target.checked })}
                      className="rounded bg-[#111126] border-white/20 text-purple-600"
                    />
                    <span>Remove Duplicate Rows</span>
                  </label>
                </div>
              )}

              {/* Features specific */}
              {newExpModule === "features" && (
                <div className="space-y-3">
                  <label className="flex items-center gap-2 text-xs text-white cursor-pointer">
                    <input
                      type="checkbox"
                      checked={newExpParams.enable_polynomial}
                      onChange={(e) => setNewExpParams({ ...newExpParams, enable_polynomial: e.target.checked })}
                      className="rounded bg-[#111126] border-white/20 text-purple-600"
                    />
                    <span>Generate Polynomial Features</span>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-white cursor-pointer">
                    <input
                      type="checkbox"
                      checked={newExpParams.enable_interactions}
                      onChange={(e) => setNewExpParams({ ...newExpParams, enable_interactions: e.target.checked })}
                      className="rounded bg-[#111126] border-white/20 text-purple-600"
                    />
                    <span>Pairwise Feature Interactions</span>
                  </label>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-white/[0.08]">
            <button
              onClick={() => handleCreateNewExperiment(false)}
              disabled={loading}
              className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08]"
            >
              Save as Draft Run
            </button>
            <button
              onClick={() => handleCreateNewExperiment(true)}
              disabled={loading}
              className="px-5 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-lg shadow-purple-900/30 flex items-center gap-2"
            >
              <Play className="w-3.5 h-3.5" />
              Create & Execute Now
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 3. ALL RUNS VIEW */}
      {/* ------------------------------------------------------------- */}
      {activeTab === "runs" && (
        <div className="space-y-4">
          {/* Controls & Filter Bar */}
          <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-[#A4A4B8] flex items-center gap-1 font-medium mr-1">
                <Filter className="w-3.5 h-3.5" /> Module:
              </span>
              <button
                onClick={() => setModuleFilter("all")}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                  moduleFilter === "all" ? "bg-purple-600 text-white" : "bg-[#171630] text-[#A4A4B8] hover:text-white"
                }`}
              >
                All
              </button>
              {MODULES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setModuleFilter(m.id)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                    moduleFilter === m.id
                      ? "bg-purple-600 text-white"
                      : "bg-[#171630] text-[#A4A4B8] hover:text-white"
                  }`}
                >
                  {m.label.split(" ")[0]}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-3">
              <div className="relative w-full sm:w-64">
                <Search className="w-3.5 h-3.5 text-[#A4A4B8] absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search runs..."
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl pl-8 pr-3 py-1.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              {selectedRunIdsForCompare.length >= 2 && (
                <button
                  onClick={handleTriggerCompare}
                  disabled={comparingLoading}
                  className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-md flex items-center gap-1.5 whitespace-nowrap"
                >
                  <ArrowLeftRight className="w-3.5 h-3.5" />
                  Compare ({selectedRunIdsForCompare.length})
                </button>
              )}
            </div>
          </div>

          {/* Runs Table */}
          <div className="bg-[#111126] border border-white/[0.08] rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-white/[0.08] bg-[#171630]/70 text-[#A4A4B8] uppercase text-[10px] tracking-wider">
                    <th className="py-3.5 px-4 w-10">Select</th>
                    <th className="py-3.5 px-4">Run Name & ID</th>
                    <th className="py-3.5 px-4">Module</th>
                    <th className="py-3.5 px-4">Status</th>
                    <th className="py-3.5 px-4">Metrics Summary</th>
                    <th className="py-3.5 px-4">Latency</th>
                    <th className="py-3.5 px-4">Created At</th>
                    <th className="py-3.5 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.06] text-white">
                  {filteredRuns.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-[#A4A4B8]">
                        No runs match the selected filters.
                      </td>
                    </tr>
                  ) : (
                    filteredRuns.map((r) => {
                      const isSelected = selectedRunIdsForCompare.includes(r.run_id);
                      return (
                        <tr key={r.run_id} className="hover:bg-white/[0.02] transition-colors">
                          <td className="py-3.5 px-4">
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => toggleSelectRunForCompare(r.run_id)}
                              className="rounded bg-[#171630] border-white/20 text-purple-600 focus:ring-0"
                            />
                          </td>
                          <td className="py-3.5 px-4">
                            <div className="font-semibold text-white">{r.run_name}</div>
                            <div className="text-[10px] text-[#A4A4B8] font-mono">{r.run_id}</div>
                          </td>
                          <td className="py-3.5 px-4">
                            <span className="uppercase text-[9px] font-bold px-2 py-0.5 rounded bg-white/[0.06] border border-white/[0.08] text-purple-300">
                              {r.module}
                            </span>
                          </td>
                          <td className="py-3.5 px-4">
                            <span
                              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-semibold border ${
                                r.status === "completed"
                                  ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/20"
                                  : r.status === "running"
                                  ? "bg-amber-500/10 text-amber-300 border-amber-500/20 animate-pulse"
                                  : r.status === "failed"
                                  ? "bg-red-500/10 text-red-300 border-red-500/20"
                                  : "bg-gray-500/10 text-gray-300 border-gray-500/20"
                              }`}
                            >
                              {r.status}
                            </span>
                          </td>
                          <td className="py-3.5 px-4">
                            {r.status === "completed" && r.metrics ? (
                              <div className="flex flex-wrap gap-1 text-[10px]">
                                {Object.entries(r.metrics)
                                  .filter(([k, v]) => typeof v === "number")
                                  .slice(0, 3)
                                  .map(([k, v]) => (
                                    <span key={k} className="px-1.5 py-0.5 rounded bg-white/[0.04] border border-white/[0.06] text-[#A4A4B8]">
                                      {k}: <span className="text-white font-semibold">{formatMetric(v)}</span>
                                    </span>
                                  ))}
                              </div>
                            ) : (
                              <span className="text-[11px] text-[#A4A4B8] italic">No metrics recorded</span>
                            )}
                          </td>
                          <td className="py-3.5 px-4 text-[#A4A4B8]">{r.duration_seconds}s</td>
                          <td className="py-3.5 px-4 text-[#A4A4B8] text-[11px]">
                            {new Date(r.created_at).toLocaleString()}
                          </td>
                          <td className="py-3.5 px-4 text-right">
                            <div className="flex items-center justify-end gap-1.5">
                              {r.status === "draft" && (
                                <button
                                  onClick={() => handleExecuteRun(r.run_id)}
                                  title="Execute Run"
                                  className="p-1.5 rounded-lg bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 border border-emerald-500/30"
                                >
                                  <Play className="w-3.5 h-3.5" />
                                </button>
                              )}
                              <button
                                onClick={() => handleViewLineage(r.run_id)}
                                title="View Lineage DAG"
                                className="p-1.5 rounded-lg bg-[#171630] text-purple-300 hover:bg-[#1f1e40] border border-white/[0.08]"
                              >
                                <GitBranch className="w-3.5 h-3.5" />
                              </button>
                              <button
                                onClick={() => setSelectedRun(r)}
                                title="Inspect Details"
                                className="p-1.5 rounded-lg bg-[#171630] text-white hover:bg-[#1f1e40] border border-white/[0.08]"
                              >
                                <Eye className="w-3.5 h-3.5" />
                              </button>
                              <button
                                onClick={() => handleDeleteRun(r.run_id)}
                                title="Delete Run"
                                className="p-1.5 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Run Details Modal / Slide-out */}
          {selectedRun && (
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 space-y-5">
                <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
                  <div>
                    <h3 className="text-base font-bold text-white">{selectedRun.run_name}</h3>
                    <p className="text-xs text-[#A4A4B8] font-mono mt-0.5">{selectedRun.run_id}</p>
                  </div>
                  <button
                    onClick={() => setSelectedRun(null)}
                    className="p-1.5 rounded-lg text-[#A4A4B8] hover:text-white hover:bg-white/[0.06]"
                  >
                    <XCircle className="w-5 h-5" />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <span className="text-[#A4A4B8]">Module:</span>{" "}
                    <span className="font-semibold uppercase text-purple-300">{selectedRun.module}</span>
                  </div>
                  <div>
                    <span className="text-[#A4A4B8]">Status:</span>{" "}
                    <span className="font-semibold text-emerald-300">{selectedRun.status}</span>
                  </div>
                  <div>
                    <span className="text-[#A4A4B8]">Execution Duration:</span>{" "}
                    <span className="font-semibold text-white">{selectedRun.duration_seconds}s</span>
                  </div>
                  <div>
                    <span className="text-[#A4A4B8]">Dataset ID:</span>{" "}
                    <span className="font-mono text-white">{selectedRun.dataset_id}</span>
                  </div>
                </div>

                {/* Parameters */}
                <div>
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Input Parameters</h4>
                  <pre className="bg-[#171630] border border-white/[0.06] p-3 rounded-xl text-[11px] text-[#A4A4B8] overflow-x-auto font-mono">
                    {JSON.stringify(selectedRun.parameters, null, 2)}
                  </pre>
                </div>

                {/* Metrics */}
                <div>
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Recorded Metrics</h4>
                  <pre className="bg-[#171630] border border-white/[0.06] p-3 rounded-xl text-[11px] text-emerald-300 overflow-x-auto font-mono">
                    {JSON.stringify(selectedRun.metrics, null, 2)}
                  </pre>
                </div>

                {/* Artifacts */}
                {selectedRun.artifacts && selectedRun.artifacts.length > 0 && (
                  <div>
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Produced Artifacts</h4>
                    <div className="space-y-1.5">
                      {selectedRun.artifacts.map((art, i) => (
                        <div
                          key={i}
                          className="p-2.5 rounded-xl bg-[#171630] border border-white/[0.06] text-xs flex items-center justify-between text-white"
                        >
                          <div className="flex items-center gap-2">
                            <FileText className="w-3.5 h-3.5 text-purple-400" />
                            <span>{art.name || art.type}</span>
                          </div>
                          <span className="text-[10px] text-[#A4A4B8] font-mono">{art.type}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 4. COMPARISONS VIEW */}
      {/* ------------------------------------------------------------- */}
      {activeTab === "compare" && (
        <div className="space-y-6">
          {!comparisonResult ? (
            <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-12 text-center text-[#A4A4B8]">
              <ArrowLeftRight className="w-10 h-10 mx-auto mb-3 opacity-40 text-purple-400" />
              <h3 className="text-sm font-bold text-white">No Active Comparison</h3>
              <p className="text-xs text-[#A4A4B8] mt-1">
                Go to the "All Runs" tab and select 2 or more runs from the same module to view side-by-side metrics.
              </p>
              <button
                onClick={() => setActiveTab("runs")}
                className="mt-4 px-4 py-2 rounded-xl text-xs font-semibold bg-purple-600 hover:bg-purple-500 text-white transition-all"
              >
                Go to All Runs
              </button>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Header card */}
              <div className="bg-[#111126] border border-white/[0.08] p-5 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <div className="text-xs text-purple-400 font-semibold uppercase tracking-wider">
                    {comparisonResult.module} Module Comparison
                  </div>
                  <h3 className="text-xl font-bold text-white mt-0.5">
                    Comparing {comparisonResult.runs_count} Runs Side-by-Side
                  </h3>
                </div>
                <button
                  onClick={() => setActiveTab("runs")}
                  className="px-3 py-1.5 rounded-xl text-xs bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08]"
                >
                  Change Selection
                </button>
              </div>

              {/* Parameter Diff Section */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6">
                <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-purple-400" /> Hyperparameter Differencing
                </h4>

                {Object.keys(comparisonResult.varying_parameters || {}).length === 0 ? (
                  <div className="text-xs text-[#A4A4B8] italic">All evaluated runs shared identical hyperparameters.</div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-white/[0.08] text-[#A4A4B8]">
                          <th className="py-2.5 px-3">Varying Parameter</th>
                          {comparisonResult.runs.map((r) => (
                            <th key={r.run_id} className="py-2.5 px-3 text-white">
                              {r.run_name}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.06] text-white">
                        {Object.entries(comparisonResult.varying_parameters).map(([paramKey, valuesMap]) => (
                          <tr key={paramKey} className="hover:bg-white/[0.02]">
                            <td className="py-2 px-3 font-mono text-purple-300 font-medium">{paramKey}</td>
                            {comparisonResult.runs.map((r) => (
                              <td key={r.run_id} className="py-2 px-3 font-mono text-[#A4A4B8]">
                                {String(valuesMap[r.run_id] ?? "—")}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Metrics Matrix Section */}
              <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6">
                <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-emerald-400" /> Metric Performance Matrix
                </h4>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-white/[0.08] text-[#A4A4B8]">
                        <th className="py-2.5 px-3">Metric Name</th>
                        <th className="py-2.5 px-3">Optimal Direction</th>
                        {comparisonResult.runs.map((r) => (
                          <th key={r.run_id} className="py-2.5 px-3 text-white">
                            {r.run_name}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.06] text-white">
                      {comparisonResult.comparison_metrics.map((cm) => (
                        <tr key={cm.metric} className="hover:bg-white/[0.02]">
                          <td className="py-2.5 px-3 font-semibold text-white">{cm.metric}</td>
                          <td className="py-2.5 px-3 text-[#A4A4B8] capitalize text-[11px]">{cm.direction} is better</td>
                          {comparisonResult.runs.map((r) => {
                            const val = cm.values[r.run_id];
                            const delta = cm.deltas[r.run_id];
                            const isWinner = cm.best_run_id === r.run_id;
                            return (
                              <td key={r.run_id} className="py-2.5 px-3">
                                <div className="flex items-center gap-2">
                                  <span
                                    className={`font-mono font-bold ${
                                      isWinner ? "text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20" : "text-white"
                                    }`}
                                  >
                                    {val !== undefined ? formatMetric(val) : "—"}
                                  </span>
                                  {delta !== 0 && delta !== undefined && (
                                    <span
                                      className={`text-[10px] font-semibold ${
                                        delta > 0 ? "text-emerald-400" : "text-red-400"
                                      }`}
                                    >
                                      {delta > 0 ? `+${delta}%` : `${delta}%`}
                                    </span>
                                  )}
                                </div>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 5. WORKFLOW LINEAGE VIEW */}
      {/* ------------------------------------------------------------- */}
      {activeTab === "lineage" && (
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <GitBranch className="w-4 h-4 text-purple-400" />
                Lineage & Provenance DAG
              </h3>
              <p className="text-xs text-[#A4A4B8] mt-0.5">
                Full end-to-end trace from raw dataset through transformations, candidate training, and reports.
              </p>
            </div>
          </div>

          {!lineageData ? (
            <div className="text-center py-12 text-[#A4A4B8]">
              <GitBranch className="w-8 h-8 mx-auto mb-2 opacity-40" />
              <p className="text-xs">No run selected for lineage analysis.</p>
              <p className="text-[11px] text-gray-500 mt-1">Select a run in "All Runs" and click the lineage button.</p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="text-xs text-purple-300 bg-purple-500/10 p-3 rounded-xl border border-purple-500/20">
                Lineage Focus Run ID: <span className="font-mono font-bold text-white">{lineageData.focus_run_id}</span>
              </div>

              {/* Node cards layout */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {lineageData.nodes.map((node) => (
                  <div
                    key={node.id}
                    className={`p-4 rounded-xl border transition-all ${
                      node.id === lineageData.focus_run_id
                        ? "bg-purple-950/30 border-purple-500 ring-1 ring-purple-500/50"
                        : "bg-[#171630] border-white/[0.06]"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="uppercase text-[9px] font-bold px-2 py-0.5 rounded bg-white/[0.06] border border-white/[0.08] text-[#A4A4B8]">
                        {node.type}
                      </span>
                      {node.id === lineageData.focus_run_id && (
                        <span className="text-[10px] text-purple-400 font-semibold">Active Focus</span>
                      )}
                    </div>
                    <div className="text-xs font-bold text-white">{node.label}</div>
                    <div className="text-[10px] text-[#A4A4B8] font-mono mt-0.5 truncate">{node.id}</div>

                    {node.details && (
                      <div className="mt-3 pt-3 border-t border-white/[0.06] text-[10px] text-[#A4A4B8] space-y-1">
                        {Object.entries(node.details).slice(0, 3).map(([k, v]) => (
                          <div key={k} className="flex justify-between">
                            <span className="capitalize">{k}:</span>
                            <span className="text-white font-medium">{String(v)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Directed Edges List */}
              <div className="bg-[#171630] border border-white/[0.06] p-4 rounded-xl">
                <h5 className="text-[11px] font-bold text-white uppercase tracking-wider mb-2">Directed Dependencies</h5>
                <div className="space-y-1 text-xs">
                  {lineageData.edges.length === 0 ? (
                    <span className="text-[#A4A4B8] text-[11px] italic">Root node with no ancestors recorded.</span>
                  ) : (
                    lineageData.edges.map((e, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-[#A4A4B8] font-mono text-[11px]">
                        <span className="text-white">{e.from}</span>
                        <ArrowRight className="w-3 h-3 text-purple-400" />
                        <span className="text-purple-300 font-semibold">{e.to}</span>
                        <span className="text-[9px] px-1.5 py-0.2 rounded bg-white/[0.05] border border-white/[0.08] text-gray-400">
                          {e.label}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 6. SAVED WORKFLOWS VIEW */}
      {/* ------------------------------------------------------------- */}
      {activeTab === "workflows" && (
        <div className="space-y-6">
          {/* Create Workflow Definition Form */}
          <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-4">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Workflow className="w-4 h-4 text-purple-400" /> Define Reusable Workflow Pipeline
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-[#A4A4B8] mb-1">Workflow Name</label>
                <input
                  type="text"
                  value={workflowName}
                  onChange={(e) => setWorkflowName(e.target.value)}
                  placeholder="e.g. Robust Production Pipeline"
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white"
                />
              </div>

              <div>
                <label className="block text-xs text-[#A4A4B8] mb-1">Description</label>
                <input
                  type="text"
                  value={workflowDesc}
                  onChange={(e) => setWorkflowDesc(e.target.value)}
                  placeholder="e.g. Cleans data, expands features, and trains candidate model"
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white"
                />
              </div>
            </div>

            {/* Configured Stages Preview */}
            <div className="space-y-2">
              <div className="text-xs font-semibold text-white">Pipeline Execution Stages:</div>
              <div className="flex flex-wrap items-center gap-2">
                {workflowStages.map((st, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className="px-3 py-1.5 rounded-xl bg-[#171630] border border-white/[0.08] text-xs text-white flex items-center gap-2">
                      <span className="w-4 h-4 rounded-full bg-purple-500/20 text-purple-300 text-[10px] flex items-center justify-center font-bold">
                        {i + 1}
                      </span>
                      <span>{st.name}</span>
                      <span className="uppercase text-[9px] text-[#A4A4B8] bg-white/[0.04] px-1 rounded">
                        {st.module}
                      </span>
                    </div>
                    {i < workflowStages.length - 1 && <ArrowRight className="w-3.5 h-3.5 text-gray-500" />}
                  </div>
                ))}
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={handleSaveWorkflow}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-purple-600 hover:bg-purple-500 text-white transition-all"
              >
                Save Workflow Definition
              </button>
            </div>
          </div>

          {/* List of Defined Workflows */}
          <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6">
            <h4 className="text-sm font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
              <FolderGit2 className="w-4 h-4 text-blue-400" /> Defined Pipelines
            </h4>

            {workflows.length === 0 ? (
              <div className="text-center py-8 text-[#A4A4B8] text-xs">
                No workflows defined yet. Create your first workflow above.
              </div>
            ) : (
              <div className="space-y-4">
                {workflows.map((wf) => (
                  <div
                    key={wf.workflow_id}
                    className="p-5 rounded-xl bg-[#171630]/60 border border-white/[0.06] space-y-3"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-bold text-white">{wf.name}</div>
                        <div className="text-xs text-[#A4A4B8] mt-0.5">{wf.description || "No description provided."}</div>
                      </div>

                      <div className="flex items-center gap-2">
                        <select
                          value={wfExecDatasetId}
                          onChange={(e) => setWfExecDatasetId(e.target.value)}
                          className="bg-[#111126] border border-white/[0.08] rounded-xl px-2.5 py-1.5 text-xs text-white"
                        >
                          {datasets.map((ds) => (
                            <option key={ds.dataset_id} value={ds.dataset_id}>
                              {ds.filename}
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={() => handleExecuteWorkflow(wf.workflow_id)}
                          className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 border border-emerald-500/30 flex items-center gap-1.5 whitespace-nowrap"
                        >
                          <Play className="w-3.5 h-3.5" /> Execute Pipeline
                        </button>
                      </div>
                    </div>

                    {/* Stages badges */}
                    <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-white/[0.04]">
                      {wf.stages.map((st, idx) => (
                        <span
                          key={idx}
                          className="text-[10px] px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06] text-[#A4A4B8]"
                        >
                          {idx + 1}. {st.name} ({st.module})
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
