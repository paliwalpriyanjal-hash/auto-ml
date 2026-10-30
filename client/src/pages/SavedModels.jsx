// client/src/pages/SavedModels.jsx
import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { 
  Layers, 
  Trash2, 
  PlayCircle, 
  CheckCircle2, 
  Clock, 
  Database, 
  Search, 
  AlertTriangle,
  RefreshCw,
  Cpu,
  BarChart2,
  ExternalLink,
  ShieldCheck,
  Sliders,
  FileText
} from "lucide-react";
import { getApiBase, formatMetric } from "../utils/api";
import PageHeader from "../components/ui/PageHeader";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Button from "../components/ui/Button";
import EmptyState from "../components/ui/EmptyState";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";

export default function SavedModels() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedTask, setSelectedTask] = useState("all");
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  const [deleting, setDeleting] = useState(false);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const base = getApiBase();
      const taskParam = selectedTask !== "all" ? `&task_type=${selectedTask}` : "";
      const res = await fetch(`${base}/models?limit=50${taskParam}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setModels(data.models || []);
    } catch (err) {
      setError("Unable to load models from AutoVeyra Model Registry.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, [selectedTask]);

  const handleDelete = async (modelId) => {
    setDeleting(true);
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/models/${modelId}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete model");
      setModels((prev) => prev.filter((m) => m.id !== modelId));
      setDeleteConfirmId(null);
    } catch (err) {
      alert("Error deleting model: " + err.message);
    } finally {
      setDeleting(false);
    }
  };

  const filteredModels = models.filter((m) => {
    const matchSearch = 
      m.model_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (m.dataset_name && m.dataset_name.toLowerCase().includes(searchTerm.toLowerCase())) ||
      m.model_type.toLowerCase().includes(searchTerm.toLowerCase());
    return matchSearch;
  });

  return (
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-6">
      
      {/* Universal Page Header */}
      <PageHeader
        category="Model Artifacts"
        title="Saved & Registered Models"
        subtitle="Catalog of trained machine learning pipelines, evaluation scores, and real-time inference endpoints."
        actions={
          <Button
            variant="secondary"
            size="sm"
            onClick={fetchModels}
            isLoading={loading}
            icon={RefreshCw}
          >
            Refresh
          </Button>
        }
      />

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between bg-[#0F0F22] border border-white/[0.08] p-4 rounded-2xl">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-[#64677E] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by model, dataset or algorithm..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-[#070711] border border-white/[0.08] rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-[#64677E] focus:outline-none focus:border-[#8265FF] transition-all"
          />
        </div>

        <div className="flex items-center gap-1.5 w-full sm:w-auto p-1 bg-[#070711] rounded-xl border border-white/[0.06]">
          {["all", "classification", "regression"].map((task) => (
            <button
              key={task}
              onClick={() => setSelectedTask(task)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold capitalize transition-all ${
                selectedTask === task
                  ? "bg-[#16152F] text-white shadow-sm border border-[#8265FF]/30 text-[#8265FF]"
                  : "text-[#9FA2B6] hover:text-white"
              }`}
            >
              {task}
            </button>
          ))}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-rose-500/10 border border-rose-500/20 text-rose-300 p-4 rounded-xl text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Models Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <LoadingSkeleton key={i} card />
          ))}
        </div>
      ) : filteredModels.length === 0 ? (
        <EmptyState
          title="No Registered Models Found"
          description="Models trained via the Classification training engine or the Multi-Algorithm Comparison engine are automatically cataloged here."
          actionLabel="Train New Model"
          onAction={() => window.location.href = "/automl"}
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredModels.map((model) => {
            const isConfirmingDelete = deleteConfirmId === model.id;
            return (
              <Card
                key={model.id}
                variant="default"
                className="flex flex-col justify-between group hover:border-[#8265FF]/40"
              >
                <div>
                  <div className="flex items-start justify-between gap-2 mb-3">
                    <Badge
                      variant={model.task_type === "classification" ? "violet" : "success"}
                      size="sm"
                    >
                      {model.task_type}
                    </Badge>
                    <span className="text-[11px] text-[#9FA2B6] font-mono">
                      #{model.id}
                    </span>
                  </div>

                  <h3 className="text-base font-bold text-white group-hover:text-[#8265FF] transition-colors leading-snug">
                    {model.model_name}
                  </h3>
                  
                  <p className="text-xs text-[#9FA2B6] mt-1">
                    Algorithm: <b className="text-white font-medium">{model.model_type}</b>
                  </p>
                  
                  <div className="mt-4 p-3 rounded-xl bg-[#070711] border border-white/[0.04] space-y-1.5 text-xs">
                    <div className="flex justify-between text-[#9FA2B6]">
                      <span>Source Dataset:</span>
                      <span className="text-white truncate max-w-[140px]">{model.dataset_name || "Custom Upload"}</span>
                    </div>
                    <div className="flex justify-between text-[#9FA2B6]">
                      <span>Target Column:</span>
                      <span className="text-[#8265FF] font-semibold">{model.target_column || "—"}</span>
                    </div>
                    <div className="flex justify-between text-[#9FA2B6]">
                      <span>Feature Count:</span>
                      <span className="text-white font-mono">{model.feature_columns?.length || 0} columns</span>
                    </div>
                  </div>

                  {/* Key Metrics Display */}
                  <div className="grid grid-cols-2 gap-2 mt-3">
                    <div className="p-2.5 rounded-xl bg-[#16152F]/70 border border-white/[0.04]">
                      <span className="text-[10px] uppercase text-[#9FA2B6] font-medium block">Accuracy / R²</span>
                      <span className="text-sm font-bold font-mono text-emerald-400">
                        {model.metrics?.accuracy !== undefined 
                          ? `${(model.metrics.accuracy * 100).toFixed(1)}%`
                          : model.metrics?.r2 !== undefined 
                          ? formatMetric(model.metrics.r2, 3) 
                          : "N/A"}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-xl bg-[#16152F]/70 border border-white/[0.04]">
                      <span className="text-[10px] uppercase text-[#9FA2B6] font-medium block">F1 / RMSE</span>
                      <span className="text-sm font-bold font-mono text-white">
                        {model.metrics?.f1_macro !== undefined 
                          ? formatMetric(model.metrics.f1_macro, 3)
                          : model.metrics?.rmse !== undefined 
                          ? formatMetric(model.metrics.rmse, 3) 
                          : "N/A"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Actions Bar */}
                <div className="pt-5 mt-5 border-t border-white/[0.06] flex items-center justify-between">
                  {isConfirmingDelete ? (
                    <div className="flex items-center gap-2 w-full justify-between">
                      <span className="text-xs text-rose-400">Confirm delete?</span>
                      <div className="flex items-center gap-1.5">
                        <button
                          onClick={() => handleDelete(model.id)}
                          disabled={deleting}
                          className="px-2.5 py-1 bg-rose-500 hover:bg-rose-600 text-white rounded-lg text-xs font-semibold"
                        >
                          {deleting ? "..." : "Delete"}
                        </button>
                        <button
                          onClick={() => setDeleteConfirmId(null)}
                          className="px-2.5 py-1 bg-[#16152F] text-gray-300 rounded-lg text-xs"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <button
                        onClick={() => setDeleteConfirmId(model.id)}
                        className="p-2 rounded-xl text-[#9FA2B6] hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                        title="Delete from registry"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>

                      <div className="flex items-center gap-1.5">
                        <Link
                          to={`/reports?model=${model.id}`}
                          className="px-2.5 py-1.5 rounded-xl text-xs font-semibold bg-[#16152F] hover:bg-[#1D1C3C] text-white border border-white/[0.08] flex items-center gap-1 transition-all"
                          title="Generate model report"
                        >
                          <FileText className="w-3.5 h-3.5 text-emerald-400" />
                          <span>Report</span>
                        </Link>
                        <Link
                          to={`/tune?model_type=${model.model_type}&task=${model.task_type}`}
                          className="px-2.5 py-1.5 rounded-xl text-xs font-semibold bg-[#16152F] hover:bg-[#1D1C3C] text-white border border-white/[0.08] flex items-center gap-1 transition-all"
                          title="Optimize hyperparameters"
                        >
                          <Sliders className="w-3.5 h-3.5 text-[#A855F7]" />
                          <span>Tune</span>
                        </Link>
                        <Link
                          to={`/explain?model=${model.id}`}
                          className="px-2.5 py-1.5 rounded-xl text-xs font-semibold bg-[#16152F] hover:bg-[#1D1C3C] text-white border border-white/[0.08] flex items-center gap-1 transition-all"
                        >
                          <ShieldCheck className="w-3.5 h-3.5 text-[#8265FF]" />
                          <span>Explain</span>
                        </Link>
                        <Link
                          to={`/predict?model=${model.id}`}
                          className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] text-white shadow-sm shadow-[#8265FF]/20 flex items-center gap-1"
                        >
                          <PlayCircle className="w-3.5 h-3.5" />
                          <span>Predict</span>
                        </Link>
                      </div>
                    </>
                  )}
                </div>
              </Card>
            );
          })}
        </div>
      )}

    </div>
  );
}
