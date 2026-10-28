import { useState, useEffect } from "react";
import { 
  Clock, 
  Trash2, 
  Search, 
  RefreshCw, 
  Eye, 
  CheckCircle2, 
  Layers, 
  Database,
  ArrowRight,
  Sliders,
  ChevronDown,
  ChevronUp
} from "lucide-react";
import { getApiBase, formatMetric } from "../utils/api";

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [expandedId, setExpandedId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const fetchExperiments = async () => {
    setLoading(true);
    setError(null);
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/experiments?limit=50`);
      if (!res.ok) throw new Error("Could not retrieve experiment records");
      const data = await res.json();
      setExperiments(data.experiments || []);
    } catch (err) {
      setError("Failed to query experiment database from AutoVeyra backend.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExperiments();
  }, []);

  const handleDelete = async (expId) => {
    if (!window.confirm(`Delete experiment #${expId}?`)) return;
    setDeletingId(expId);
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/experiments/${expId}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete experiment");
      setExperiments((prev) => prev.filter((e) => e.id !== expId));
      if (expandedId === expId) setExpandedId(null);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setDeletingId(null);
    }
  };

  const filteredExperiments = experiments.filter((e) => {
    const text = `${e.experiment_name} ${e.model_type} ${e.dataset_name} ${e.task_type}`.toLowerCase();
    return text.includes(searchTerm.toLowerCase());
  });

  return (
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-6">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.08] pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[#A855F7] bg-[#A855F7]/10 px-2 py-0.5 rounded-full border border-[#A855F7]/20">
              Audit & Reproducibility Logs
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Experiment History</h1>
          <p className="text-sm text-[#A4A4B8] mt-1">
            Complete audit trail of hyperparameter selections, preprocessing configurations, and evaluation metrics.
          </p>
        </div>

        <button
          onClick={fetchExperiments}
          disabled={loading}
          className="self-start md:self-auto px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] transition-all flex items-center gap-2"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          Reload History
        </button>
      </div>

      {/* Search Bar */}
      <div className="bg-[#111126] border border-white/[0.08] p-4 rounded-2xl flex items-center justify-between">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-[#A4A4B8] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search experiments by name, model or dataset..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-[#171630] border border-white/[0.08] rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-[#A4A4B8]/60 focus:outline-none focus:border-[#8265FF]"
          />
        </div>
        <span className="text-xs text-[#A4A4B8] font-mono hidden sm:inline-block">
          Total Logs: {experiments.length}
        </span>
      </div>

      {/* Error notification */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
          {error}
        </div>
      )}

      {/* List */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-20 bg-[#111126] border border-white/[0.08] rounded-2xl animate-pulse" />
          ))}
        </div>
      ) : filteredExperiments.length === 0 ? (
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-12 text-center text-[#A4A4B8] space-y-2">
          <Clock className="w-8 h-8 mx-auto text-[#A4A4B8]/40" />
          <p className="text-sm">No experiment records found.</p>
          <p className="text-xs text-[#A4A4B8]/80">Run model training or model comparison to automatically create records.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredExperiments.map((exp) => {
            const isExpanded = expandedId === exp.id;
            return (
              <div
                key={exp.id}
                className="bg-[#111126] border border-white/[0.08] hover:border-[#8265FF]/30 rounded-2xl overflow-hidden transition-all shadow-md shadow-black/20"
              >
                {/* Collapsed summary header */}
                <div 
                  className="p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 cursor-pointer hover:bg-white/[0.02] transition-colors"
                  onClick={() => setExpandedId(isExpanded ? null : exp.id)}
                >
                  <div className="flex items-start md:items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-[#8265FF]/10 text-[#8265FF] flex items-center justify-center font-mono font-bold text-xs shrink-0">
                      #{exp.id}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-bold text-white">{exp.experiment_name}</h3>
                        <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#171630] border border-white/[0.08] text-[#A855F7]">
                          {exp.task_type}
                        </span>
                      </div>
                      <p className="text-xs text-[#A4A4B8] mt-0.5">
                        {exp.dataset_name} • Algorithm: <b className="text-gray-200">{exp.model_type}</b> • Target: <span className="text-[#8265FF]">{exp.target_column || "—"}</span>
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center justify-between md:justify-end gap-6 text-xs">
                    <div className="text-left md:text-right">
                      <span className="text-xs font-mono font-bold text-emerald-400 block">
                        {exp.metrics?.accuracy !== undefined
                          ? `Acc: ${(exp.metrics.accuracy * 100).toFixed(1)}%`
                          : exp.metrics?.r2 !== undefined
                          ? `R²: ${formatMetric(exp.metrics.r2, 3)}`
                          : "Completed"}
                      </span>
                      <span className="text-[10px] text-[#A4A4B8]">
                        {new Date(exp.created_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(exp.id);
                        }}
                        disabled={deletingId === exp.id}
                        className="p-2 rounded-lg text-[#A4A4B8] hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                        title="Delete record"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                      <button className="p-2 text-white">
                        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Expanded Details Section */}
                {isExpanded && (
                  <div className="border-t border-white/[0.06] bg-[#0D0D1B] p-6 space-y-6">
                    
                    {/* Pipeline Parameters */}
                    <div>
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-white mb-3 flex items-center gap-2">
                        <Sliders className="w-3.5 h-3.5 text-[#8265FF]" />
                        Preprocessing Pipeline Parameters
                      </h4>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                        <div className="p-3 rounded-xl bg-[#111126] border border-white/[0.04]">
                          <span className="text-[#A4A4B8] block text-[10px]">Scaler</span>
                          <span className="font-semibold text-white">{exp.scaler_type || "None"}</span>
                        </div>
                        <div className="p-3 rounded-xl bg-[#111126] border border-white/[0.04]">
                          <span className="text-[#A4A4B8] block text-[10px]">Missing Strategy</span>
                          <span className="font-semibold text-white">{exp.missing_strategy || "None"}</span>
                        </div>
                        <div className="p-3 rounded-xl bg-[#111126] border border-white/[0.04]">
                          <span className="text-[#A4A4B8] block text-[10px]">Categorical Encoding</span>
                          <span className="font-semibold text-white">{exp.encoding_strategy || "None"}</span>
                        </div>
                        <div className="p-3 rounded-xl bg-[#111126] border border-white/[0.04]">
                          <span className="text-[#A4A4B8] block text-[10px]">Train / Test Ratio</span>
                          <span className="font-semibold text-white">{exp.split_ratio ? `${(exp.split_ratio * 100).toFixed(0)}% / ${((1 - exp.split_ratio) * 100).toFixed(0)}%` : "N/A"}</span>
                        </div>
                      </div>
                    </div>

                    {/* All Evaluation Metrics */}
                    <div>
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-white mb-3 flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                        Holdout Test Metrics
                      </h4>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                        {Object.entries(exp.metrics || {}).map(([key, val]) => {
                          if (typeof val === "object" && val !== null) return null;
                          return (
                            <div key={key} className="p-3 rounded-xl bg-[#111126] border border-white/[0.04]">
                              <span className="text-[#A4A4B8] block text-[10px] uppercase font-sans">{key.replace(/_/g, " ")}</span>
                              <span className="font-bold text-white text-sm">{formatMetric(val)}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

    </div>
  );
}
