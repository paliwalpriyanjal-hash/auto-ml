import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  Zap,
  Clock,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Layers,
  Search,
  Database,
  Cpu,
  ShieldCheck,
  TrendingUp,
  Sliders,
  ChevronRight,
  Info
} from "lucide-react";
import { getApiBase, formatMetric, formatPercent } from "../utils/api";

export default function Monitoring() {
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const [logsLoading, setLogsLoading] = useState(true);
  const [statsLoading, setStatsLoading] = useState(true);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [driftData, setDriftData] = useState(null);
  const [driftLoading, setDriftLoading] = useState(false);
  const [logFilterStatus, setLogFilterStatus] = useState("all");

  // Fetch telemetry summary
  const fetchStats = async () => {
    setStatsLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/monitoring/stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
        if (data.requests_by_model?.length > 0 && !selectedModelId) {
          setSelectedModelId(String(data.requests_by_model[0].model_id));
        }
      }
    } catch (err) {
      console.error("Failed to load monitoring stats", err);
    } finally {
      setStatsLoading(false);
    }
  };

  // Fetch paginated inference logs
  const fetchLogs = async () => {
    setLogsLoading(true);
    try {
      let url = `${getApiBase()}/monitoring/logs?limit=40`;
      if (logFilterStatus !== "all") {
        url += `&status=${logFilterStatus}`;
      }
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setLogs(data.items || []);
      }
    } catch (err) {
      console.error("Failed to load inference logs", err);
    } finally {
      setLogsLoading(false);
    }
  };

  // Fetch drift diagnostic for selected model
  const fetchDrift = async (mId) => {
    if (!mId) return;
    setDriftLoading(true);
    try {
      const res = await fetch(`${getApiBase()}/monitoring/drift/${mId}`);
      if (res.ok) {
        const data = await res.json();
        setDriftData(data);
      }
    } catch (err) {
      console.error("Failed to load drift diagnostics", err);
    } finally {
      setDriftLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchLogs();
  }, [logFilterStatus]);

  useEffect(() => {
    if (selectedModelId) {
      fetchDrift(selectedModelId);
    }
  }, [selectedModelId]);

  return (
    <div className="min-h-screen bg-[#070711] text-[#F7F7FC] pt-20 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="mb-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.08] pb-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5" />
                Live Telemetry
              </span>
              <span className="text-xs text-[#A4A4B8]">• Operational Diagnostics & Serving Metrics</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
              Model Serving & Health Monitoring
            </h1>
            <p className="mt-1 text-xs sm:text-sm text-[#A4A4B8] max-w-2xl">
              Track real-time inference throughput, p50/p95 serving latencies, in-memory LRU model cache utilization, and input distribution drift indicators.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                fetchStats();
                fetchLogs();
                if (selectedModelId) fetchDrift(selectedModelId);
              }}
              disabled={statsLoading}
              className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#201f44] text-white border border-white/[0.08] flex items-center gap-2 transition-all shadow-sm"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${statsLoading ? "animate-spin" : ""}`} />
              Refresh Metrics
            </button>
            <Link
              to="/models"
              className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-[#8265FF] hover:bg-[#7152f8] text-white shadow-sm shadow-[#8265FF]/20 flex items-center gap-2 transition-all"
            >
              <Layers className="w-3.5 h-3.5" />
              Model Registry
            </Link>
          </div>
        </div>
      </div>

      {/* Top Telemetry KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
        <div className="p-4 rounded-2xl bg-[#111126] border border-white/[0.08] shadow-lg">
          <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block mb-1">
            Total Predictions
          </span>
          <span className="text-xl sm:text-2xl font-black text-white font-mono block">
            {stats ? stats.total_inferences.toLocaleString() : "..."}
          </span>
          <span className="text-[11px] text-[#A4A4B8] mt-1 block">
            {stats ? `${stats.successful_inferences} successful` : "Loading..."}
          </span>
        </div>

        <div className="p-4 rounded-2xl bg-[#111126] border border-white/[0.08] shadow-lg">
          <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block mb-1">
            Error Rate
          </span>
          <span className={`text-xl sm:text-2xl font-black font-mono block ${
            stats && stats.error_rate > 0.05 ? "text-rose-400" : "text-emerald-400"
          }`}>
            {stats ? formatPercent(stats.error_rate, 2) : "..."}
          </span>
          <span className="text-[11px] text-[#A4A4B8] mt-1 block">
            {stats ? `${stats.failed_inferences} failures recorded` : "Loading..."}
          </span>
        </div>

        <div className="p-4 rounded-2xl bg-[#111126] border border-white/[0.08] shadow-lg">
          <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block mb-1">
            Avg / P50 Latency
          </span>
          <span className="text-xl sm:text-2xl font-black text-[#8265FF] font-mono block">
            {stats ? `${stats.avg_latency_ms} ms` : "..."}
          </span>
          <span className="text-[11px] text-[#A4A4B8] mt-1 block">
            {stats ? `Median: ${stats.p50_latency_ms} ms` : "Loading..."}
          </span>
        </div>

        <div className="p-4 rounded-2xl bg-[#111126] border border-white/[0.08] shadow-lg">
          <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block mb-1">
            P95 / P99 Latency
          </span>
          <span className="text-xl sm:text-2xl font-black text-purple-300 font-mono block">
            {stats ? `${stats.p90_latency_ms} ms` : "..."}
          </span>
          <span className="text-[11px] text-[#A4A4B8] mt-1 block">
            {stats ? `P99: ${stats.p99_latency_ms} ms` : "Loading..."}
          </span>
        </div>

        <div className="p-4 rounded-2xl bg-[#111126] border border-white/[0.08] shadow-lg col-span-2 lg:col-span-1">
          <span className="text-[10px] uppercase font-bold text-[#A4A4B8] block mb-1">
            LRU Cache Serving
          </span>
          <span className="text-xl sm:text-2xl font-black text-emerald-400 font-mono block">
            {stats?.serving_inventory ? `${stats.serving_inventory.length} / 8` : "0 / 8"}
          </span>
          <span className="text-[11px] text-[#A4A4B8] mt-1 block">
            Active in-memory pipelines
          </span>
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-8">
        {/* Left Column: Active Model Serving Inventory */}
        <div className="lg:col-span-6 bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-4">
          <h2 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] flex items-center gap-2">
            <Cpu className="w-4 h-4" />
            1. Model Serving Inventory & Latency Breakdown
          </h2>

          {!stats || stats.requests_by_model.length === 0 ? (
            <div className="p-8 text-center bg-[#171630]/40 rounded-xl border border-white/[0.04]">
              <Database className="w-8 h-8 mx-auto text-[#A4A4B8]/40 mb-2" />
              <p className="text-xs text-[#A4A4B8]">No serving traffic recorded yet.</p>
              <p className="text-[11px] text-[#A4A4B8]/60 mt-1">
                Trigger prediction requests via the Predictions workspace to track serving performance.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-white/[0.08] bg-[#171630]">
                    <th className="py-2 px-3 font-bold uppercase text-[10px] text-[#A4A4B8]">Model</th>
                    <th className="py-2 px-3 font-bold uppercase text-[10px] text-[#A4A4B8]">Task</th>
                    <th className="py-2 px-3 font-bold uppercase text-[10px] text-[#A4A4B8] text-right">Inferences</th>
                    <th className="py-2 px-3 font-bold uppercase text-[10px] text-[#A4A4B8] text-right">Avg Latency</th>
                    <th className="py-2 px-3 font-bold uppercase text-[10px] text-[#A4A4B8] text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {stats.requests_by_model.map((m) => {
                    const isSelected = String(m.model_id) === selectedModelId;
                    return (
                      <tr
                        key={m.model_id}
                        onClick={() => setSelectedModelId(String(m.model_id))}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? "bg-[#171630] text-white" : "hover:bg-white/[0.02]"
                        }`}
                      >
                        <td className="py-2.5 px-3">
                          <span className="font-bold text-white block truncate max-w-[150px]">
                            {m.model_name}
                          </span>
                          <span className="text-[10px] font-mono text-[#A4A4B8]">#{m.model_id}</span>
                        </td>
                        <td className="py-2.5 px-3">
                          <span className="px-1.5 py-0.5 rounded text-[10px] uppercase font-semibold bg-white/[0.06] text-[#A4A4B8]">
                            {m.task_type}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-mono font-bold text-white text-right">
                          {m.request_count.toLocaleString()}
                        </td>
                        <td className="py-2.5 px-3 font-mono font-bold text-[#8265FF] text-right">
                          {m.avg_latency_ms} ms
                        </td>
                        <td className="py-2.5 px-3 text-center">
                          <Link
                            to={`/predict?model=${m.model_id}`}
                            className="text-[11px] text-emerald-400 hover:underline inline-flex items-center gap-0.5"
                          >
                            Predict <ChevronRight className="w-3 h-3" />
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Right Column: Feature Drift Diagnostic Inspector */}
        <div className="lg:col-span-6 bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              2. Input Covariate Drift Diagnostics
            </h2>
            {selectedModelId && (
              <span className="text-[11px] font-mono text-[#A4A4B8]">
                Model #{selectedModelId}
              </span>
            )}
          </div>

          <p className="text-xs text-[#A4A4B8]">
            Statistical variation ratio computed across recent incoming inference batches. High ratio indicates potential covariate shift.
          </p>

          {driftLoading ? (
            <div className="p-8 text-center">
              <RefreshCw className="w-6 h-6 animate-spin mx-auto text-[#8265FF] mb-2" />
              <p className="text-xs text-[#A4A4B8]">Analyzing feature distributions...</p>
            </div>
          ) : !driftData || driftData.status === "insufficient_data" ? (
            <div className="p-6 rounded-xl bg-[#171630]/40 border border-white/[0.04] text-xs text-[#A4A4B8] space-y-2">
              <div className="flex items-center gap-2 text-amber-300 font-semibold">
                <Info className="w-4 h-4 shrink-0" />
                <span>Baseline Warm-Up Required</span>
              </div>
              <p className="text-[11px] text-[#A4A4B8]/80 leading-relaxed">
                {driftData?.message || "Select a model from the inventory and run predictions to collect telemetry."}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              <div className={`p-3 rounded-xl border text-xs flex items-center gap-2 ${
                driftData.status === "shift_warning"
                  ? "bg-amber-500/10 border-amber-500/20 text-amber-300"
                  : "bg-emerald-500/10 border-emerald-500/20 text-emerald-300"
              }`}>
                {driftData.status === "shift_warning" ? (
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                ) : (
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                )}
                <span>{driftData.message}</span>
              </div>

              <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1">
                {driftData.feature_indicators.map((ind, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-xl bg-[#171630] border border-white/[0.04] flex items-center justify-between text-xs"
                  >
                    <div>
                      <span className="font-mono text-white font-semibold">{ind.feature}</span>
                      <span className="text-[10px] text-[#A4A4B8] block">
                        Recent Mean: {ind.recent_mean} • Std: {ind.std_deviation}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase font-mono ${
                        ind.status === "shift_detected"
                          ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                          : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                      }`}>
                        {ind.status} (cv: {ind.variance_ratio})
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Inference Audit Event Logs Feed */}
      <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-xs font-bold uppercase tracking-wider text-[#8265FF] flex items-center gap-2">
              <Zap className="w-4 h-4" />
              3. Recent Inference Audit Trail
            </h2>
            <p className="text-xs text-[#A4A4B8] mt-0.5">
              Live request telemetry logs with latency, row counts, and correlation IDs.
            </p>
          </div>

          <div className="flex items-center gap-2">
            {["all", "success", "error"].map((status) => (
              <button
                key={status}
                onClick={() => setLogFilterStatus(status)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition-all ${
                  logFilterStatus === status
                    ? "bg-[#8265FF] text-white shadow-sm"
                    : "bg-[#171630] text-[#A4A4B8] hover:text-white"
                }`}
              >
                {status}
              </button>
            ))}
          </div>
        </div>

        {logsLoading ? (
          <div className="p-12 text-center">
            <RefreshCw className="w-6 h-6 animate-spin mx-auto text-[#8265FF] mb-2" />
            <p className="text-xs text-[#A4A4B8]">Loading telemetry stream...</p>
          </div>
        ) : logs.length === 0 ? (
          <div className="p-8 text-center bg-[#171630]/30 rounded-xl">
            <p className="text-xs text-[#A4A4B8]">No inference logs matching filter.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-white/[0.08] bg-[#171630]">
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#A4A4B8]">Time</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#A4A4B8]">Model</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#A4A4B8]">Rows</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#A4A4B8] text-right">Latency</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#A4A4B8] text-center">Status</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#A4A4B8]">Correlation ID</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {logs.map((log) => (
                  <tr key={log.id} className="hover:bg-white/[0.02]">
                    <td className="py-2.5 px-4 text-[#A4A4B8] font-mono text-[11px]">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 px-4 font-semibold text-white">
                      {log.model_name} <span className="text-[10px] text-[#A4A4B8] font-mono">#{log.model_id}</span>
                    </td>
                    <td className="py-2.5 px-4 font-mono text-white">
                      {log.row_count} rows
                    </td>
                    <td className="py-2.5 px-4 font-mono font-bold text-emerald-400 text-right">
                      {log.latency_ms} ms
                    </td>
                    <td className="py-2.5 px-4 text-center">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                        log.status === "success"
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                          : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                      }`}>
                        {log.status}
                      </span>
                    </td>
                    <td className="py-2.5 px-4 font-mono text-[11px] text-[#6E6E88] truncate max-w-[120px]">
                      {log.correlation_id}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
