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
  Database,
  Cpu,
  TrendingUp,
  ChevronRight,
  Info
} from "lucide-react";
import { getApiBase, formatPercent } from "../utils/api";
import PageHeader from "../components/ui/PageHeader";
import Card from "../components/ui/Card";
import MetricCard from "../components/ui/MetricCard";
import Badge from "../components/ui/Badge";
import Button from "../components/ui/Button";

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
    <div className="min-h-screen bg-[#070711] text-[#F8F9FD] pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
      {/* Universal Page Header */}
      <PageHeader
        category="Intelligence"
        title="Model Serving & Telemetry"
        subtitle="Track real-time inference throughput, p50/p95 serving latencies, in-memory LRU model cache utilization, and input distribution drift indicators."
        badge={
          <Badge variant="success" size="sm">
            <Activity className="w-3.5 h-3.5 mr-1 inline" /> Live Telemetry
          </Badge>
        }
        actions={
          <>
            <Button
              variant="secondary"
              size="sm"
              icon={RefreshCw}
              loading={statsLoading}
              onClick={() => {
                fetchStats();
                fetchLogs();
                if (selectedModelId) fetchDrift(selectedModelId);
              }}
            >
              Refresh Metrics
            </Button>
            <Link to="/models">
              <Button variant="primary" size="sm" icon={Layers}>
                Model Registry
              </Button>
            </Link>
          </>
        }
      />

      {/* Top Telemetry KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          title="Total Predictions"
          value={stats ? stats.total_inferences.toLocaleString() : "..."}
          subtitle={stats ? `${stats.successful_inferences} successful` : "Loading..."}
          icon={Zap}
        />
        <MetricCard
          title="Error Rate"
          value={stats ? formatPercent(stats.error_rate, 2) : "..."}
          subtitle={stats ? `${stats.failed_inferences} failures` : "Loading..."}
          change={stats && stats.error_rate > 0.05 ? "Elevated" : "Optimal"}
          changeType={stats && stats.error_rate > 0.05 ? "negative" : "positive"}
          icon={AlertTriangle}
        />
        <MetricCard
          title="Avg Latency"
          value={stats ? `${stats.avg_latency_ms} ms` : "..."}
          subtitle={stats ? `Median p50: ${stats.p50_latency_ms} ms` : "Loading..."}
          icon={Clock}
        />
        <MetricCard
          title="P95 / P99 Latency"
          value={stats ? `${stats.p90_latency_ms} ms` : "..."}
          subtitle={stats ? `P99: ${stats.p99_latency_ms} ms` : "Loading..."}
          icon={Activity}
        />
        <MetricCard
          title="LRU Cache Serving"
          value={stats?.serving_inventory ? `${stats.serving_inventory.length} / 8` : "0 / 8"}
          subtitle="Active in-memory pipelines"
          icon={Cpu}
          className="col-span-2 lg:col-span-1"
        />
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Active Model Serving Inventory */}
        <Card
          className="lg:col-span-6 space-y-4"
          header={
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#8265FF]" />
              <h2 className="text-sm font-bold text-white tracking-wide">
                1. Serving Inventory & Latency Breakdown
              </h2>
            </div>
          }
        >
          {!stats || stats.requests_by_model.length === 0 ? (
            <div className="p-8 text-center bg-white/[0.02] rounded-xl border border-white/[0.04]">
              <Database className="w-8 h-8 mx-auto text-[#9FA2B6]/40 mb-2" />
              <p className="text-xs text-[#9FA2B6]">No serving traffic recorded yet.</p>
              <p className="text-[11px] text-[#9FA2B6]/60 mt-1">
                Trigger prediction requests via the Predictions workspace to track serving performance.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-white/[0.08] bg-white/[0.02]">
                    <th className="py-2.5 px-3 font-bold uppercase text-[10px] text-[#9FA2B6]">Model</th>
                    <th className="py-2.5 px-3 font-bold uppercase text-[10px] text-[#9FA2B6]">Task</th>
                    <th className="py-2.5 px-3 font-bold uppercase text-[10px] text-[#9FA2B6] text-right">Inferences</th>
                    <th className="py-2.5 px-3 font-bold uppercase text-[10px] text-[#9FA2B6] text-right">Avg Latency</th>
                    <th className="py-2.5 px-3 font-bold uppercase text-[10px] text-[#9FA2B6] text-center">Action</th>
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
                          isSelected ? "bg-[#8265FF]/10 text-white" : "hover:bg-white/[0.02]"
                        }`}
                      >
                        <td className="py-2.5 px-3">
                          <span className="font-bold text-white block truncate max-w-[150px]">
                            {m.model_name}
                          </span>
                          <span className="text-[10px] font-mono text-[#9FA2B6]">#{m.model_id}</span>
                        </td>
                        <td className="py-2.5 px-3">
                          <Badge variant="neutral" size="sm">
                            {m.task_type}
                          </Badge>
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
        </Card>

        {/* Right Column: Feature Drift Diagnostic Inspector */}
        <Card
          className="lg:col-span-6 space-y-4"
          header={
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-[#8265FF]" />
                <h2 className="text-sm font-bold text-white tracking-wide">
                  2. Input Covariate Drift Diagnostics
                </h2>
              </div>
              {selectedModelId && (
                <Badge variant="violet" size="sm">
                  Model #{selectedModelId}
                </Badge>
              )}
            </div>
          }
        >
          <p className="text-xs text-[#9FA2B6]">
            Statistical variation ratio computed across recent incoming inference batches. High ratio indicates potential covariate shift.
          </p>

          {driftLoading ? (
            <div className="p-8 text-center">
              <RefreshCw className="w-6 h-6 animate-spin mx-auto text-[#8265FF] mb-2" />
              <p className="text-xs text-[#9FA2B6]">Analyzing feature distributions...</p>
            </div>
          ) : !driftData || driftData.status === "insufficient_data" ? (
            <div className="p-5 rounded-xl bg-white/[0.02] border border-white/[0.06] text-xs text-[#9FA2B6] space-y-2">
              <div className="flex items-center gap-2 text-amber-300 font-semibold">
                <Info className="w-4 h-4 shrink-0" />
                <span>Baseline Warm-Up Required</span>
              </div>
              <p className="text-[11px] text-[#9FA2B6]/80 leading-relaxed">
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

              <div className="max-h-56 overflow-y-auto space-y-2 pr-1">
                {driftData.feature_indicators.map((ind, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.04] flex items-center justify-between text-xs"
                  >
                    <div>
                      <span className="font-mono text-white font-semibold">{ind.feature}</span>
                      <span className="text-[10px] text-[#9FA2B6] block">
                        Recent Mean: {ind.recent_mean} • Std: {ind.std_deviation}
                      </span>
                    </div>
                    <div className="text-right">
                      <Badge
                        variant={ind.status === "shift_detected" ? "warning" : "success"}
                        size="sm"
                      >
                        {ind.status} (cv: {ind.variance_ratio})
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* Inference Audit Event Logs Feed */}
      <Card
        header={
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 w-full">
            <div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-[#8265FF]" />
                <h2 className="text-sm font-bold text-white tracking-wide">
                  3. Recent Inference Audit Trail
                </h2>
              </div>
              <p className="text-xs text-[#9FA2B6] mt-0.5">
                Live request telemetry logs with latency, row counts, and correlation IDs.
              </p>
            </div>

            <div className="flex items-center gap-1.5 p-1 bg-white/[0.04] border border-white/[0.08] rounded-xl">
              {["all", "success", "error"].map((status) => (
                <button
                  key={status}
                  onClick={() => setLogFilterStatus(status)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold capitalize transition-all ${
                    logFilterStatus === status
                      ? "bg-[#8265FF] text-white shadow-sm"
                      : "text-[#9FA2B6] hover:text-white"
                  }`}
                >
                  {status}
                </button>
              ))}
            </div>
          </div>
        }
      >
        {logsLoading ? (
          <div className="p-12 text-center">
            <RefreshCw className="w-6 h-6 animate-spin mx-auto text-[#8265FF] mb-2" />
            <p className="text-xs text-[#9FA2B6]">Loading telemetry stream...</p>
          </div>
        ) : logs.length === 0 ? (
          <div className="p-8 text-center bg-white/[0.02] rounded-xl">
            <p className="text-xs text-[#9FA2B6]">No inference logs matching filter.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-white/[0.08] bg-white/[0.02]">
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#9FA2B6]">Time</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#9FA2B6]">Model</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#9FA2B6]">Rows</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#9FA2B6] text-right">Latency</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#9FA2B6] text-center">Status</th>
                  <th className="py-2.5 px-4 font-bold uppercase text-[10px] text-[#9FA2B6]">Correlation ID</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {logs.map((log) => (
                  <tr key={log.id} className="hover:bg-white/[0.02]">
                    <td className="py-2.5 px-4 text-[#9FA2B6] font-mono text-[11px]">
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 px-4 font-semibold text-white">
                      {log.model_name} <span className="text-[10px] text-[#9FA2B6] font-mono">#{log.model_id}</span>
                    </td>
                    <td className="py-2.5 px-4 font-mono text-white">
                      {log.row_count} rows
                    </td>
                    <td className="py-2.5 px-4 font-mono font-bold text-emerald-400 text-right">
                      {log.latency_ms} ms
                    </td>
                    <td className="py-2.5 px-4 text-center">
                      <Badge
                        variant={log.status === "success" ? "success" : "danger"}
                        size="sm"
                      >
                        {log.status}
                      </Badge>
                    </td>
                    <td className="py-2.5 px-4 font-mono text-[11px] text-[#9FA2B6]/60 truncate max-w-[120px]">
                      {log.correlation_id}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
