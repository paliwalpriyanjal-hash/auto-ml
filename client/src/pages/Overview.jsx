import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { 
  Activity, 
  Cpu, 
  GitCompare, 
  Layers, 
  Clock, 
  PlayCircle, 
  TrendingUp, 
  ArrowUpRight,
  Database,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  ChevronRight,
  Sparkles
} from "lucide-react";
import { getApiBase, formatMetric } from "../utils/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

export default function Overview() {
  const [modelsData, setModelsData] = useState({ total: 0, models: [] });
  const [experimentsData, setExperimentsData] = useState({ total: 0, experiments: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const base = getApiBase();
        const [modelsRes, expsRes] = await Promise.all([
          fetch(`${base}/models?limit=10`).then((r) => r.ok ? r.json() : { total: 0, models: [] }),
          fetch(`${base}/experiments?limit=10`).then((r) => r.ok ? r.json() : { total: 0, experiments: [] })
        ]);
        if (isMounted) {
          setModelsData(modelsRes);
          setExperimentsData(expsRes);
        }
      } catch (err) {
        if (isMounted) setError("Failed to communicate with AutoVeyra backend service.");
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    fetchData();
    return () => { isMounted = false; };
  }, []);

  // Compute breakdown for visual chart
  const taskDistribution = [
    { 
      name: "Classification", 
      value: experimentsData.experiments.filter((e) => e.task_type === "classification").length,
      color: "#8265FF" 
    },
    { 
      name: "Regression", 
      value: experimentsData.experiments.filter((e) => e.task_type === "regression").length,
      color: "#A855F7" 
    },
  ];

  const recentModelsChartData = modelsData.models.slice(0, 5).map((m) => {
    const accuracy = m.metrics?.accuracy ?? m.metrics?.r2 ?? 0;
    return {
      name: m.model_name.length > 18 ? m.model_name.slice(0, 16) + "..." : m.model_name,
      score: accuracy,
      type: m.task_type,
    };
  });

  return (
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
      
      {/* Page Title & Context Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.08] pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[#8265FF] bg-[#8265FF]/10 px-2 py-0.5 rounded-full border border-[#8265FF]/20">
              Workspace Analytics
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">AutoML Operations Overview</h1>
          <p className="text-sm text-[#A4A4B8] mt-1">
            Live telemetry of registered models, evaluation benchmarks, and training experiment runs.
          </p>
        </div>

        {/* Quick Launch Buttons */}
        <div className="flex items-center gap-3">
          <Link
            to="/automl"
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-[#171630] hover:bg-[#1f1e40] text-white border border-white/[0.08] transition-all flex items-center gap-2"
          >
            <Cpu className="w-3.5 h-3.5 text-[#8265FF]" />
            New Model Training
          </Link>
          <Link
            to="/compare"
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/20 transition-all flex items-center gap-2"
          >
            <GitCompare className="w-3.5 h-3.5" />
            Benchmark Algorithms
          </Link>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        {/* Metric 1 */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-5 hover:border-[#8265FF]/30 transition-all group">
          <div className="flex items-center justify-between text-[#A4A4B8] mb-3">
            <span className="text-xs font-medium">Registered Models</span>
            <div className="w-8 h-8 rounded-lg bg-[#8265FF]/10 flex items-center justify-center text-[#8265FF]">
              <Layers className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white tracking-tight">
            {loading ? "..." : modelsData.total}
          </div>
          <div className="flex items-center gap-1.5 text-xs text-emerald-400 mt-2">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Ready for inference</span>
          </div>
        </div>

        {/* Metric 2 */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-5 hover:border-[#8265FF]/30 transition-all group">
          <div className="flex items-center justify-between text-[#A4A4B8] mb-3">
            <span className="text-xs font-medium">Total Experiments</span>
            <div className="w-8 h-8 rounded-lg bg-[#A855F7]/10 flex items-center justify-center text-[#A855F7]">
              <Clock className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white tracking-tight">
            {loading ? "..." : experimentsData.total}
          </div>
          <div className="flex items-center gap-1.5 text-xs text-[#A4A4B8] mt-2">
            <Database className="w-3.5 h-3.5 text-[#8265FF]" />
            <span>Recorded SQLite runs</span>
          </div>
        </div>

        {/* Metric 3 */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-5 hover:border-[#8265FF]/30 transition-all group">
          <div className="flex items-center justify-between text-[#A4A4B8] mb-3">
            <span className="text-xs font-medium">Model Benchmarks</span>
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400">
              <GitCompare className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-white tracking-tight">
            {loading ? "..." : experimentsData.experiments.filter((e) => e.experiment_name.includes("Comparison")).length}
          </div>
          <div className="flex items-center gap-1.5 text-xs text-emerald-400 mt-2">
            <TrendingUp className="w-3.5 h-3.5" />
            <span>Multi-fold cross-validated</span>
          </div>
        </div>

        {/* Metric 4 */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-5 hover:border-[#8265FF]/30 transition-all group">
          <div className="flex items-center justify-between text-[#A4A4B8] mb-3">
            <span className="text-xs font-medium">FastAPI Engine</span>
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="text-3xl font-bold text-emerald-400 tracking-tight flex items-center gap-2">
            Active
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
          </div>
          <div className="flex items-center gap-1.5 text-xs text-[#A4A4B8] mt-2">
            <span>Uvicorn 0.54 • Py 3.13</span>
          </div>
        </div>

      </div>

      {/* Visual Analytics Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Chart 1: Model Score Performance */}
        <div className="lg:col-span-2 bg-[#111126] border border-white/[0.08] rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-sm font-semibold text-white">Registered Model Primary Performance</h3>
              <p className="text-xs text-[#A4A4B8] mt-0.5">Top registered model holdout metric (Accuracy / R²)</p>
            </div>
            <Link to="/models" className="text-xs text-[#8265FF] hover:text-[#A855F7] flex items-center gap-1">
              View Registry <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          {recentModelsChartData.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={recentModelsChartData}>
                  <XAxis dataKey="name" stroke="#6B6B8A" fontSize={11} tickLine={false} />
                  <YAxis stroke="#6B6B8A" fontSize={11} domain={[0, 1]} tickLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#171630",
                      borderColor: "rgba(255,255,255,0.1)",
                      borderRadius: "12px",
                      color: "#F7F7FC",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="score" fill="#8265FF" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-center p-6 text-[#A4A4B8]">
              <Layers className="w-8 h-8 text-[#A4A4B8]/40 mb-2" />
              <p className="text-sm">No models registered yet</p>
              <Link to="/automl" className="mt-3 text-xs text-[#8265FF] font-medium hover:underline">
                Train your first model →
              </Link>
            </div>
          )}
        </div>

        {/* Chart 2: Task Distribution */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white mb-1">Experiment Task Breakdown</h3>
            <p className="text-xs text-[#A4A4B8]">Classification vs. Regression runs</p>
          </div>

          <div className="h-44 w-full my-auto flex items-center justify-center">
            {experimentsData.total > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={taskDistribution}
                    innerRadius={45}
                    outerRadius={65}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {taskDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: "#171630",
                      borderColor: "rgba(255,255,255,0.1)",
                      borderRadius: "10px",
                      color: "#F7F7FC",
                      fontSize: "11px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-xs text-[#A4A4B8]">No experiment telemetry</p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-2 pt-4 border-t border-white/[0.06] text-xs">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#8265FF]" />
              <span className="text-[#A4A4B8]">Classification: <b className="text-white">{taskDistribution[0].value}</b></span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#A855F7]" />
              <span className="text-[#A4A4B8]">Regression: <b className="text-white">{taskDistribution[1].value}</b></span>
            </div>
          </div>
        </div>

      </div>

      {/* Two Column Table Previews */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Column 1: Recent Registered Models */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#8265FF]" />
              Active Model Registry
            </h3>
            <Link to="/models" className="text-xs text-[#8265FF] hover:underline">
              All Models ({modelsData.total})
            </Link>
          </div>

          <div className="space-y-3">
            {modelsData.models.length > 0 ? (
              modelsData.models.slice(0, 4).map((model) => (
                <div 
                  key={model.id}
                  className="p-3.5 rounded-xl bg-[#171630]/60 border border-white/[0.06] hover:border-[#8265FF]/40 transition-all flex items-center justify-between"
                >
                  <div>
                    <h4 className="text-xs font-semibold text-white">{model.model_name}</h4>
                    <p className="text-[11px] text-[#A4A4B8] mt-0.5">
                      {model.dataset_name} • Target: <span className="text-[#8265FF]">{model.target_column}</span>
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-500/20">
                      {model.metrics?.accuracy ? `Acc: ${formatMetric(model.metrics.accuracy, 2)}` : model.metrics?.r2 ? `R²: ${formatMetric(model.metrics.r2, 2)}` : "Trained"}
                    </span>
                    <Link
                      to={`/predict?model=${model.id}`}
                      className="p-1.5 rounded-lg bg-[#8265FF]/20 text-[#8265FF] hover:bg-[#8265FF] hover:text-white transition-colors"
                      title="Test Inference"
                    >
                      <PlayCircle className="w-4 h-4" />
                    </Link>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-xs text-[#A4A4B8]">
                No registered models yet. Train or compare to auto-register.
              </div>
            )}
          </div>
        </div>

        {/* Column 2: Recent Experiments */}
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Clock className="w-4 h-4 text-[#A855F7]" />
              Recent Experiment Runs
            </h3>
            <Link to="/experiments" className="text-xs text-[#8265FF] hover:underline">
              All Experiments ({experimentsData.total})
            </Link>
          </div>

          <div className="space-y-3">
            {experimentsData.experiments.length > 0 ? (
              experimentsData.experiments.slice(0, 4).map((exp) => (
                <div 
                  key={exp.id}
                  className="p-3.5 rounded-xl bg-[#171630]/60 border border-white/[0.06] hover:border-[#A855F7]/40 transition-all flex items-center justify-between"
                >
                  <div>
                    <h4 className="text-xs font-semibold text-white">{exp.experiment_name}</h4>
                    <p className="text-[11px] text-[#A4A4B8] mt-0.5">
                      {exp.model_type} • {exp.task_type} • Split {exp.split_ratio}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-[#A4A4B8] block">
                      {new Date(exp.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <Link
                      to="/experiments"
                      className="text-[11px] text-[#8265FF] hover:underline flex items-center gap-0.5 justify-end mt-0.5"
                    >
                      Audit run <ArrowUpRight className="w-3 h-3" />
                    </Link>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-xs text-[#A4A4B8]">
                No experiments logged. Every training and comparison run is preserved here.
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  );
}
