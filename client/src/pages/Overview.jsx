// client/src/pages/Overview.jsx
import React, { useState, useEffect } from "react";
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
  Sparkles,
  Compass,
  FileText,
  BarChart3
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
import PageHeader from "../components/ui/PageHeader";
import MetricCard from "../components/ui/MetricCard";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import Badge from "../components/ui/Badge";
import EmptyState from "../components/ui/EmptyState";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";

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
      
      {/* Universal Page Header */}
      <PageHeader
        category="Operations & Analytics"
        title="ML Operations Overview"
        subtitle="Live telemetry of registered models, evaluation benchmarks, and training experiment runs."
        actions={
          <>
            <Link to="/workspace">
              <Button variant="primary" size="sm" icon={Compass}>
                Launch Guided Studio
              </Button>
            </Link>
            <Link to="/automl">
              <Button variant="secondary" size="sm" icon={Cpu}>
                Model Training
              </Button>
            </Link>
            <Link to="/compare">
              <Button variant="outline" size="sm" icon={GitCompare}>
                Benchmarks
              </Button>
            </Link>
          </>
        }
      />

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Active Model Registry"
          value={loading ? "..." : modelsData.total}
          subtitle="Trained pipelines ready for inference"
          icon={Layers}
          change="Available"
          changeType="positive"
        />

        <MetricCard
          title="Experiment Runs"
          value={loading ? "..." : experimentsData.total}
          subtitle="Preserved SQLite training history"
          icon={Clock}
          change="Logged"
          changeType="neutral"
        />

        <MetricCard
          title="Comparison Benchmarks"
          value={loading ? "..." : experimentsData.experiments.filter((e) => e.experiment_name.includes("Comparison")).length}
          subtitle="Multi-model evaluations with holdouts"
          icon={GitCompare}
          change="Validated"
          changeType="positive"
        />

        <MetricCard
          title="API Engine Status"
          value="Online"
          subtitle="FastAPI 0.162+ • Port 8000"
          icon={Activity}
          change="Port 8000"
          changeType="positive"
        />
      </div>

      {/* Visual Analytics Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Chart 1: Model Score Performance */}
        <div className="lg:col-span-2">
          <Card
            title="Registered Model Primary Performance"
            subtitle="Top registered model holdout metric (Accuracy / R²)"
            variant="default"
            action={
              <Link to="/models" className="text-xs text-[#8265FF] hover:text-[#C084FC] flex items-center gap-1 font-semibold">
                View Registry <ChevronRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            {loading ? (
              <LoadingSkeleton lines={6} />
            ) : recentModelsChartData.length > 0 ? (
              <div className="h-64 w-full pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={recentModelsChartData}>
                    <XAxis dataKey="name" stroke="#64677E" fontSize={11} tickLine={false} />
                    <YAxis stroke="#64677E" fontSize={11} domain={[0, 1]} tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#16152F",
                        borderColor: "rgba(255,255,255,0.12)",
                        borderRadius: "12px",
                        color: "#F8F9FD",
                        fontSize: "12px",
                      }}
                    />
                    <Bar dataKey="score" fill="#8265FF" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState
                title="No Models Registered Yet"
                description="Train or compare models to populate the performance benchmark chart."
                actionLabel="Train First Model"
                onAction={() => window.location.href = "/automl"}
                className="py-10 bg-transparent border-0"
              />
            )}
          </Card>
        </div>

        {/* Chart 2: Task Distribution */}
        <div className="lg:col-span-1">
          <Card
            title="Task Distribution"
            subtitle="Classification vs. Regression runs"
            variant="default"
          >
            <div className="h-44 w-full flex items-center justify-center">
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
                        backgroundColor: "#16152F",
                        borderColor: "rgba(255,255,255,0.12)",
                        borderRadius: "10px",
                        color: "#F8F9FD",
                        fontSize: "11px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-xs text-[#9FA2B6]">No experiment runs yet</p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2 pt-4 border-t border-white/[0.06] text-xs">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-[#8265FF]" />
                <span className="text-[#9FA2B6]">Classification: <b className="text-white">{taskDistribution[0].value}</b></span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-[#A855F7]" />
                <span className="text-[#9FA2B6]">Regression: <b className="text-white">{taskDistribution[1].value}</b></span>
              </div>
            </div>
          </Card>
        </div>

      </div>

      {/* Two Column Table Previews */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Column 1: Active Model Registry */}
        <Card
          title="Active Model Registry"
          subtitle="Top registered models ready for deployment"
          variant="default"
          icon={Layers}
          action={
            <Link to="/models" className="text-xs text-[#8265FF] hover:text-[#C084FC] font-semibold">
              All Models ({modelsData.total})
            </Link>
          }
        >
          <div className="space-y-3">
            {loading ? (
              <LoadingSkeleton lines={4} />
            ) : modelsData.models.length > 0 ? (
              modelsData.models.slice(0, 4).map((model) => (
                <div 
                  key={model.id}
                  className="p-3.5 rounded-xl bg-[#16152F]/70 border border-white/[0.06] hover:border-[#8265FF]/40 transition-all flex items-center justify-between"
                >
                  <div>
                    <h4 className="text-xs font-semibold text-white">{model.model_name}</h4>
                    <p className="text-[11px] text-[#9FA2B6] mt-0.5">
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
              <EmptyState
                title="No Models Registered"
                description="Train or compare models to view registered artifacts."
                actionLabel="Train Model"
                onAction={() => window.location.href = "/automl"}
                className="py-6 bg-transparent border-0"
              />
            )}
          </div>
        </Card>

        {/* Column 2: Recent Experiment Runs */}
        <Card
          title="Recent Experiment Runs"
          subtitle="Audit ledger of past model training runs"
          variant="default"
          icon={Clock}
          action={
            <Link to="/experiments" className="text-xs text-[#8265FF] hover:text-[#C084FC] font-semibold">
              All Runs ({experimentsData.total})
            </Link>
          }
        >
          <div className="space-y-3">
            {loading ? (
              <LoadingSkeleton lines={4} />
            ) : experimentsData.experiments.length > 0 ? (
              experimentsData.experiments.slice(0, 4).map((exp) => (
                <div 
                  key={exp.id}
                  className="p-3.5 rounded-xl bg-[#16152F]/70 border border-white/[0.06] hover:border-[#8265FF]/40 transition-all flex items-center justify-between"
                >
                  <div>
                    <h4 className="text-xs font-semibold text-white">{exp.experiment_name}</h4>
                    <p className="text-[11px] text-[#9FA2B6] mt-0.5">
                      {exp.model_type} • {exp.task_type} • Split {exp.split_ratio}
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-[#9FA2B6] block">
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
              <EmptyState
                title="No Experiments Logged"
                description="Every training, tuning, and comparison run is preserved here."
                className="py-6 bg-transparent border-0"
              />
            )}
          </div>
        </Card>

      </div>

    </div>
  );
}
