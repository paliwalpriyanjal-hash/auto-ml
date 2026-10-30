import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { 
  Sparkles, 
  ArrowRight, 
  Cpu, 
  GitCompare, 
  Wand2, 
  Layers, 
  ShieldCheck, 
  CheckCircle2, 
  Database,
  BarChart2,
  Terminal,
  PlayCircle,
  Zap,
  TrendingUp,
  Sliders,
  ChevronRight,
  Clock
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const benchmarkDemoData = [
  { model: "RandomForest", cv: 0.942, holdout: 0.938 },
  { model: "GradientBoost", cv: 0.925, holdout: 0.919 },
  { model: "LogisticReg", cv: 0.884, holdout: 0.875 },
  { model: "DecisionTree", cv: 0.861, holdout: 0.852 },
  { model: "ExtraTrees", cv: 0.931, holdout: 0.924 },
];

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#070711] text-[#F7F7FC] relative overflow-hidden">
      
      {/* Background Lighting & Radial Violet Gradients */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[550px] bg-gradient-to-b from-[#8265FF]/18 via-[#A855F7]/10 to-transparent blur-[120px] pointer-events-none -z-0" />
      <div className="absolute top-[35%] -right-40 w-[600px] h-[600px] bg-[#8265FF]/10 blur-[140px] pointer-events-none -z-0" />
      <div className="absolute bottom-[20%] -left-40 w-[600px] h-[600px] bg-[#A855F7]/10 blur-[140px] pointer-events-none -z-0" />

      {/* ─────────────────────────────────────────────────────────────
          1. HERO SECTION
      ───────────────────────────────────────────────────────────── */}
      <section className="relative z-10 pt-36 pb-24 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          
          {/* Left Column: Headlines & CTAs */}
          <div className="lg:col-span-7 space-y-6 text-center lg:text-left">
            
            {/* Eyebrow badge */}
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#111126] border border-[#8265FF]/30 text-xs font-semibold text-[#8265FF] tracking-wide shadow-sm shadow-[#8265FF]/10">
              <Sparkles className="w-3.5 h-3.5 text-[#A855F7]" />
              <span className="uppercase tracking-wider">Automated Machine Learning Platform</span>
            </div>

            {/* Headline */}
            <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white leading-[1.12]">
              From Raw Data to{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#8265FF] via-[#C084FC] to-white">
                Production-Ready
              </span>{" "}
              Models.
            </h1>

            {/* Subtitle */}
            <p className="text-base sm:text-lg text-[#A4A4B8] max-w-2xl mx-auto lg:mx-0 leading-relaxed">
              Prepare datasets, compare machine learning algorithms, evaluate results, and turn experiments into reusable models—all in one intelligent workspace.
            </p>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 pt-2">
              <button
                onClick={() => navigate("/workspace")}
                className="w-full sm:w-auto px-7 py-3.5 rounded-xl font-bold text-sm bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-xl shadow-[#8265FF]/25 hover:shadow-[#8265FF]/40 transition-all flex items-center justify-center gap-2 group"
              >
                Start Guided Project
                <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </button>

              <button
                onClick={() => navigate("/compare")}
                className="w-full sm:w-auto px-7 py-3.5 rounded-xl font-bold text-sm bg-[#111126] hover:bg-[#171630] text-white border border-white/[0.12] hover:border-[#8265FF]/40 transition-all flex items-center justify-center gap-2"
              >
                <GitCompare className="w-4 h-4 text-[#8265FF]" />
                Explore Platform
              </button>
            </div>

            {/* Micro Highlights */}
            <div className="pt-6 border-t border-white/[0.06] flex flex-wrap items-center justify-center lg:justify-start gap-6 text-xs text-[#A4A4B8]">
              <span className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Zero code training
              </span>
              <span className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Multi-algorithm cross-validation
              </span>
              <span className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Integrated Model Registry
              </span>
            </div>

          </div>

          {/* Right Column: Floating Dark Preview Deck */}
          <div className="lg:col-span-5 relative">
            <div className="relative mx-auto max-w-md lg:max-w-none">
              
              {/* Violet Glow under card */}
              <div className="absolute inset-0 bg-gradient-to-tr from-[#8265FF]/30 to-[#A855F7]/20 rounded-3xl blur-2xl -z-10 transform scale-95" />

              {/* Main Preview Container */}
              <div className="bg-[#0D0D1B] border border-white/[0.12] rounded-3xl p-6 shadow-2xl space-y-5">
                
                {/* Header bar */}
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
                  <div className="flex items-center gap-2.5">
                    <span className="w-3 h-3 rounded-full bg-rose-500/80" />
                    <span className="w-3 h-3 rounded-full bg-amber-500/80" />
                    <span className="w-3 h-3 rounded-full bg-emerald-500/80" />
                    <span className="text-xs font-mono text-[#A4A4B8] ml-2">autoveyra.pipeline.live</span>
                  </div>
                  <span className="text-[10px] font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                    CV Benchmarking
                  </span>
                </div>

                {/* Benchmark Leaderboard Preview */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-[#A4A4B8] px-1">
                    <span>Candidate Algorithm</span>
                    <span>Holdout Accuracy</span>
                  </div>

                  {benchmarkDemoData.slice(0, 3).map((item, idx) => (
                    <div 
                      key={item.model}
                      className={`p-3 rounded-xl border flex items-center justify-between text-xs transition-all ${
                        idx === 0 
                          ? "bg-[#171630] border-[#8265FF]/50 shadow-md shadow-[#8265FF]/15" 
                          : "bg-[#111126] border-white/[0.06]"
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <span className={`w-5 h-5 rounded-md flex items-center justify-center font-bold text-[10px] ${
                          idx === 0 ? "bg-[#8265FF] text-white" : "bg-white/[0.06] text-[#A4A4B8]"
                        }`}>
                          #{idx + 1}
                        </span>
                        <span className="font-semibold text-white">{item.model}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-[11px] font-mono text-[#A4A4B8]">CV {item.cv}</span>
                        <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                          {(item.holdout * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Metric Summary Spark */}
                <div className="p-4 rounded-2xl bg-[#111126] border border-white/[0.06] space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-[#A4A4B8]">Auto-Registered Winner</span>
                    <span className="text-emerald-400 font-semibold">Active in Registry</span>
                  </div>
                  <div className="text-sm font-bold text-white flex items-center justify-between">
                    <span>RandomForestClassifier</span>
                    <span className="font-mono text-[#8265FF]">5-Fold Stratified</span>
                  </div>
                </div>

              </div>
            </div>
          </div>

        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          2. CORE PLATFORM WORKFLOW (How It Works)
      ───────────────────────────────────────────────────────────── */}
      <section className="relative z-10 py-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-white/[0.06]">
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#8265FF] bg-[#8265FF]/10 px-3 py-1 rounded-full border border-[#8265FF]/20">
            End-To-End Architecture
          </span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white">How AutoVeyra Works</h2>
          <p className="text-sm text-[#A4A4B8]">
            A transparent four-stage pipeline designed for reproducibility, leak-free validation, and zero-code execution.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            {
              step: "01",
              title: "Dataset Ingestion",
              desc: "Upload CSV data with instant schema inference, missing value inspection, and target column isolation.",
              icon: Database,
            },
            {
              step: "02",
              title: "Leakage-Safe Prep",
              desc: "Configure imputers, standard or min-max scalers, and categorical encoders fitted strictly on training splits.",
              icon: Sliders,
            },
            {
              step: "03",
              title: "Parallel Benchmarking",
              desc: "Execute cross-validated model trials simultaneously across decision trees, ensembles, and linear models.",
              icon: GitCompare,
            },
            {
              step: "04",
              title: "Registry & Scoring",
              desc: "Deploy winning pipelines with immutable artifact IDs and execute scoring through real-time REST endpoints.",
              icon: Layers,
            },
          ].map((item) => {
            const Icon = item.icon;
            return (
              <div 
                key={item.step}
                className="bg-[#0D0D1B] border border-white/[0.08] hover:border-[#8265FF]/40 rounded-2xl p-6 transition-all space-y-3 relative group"
              >
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-black font-mono text-[#8265FF]/30 group-hover:text-[#8265FF] transition-colors">
                    {item.step}
                  </span>
                  <div className="w-8 h-8 rounded-lg bg-[#171630] flex items-center justify-center text-[#8265FF]">
                    <Icon className="w-4 h-4" />
                  </div>
                </div>
                <h3 className="text-base font-bold text-white">{item.title}</h3>
                <p className="text-xs text-[#A4A4B8] leading-relaxed">{item.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          3. FEATURE SUITE
      ───────────────────────────────────────────────────────────── */}
      <section className="relative z-10 py-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-white/[0.06]">
        <div className="text-center max-w-3xl mx-auto mb-16 space-y-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#A855F7] bg-[#A855F7]/10 px-3 py-1 rounded-full border border-[#A855F7]/20">
            Industrial Capabilities
          </span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white">Full-Stack AutoML Features</h2>
          <p className="text-sm text-[#A4A4B8]">
            Every tool needed by developers, data analysts, and researchers to transform tabular datasets into deployed models.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          <div className="bg-[#111126] border border-white/[0.08] hover:border-[#8265FF]/40 rounded-2xl p-6 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-[#8265FF]/10 text-[#8265FF] flex items-center justify-center mb-4">
              <Cpu className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-white">Dedicated Model Training</h3>
            <p className="text-xs text-[#A4A4B8] leading-relaxed">
              Fine-tune individual models with custom preprocessing strategies, classification class weighting, and holdout performance matrices.
            </p>
            <Link to="/automl" className="text-xs text-[#8265FF] hover:underline font-semibold flex items-center gap-1 pt-2">
              Launch Training Engine <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="bg-[#111126] border border-white/[0.08] hover:border-[#8265FF]/40 rounded-2xl p-6 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-[#A855F7]/10 text-[#A855F7] flex items-center justify-center mb-4">
              <GitCompare className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-white">Multi-Algorithm Comparison</h3>
            <p className="text-xs text-[#A4A4B8] leading-relaxed">
              Evaluate Random Forests, Gradient Boosters, Decision Trees, and Regression algorithms in a single run with cross-validated ranking.
            </p>
            <Link to="/compare" className="text-xs text-[#A855F7] hover:underline font-semibold flex items-center gap-1 pt-2">
              Benchmark Models <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="bg-[#111126] border border-white/[0.08] hover:border-[#8265FF]/40 rounded-2xl p-6 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center mb-4">
              <Wand2 className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-white">Advanced Data Cleaning</h3>
            <p className="text-xs text-[#A4A4B8] leading-relaxed">
              KNN imputation, IQR outlier truncation, SMOTE class balancing, and TF-IDF feature extraction powered by the backend cleaner.
            </p>
            <Link to="/clean" className="text-xs text-emerald-400 hover:underline font-semibold flex items-center gap-1 pt-2">
              Clean Raw Datasets <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="bg-[#111126] border border-white/[0.08] hover:border-[#8265FF]/40 rounded-2xl p-6 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mb-4">
              <Layers className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-white">Model Registry & Artifacts</h3>
            <p className="text-xs text-[#A4A4B8] leading-relaxed">
              Inspect model artifacts, feature columns, and training lineage stored persistently in SQLite with deletion and lifecycle controls.
            </p>
            <Link to="/models" className="text-xs text-indigo-400 hover:underline font-semibold flex items-center gap-1 pt-2">
              Explore Registry <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="bg-[#111126] border border-white/[0.08] hover:border-[#8265FF]/40 rounded-2xl p-6 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center mb-4">
              <PlayCircle className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-white">Instant Model Inference</h3>
            <p className="text-xs text-[#A4A4B8] leading-relaxed">
              Dynamically generated UI forms that bind to any registered model's expected feature inputs and execute live inference scoring.
            </p>
            <Link to="/predict" className="text-xs text-cyan-400 hover:underline font-semibold flex items-center gap-1 pt-2">
              Run Live Predictions <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="bg-[#111126] border border-white/[0.08] hover:border-[#8265FF]/40 rounded-2xl p-6 space-y-3">
            <div className="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center mb-4">
              <Clock className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-white">Experiment History & Audit</h3>
            <p className="text-xs text-[#A4A4B8] leading-relaxed">
              Review every historical training job, compare runs over time, and inspect complete hyperparameter and holdout records.
            </p>
            <Link to="/experiments" className="text-xs text-amber-400 hover:underline font-semibold flex items-center gap-1 pt-2">
              Audit Experiments <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>

        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          4. CONVERSION CTA
      ───────────────────────────────────────────────────────────── */}
      <section className="relative z-10 py-20 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto">
        <div className="bg-gradient-to-tr from-[#111126] via-[#171630] to-[#0D0D1B] border border-[#8265FF]/30 rounded-3xl p-10 sm:p-14 text-center relative overflow-hidden shadow-2xl glow-purple">
          
          <div className="space-y-4 max-w-xl mx-auto">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#A855F7] bg-[#A855F7]/10 px-3 py-1 rounded-full border border-[#A855F7]/20">
              Instant Workspace Access
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
              Ready to Accelerate Your Machine Learning Workflow?
            </h2>
            <p className="text-sm text-[#A4A4B8] leading-relaxed">
              Select your CSV dataset, benchmark algorithms against each other, and deploy reliable models in minutes.
            </p>
            <div className="pt-4 flex flex-col sm:flex-row justify-center gap-4">
              <button
                onClick={() => navigate("/automl")}
                className="px-8 py-3.5 rounded-xl font-bold text-sm bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-xl shadow-[#8265FF]/30 transition-all flex items-center justify-center gap-2"
              >
                Launch AutoVeyra Studio
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>

        </div>
      </section>

    </div>
  );
}
