import React, { useState } from "react";
import Papa from "papaparse";
import Result from "../components/ResultDisplay";
import Model from "../components/ModelSelection";
import Preprocess from "../components/PreprocessingOptions";
import { getApiBase } from "../utils/api";
import {
  FiUpload,
  FiSettings,
  FiCpu,
  FiBarChart2,
  FiCheckCircle,
  FiAlertCircle,
  FiSliders,
  FiZap,
  FiActivity,
  FiHelpCircle
} from "react-icons/fi";
import PageHeader from "../components/ui/PageHeader";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Button from "../components/ui/Button";

function AutoML() {
  const [submit, setSubmit] = useState(false);
  const [csvFile, setCsvFile] = useState(null);
  const [previewData, setPreviewData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [activeStep, setActiveStep] = useState(1);

  const [taskType, setTaskType] = useState("regression");
  const [targetColumn, setTargetColumn] = useState("");

  const [autoMode, setAutoMode] = useState(true);
  const [intelligence, setIntelligence] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);

  const [missingStrategy, setMissingStrategy] = useState("median");
  const [encodingStrategy, setEncodingStrategy] = useState("onehot");
  const [scaler, setScaler] = useState("standard");
  const [splitRatio, setSplitRatio] = useState("0.8");
  const [model, setModel] = useState("RandomForestClassifier");

  const [error, setError] = useState("");

  const fetchDatasetIntelligence = async (file, currentTarget) => {
    try {
      setAnalyzing(true);
      const apiBase = getApiBase();
      const fd = new FormData();
      fd.append("file", file);
      if (currentTarget) {
        fd.append("target_column", currentTarget);
      }

      const res = await fetch(`${apiBase}/train/analyze`, {
        method: "POST",
        body: fd,
      });

      if (res.ok) {
        const data = await res.json();
        setIntelligence(data);

        // Intelligently adapt defaults based on dataset analysis
        if (data?.target_analysis?.inferred_task) {
          const task = data.target_analysis.inferred_task === "regression" ? "regression" : "classification";
          setTaskType(task);
          if (task === "classification") {
            setModel("RandomForestClassifier");
            setEncodingStrategy("onehot");
          } else {
            setModel("RandomForest");
            setEncodingStrategy("label");
          }
        }

        if (data?.target_analysis?.inferred_target && !currentTarget) {
          setTargetColumn(data.target_analysis.inferred_target);
        }
      }
    } catch (err) {
      console.warn("Dataset intelligence analysis note:", err);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleTaskTypeChange = (newTask) => {
    setTaskType(newTask);
    setError("");

    if (newTask === "classification") {
      setModel("RandomForestClassifier");
      setEncodingStrategy("onehot");
    } else {
      setModel("RandomForest");
      setEncodingStrategy("label");
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setSubmit(false);
    setCsvFile(null);
    setPreviewData([]);
    setColumns([]);
    setTargetColumn("");
    setIntelligence(null);
    setActiveStep(1);

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Please upload a valid CSV file.");
      return;
    }

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (header) => header.trim(),
      complete: (result) => {
        const fields = (result.meta.fields || []).filter(Boolean);
        const data = result.data || [];

        if (fields.length < 2) {
          setError("The CSV must contain a header row and at least two columns.");
          return;
        }

        const initialTarget = fields[fields.length - 1];
        setCsvFile(file);
        setColumns(fields);
        setPreviewData(data.slice(0, 5));
        setTargetColumn(initialTarget);
        setActiveStep(2);

        // Automatically trigger intelligent dataset analysis
        fetchDatasetIntelligence(file, initialTarget);
      },
    });
  };

  const handleTargetChange = (newTarget) => {
    setTargetColumn(newTarget);
    if (csvFile && newTarget) {
      fetchDatasetIntelligence(csvFile, newTarget);
    }
  };

  const handleSubmit = () => {
    if (!csvFile) {
      setError("Please choose a CSV file.");
      return;
    }

    if (!targetColumn) {
      setError("Please choose a target column.");
      return;
    }

    setError("");
    setSubmit(true);
    setActiveStep(4);
  };

  const handleReset = () => {
    setSubmit(false);
    setActiveStep(csvFile ? 2 : 1);
  };

  return (
    <div className="min-h-screen pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
      {/* Universal Page Header */}
      <PageHeader
        category="Machine Learning"
        title="AutoML Pipeline Training"
        subtitle="Train a specific algorithm with deterministic preprocessing, evaluate holdout metrics, and auto-register the fitted pipeline."
        badge={
          <Badge variant="violet" size="sm">
            Dataset-Adaptive Architecture
          </Badge>
        }
      />

      {/* Progress Steps Header */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { num: 1, title: "1. Dataset Upload", icon: FiUpload },
          { num: 2, title: "2. Task & Target", icon: FiSettings },
          { num: 3, title: "3. Algorithm & Prep", icon: FiCpu },
          { num: 4, title: "4. Training Results", icon: FiBarChart2 },
        ].map(({ num, title, icon: Icon }) => (
          <div
            key={num}
            className={`p-3.5 rounded-xl border flex items-center gap-2.5 text-xs font-semibold transition-all ${
              activeStep >= num
                ? "bg-[#171630] border-[#8265FF]/40 text-white"
                : "bg-[#111126] border-white/[0.06] text-[#A4A4B8]"
            }`}
          >
            <Icon className={activeStep >= num ? "text-[#8265FF]" : "text-[#A4A4B8]"} />
            <span>{title}</span>
          </div>
        ))}
      </div>

      {/* Step 1: Upload */}
      {!submit && (
        <section className="bg-[#111126] p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-4">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <FiUpload className="text-[#8265FF]" /> Select Dataset File
          </h3>

          <div className="flex flex-col sm:flex-row items-center gap-4">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="block w-full text-xs text-[#A4A4B8] file:mr-4 file:py-2.5 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-[#8265FF]/15 file:text-[#8265FF] hover:file:bg-[#8265FF]/25 cursor-pointer bg-[#171630] p-2 rounded-xl border border-white/[0.08]"
            />
            {csvFile && (
              <span className="text-xs text-emerald-400 flex items-center gap-1.5 shrink-0">
                <FiCheckCircle /> {csvFile.name} loaded
              </span>
            )}
          </div>

          {previewData.length > 0 && (
            <div className="mt-4 pt-4 border-t border-white/[0.06] overflow-x-auto">
              <p className="text-xs font-semibold text-[#A4A4B8] mb-2">Dataset Preview (First 5 Rows):</p>
              <table className="min-w-full text-xs text-left divide-y divide-white/[0.06] border border-white/[0.06] rounded-xl overflow-hidden">
                <thead className="bg-[#0D0D1B]">
                  <tr>
                    {columns.map((c) => (
                      <th key={c} className="px-3 py-2 text-[#A4A4B8] font-medium">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04]">
                  {previewData.map((row, i) => (
                    <tr key={i}>
                      {columns.map((c) => (
                        <td key={c} className="px-3 py-1.5 font-mono text-[#F7F7FC]">{row[c]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {/* Step 2: Task & Target Selection + Pre-Training Intelligence */}
      {!submit && csvFile && (
        <section className="bg-[#111126] p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-5">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <FiSettings className="text-[#A855F7]" /> Task & Target Configuration
            </h3>
            {analyzing && (
              <span className="text-xs text-[#8265FF] flex items-center gap-1.5 animate-pulse">
                <FiActivity /> Analyzing dataset intelligence...
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
                Task Type
              </label>
              <select
                value={taskType}
                onChange={(e) => handleTaskTypeChange(e.target.value)}
                className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                <option value="regression">Regression (Continuous values)</option>
                <option value="classification">Classification (Discrete labels)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
                Target Column (y)
              </label>
              <select
                value={targetColumn}
                onChange={(e) => handleTargetChange(e.target.value)}
                className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                <option value="">Select target column...</option>
                {columns.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Dataset Intelligence Pre-training Report */}
          {intelligence && (
            <div className="bg-[#171630]/80 p-4 rounded-xl border border-[#8265FF]/20 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.06] pb-2">
                <span className="text-xs font-semibold text-white flex items-center gap-2">
                  <FiActivity className="text-[#8265FF]" /> Dataset Pre-Training Intelligence
                </span>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="px-2 py-0.5 rounded-full bg-[#8265FF]/20 text-[#8265FF] border border-[#8265FF]/30 font-medium">
                    {intelligence.dimensions?.rows} rows × {intelligence.dimensions?.columns} cols
                  </span>
                  {intelligence.target_analysis?.imbalance_severity && (
                    <span
                      className={`px-2 py-0.5 rounded-full font-medium border ${
                        intelligence.target_analysis.imbalance_severity === "severe"
                          ? "bg-rose-500/10 text-rose-400 border-rose-500/30"
                          : intelligence.target_analysis.imbalance_severity === "moderate"
                          ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                          : "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                      }`}
                    >
                      Imbalance: {intelligence.target_analysis.imbalance_severity} ({intelligence.target_analysis.imbalance_ratio}:1)
                    </span>
                  )}
                </div>
              </div>

              {intelligence.recommendations && intelligence.recommendations.length > 0 && (
                <div className="space-y-1">
                  <p className="text-[11px] font-semibold text-[#8265FF] uppercase tracking-wider">
                    Automated Pre-Training Recommendations:
                  </p>
                  <ul className="list-disc list-inside text-xs text-[#A4A4B8] space-y-0.5">
                    {intelligence.recommendations.slice(0, 3).map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {/* Mode Selector: Auto vs Advanced */}
      {!submit && csvFile && (
        <section className="bg-[#111126] p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/[0.06] pb-4">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <FiSliders className="text-[#8265FF]" /> Training Mode
              </h3>
              <p className="text-xs text-[#A4A4B8]">
                Choose between dataset-adaptive automation and granular engineer control.
              </p>
            </div>
            <div className="inline-flex rounded-xl bg-[#171630] p-1 border border-white/[0.08]">
              <button
                type="button"
                onClick={() => setAutoMode(true)}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition ${
                  autoMode
                    ? "bg-[#8265FF] text-white shadow-md shadow-[#8265FF]/30"
                    : "text-[#A4A4B8] hover:text-white"
                }`}
              >
                <FiZap /> Auto Mode (Recommended)
              </button>
              <button
                type="button"
                onClick={() => setAutoMode(false)}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition ${
                  !autoMode
                    ? "bg-[#8265FF] text-white shadow-md shadow-[#8265FF]/30"
                    : "text-[#A4A4B8] hover:text-white"
                }`}
              >
                <FiSliders /> Advanced Mode
              </button>
            </div>
          </div>

          {autoMode ? (
            <div className="bg-[#171630] border border-white/[0.06] p-5 rounded-xl space-y-3">
              <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold">
                <FiCheckCircle /> Intelligent Dataset-Adaptive Presets Active
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-[#A4A4B8]">
                <div className="bg-[#111126] p-3 rounded-lg border border-white/[0.04]">
                  <span className="text-[#8265FF] font-semibold block mb-1">Recommended Estimator</span>
                  <span className="text-white font-mono">{model}</span>
                </div>
                <div className="bg-[#111126] p-3 rounded-lg border border-white/[0.04]">
                  <span className="text-[#8265FF] font-semibold block mb-1">Preprocessing Pipeline</span>
                  <span className="text-white">Leak-free Median Imputation + {scaler} Scaler</span>
                </div>
                <div className="bg-[#111126] p-3 rounded-lg border border-white/[0.04]">
                  <span className="text-[#8265FF] font-semibold block mb-1">Evaluation Methodology</span>
                  <span className="text-white">5-Fold Stratified CV + 20% Untouched Holdout</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4 pt-2">
              <Model model={model} setModel={setModel} taskType={taskType} />
              <Preprocess
                missingStrategy={missingStrategy}
                setMissingStrategy={setMissingStrategy}
                encodingStrategy={encodingStrategy}
                setEncodingStrategy={setEncodingStrategy}
                scaler={scaler}
                setScaler={setScaler}
                splitRatio={splitRatio}
                setSplitRatio={setSplitRatio}
              />
            </div>
          )}

          <div className="flex justify-end pt-2">
            <button
              onClick={handleSubmit}
              className="px-8 py-3.5 rounded-xl font-bold text-sm bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-xl shadow-[#8265FF]/25 flex items-center gap-2 transition-all cursor-pointer"
            >
              <FiBarChart2 />
              Train Model & Verify Generalization
            </button>
          </div>
        </section>
      )}

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
          <FiAlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Step 4: Results Display */}
      {submit && (
        <Result
          csv={csvFile}
          taskType={taskType}
          targetColumn={targetColumn}
          model={model}
          scaler={scaler}
          splitRatio={splitRatio}
          missing={missingStrategy}
          encoding={encodingStrategy}
          onReset={handleReset}
        />
      )}

    </div>
  );
}

export default AutoML;
