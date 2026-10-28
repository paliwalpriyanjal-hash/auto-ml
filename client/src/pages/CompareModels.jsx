import React, { useState } from "react";
import Papa from "papaparse";
import Preprocess from "../components/PreprocessingOptions";
import ComparisonResultTable from "../components/ComparisonResultTable";
import BestModelSummary from "../components/BestModelSummary";
import {
  FiUpload,
  FiSettings,
  FiCpu,
  FiBarChart2,
  FiCheckCircle,
  FiLayers,
  FiRefreshCw,
  FiAlertCircle
} from "react-icons/fi";
import { getApiBase } from "../utils/api";

const CLASSIFICATION_METRICS = [
  { value: "f1_macro", label: "F1 Macro", higherIsBetter: true },
  { value: "balanced_accuracy", label: "Balanced Accuracy", higherIsBetter: true },
  { value: "accuracy", label: "Accuracy", higherIsBetter: true },
  { value: "f1_weighted", label: "F1 Weighted", higherIsBetter: true },
];

const REGRESSION_METRICS = [
  { value: "rmse", label: "RMSE", higherIsBetter: false },
  { value: "mae", label: "MAE", higherIsBetter: false },
  { value: "r2", label: "R² Score", higherIsBetter: true },
];

function CompareModels() {
  const [csvFile, setCsvFile] = useState(null);
  const [previewData, setPreviewData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [activeStep, setActiveStep] = useState(1);

  const [taskType, setTaskType] = useState("regression");
  const [targetColumn, setTargetColumn] = useState("");
  const [rankingMetric, setRankingMetric] = useState("rmse");

  const [missingStrategy, setMissingStrategy] = useState("drop");
  const [encodingStrategy, setEncodingStrategy] = useState("label");
  const [scaler, setScaler] = useState("standard");
  const [splitRatio, setSplitRatio] = useState("0.8");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [comparisonData, setComparisonData] = useState(null);

  const handleTaskTypeChange = (newTask) => {
    setTaskType(newTask);
    if (newTask === "classification") {
      setRankingMetric("f1_macro");
      setEncodingStrategy("onehot");
    } else {
      setRankingMetric("rmse");
      setEncodingStrategy("label");
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setCsvFile(file);
      setActiveStep(2);
      const reader = new FileReader();
      reader.onload = (evt) => {
        Papa.parse(evt.target.result, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const data = results.data;
            const fields = results.meta.fields || [];
            setColumns(fields);
            setPreviewData(data.slice(0, 5));
            if (fields.length > 0) {
              setTargetColumn(fields[fields.length - 1]);
            }
          },
        });
      };
      reader.readAsText(file);
    }
  };

  const handleSubmit = async () => {
    if (!targetColumn) {
      setError("Please select a target column.");
      return;
    }

    setLoading(true);
    setError("");
    setComparisonData(null);
    setActiveStep(4);

    const apiBase = getApiBase();
    const formData = new FormData();
    formData.append("file", csvFile);
    formData.append("task_type", taskType);
    formData.append("target_column", targetColumn);
    formData.append("scaler", scaler);
    formData.append("splitRatio", splitRatio);
    formData.append("missing", missingStrategy);
    formData.append("encoding", encodingStrategy);
    if (rankingMetric) {
      formData.append("ranking_metric", rankingMetric);
    }

    try {
      const res = await fetch(`${apiBase}/compare`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with ${res.status}`);
      }

      const data = await res.json();
      setComparisonData(data);
    } catch (err) {
      setError(err.message || "Failed to compare models.");
    } finally {
      setLoading(false);
    }
  };

  const currentMetricOptions =
    taskType === "classification" ? CLASSIFICATION_METRICS : REGRESSION_METRICS;
  const currentMetricObj = currentMetricOptions.find(
    (m) => m.value === rankingMetric
  );
  const higherIsBetter = currentMetricObj ? currentMetricObj.higherIsBetter : true;

  return (
    <div className="min-h-screen pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
      
      {/* Title */}
      <div className="border-b border-white/[0.08] pb-6">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-[#8265FF] bg-[#8265FF]/10 px-2 py-0.5 rounded-full border border-[#8265FF]/20">
            Automated Cross-Validation Benchmarking
          </span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Multi-Algorithm Comparison</h1>
        <p className="text-sm text-[#A4A4B8] mt-1">
          Train and evaluate multiple candidate models simultaneously. The highest-ranking model is verified on holdout data and registered automatically.
        </p>
      </div>

      {/* Progress Steps Header */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { num: 1, title: "1. Dataset Upload", icon: FiUpload },
          { num: 2, title: "2. Task & Target", icon: FiSettings },
          { num: 3, title: "3. Preprocessing", icon: FiCpu },
          { num: 4, title: "4. Leaderboard", icon: FiBarChart2 },
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

      {/* Step 2: Task Configuration */}
      {csvFile && (
        <section className="bg-[#111126] p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-4">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <FiSettings className="text-[#A855F7]" /> Task & Target Configuration
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
                ML Task Type
              </label>
              <select
                value={taskType}
                onChange={(e) => handleTaskTypeChange(e.target.value)}
                className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                <option value="regression">Regression (Continuous values)</option>
                <option value="classification">Classification (Discrete classes)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
                Target Column (y)
              </label>
              <select
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                <option value="">Select target...</option>
                {columns.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
                Leaderboard Ranking Metric
              </label>
              <select
                value={rankingMetric}
                onChange={(e) => setRankingMetric(e.target.value)}
                className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                {currentMetricOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label} ({opt.higherIsBetter ? "↑ higher is better" : "↓ lower is better"})
                  </option>
                ))}
              </select>
            </div>
          </div>
        </section>
      )}

      {/* Step 3: Preprocessing Setup */}
      {csvFile && (
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
      )}

      {/* Run Action */}
      {csvFile && (
        <div className="flex justify-end">
          <button
            onClick={handleSubmit}
            disabled={loading || !targetColumn}
            className="px-8 py-3.5 rounded-xl font-bold text-sm bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-xl shadow-[#8265FF]/25 flex items-center gap-2 disabled:opacity-50 transition-all"
          >
            {loading ? (
              <>
                <FiRefreshCw className="animate-spin" />
                Training & Cross-Validating Models...
              </>
            ) : (
              <>
                <FiBarChart2 />
                Run Multi-Algorithm Benchmark
              </>
            )}
          </button>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
          <FiAlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Benchmark Results Display */}
      {comparisonData && (
        <div className="space-y-8 animate-fade-in">
          <ComparisonResultTable
            modelResults={comparisonData.model_results}
            rankingMetric={comparisonData.ranking_metric || rankingMetric}
            higherIsBetter={higherIsBetter}
          />

          <BestModelSummary
            bestModel={comparisonData.best_model}
            rankingMetric={comparisonData.ranking_metric || rankingMetric}
            cvFolds={comparisonData.cv_folds || 5}
            taskType={comparisonData.task_type || taskType}
            classes={comparisonData.classes || []}
            warnings={comparisonData.warnings || []}
            dataQuality={comparisonData.data_quality}
            registeredModel={comparisonData.registered_model}
          />
        </div>
      )}

    </div>
  );
}

export default CompareModels;
