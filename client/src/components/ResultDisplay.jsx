import React, { useCallback, useEffect, useState } from "react";
import {
  FiBarChart2,
  FiTrendingUp,
  FiCheck,
  FiAlertTriangle,
  FiDownload,
  FiRefreshCw,
  FiLoader,
  FiCheckCircle,
} from "react-icons/fi";
import { Link } from "react-router-dom";
import Dashboard from "./Dashboard";
import { getApiBase } from "../utils/api";

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "N/A";
}

function formatPercent(value, digits = 2) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "N/A";
}

function escapeCsvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function MetricCard({ title, value, description, color = "purple" }) {
  const colors = {
    purple: "bg-[#171630] border-[#8265FF]/40 text-[#8265FF]",
    green: "bg-[#171630] border-emerald-500/40 text-emerald-400",
    blue: "bg-[#171630] border-cyan-500/40 text-cyan-400",
    amber: "bg-[#171630] border-amber-500/40 text-amber-400",
    rose: "bg-[#171630] border-rose-500/40 text-rose-400",
    indigo: "bg-[#171630] border-indigo-500/40 text-indigo-400",
    emerald: "bg-[#171630] border-emerald-500/40 text-emerald-400",
    red: "bg-[#171630] border-rose-500/40 text-rose-400",
  };

  return (
    <div className={`rounded-xl border p-4 ${colors[color] || colors.purple} shadow-md`}>
      <h4 className="mb-1 text-xs font-semibold uppercase tracking-wider text-[#A4A4B8]">{title}</h4>
      <p className="text-2xl font-bold font-mono tracking-tight text-white">{value}</p>
      {description && <p className="mt-1 text-[11px] text-[#A4A4B8]">{description}</p>}
    </div>
  );
}

function Result({
  csv,
  taskType = "regression",
  targetColumn = "",
  model,
  scaler,
  splitRatio,
  missing,
  encoding,
  onReset,
}) {
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [retryCount, setRetryCount] = useState(0);

  const runTraining = useCallback(
    async (signal) => {
      if (!(csv instanceof File)) {
        setError("Please upload a valid CSV file before training.");
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");
      setResponse(null);

      const formData = new FormData();
      formData.append("file", csv);
      formData.append("model", model || "");
      formData.append("scaler", scaler || "standard");
      formData.append("splitRatio", String(splitRatio ?? 0.8));
      formData.append("missing", missing || "median");
      formData.append("encoding", encoding || "onehot");
      formData.append("task_type", taskType || "regression");

      if (taskType === "classification") {
        if (!targetColumn) {
          setError("Select a target column before training a classifier.");
          setLoading(false);
          return;
        }
        formData.append("target_column", targetColumn);
      }

      try {
        const apiBase = getApiBase();
        const res = await fetch(`${apiBase}/train`, {
          method: "POST",
          body: formData,
          signal,
        });

        const contentType = res.headers.get("content-type") || "";
        let data;

        if (contentType.includes("application/json")) {
          data = await res.json();
        } else {
          const bodyText = await res.text();
          data = { detail: bodyText };
        }

        if (!res.ok) {
          const detail = data?.detail;
          const message = Array.isArray(detail)
            ? detail.map((item) => item.msg || JSON.stringify(item)).join("; ")
            : typeof detail === "string"
            ? detail
            : `Training request failed with HTTP ${res.status}.`;
          throw new Error(message);
        }

        if (!data || typeof data !== "object") {
          throw new Error("The server returned an invalid response.");
        }

        setResponse(data);
      } catch (err) {
        if (err.name === "AbortError") return;
        console.error("AutoML training request failed:", err);
        setError(err.message || "Failed to train model.");
      } finally {
        setLoading(false);
      }
    },
    [csv, model, scaler, splitRatio, missing, encoding, taskType, targetColumn]
  );

  useEffect(() => {
    const controller = new AbortController();
    runTraining(controller.signal);
    return () => controller.abort();
  }, [runTraining, retryCount]);

  const handleRetry = () => {
    setRetryCount((count) => count + 1);
  };

  const handleDownload = () => {
    if (!response?.columns || !Array.isArray(response?.rows)) {
      setError("No processed data is available to download.");
      return;
    }

    const columns = response.columns;
    const lines = [
      columns.map(escapeCsvCell).join(","),
      ...response.rows.map((row) =>
        columns.map((column) => escapeCsvCell(row?.[column])).join(",")
      ),
    ];

    const blob = new Blob(["\uFEFF", lines.join("\r\n")], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${String(model || "model").replace(/[^a-z0-9_-]/gi, "_")}_results.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  if (loading) {
    return (
      <section className="bg-[#111126] border border-white/[0.08] rounded-2xl p-16 flex flex-col items-center justify-center text-center space-y-4">
        <FiLoader className="h-10 w-10 animate-spin text-[#8265FF]" />
        <h3 className="text-xl font-bold text-white">Fitting Pipeline & Estimators...</h3>
        <p className="text-xs text-[#A4A4B8] max-w-sm leading-relaxed">
          Preprocessing dataset, applying {scaler} scaling, handling missing values, and validating holdout partitions.
        </p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-2xl border border-rose-500/30 bg-rose-950/20 p-6 space-y-4">
        <div className="flex items-start gap-3">
          <FiAlertTriangle className="mt-1 h-5 w-5 shrink-0 text-rose-400" />
          <div className="flex-1">
            <h3 className="text-sm font-bold text-rose-300">Training Pipeline Failed</h3>
            <p className="mt-1 text-xs text-rose-200/90 leading-relaxed font-mono whitespace-pre-wrap">{error}</p>
            <div className="mt-4 flex gap-3">
              <button
                type="button"
                onClick={handleRetry}
                className="rounded-xl bg-rose-500 hover:bg-rose-600 px-4 py-2 text-xs font-semibold text-white transition-colors"
              >
                Retry Run
              </button>
              {onReset && (
                <button
                  type="button"
                  onClick={onReset}
                  className="rounded-xl bg-[#171630] border border-white/[0.08] px-4 py-2 text-xs font-semibold text-[#A4A4B8] hover:text-white"
                >
                  Adjust Parameters
                </button>
              )}
            </div>
          </div>
        </div>
      </section>
    );
  }

  if (!response) return null;

  const metrics = response.metrics || {};
  const isClassification = (response.task_type || taskType) === "classification";
  const confusion = response.confusion_matrix;
  const registeredModel = response.registered_model;

  const metricCards = isClassification
    ? [
        {
          title: "Holdout Accuracy",
          value: formatPercent(metrics.accuracy),
          description: "Overall correctness on unseen split",
          color: "green",
        },
        {
          title: "Balanced Accuracy",
          value: formatPercent(metrics.balanced_accuracy),
          description: "Average recall across classes",
          color: "purple",
        },
        {
          title: "Macro F1 Score",
          value: formatNumber(metrics.f1_macro),
          description: "Unweighted harmonic mean",
          color: "blue",
        },
        {
          title: "Weighted F1",
          value: formatNumber(metrics.f1_weighted),
          description: "Weighted by class support",
          color: "indigo",
        },
        {
          title: "Macro Precision",
          value: formatPercent(metrics.precision_macro),
          description: "Average positive predictive value",
          color: "purple",
        },
        {
          title: "Macro Recall",
          value: formatPercent(metrics.recall_macro),
          description: "Average sensitivity across classes",
          color: "purple",
        },
      ]
    : [
        {
          title: "Root Mean Squared Error",
          value: formatNumber(metrics.rmse),
          description: "Quadratic mean of residuals",
          color: "green",
        },
        {
          title: "R² Coefficient",
          value: formatNumber(metrics.r2),
          description: "Variance explained by model",
          color: "purple",
        },
        {
          title: "Mean Absolute Error",
          value: formatNumber(metrics.mae),
          description: "Average absolute difference",
          color: "blue",
        },
        {
          title: "Explained Variance",
          value: formatNumber(metrics.explained_variance),
          description: "Proportion of dataset variance",
          color: "indigo",
        },
        {
          title: "Max Error",
          value: formatNumber(metrics.max_error),
          description: "Worst-case residual deviation",
          color: "rose",
        },
      ];

  return (
    <section className="overflow-hidden rounded-2xl bg-[#111126] border border-white/[0.08] shadow-2xl space-y-6">
      
      {/* Header Banner */}
      <header className="bg-gradient-to-tr from-[#171630] to-[#111126] border-b border-white/[0.08] p-6 text-white flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400">
              Training Pipeline Succeeded
            </span>
          </div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            {response.model_type || model || "Pipeline"}
          </h2>
          <p className="text-xs text-[#A4A4B8] mt-1">
            Task: <b className="text-white capitalize">{isClassification ? "Classification" : "Regression"}</b>
            {response.target_column ? ` • Target: ${response.target_column}` : ""}
          </p>
        </div>

        {registeredModel && (
          <div className="flex flex-wrap items-center gap-2 bg-[#8265FF]/10 border border-[#8265FF]/30 p-2.5 rounded-xl">
            <FiCheckCircle className="text-[#8265FF] w-4 h-4 shrink-0" />
            <div className="text-left text-xs mr-2">
              <span className="font-semibold text-white block">Registered #{registeredModel.id}</span>
            </div>
            <Link
              to={`/predict?model=${registeredModel.id}`}
              className="px-2.5 py-1 text-[11px] font-semibold rounded-lg bg-[#8265FF]/20 border border-[#8265FF]/40 text-[#8265FF] hover:bg-[#8265FF]/30 transition"
            >
              Predict Online →
            </Link>
            <Link
              to={`/explain?model=${registeredModel.id}`}
              className="px-2.5 py-1 text-[11px] font-semibold rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/30 transition"
            >
              Explainability →
            </Link>
            <Link
              to={`/reports?model=${registeredModel.id}`}
              className="px-2.5 py-1 text-[11px] font-semibold rounded-lg bg-cyan-500/20 border border-cyan-500/40 text-cyan-400 hover:bg-cyan-500/30 transition"
            >
              Report Card →
            </Link>
          </div>
        )}
      </header>

      {/* Validation & Generalization Hierarchy */}
      <div className="p-6 pt-0 space-y-4">
        <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[#A4A4B8] flex items-center gap-2">
            <FiTrendingUp className="text-[#8265FF]" /> Validation & Generalization Hierarchy
          </h3>
          {response.overfitting_gap !== undefined && (
            <span
              className={`text-xs px-2.5 py-0.5 rounded-full border font-mono ${
                Math.abs(response.overfitting_gap) > 0.15
                  ? "bg-rose-500/10 border-rose-500/30 text-rose-400"
                  : Math.abs(response.overfitting_gap) > 0.08
                  ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                  : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
              }`}
            >
              Overfitting Gap: {formatPercent(response.overfitting_gap)}
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#171630] border border-white/[0.08] p-4 rounded-xl">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[#A4A4B8] block mb-1">
              1. Training Fit Metric
            </span>
            <p className="text-xl font-bold font-mono text-white">
              {isClassification
                ? formatPercent(response.training_metrics?.accuracy)
                : formatNumber(response.training_metrics?.r2)}
            </p>
            <p className="text-[11px] text-[#A4A4B8] mt-1">
              {isClassification ? "Accuracy on Training Split" : "R² Score on Training Split"}
            </p>
          </div>

          <div className="bg-[#171630] border border-[#8265FF]/40 p-4 rounded-xl shadow-lg shadow-[#8265FF]/5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[#8265FF] block mb-1">
              2. K-Fold Cross-Validation
            </span>
            <p className="text-xl font-bold font-mono text-white">
              {isClassification
                ? response.cv_metrics?.accuracy_mean !== undefined
                  ? `${formatPercent(response.cv_metrics.accuracy_mean)} ± ${formatPercent(response.cv_metrics.accuracy_std)}`
                  : "N/A"
                : response.cv_metrics?.r2_mean !== undefined
                ? `${formatNumber(response.cv_metrics.r2_mean)} ± ${formatNumber(response.cv_metrics.r2_std)}`
                : "N/A"}
            </p>
            <p className="text-[11px] text-[#A4A4B8] mt-1">
              Mean CV Score ({response.cv_metrics?.cv_folds || 5} Folds on Training Data)
            </p>
          </div>

          <div className="bg-[#171630] border border-emerald-500/40 p-4 rounded-xl">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-400 block mb-1">
              3. Untouched Holdout Test
            </span>
            <p className="text-xl font-bold font-mono text-white">
              {isClassification
                ? formatPercent(metrics.accuracy)
                : formatNumber(metrics.r2)}
            </p>
            <p className="text-[11px] text-[#A4A4B8] mt-1">
              Final Unseen Split ({response.data_quality?.testing_samples || 0} held-out samples)
            </p>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="p-6 pt-0 space-y-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-300 flex items-center gap-2">
          <FiBarChart2 className="text-[#8265FF]" /> Performance Metrics
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {metricCards.map((metric) => (
            <MetricCard key={metric.title} {...metric} />
          ))}

          {isClassification && metrics.roc_auc != null && (
            <MetricCard
              title="ROC-AUC"
              value={formatNumber(metrics.roc_auc)}
              description="Area under the ROC curve"
              color="amber"
            />
          )}

          {isClassification && metrics.log_loss != null && (
            <MetricCard
              title="Log Loss"
              value={formatNumber(metrics.log_loss)}
              description="Lower indicates calibrated probabilities"
              color="rose"
            />
          )}
        </div>
      </div>

      {/* Confusion Matrix (classification) */}
      {isClassification && confusion && Array.isArray(confusion.labels) && (
        <div className="p-6 border-t border-white/[0.06] space-y-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-300">
            Confusion Matrix (Holdout Split)
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full border border-white/[0.08] text-center text-xs divide-y divide-white/[0.06]">
              <thead className="bg-[#0D0D1B]">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-[#A4A4B8]">True \ Pred</th>
                  {confusion.labels.map((lbl) => (
                    <th key={lbl} className="px-4 py-3 font-semibold text-white uppercase">{lbl}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {confusion.matrix.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <td className="px-4 py-3 font-bold text-left text-white bg-[#0D0D1B]/60">
                      {confusion.labels[rIdx]}
                    </td>
                    {row.map((cell, cIdx) => (
                      <td
                        key={cIdx}
                        className={`px-4 py-3 font-mono font-bold ${
                          rIdx === cIdx ? "bg-emerald-500/10 text-emerald-400" : "text-[#A4A4B8]"
                        }`}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Processed Data Preview */}
      {Array.isArray(response.columns) && Array.isArray(response.rows) && response.rows.length > 0 && (
        <div className="p-6 border-t border-white/[0.06] space-y-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-300">
            Engineered Features Preview
          </h3>
          <div className="overflow-x-auto border border-white/[0.06] rounded-xl">
            <table className="min-w-full text-xs text-left divide-y divide-white/[0.06]">
              <thead className="bg-[#0D0D1B]">
                <tr>
                  {response.columns.slice(0, 10).map((col) => (
                    <th key={col} className="px-4 py-3 text-[#A4A4B8] font-medium uppercase">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {response.rows.slice(0, 5).map((row, rI) => (
                  <tr key={rI} className="hover:bg-white/[0.02]">
                    {response.columns.slice(0, 10).map((col) => (
                      <td key={col} className="px-4 py-2 font-mono text-[#F7F7FC]">{String(row?.[col] ?? "")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Diagnostics & Generalization Assessment */}
      {response.diagnostics && (
        <div className="p-6 border-t border-white/[0.06] space-y-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-gray-300 flex items-center gap-2">
            <FiAlertTriangle className="text-amber-400" /> Model Intelligence & Failure Diagnostics
          </h3>
          <div className="bg-[#171630] border border-white/[0.08] p-4 rounded-xl space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/[0.06] pb-2">
              <span className="text-xs font-semibold text-white">
                Generalization Status:{" "}
                <span className="capitalize text-emerald-400 font-mono">
                  {response.diagnostics.status || "Healthy"}
                </span>
              </span>
              <span className="text-xs text-[#A4A4B8]">
                {response.diagnostics.summary}
              </span>
            </div>

            {response.imbalance_info && (
              <div className="text-xs bg-[#111126] p-3 rounded-lg border border-white/[0.06] flex items-center justify-between">
                <span className="text-[#A4A4B8]">Class Imbalance Ratio:</span>
                <span className="font-mono text-white font-bold">
                  {response.imbalance_info.imbalance_ratio}:1 ({response.imbalance_info.imbalance_severity})
                </span>
                <span className="text-emerald-400 text-[11px]">
                  Applied: {response.imbalance_info.applied_strategy}
                </span>
              </div>
            )}

            {Array.isArray(response.diagnostics.findings) && response.diagnostics.findings.length > 0 && (
              <div className="space-y-1">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-amber-400">Diagnostic Findings:</p>
                <ul className="list-disc list-inside text-xs text-[#A4A4B8] space-y-0.5">
                  {response.diagnostics.findings.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </div>
            )}

            {Array.isArray(response.diagnostics.recommendations) && response.diagnostics.recommendations.length > 0 && (
              <div className="space-y-1">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-[#8265FF]">Actionable Recommendations:</p>
                <ul className="list-disc list-inside text-xs text-[#F7F7FC] space-y-0.5">
                  {response.diagnostics.recommendations.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Action Footer */}
      <footer className="flex flex-wrap items-center justify-between gap-4 bg-[#0D0D1B] border-t border-white/[0.08] px-6 py-4">
        <button
          type="button"
          onClick={handleDownload}
          disabled={!Array.isArray(response.columns) || !Array.isArray(response.rows)}
          className="flex items-center gap-2 rounded-xl bg-[#171630] border border-white/[0.08] hover:bg-[#1f1e40] px-4 py-2 text-xs font-semibold text-white transition-all disabled:opacity-40"
        >
          <FiDownload /> Download Processed Results CSV
        </button>

        <button
          type="button"
          onClick={onReset}
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] px-5 py-2 text-xs font-semibold text-white transition-all shadow-md shadow-[#8265FF]/20"
        >
          <FiRefreshCw /> Configure New Experiment
        </button>
      </footer>
    </section>
  );
}

export default Result;
