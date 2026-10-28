import React from "react";
import { FiTrendingUp, FiCheckCircle, FiLayers } from "react-icons/fi";
import { Link } from "react-router-dom";

const ClassificationHoldout = ({ holdout }) => (
  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div className="p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
      <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">Accuracy</p>
      <p className="text-xl font-bold font-mono text-emerald-400 mt-1">
        {holdout.accuracy != null ? `${(holdout.accuracy * 100).toFixed(1)}%` : "N/A"}
      </p>
      <p className="text-[10px] text-[#A4A4B8] mt-1">Overall correctness</p>
    </div>
    <div className="p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
      <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">F1 Macro</p>
      <p className="text-xl font-bold font-mono text-[#8265FF] mt-1">
        {holdout.f1_macro != null ? holdout.f1_macro.toFixed(4) : "N/A"}
      </p>
      <p className="text-[10px] text-[#A4A4B8] mt-1">Unweighted harmonic mean</p>
    </div>
    <div className="p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
      <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">Balanced Accuracy</p>
      <p className="text-xl font-bold font-mono text-white mt-1">
        {holdout.balanced_accuracy != null ? `${(holdout.balanced_accuracy * 100).toFixed(1)}%` : "N/A"}
      </p>
      <p className="text-[10px] text-[#A4A4B8] mt-1">Per-class recall average</p>
    </div>
    <div className="p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
      <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">ROC-AUC</p>
      <p className="text-xl font-bold font-mono text-white mt-1">
        {holdout.roc_auc != null ? holdout.roc_auc.toFixed(4) : "N/A"}
      </p>
      <p className="text-[10px] text-[#A4A4B8] mt-1">Discrimination power</p>
    </div>
  </div>
);

const RegressionHoldout = ({ holdout }) => (
  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div className="p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
      <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">RMSE</p>
      <p className="text-xl font-bold font-mono text-emerald-400 mt-1">
        {holdout.rmse != null ? holdout.rmse.toFixed(4) : "N/A"}
      </p>
      <p className="text-[10px] text-[#A4A4B8] mt-1">Root Mean Squared Error</p>
    </div>
    <div className="p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
      <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">R² Score</p>
      <p className="text-xl font-bold font-mono text-[#8265FF] mt-1">
        {holdout.r2 != null ? holdout.r2.toFixed(4) : "N/A"}
      </p>
      <p className="text-[10px] text-[#A4A4B8] mt-1">Variance explained</p>
    </div>
    <div className="p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
      <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">MAE</p>
      <p className="text-xl font-bold font-mono text-white mt-1">
        {holdout.mae != null ? holdout.mae.toFixed(4) : "N/A"}
      </p>
      <p className="text-[10px] text-[#A4A4B8] mt-1">Mean Absolute Error</p>
    </div>
    <div className="p-4 rounded-xl bg-[#0D0D1B] border border-white/[0.06]">
      <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">Max Error</p>
      <p className="text-xl font-bold font-mono text-white mt-1">
        {holdout.max_error != null ? holdout.max_error.toFixed(4) : "N/A"}
      </p>
      <p className="text-[10px] text-[#A4A4B8] mt-1">Worst-case deviation</p>
    </div>
  </div>
);

const BestModelSummary = ({
  bestModel,
  rankingMetric = "",
  cvFolds = 5,
  taskType = "regression",
  classes = [],
  warnings = [],
  dataQuality = null,
  registeredModel = null,
}) => {
  if (!bestModel) return null;

  const holdout = bestModel.holdout_metrics || {};
  const metricLabel = rankingMetric
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  const isClassification = taskType === "classification";

  return (
    <div className="space-y-6">
      {/* Winner Spotlight Card */}
      <section className="bg-gradient-to-tr from-[#111126] via-[#171630] to-[#0D0D1B] rounded-2xl border border-[#8265FF]/40 p-6 shadow-2xl glow-purple relative overflow-hidden">
        <div className="relative z-10">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#8265FF]/20 text-[#8265FF] border border-[#8265FF]/30">
                  Winning Algorithm
                </span>
                <span className="text-xs text-[#A4A4B8]">
                  Across {cvFolds}-fold cross validation
                </span>
              </div>
              <h3 className="text-2xl font-extrabold text-white flex items-center gap-2">
                🏆 {bestModel.model}
              </h3>
            </div>

            {registeredModel && (
              <div className="flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/30 px-3.5 py-2 rounded-xl">
                <FiCheckCircle className="text-emerald-400 w-4 h-4" />
                <div className="text-left text-xs">
                  <span className="font-semibold text-emerald-300 block">Saved to Model Registry</span>
                  <Link to={`/predict?model=${registeredModel.id}`} className="text-[#8265FF] hover:underline font-medium">
                    Test Live Prediction →
                  </Link>
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-[#070711]/60 border border-white/[0.08] rounded-xl p-3.5">
              <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">
                CV Mean ({metricLabel})
              </p>
              <p className="text-2xl font-bold font-mono text-emerald-400 mt-1">
                {bestModel.cv_mean != null ? bestModel.cv_mean.toFixed(4) : "N/A"}
              </p>
            </div>
            <div className="bg-[#070711]/60 border border-white/[0.08] rounded-xl p-3.5">
              <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">
                CV Std Deviation
              </p>
              <p className="text-2xl font-bold font-mono text-white mt-1">
                ± {bestModel.cv_std != null ? bestModel.cv_std.toFixed(4) : "N/A"}
              </p>
            </div>
            <div className="bg-[#070711]/60 border border-white/[0.08] rounded-xl p-3.5">
              <p className="text-[10px] uppercase tracking-wider text-[#A4A4B8]">
                Validation Folds
              </p>
              <p className="text-2xl font-bold font-mono text-[#8265FF] mt-1">
                {cvFolds}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Holdout Test Metrics */}
      <section className="bg-[#111126] rounded-2xl border border-white/[0.08] p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <FiTrendingUp className="text-[#8265FF]" />
            Holdout Test Metrics — {bestModel.model}
          </h3>
          <span className="text-xs text-[#A4A4B8]">Evaluated on unseen test partition</span>
        </div>

        {isClassification ? (
          <ClassificationHoldout holdout={holdout} />
        ) : (
          <RegressionHoldout holdout={holdout} />
        )}
      </section>

      {/* Confusion Matrix (classification only) */}
      {isClassification && holdout.confusion_matrix && (
        <section className="bg-[#111126] rounded-2xl border border-white/[0.08] p-6 shadow-xl space-y-4">
          <div className="border-b border-white/[0.06] pb-3">
            <h3 className="text-base font-bold text-white">Confusion Matrix</h3>
            <p className="text-xs text-[#A4A4B8] mt-0.5">Rows = True Class, Columns = Predicted Class</p>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full border border-white/[0.08] text-center text-xs divide-y divide-white/[0.08]">
              <thead className="bg-[#0D0D1B]">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-[#A4A4B8]">True \ Pred</th>
                  {holdout.confusion_matrix.labels.map((lbl) => (
                    <th key={lbl} className="px-4 py-3 font-semibold text-white uppercase">{lbl}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {holdout.confusion_matrix.matrix.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <td className="px-4 py-3 font-bold text-left text-white bg-[#0D0D1B]/60">
                      {holdout.confusion_matrix.labels[rIdx]}
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
        </section>
      )}

      {/* Warnings & Data Quality Note */}
      {warnings && warnings.length > 0 && (
        <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs space-y-1">
          <span className="font-bold">Evaluation Warnings:</span>
          <ul className="list-disc list-inside space-y-0.5">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default BestModelSummary;
