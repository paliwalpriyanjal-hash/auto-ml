import React from 'react';
import { FiTrendingUp, FiAlertTriangle, FiInfo } from 'react-icons/fi';

/**
 * Renders the best-model summary card with holdout metrics.
 *
 * Props:
 *   bestModel      – { model, cv_mean, cv_std, holdout_metrics: {...} }
 *   rankingMetric  – string (e.g. "f1_macro")
 *   cvFolds        – number
 *   taskType       – "classification" | "regression"
 *   classes        – array of class labels (classification only)
 *   warnings       – array of warning strings
 *   dataQuality    – data_quality object from backend response
 */
const BestModelSummary = ({
  bestModel,
  rankingMetric = '',
  cvFolds = 5,
  taskType = 'regression',
  classes = [],
  warnings = [],
  dataQuality = null,
}) => {
  if (!bestModel) return null;

  const holdout = bestModel.holdout_metrics || {};
  const metricLabel = rankingMetric.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  const isClassification = taskType === 'classification';

  return (
    <div className="space-y-6">
      {/* Winner Card */}
      <section className="bg-gradient-to-r from-yellow-600 to-gray-600 rounded-xl shadow-lg overflow-hidden text-white">
        <div className="p-6">
          <h3 className="text-2xl font-bold flex items-center">
            🏆 Best Model: {bestModel.model}
          </h3>
          <p className="opacity-90 mt-1 text-sm">
            Ranked #1 by {metricLabel} across {cvFolds}-fold cross-validation
          </p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
            <div className="bg-white/20 rounded-lg p-3">
              <p className="text-xs uppercase opacity-75">CV Mean ({metricLabel})</p>
              <p className="text-xl font-bold font-mono">
                {bestModel.cv_mean != null ? bestModel.cv_mean.toFixed(4) : 'N/A'}
              </p>
            </div>
            <div className="bg-white/20 rounded-lg p-3">
              <p className="text-xs uppercase opacity-75">CV Std Dev</p>
              <p className="text-xl font-bold font-mono">
                ± {bestModel.cv_std != null ? bestModel.cv_std.toFixed(4) : 'N/A'}
              </p>
            </div>
            <div className="bg-white/20 rounded-lg p-3">
              <p className="text-xs uppercase opacity-75">CV Folds</p>
              <p className="text-xl font-bold">{cvFolds}</p>
            </div>
          </div>
        </div>
      </section>

      {/* Holdout Metrics */}
      <section className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-xl font-semibold text-gray-800 flex items-center">
            <FiTrendingUp className="text-purple-500 mr-2" />
            Holdout Test Metrics — {bestModel.model}
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            Evaluated on the unseen test split
          </p>
        </div>

        <div className="p-6">
          {isClassification ? (
            <ClassificationHoldout holdout={holdout} />
          ) : (
            <RegressionHoldout holdout={holdout} />
          )}
        </div>
      </section>

      {/* Confusion Matrix (classification only) */}
      {isClassification && holdout.confusion_matrix && (
        <section className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-xl font-semibold text-gray-800">Confusion Matrix</h3>
            <p className="text-xs text-gray-500 mt-1">Rows = True Class, Columns = Predicted Class</p>
          </div>
          <div className="p-6 overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 border border-gray-200 text-center">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-3 py-2 text-xs font-semibold text-gray-600 uppercase">True \ Pred</th>
                  {holdout.confusion_matrix.labels.map((lbl) => (
                    <th key={lbl} className="px-3 py-2 text-xs font-semibold text-gray-700 uppercase">{lbl}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {holdout.confusion_matrix.matrix.map((row, rowIdx) => (
                  <tr key={rowIdx}>
                    <td className="px-3 py-2 text-xs font-bold text-gray-700 bg-gray-50">
                      {holdout.confusion_matrix.labels[rowIdx]}
                    </td>
                    {row.map((cell, colIdx) => (
                      <td
                        key={colIdx}
                        className={`px-4 py-2 text-sm font-medium ${
                          rowIdx === colIdx
                            ? 'bg-green-50 text-green-700 font-bold'
                            : cell > 0
                            ? 'bg-red-50 text-red-600'
                            : 'text-gray-500'
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

      {/* Data Quality */}
      {dataQuality && (
        <section className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-xl font-semibold text-gray-800 flex items-center">
              <FiInfo className="text-blue-500 mr-2" />
              Data Quality Summary
            </h3>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Stat label="Initial Shape" value={dataQuality.initial_shape} />
              <Stat label="Final Shape" value={dataQuality.final_shape} />
              <Stat label="Training Samples" value={dataQuality.training_samples} />
              <Stat label="Testing Samples" value={dataQuality.testing_samples} />
              <Stat label="Dropped (Target NaN)" value={dataQuality.dropped_target_nans} />
              <Stat label="Dropped (Feature NaN)" value={dataQuality.dropped_feature_nans} />
              <Stat label="Numeric Features" value={dataQuality.feature_counts?.numeric} />
              <Stat label="Categorical Features" value={dataQuality.feature_counts?.categorical} />
            </div>
            {dataQuality.class_distribution && (
              <div className="mt-4 bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Class Distribution</h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(dataQuality.class_distribution).map(([cls, count]) => (
                    <span key={cls} className="px-3 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                      {cls}: {count}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Warnings */}
      {warnings && warnings.length > 0 && (
        <section className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-md">
          <h4 className="text-xs font-bold text-yellow-800 uppercase tracking-wider mb-2 flex items-center">
            <FiAlertTriangle className="mr-1" size={14} />
            Evaluation Diagnostics
          </h4>
          <ul className="list-disc list-inside text-xs text-yellow-700 space-y-1">
            {warnings.map((w, idx) => (
              <li key={idx}>{w}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
};

/* ---- Sub-components ---- */

const Stat = ({ label, value }) => (
  <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
    <p className="text-xs text-gray-500 uppercase">{label}</p>
    <p className="text-lg font-semibold text-gray-800">{value ?? '–'}</p>
  </div>
);

const MetricCard = ({ label, value, color = 'blue', hint = '' }) => {
  const colorMap = {
    blue:    'bg-blue-50 border-blue-100 text-blue-800 text-blue-600',
    green:   'bg-green-50 border-green-100 text-green-800 text-green-600',
    purple:  'bg-purple-50 border-purple-100 text-purple-800 text-purple-600',
    amber:   'bg-amber-50 border-amber-100 text-amber-800 text-amber-600',
    red:     'bg-red-50 border-red-100 text-red-800 text-red-600',
    indigo:  'bg-indigo-50 border-indigo-100 text-indigo-800 text-indigo-600',
    emerald: 'bg-emerald-50 border-emerald-100 text-emerald-800 text-emerald-600',
    rose:    'bg-rose-50 border-rose-100 text-rose-800 text-rose-600',
  };
  const [bg, border, textLabel, textValue] = (colorMap[color] || colorMap.blue).split(' ');

  return (
    <div className={`${bg} p-4 rounded-lg border ${border}`}>
      <h4 className={`text-sm font-medium ${textLabel} mb-1`}>{label}</h4>
      <p className={`text-2xl font-bold ${textValue}`}>
        {value != null ? (typeof value === 'number' ? value.toFixed(4) : value) : 'N/A'}
      </p>
      {hint && <p className={`text-xs ${textLabel} mt-1 opacity-70`}>{hint}</p>}
    </div>
  );
};

const ClassificationHoldout = ({ holdout }) => (
  <div className="space-y-6">
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <MetricCard label="Accuracy" value={holdout.accuracy != null ? (holdout.accuracy * 100).toFixed(2) + '%' : null} color="blue" hint="Overall correct predictions" />
      <MetricCard label="Balanced Accuracy" value={holdout.balanced_accuracy != null ? (holdout.balanced_accuracy * 100).toFixed(2) + '%' : null} color="green" hint="Balanced across class frequencies" />
      <MetricCard label="F1 Score (Macro)" value={holdout.f1_macro} color="purple" hint="Unweighted harmonic mean" />
      <MetricCard label="F1 Score (Weighted)" value={holdout.f1_weighted} color="indigo" hint="Support-weighted harmonic mean" />
      <MetricCard label="Precision / Recall (Macro)" value={
        holdout.precision_macro != null && holdout.recall_macro != null
          ? `${(holdout.precision_macro * 100).toFixed(1)}% / ${(holdout.recall_macro * 100).toFixed(1)}%`
          : null
      } color="emerald" hint="Precision / Recall macro-averaged" />
      {holdout.roc_auc != null && (
        <MetricCard label="ROC-AUC" value={holdout.roc_auc} color="amber" hint="Area under ROC curve" />
      )}
      {holdout.log_loss != null && (
        <MetricCard label="Log Loss" value={holdout.log_loss} color="rose" hint="Cross-entropy loss (lower is better)" />
      )}
    </div>
  </div>
);

const RegressionHoldout = ({ holdout }) => (
  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
    <MetricCard label="RMSE" value={holdout.rmse} color="blue" hint="Lower is better" />
    <MetricCard label="R² Score" value={holdout.r2} color="green" hint="Closer to 1 is better" />
    <MetricCard label="MAE" value={holdout.mae} color="purple" hint="Lower is better" />
    <MetricCard label="Explained Variance" value={holdout.explained_variance} color="amber" hint="Closer to 1 is better" />
    <MetricCard label="Max Error" value={holdout.max_error} color="red" hint="Largest single error" />
  </div>
);

export default BestModelSummary;
