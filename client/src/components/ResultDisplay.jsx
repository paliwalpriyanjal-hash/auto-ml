import React, { useEffect, useState } from 'react';
import { FiBarChart2, FiTrendingUp, FiCheck, FiAlertTriangle, FiDownload, FiRefreshCw } from 'react-icons/fi';
import Dashboard from './Dashboard';
import Dashboard2 from './Dashboard2';

function Result({
  csv,
  taskType = 'regression',
  targetColumn = '',
  model,
  scaler,
  splitRatio,
  missing,
  encoding,
  onReset
}) {
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const sendRequest = async () => {
      if (!csv) return;

      // Resolve the backend base URL from the Vite environment variable with local fallback.
      const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';

      const formData = new FormData();
      formData.append('file', csv);
      formData.append('missing', missing);
      formData.append('encoding', encoding);
      formData.append('scaler', scaler);
      formData.append('model', model);
      formData.append('splitRatio', splitRatio);
      formData.append('task_type', taskType);
      if (taskType === 'classification' && targetColumn) {
        formData.append('target_column', targetColumn);
      }

      try {
        const res = await fetch(`${apiBase}/train`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          let errDetail = `HTTP error! Status: ${res.status}`;
          try {
            const errData = await res.json();
            if (errData?.detail) {
              errDetail = errData.detail;
            }
          } catch (_) {}
          throw new Error(errDetail);
        }
        const data = await res.json();
        setResponse(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    sendRequest();
  }, [csv, taskType, targetColumn, model, scaler, splitRatio, missing, encoding]);

  const handleTryAnother = () => {
    if (onReset) {
      onReset();
    }
  };

  const handleDownload = () => {
    if (!response?.columns || !response?.rows) return;

    const header = response.columns.join(',');
    const dataLines = response.rows.map(row =>
      response.columns.map(col => {
        const cell = String(row[col] ?? '');
        return cell.includes(',') ? `"${cell}"` : cell;
      }).join(',')
    );
    const csvContent = [header, ...dataLines].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `${model}_results.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500 mb-4"></div>
        <h3 className="text-xl font-semibold text-gray-700">Training your model</h3>
        <p className="text-gray-500 mt-2">This may take a few moments...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-lg shadow-md">
        <div className="flex">
          <div className="flex-shrink-0">
            <FiAlertTriangle className="h-6 w-6 text-red-500" />
          </div>
          <div className="ml-4">
            <h3 className="text-lg font-medium text-red-800">Error Processing Request</h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
            <div className="mt-4">
              <button
                onClick={() => window.location.reload()}
                className="text-sm font-medium text-red-700 hover:text-red-600"
              >
                Try again <span aria-hidden="true">→</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!response) return null;

  const { metrics, visualization, prediction_plot, columns, rows, dashboard_metrics } = response;

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden" style={{ marginTop: '10px' }}>
      {/* Header */}
      <div className="bg-gradient-to-r from-yellow-600 to-gray-600 p-6 text-white">
        <h2 className="text-2xl font-bold flex items-center">
          <FiCheck className="mr-2" /> Model Training Complete
        </h2>
        <p className="opacity-90 mt-1">Here are your results for the {model} model</p>
      </div>

      {/* Metrics */}
      <div className="p-6 border-b border-gray-200">
        <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          <FiBarChart2 className="mr-2 text-blue-500" /> Performance Metrics
        </h3>

        {metrics ? (
          response.task_type === 'classification' ? (
            <div className="space-y-6">
              {/* Classification Metrics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                  <h4 className="text-sm font-medium text-blue-800 mb-1">Accuracy</h4>
                  <p className="text-2xl font-bold text-blue-600">
                    {metrics.accuracy !== null && metrics.accuracy !== undefined
                      ? (metrics.accuracy * 100).toFixed(2) + '%'
                      : 'N/A'}
                  </p>
                  <p className="text-xs text-blue-700 mt-1">Overall correct predictions</p>
                </div>

                <div className="bg-green-50 p-4 rounded-lg border border-green-100">
                  <h4 className="text-sm font-medium text-green-800 mb-1">Balanced Accuracy</h4>
                  <p className="text-2xl font-bold text-green-600">
                    {metrics.balanced_accuracy !== null && metrics.balanced_accuracy !== undefined
                      ? (metrics.balanced_accuracy * 100).toFixed(2) + '%'
                      : 'N/A'}
                  </p>
                  <p className="text-xs text-green-700 mt-1">Balanced across class frequencies</p>
                </div>

                <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
                  <h4 className="text-sm font-medium text-purple-800 mb-1">F1 Score (Macro)</h4>
                  <p className="text-2xl font-bold text-purple-600">
                    {metrics.f1_macro !== null && metrics.f1_macro !== undefined
                      ? metrics.f1_macro.toFixed(4)
                      : 'N/A'}
                  </p>
                  <p className="text-xs text-purple-700 mt-1">Unweighted harmonic mean</p>
                </div>

                <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-100">
                  <h4 className="text-sm font-medium text-indigo-800 mb-1">F1 Score (Weighted)</h4>
                  <p className="text-2xl font-bold text-indigo-600">
                    {metrics.f1_weighted !== null && metrics.f1_weighted !== undefined
                      ? metrics.f1_weighted.toFixed(4)
                      : 'N/A'}
                  </p>
                  <p className="text-xs text-indigo-700 mt-1">Support-weighted harmonic mean</p>
                </div>

                <div className="bg-emerald-50 p-4 rounded-lg border border-emerald-100">
                  <h4 className="text-sm font-medium text-emerald-800 mb-1">Precision / Recall (Macro)</h4>
                  <p className="text-xl font-bold text-emerald-600">
                    {metrics.precision_macro !== null && metrics.precision_macro !== undefined
                      ? (metrics.precision_macro * 100).toFixed(1) + '%'
                      : 'N/A'}{' '}
                    /{' '}
                    {metrics.recall_macro !== null && metrics.recall_macro !== undefined
                      ? (metrics.recall_macro * 100).toFixed(1) + '%'
                      : 'N/A'}
                  </p>
                  <p className="text-xs text-emerald-700 mt-1">Precision / Recall macro-averaged</p>
                </div>

                {metrics.roc_auc !== null && metrics.roc_auc !== undefined && (
                  <div className="bg-amber-50 p-4 rounded-lg border border-amber-100">
                    <h4 className="text-sm font-medium text-amber-800 mb-1">ROC-AUC</h4>
                    <p className="text-2xl font-bold text-amber-600">
                      {metrics.roc_auc.toFixed(4)}
                    </p>
                    <p className="text-xs text-amber-700 mt-1">Area under ROC curve</p>
                  </div>
                )}

                {metrics.log_loss !== null && metrics.log_loss !== undefined && (
                  <div className="bg-rose-50 p-4 rounded-lg border border-rose-100">
                    <h4 className="text-sm font-medium text-rose-800 mb-1">Log Loss</h4>
                    <p className="text-2xl font-bold text-rose-600">
                      {metrics.log_loss.toFixed(4)}
                    </p>
                    <p className="text-xs text-rose-700 mt-1">Cross-entropy loss (lower is better)</p>
                  </div>
                )}
              </div>

              {/* Confusion Matrix Display */}
              {response.confusion_matrix && (
                <div className="mt-6 bg-gray-50 p-5 rounded-xl border border-gray-200">
                  <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
                    Confusion Matrix
                  </h4>
                  <p className="text-xs text-gray-500 mb-4">
                    Rows represent True Class, Columns represent Predicted Class.
                  </p>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 border border-gray-200 text-center">
                      <thead className="bg-gray-100">
                        <tr>
                          <th className="px-3 py-2 text-xs font-semibold text-gray-600 uppercase">
                            True \ Pred
                          </th>
                          {response.confusion_matrix.labels.map((lbl) => (
                            <th
                              key={lbl}
                              className="px-3 py-2 text-xs font-semibold text-gray-700 uppercase"
                            >
                              {lbl}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {response.confusion_matrix.matrix.map((row, rowIdx) => (
                          <tr key={rowIdx}>
                            <td className="px-3 py-2 text-xs font-bold text-gray-700 bg-gray-50">
                              {response.confusion_matrix.labels[rowIdx]}
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
                </div>
              )}

              {/* Warnings / Diagnostic notices if any */}
              {response.warnings && response.warnings.length > 0 && (
                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-md">
                  <h4 className="text-xs font-bold text-yellow-800 uppercase tracking-wider mb-1">
                    Evaluation Diagnostics
                  </h4>
                  <ul className="list-disc list-inside text-xs text-yellow-700 space-y-1">
                    {response.warnings.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                <h4 className="text-sm font-medium text-blue-800 mb-1">Root Mean Squared Error</h4>
                <p className="text-2xl font-bold text-blue-600">
                  {metrics.rmse?.toFixed(4) || 'N/A'}
                </p>
                <p className="text-xs text-blue-700 mt-1">Lower is better</p>
              </div>

              <div className="bg-green-50 p-4 rounded-lg border border-green-100">
                <h4 className="text-sm font-medium text-green-800 mb-1">R² Score</h4>
                <p className="text-2xl font-bold text-green-600">
                  {metrics.r2?.toFixed(4) || 'N/A'}
                </p>
                <p className="text-xs text-green-700 mt-1">Closer to 1 is better</p>
              </div>

              <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
                <h4 className="text-sm font-medium text-purple-800 mb-1">Mean Absolute Error</h4>
                <p className="text-2xl font-bold text-purple-600">
                  {metrics.mae?.toFixed(4) || 'N/A'}
                </p>
                <p className="text-xs text-purple-700 mt-1">Lower is better</p>
              </div>

              <div className="bg-amber-50 p-4 rounded-lg border border-amber-100">
                <h4 className="text-sm font-medium text-amber-800 mb-1">Explained Variance</h4>
                <p className="text-2xl font-bold text-amber-600">
                  {metrics.explained_variance?.toFixed(4) || 'N/A'}
                </p>
                <p className="text-xs text-amber-700 mt-1">Closer to 1 is better</p>
              </div>

              <div className="bg-red-50 p-4 rounded-lg border border-red-100">
                <h4 className="text-sm font-medium text-red-800 mb-1">Max Error</h4>
                <p className="text-2xl font-bold text-red-600">
                  {metrics.max_error?.toFixed(4) || 'N/A'}
                </p>
                <p className="text-xs text-red-700 mt-1">Largest error</p>
              </div>

              <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-100">
                <h4 className="text-sm font-medium text-indigo-800 mb-1">Mean Absolute % Error</h4>
                <p className="text-2xl font-bold text-indigo-600">
                  {metrics.mape ? (metrics.mape * 100).toFixed(2) + '%' : 'N/A'}
                </p>
                <p className="text-xs text-indigo-700 mt-1">Percentage error</p>
              </div>
            </div>
          )
        ) : (
          <p className="text-gray-500">No performance metrics available.</p>
        )}
      </div>

      {/* Visualizations */}
      <div className="p-6 border-b border-gray-200">
        <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          <FiTrendingUp className="mr-2 text-purple-500" /> Model Insights
        </h3>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {visualization && (
            <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
              <h4 className="font-semibold text-gray-700 mb-3">Feature Importance</h4>
              <img
                src={`data:image/png;base64,${visualization}`}
                alt="Feature Importance"
                className="w-full h-auto rounded"
              />
            </div>
          )}

          {prediction_plot && (
            <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
              <h4 className="font-semibold text-gray-700 mb-3">Actual vs Predicted Values</h4>
              <img
                src={`data:image/png;base64,${prediction_plot}`}
                alt="Actual vs Predicted"
                className="w-full h-auto rounded"
              />
            </div>
          )}
        </div>
      </div>

      {/* Processed Data */}
      {columns && rows && (
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Processed Data Sample</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {columns.map((col) => (
                    <th
                      key={col}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {rows.slice(0, 5).map((row, i) => (
                  <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {columns.map((col, j) => (
                      <td key={j} className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                        {isNaN(Number(row[col])) ? '-' : Number(row[col]).toFixed(3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mt-2">Showing first 5 rows of processed data</p>
        </div>
      )}

      {/* Dashboard Section */}
      {/* // In your Result component's return statement, update the Dashboard rendering: */}
      {dashboard_metrics ? (
        <Dashboard
          metrics={metrics}
          dashboardMetrics={dashboard_metrics}
        />
      ) : (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <FiAlertTriangle className="h-5 w-5 text-yellow-400" aria-hidden="true" />
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                Dashboard data not available.
              </p>
            </div>
          </div>
        </div>
      )}
 
 {/* <div className="p-6 border-b border-gray-200">
    <h3 className="text-xl font-semibold text-gray-800 mb-4">Data Analysis Dashboard</h3>
    {response.data_analysis_metrics ? (
        <Dashboard2 dataAnalysis={response.data_analysis_metrics} />
    ) : (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
            <div className="flex">
                <div className="flex-shrink-0">
                    <FiAlertTriangle className="h-5 w-5 text-yellow-400" aria-hidden="true" />
                </div>
                <div className="ml-3">
                    <p className="text-sm text-yellow-700">
                        Data analysis metrics not available.
                    </p>
                </div>
            </div>
        </div>
    )} 
</div>*/}

      {/* Actions */}
      <div className="bg-gray-50 px-6 py-4 flex justify-between">
        <button
          onClick={handleDownload}
          className="text-gray-600 hover:text-gray-800 font-medium flex items-center"
        >
          <FiDownload className="mr-2" />
          Download Results
        </button>
        <button
          onClick={handleTryAnother}
          className="bg-yellow-600 hover:bg-orange-700 text-white font-medium py-2 px-6 rounded-lg transition duration-200 flex items-center"
        >
          <FiRefreshCw className="mr-2" />
          Try Another Model
        </button>
      </div>
    </div>
  );
}

export default Result;
