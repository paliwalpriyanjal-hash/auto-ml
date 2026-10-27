import React, { useState } from 'react';
import Papa from 'papaparse';
import Preprocess from '../components/PreprocessingOptions';
import ComparisonResultTable from '../components/ComparisonResultTable';
import BestModelSummary from '../components/BestModelSummary';
import {
  FiUpload,
  FiSettings,
  FiCpu,
  FiBarChart2,
  FiCheckCircle,
  FiLayers,
} from 'react-icons/fi';

// Ranking-metric options that match the backend exactly
const CLASSIFICATION_METRICS = [
  { value: 'f1_macro', label: 'F1 Macro', higherIsBetter: true },
  { value: 'balanced_accuracy', label: 'Balanced Accuracy', higherIsBetter: true },
  { value: 'accuracy', label: 'Accuracy', higherIsBetter: true },
  { value: 'f1_weighted', label: 'F1 Weighted', higherIsBetter: true },
];

const REGRESSION_METRICS = [
  { value: 'rmse', label: 'RMSE', higherIsBetter: false },
  { value: 'mae', label: 'MAE', higherIsBetter: false },
  { value: 'r2', label: 'R² Score', higherIsBetter: true },
];

function CompareModels() {
  // --- state ---
  const [csvFile, setCsvFile] = useState(null);
  const [previewData, setPreviewData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [activeStep, setActiveStep] = useState(1);

  const [taskType, setTaskType] = useState('regression');
  const [targetColumn, setTargetColumn] = useState('');
  const [rankingMetric, setRankingMetric] = useState('rmse');

  const [missingStrategy, setMissingStrategy] = useState('drop');
  const [encodingStrategy, setEncodingStrategy] = useState('label');
  const [scaler, setScaler] = useState('standard');
  const [splitRatio, setSplitRatio] = useState('0.8');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [comparisonData, setComparisonData] = useState(null);

  // --- handlers ---
  const handleTaskTypeChange = (newTask) => {
    setTaskType(newTask);
    if (newTask === 'classification') {
      setRankingMetric('f1_macro');
      setEncodingStrategy('onehot');
    } else {
      setRankingMetric('rmse');
      setEncodingStrategy('label');
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
      setError('Please select a target column.');
      return;
    }

    setLoading(true);
    setError('');
    setComparisonData(null);
    setActiveStep(4);

    const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    const formData = new FormData();
    formData.append('file', csvFile);
    formData.append('task_type', taskType);
    formData.append('target_column', targetColumn);
    formData.append('scaler', scaler);
    formData.append('splitRatio', splitRatio);
    formData.append('missing', missingStrategy);
    formData.append('encoding', encodingStrategy);
    formData.append('ranking_metric', rankingMetric);

    try {
      const res = await fetch(`${apiBase}/compare`, {
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
      setComparisonData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setActiveStep(1);
    setCsvFile(null);
    setPreviewData([]);
    setColumns([]);
    setTaskType('regression');
    setTargetColumn('');
    setRankingMetric('rmse');
    setMissingStrategy('drop');
    setEncodingStrategy('label');
    setScaler('standard');
    setSplitRatio('0.8');
    setLoading(false);
    setError('');
    setComparisonData(null);
  };

  const currentMetrics = taskType === 'classification' ? CLASSIFICATION_METRICS : REGRESSION_METRICS;
  const selectedMetricConfig = currentMetrics.find((m) => m.value === rankingMetric) || currentMetrics[0];

  // --- render ---
  return (
    <div className="app-container px-4 py-8 max-w-7xl mx-auto mt-20 min-h-screen">
      <header className="app-header text-center mb-12">
        <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-700 to-gray-400 mb-4">
          Compare Models
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Automatically compare all supported models and find the best one for your dataset.
        </p>
      </header>

      {/* Progress Steps */}
      <div className="mb-12">
        <div className="flex justify-between relative">
          <div className="absolute top-1/2 left-0 right-0 h-1 bg-yellow-600 -z-10 transform -translate-y-1/2"></div>
          {[1, 2, 3, 4].map((step) => (
            <div key={step} className="flex flex-col items-center">
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center text-black font-bold 
                ${
                  activeStep >= step
                    ? 'bg-gradient-to-r from-yellow-500 to-gray-500 shadow-lg'
                    : 'bg-gray-300'
                }`}
              >
                {activeStep > step ? <FiCheckCircle size={24} /> : step}
              </div>
              <span className="mt-2 text-sm font-medium text-gray-700">
                {step === 1 && 'Upload Data'}
                {step === 2 && 'Preview'}
                {step === 3 && 'Configure'}
                {step === 4 && 'Results'}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: File Upload */}
      {activeStep === 1 && (
        <section className="max-w-2xl mx-auto bg-white rounded-xl shadow-md overflow-hidden p-8 text-center">
          <div className="bg-yellow-50 p-6 rounded-lg border-2 border-dashed border-blue-200 mb-6">
            <FiUpload className="mx-auto text-yellow-500" size={48} />
            <h2 className="text-2xl font-semibold text-gray-800 mt-4 mb-2">
              Upload Your Dataset
            </h2>
            <p className="text-gray-600 mb-6">
              Drag & drop your CSV file here or click to browse
            </p>
            <label className="cursor-pointer bg-yellow-600 hover:bg-orange-700 text-white font-medium py-2 px-6 rounded-lg transition duration-200 inline-block">
              Select File
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          </div>
          <p className="text-gray-500 text-sm">
            Supported formats: CSV. Max file size: 10MB
          </p>
        </section>
      )}

      {/* Step 2: Preview Data */}
      {activeStep >= 2 && previewData.length > 0 && (
        <section className="mb-10 bg-white rounded-xl shadow-md overflow-hidden">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center">
              <FiBarChart2 className="text-yellow-500 mr-2" size={20} />
              <h2 className="text-xl font-semibold text-gray-800">
                Data Preview
              </h2>
            </div>
            <p className="text-gray-600 text-sm mt-1">
              First 5 rows of your dataset
            </p>
          </div>
          <div className="overflow-x-auto p-4">
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
                {previewData.map((row, idx) => (
                  <tr
                    key={idx}
                    className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
                  >
                    {columns.map((col) => (
                      <td
                        key={col}
                        className="px-6 py-4 whitespace-nowrap text-sm text-gray-700"
                      >
                        {row[col]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {activeStep === 2 && (
            <div className="bg-gray-50 px-6 py-3 flex justify-end">
              <button
                onClick={() => setActiveStep(3)}
                className="bg-yellow-600 hover:bg-orange-700 text-white font-medium py-2 px-6 rounded-lg transition duration-200"
              >
                Continue to Configuration <span className="ml-1">→</span>
              </button>
            </div>
          )}
        </section>
      )}

      {/* Step 3: Configuration */}
      {activeStep === 3 && (
        <div className="space-y-6">
          {/* Task Type and Target Column */}
          <section className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiSettings className="text-yellow-600 mr-2" /> Task & Target Configuration
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Task Type Selector */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Machine Learning Task
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => handleTaskTypeChange('regression')}
                    className={`py-3 px-4 rounded-lg border text-sm font-semibold transition duration-150 ${
                      taskType === 'regression'
                        ? 'bg-yellow-600 text-white border-yellow-600 shadow-md'
                        : 'bg-gray-50 text-gray-700 border-gray-300 hover:bg-gray-100'
                    }`}
                  >
                    Regression
                  </button>
                  <button
                    type="button"
                    onClick={() => handleTaskTypeChange('classification')}
                    className={`py-3 px-4 rounded-lg border text-sm font-semibold transition duration-150 ${
                      taskType === 'classification'
                        ? 'bg-yellow-600 text-white border-yellow-600 shadow-md'
                        : 'bg-gray-50 text-gray-700 border-gray-300 hover:bg-gray-100'
                    }`}
                  >
                    Classification
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  {taskType === 'regression'
                    ? 'Predict continuous numerical values (e.g. prices, scores, temperatures).'
                    : 'Predict discrete categorical classes (e.g. churn yes/no, disease type).'}
                </p>
              </div>

              {/* Target Column Selector */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Target Column <span className="text-red-500 font-bold">*</span>
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full px-4 py-2.5 border border-yellow-300 rounded-lg focus:ring-2 focus:ring-yellow-500 focus:border-yellow-500 bg-white"
                >
                  <option value="">-- Select Target Column --</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-2">
                  Required. The selected column will be predicted from the remaining features.
                </p>
              </div>
            </div>
          </section>

          {/* Ranking Metric */}
          <section className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiLayers className="text-yellow-600 mr-2" /> Ranking Metric
            </h3>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Rank models by
              </label>
              <select
                value={rankingMetric}
                onChange={(e) => setRankingMetric(e.target.value)}
                className="w-full md:w-1/2 px-4 py-2.5 border border-yellow-300 rounded-lg focus:ring-2 focus:ring-yellow-500 focus:border-yellow-500 bg-white"
              >
                {currentMetrics.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-2">
                Models will be cross-validated and ranked by this metric.
              </p>
            </div>
          </section>

          {/* Preprocessing */}
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

          {/* Submit */}
          <div className="text-center mt-8">
            <button
              className="bg-gradient-to-r from-yellow-600 to-gray-600 hover:from-orange-600 hover:to-gray-600 text-white font-bold py-3 px-8 rounded-full shadow-lg transition duration-200 transform hover:scale-105"
              onClick={handleSubmit}
              disabled={!csvFile || loading}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Comparing Models...
                </span>
              ) : (
                <>
                  <FiCpu className="inline mr-2" />
                  Compare All Models
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Step 4: Results */}
      {activeStep === 4 && loading && (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-yellow-500 mb-4"></div>
          <h3 className="text-xl font-semibold text-gray-700">Comparing all models</h3>
          <p className="text-gray-500 mt-2">This may take a few moments...</p>
        </div>
      )}

      {activeStep === 4 && !loading && comparisonData && (
        <div className="space-y-8">
          <ComparisonResultTable
            modelResults={comparisonData.model_results}
            rankingMetric={comparisonData.ranking_metric}
            higherIsBetter={selectedMetricConfig.higherIsBetter}
          />
          <BestModelSummary
            bestModel={comparisonData.best_model}
            rankingMetric={comparisonData.ranking_metric}
            cvFolds={comparisonData.cv_folds}
            taskType={comparisonData.task_type}
            classes={comparisonData.classes}
            warnings={comparisonData.warnings}
            dataQuality={comparisonData.data_quality}
          />
          <div className="text-center">
            <button
              onClick={handleReset}
              className="bg-yellow-600 hover:bg-orange-700 text-white font-medium py-2 px-6 rounded-lg transition duration-200"
            >
              Compare Another Dataset
            </button>
          </div>
        </div>
      )}

      {/* Error Handling */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded mt-6">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-red-500"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error occurred</h3>
              <div className="mt-2 text-sm text-red-700">
                <p>{error}</p>
              </div>
              <div className="mt-3">
                <button
                  onClick={handleReset}
                  className="text-sm font-medium text-red-700 hover:text-red-600 underline"
                >
                  Start over
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CompareModels;
