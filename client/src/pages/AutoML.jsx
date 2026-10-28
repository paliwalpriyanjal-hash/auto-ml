import React, { useState } from "react";
import Papa from "papaparse";
import Result from "../components/ResultDisplay";
import Model from "../components/ModelSelection";
import Preprocess from "../components/PreprocessingOptions";
import {
  FiUpload,
  FiSettings,
  FiCpu,
  FiBarChart2,
  FiCheckCircle,
  FiAlertCircle
} from "react-icons/fi";

function AutoML() {
  const [submit, setSubmit] = useState(false);
  const [csvFile, setCsvFile] = useState(null);
  const [previewData, setPreviewData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [activeStep, setActiveStep] = useState(1);

  const [taskType, setTaskType] = useState("regression");
  const [targetColumn, setTargetColumn] = useState("");

  const [missingStrategy, setMissingStrategy] = useState("drop");
  const [encodingStrategy, setEncodingStrategy] = useState("label");
  const [scaler, setScaler] = useState("standard");
  const [splitRatio, setSplitRatio] = useState("0.8");
  const [model, setModel] = useState("LinearRegression");

  const [error, setError] = useState("");

  const handleTaskTypeChange = (newTask) => {
    setTaskType(newTask);
    setError("");

    if (newTask === "classification") {
      setModel("RandomForestClassifier");
      setEncodingStrategy("onehot");
    } else {
      setModel("LinearRegression");
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

        setCsvFile(file);
        setColumns(fields);
        setPreviewData(data.slice(0, 5));
        setTargetColumn(fields[fields.length - 1]);
        setActiveStep(2);
      },
    });
  };

  const handleSubmit = () => {
    if (!csvFile) {
      setError("Please choose a CSV file.");
      return;
    }

    if (taskType === "classification" && !targetColumn) {
      setError("Please choose a target column for classification.");
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
      
      {/* Title */}
      <div className="border-b border-white/[0.08] pb-6">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-[#8265FF] bg-[#8265FF]/10 px-2 py-0.5 rounded-full border border-[#8265FF]/20">
            Pipeline Configuration
          </span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Dedicated Model Training</h1>
        <p className="text-sm text-[#A4A4B8] mt-1">
          Train a specific algorithm with deterministic preprocessing, evaluate holdout metrics, and auto-register the fitted pipeline.
        </p>
      </div>

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

      {/* Step 2: Task Selection */}
      {!submit && csvFile && (
        <section className="bg-[#111126] p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-4">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <FiSettings className="text-[#A855F7]" /> Task & Target Configuration
          </h3>

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
                <option value="regression">Regression (Predict numbers)</option>
                <option value="classification">Classification (Predict classes)</option>
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
                <option value="">Select target column...</option>
                {columns.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          </div>
        </section>
      )}

      {/* Step 3: Model & Preprocessing Options */}
      {!submit && csvFile && (
        <>
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

          <div className="flex justify-end">
            <button
              onClick={handleSubmit}
              className="px-8 py-3.5 rounded-xl font-bold text-sm bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-xl shadow-[#8265FF]/25 flex items-center gap-2 transition-all"
            >
              <FiBarChart2 />
              Start Training Pipeline
            </button>
          </div>
        </>
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
