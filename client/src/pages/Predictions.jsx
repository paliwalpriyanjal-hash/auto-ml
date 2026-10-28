import { useState, useEffect } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { 
  PlayCircle, 
  Layers, 
  Send, 
  CheckCircle2, 
  AlertCircle, 
  RefreshCw,
  Cpu,
  Database,
  ArrowRight
} from "lucide-react";
import { getApiBase } from "../utils/api";

export default function Predictions() {
  const [searchParams] = useSearchParams();
  const preselectedModelId = searchParams.get("model");

  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(preselectedModelId || "");
  const [modelDetails, setModelDetails] = useState(null);
  
  // Dynamic form values for the model features
  const [formInputs, setFormInputs] = useState({});
  const [predicting, setPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Load all available registered models
  useEffect(() => {
    let isMounted = true;
    const fetchModels = async () => {
      setModelsLoading(true);
      try {
        const base = getApiBase();
        const res = await fetch(`${base}/models?limit=50`);
        if (!res.ok) throw new Error("Could not fetch models");
        const data = await res.json();
        if (isMounted) {
          setModels(data.models || []);
          if (!selectedModelId && data.models?.length > 0) {
            setSelectedModelId(String(data.models[0].id));
          }
        }
      } catch (err) {
        if (isMounted) setError("Failed to retrieve registered models list.");
      } finally {
        if (isMounted) setModelsLoading(false);
      }
    };
    fetchModels();
    return () => { isMounted = false; };
  }, []);

  // When a model is selected, fetch its full contract & features
  useEffect(() => {
    if (!selectedModelId) return;
    let isMounted = true;
    const fetchModel = async () => {
      try {
        const base = getApiBase();
        const res = await fetch(`${base}/models/${selectedModelId}`);
        if (!res.ok) throw new Error("Failed to load model details");
        const data = await res.json();
        if (isMounted) {
          setModelDetails(data);
          setPredictionResult(null);
          setError(null);
          // Initialize empty input fields for each feature
          const initial = {};
          (data.feature_columns || []).forEach((col) => {
            initial[col] = "";
          });
          setFormInputs(initial);
        }
      } catch (err) {
        if (isMounted) setError("Failed to load model architecture specifications.");
      }
    };
    fetchModel();
    return () => { isMounted = false; };
  }, [selectedModelId]);

  const handleInputChange = (col, val) => {
    setFormInputs((prev) => ({ ...prev, [col]: val }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!selectedModelId || !modelDetails) return;

    setPredicting(true);
    setError(null);
    setPredictionResult(null);

    // Convert numeric strings into numbers if applicable
    const rowPayload = {};
    for (const [key, value] of Object.entries(formInputs)) {
      if (value === "") {
        rowPayload[key] = null;
      } else if (!isNaN(Number(value)) && value.trim() !== "") {
        rowPayload[key] = Number(value);
      } else {
        rowPayload[key] = value;
      }
    }

    try {
      const base = getApiBase();
      const res = await fetch(`${base}/models/${selectedModelId}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: [rowPayload] }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail?.message || data.detail || "Prediction request rejected.");
      }

      setPredictionResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto space-y-8">
      
      {/* Title */}
      <div className="border-b border-white/[0.08] pb-6">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-[#8265FF] bg-[#8265FF]/10 px-2 py-0.5 rounded-full border border-[#8265FF]/20">
            Real-Time Inference Interface
          </span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Run Model Predictions</h1>
        <p className="text-sm text-[#A4A4B8] mt-1">
          Select an active pipeline from your Model Registry, fill the required feature schema, and execute live scoring.
        </p>
      </div>

      {/* Model Selector Card */}
      <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-4">
        <label className="text-xs font-semibold uppercase tracking-wider text-white flex items-center gap-2">
          <Layers className="w-4 h-4 text-[#8265FF]" />
          Select Deployed Model
        </label>

        {modelsLoading ? (
          <div className="h-10 bg-[#171630] rounded-xl animate-pulse" />
        ) : models.length === 0 ? (
          <div className="text-center py-6 text-xs text-[#A4A4B8] space-y-2">
            <p>No models registered in AutoVeyra yet.</p>
            <Link to="/automl" className="text-[#8265FF] hover:underline font-medium inline-block">
              Train a classification or regression model to enable inference →
            </Link>
          </div>
        ) : (
          <div className="flex flex-col sm:flex-row gap-4 items-center">
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-[#8265FF]"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  #{m.id} — {m.model_name} ({m.task_type} • {m.model_type})
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Model Spec Metadata Badge */}
        {modelDetails && (
          <div className="pt-3 border-t border-white/[0.06] flex flex-wrap items-center gap-4 text-xs text-[#A4A4B8]">
            <span>Task: <b className="text-white capitalize">{modelDetails.task_type}</b></span>
            <span>•</span>
            <span>Target: <b className="text-[#8265FF]">{modelDetails.target_column}</b></span>
            <span>•</span>
            <span>Features Required: <b className="text-white font-mono">{modelDetails.feature_columns?.length || 0}</b></span>
          </div>
        )}
      </div>

      {/* Dynamic Feature Inputs Form */}
      {modelDetails && (
        <form onSubmit={handlePredict} className="bg-[#111126] border border-white/[0.08] rounded-2xl p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
            <div>
              <h2 className="text-base font-bold text-white">Input Feature Values</h2>
              <p className="text-xs text-[#A4A4B8] mt-0.5">
                Supply values matching this model's required feature schema.
              </p>
            </div>
            <span className="text-xs font-mono text-[#8265FF] bg-[#8265FF]/10 px-2.5 py-1 rounded-lg border border-[#8265FF]/20">
              POST /models/{selectedModelId}/predict
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {modelDetails.feature_columns?.map((col) => (
              <div key={col} className="space-y-1.5">
                <label className="text-xs font-medium text-gray-300 block truncate" title={col}>
                  {col}
                </label>
                <input
                  type="text"
                  placeholder={`Enter ${col}`}
                  value={formInputs[col] ?? ""}
                  onChange={(e) => handleInputChange(col, e.target.value)}
                  className="w-full bg-[#171630] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white placeholder-[#A4A4B8]/40 focus:outline-none focus:border-[#8265FF]"
                  required
                />
              </div>
            ))}
          </div>

          {/* Error notification */}
          {error && (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Submit Button */}
          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              type="submit"
              disabled={predicting}
              className="px-6 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/25 flex items-center gap-2 transition-all disabled:opacity-50"
            >
              {predicting ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Generating Prediction...
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  Execute Prediction
                </>
              )}
            </button>
          </div>
        </form>
      )}

      {/* Prediction Result Display */}
      {predictionResult && (
        <div className="bg-[#111126] border border-[#8265FF]/40 rounded-2xl p-6 glow-purple space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle2 className="w-5 h-5" />
              <h3 className="text-base font-bold text-white">Prediction Computed Successfully</h3>
            </div>
            <span className="text-xs text-[#A4A4B8] font-mono">
              Model #{predictionResult.model_id} ({predictionResult.model_type})
            </span>
          </div>

          <div className="p-6 rounded-xl bg-[#171630] border border-white/[0.06] flex flex-col items-center justify-center text-center space-y-2">
            <span className="text-xs uppercase tracking-wider text-[#A4A4B8] font-medium">
              Predicted Target Value ({modelDetails?.target_column})
            </span>
            <div className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-[#8265FF] font-mono tracking-tight">
              {Array.isArray(predictionResult.predictions)
                ? String(predictionResult.predictions[0])
                : String(predictionResult.predictions)}
            </div>
            <span className="text-[11px] text-[#A4A4B8]">
              Computed via fitted {predictionResult.model_type} pipeline
            </span>
          </div>
        </div>
      )}

    </div>
  );
}
