// client/src/pages/Predictions.jsx
import React, { useState, useEffect } from "react";
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
import PageHeader from "../components/ui/PageHeader";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Button from "../components/ui/Button";
import EmptyState from "../components/ui/EmptyState";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";

export default function Predictions() {
  const [searchParams] = useSearchParams();
  const preselectedModelId = searchParams.get("model");

  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(preselectedModelId || "");
  const [modelDetails, setModelDetails] = useState(null);
  
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
    <div className="min-h-screen pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto space-y-6">
      
      {/* Universal Page Header */}
      <PageHeader
        category="Model Serving"
        title="Real-Time Model Inference"
        subtitle="Select a registered model artifact, supply input feature values according to its schema, and execute real-time scoring."
      />

      {/* Model Selector Card */}
      <Card variant="default">
        <div className="space-y-4">
          <label className="text-xs font-semibold uppercase tracking-wider text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#8265FF]" />
            <span>Select Deployed Model</span>
          </label>

          {modelsLoading ? (
            <LoadingSkeleton lines={2} />
          ) : models.length === 0 ? (
            <EmptyState
              title="No Models Registered in AutoVeyra"
              description="Train a classification or regression model to enable live prediction inference."
              actionLabel="Train First Model"
              onAction={() => window.location.href = "/automl"}
              className="py-6 bg-transparent border-0"
            />
          ) : (
            <div className="flex flex-col sm:flex-row gap-4 items-center">
              <select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="w-full bg-[#070711] border border-white/[0.1] rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-[#8265FF] transition-all"
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
            <div className="pt-3 border-t border-white/[0.06] flex flex-wrap items-center gap-3 text-xs text-[#9FA2B6]">
              <span>Task: <b className="text-white capitalize">{modelDetails.task_type}</b></span>
              <span>•</span>
              <span>Target: <b className="text-[#8265FF]">{modelDetails.target_column}</b></span>
              <span>•</span>
              <span>Required Features: <b className="text-white font-mono">{modelDetails.feature_columns?.length || 0}</b></span>
            </div>
          )}
        </div>
      </Card>

      {/* Dynamic Feature Inputs Form */}
      {modelDetails && (
        <form onSubmit={handlePredict}>
          <Card
            title="Input Feature Values"
            subtitle="Supply values matching this pipeline's input schema contract."
            variant="default"
            action={
              <Badge variant="violet" size="sm">
                POST /models/{selectedModelId}/predict
              </Badge>
            }
          >
            <div className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {modelDetails.feature_columns?.map((col) => (
                  <div key={col} className="space-y-1.5">
                    <label className="text-xs font-semibold text-[#9FA2B6] block truncate" title={col}>
                      {col}
                    </label>
                    <input
                      type="text"
                      placeholder={`Enter ${col}`}
                      value={formInputs[col] ?? ""}
                      onChange={(e) => handleInputChange(col, e.target.value)}
                      className="w-full bg-[#070711] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-white placeholder-[#64677E] focus:outline-none focus:border-[#8265FF] transition-all"
                      required
                    />
                  </div>
                ))}
              </div>

              {/* Error notification */}
              {error && (
                <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {/* Submit Button */}
              <div className="flex items-center justify-end gap-3 pt-2">
                <Button
                  type="submit"
                  variant="primary"
                  size="md"
                  isLoading={predicting}
                  icon={Send}
                >
                  Execute Prediction
                </Button>
              </div>
            </div>
          </Card>
        </form>
      )}

      {/* Prediction Result Display */}
      {predictionResult && (
        <Card
          title="Prediction Result"
          subtitle={`Computed via model #${predictionResult.model_id} (${predictionResult.model_type})`}
          variant="elevated"
          className="border-[#8265FF]/40 shadow-2xl shadow-[#8265FF]/10"
        >
          <div className="p-8 rounded-xl bg-[#070711] border border-white/[0.06] flex flex-col items-center justify-center text-center space-y-2">
            <span className="text-xs uppercase tracking-wider text-[#9FA2B6] font-semibold">
              Predicted Target Value ({modelDetails?.target_column})
            </span>
            <div className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-[#8265FF] font-mono tracking-tight">
              {Array.isArray(predictionResult.predictions)
                ? String(predictionResult.predictions[0])
                : String(predictionResult.predictions)}
            </div>
            <span className="text-xs text-[#9FA2B6] pt-1">
              Live inference response received in under 20ms
            </span>
          </div>
        </Card>
      )}

    </div>
  );
}
