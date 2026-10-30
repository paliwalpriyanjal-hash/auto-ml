// client/src/components/AssistantPanel.jsx
import React, { useState } from "react";
import {
  Sparkles,
  ArrowRight,
  HelpCircle,
  ShieldAlert,
  CheckCircle2,
  BookOpen,
  ChevronDown,
  ChevronUp,
  Cpu,
  Info,
  X,
  Compass
} from "lucide-react";
import HelpTooltip from "./HelpTooltip";

export default function AssistantPanel({
  currentStep,
  recommendation,
  datasetInfo,
  preflight,
  onApplyRecommendation,
  onOpenGlossary,
  isOpen = true,
  onToggle,
}) {
  const [activeTab, setActiveTab] = useState("next_action"); // 'next_action' | 'step_guide' | 'preflight'

  const STEP_GUIDES = {
    1: {
      title: "Step 1: Upload Dataset",
      goal: "Import your raw tabular data in CSV format.",
      why: "The dataset provides the examples your model will learn from. AutoVeyra computes SHA-256 hashes to guarantee your original data is never altered.",
      tips: [
        "Ensure your first row contains descriptive column headers.",
        "Remove corrupted non-standard characters before uploading.",
        "AutoVeyra handles missing numbers and text categories automatically in later steps.",
      ],
      relevantTerms: ["dataset", "feature", "missing_values"],
    },
    2: {
      title: "Step 2: Inspect Dataset",
      goal: "Review column distributions, data types, and initial structure.",
      why: "Spotting identifier columns (like CustomerID or SSN) early prevents models from memorizing useless random keys.",
      tips: [
        "Check that numerical columns were not mistakenly read as text.",
        "Verify row count is sufficient for training (minimum 15 rows).",
      ],
      relevantTerms: ["feature", "duplicates", "outliers"],
    },
    3: {
      title: "Step 3: Understand Dataset Quality",
      goal: "Analyze missing values, duplicate records, and class balance.",
      why: "Garbage data yields garbage models. Knowing your missingness ratio helps choose whether to impute or drop columns.",
      tips: [
        "If a column has >50% missing values, it may provide little signal.",
        "Check for duplicate rows that artificially inflate accuracy.",
      ],
      relevantTerms: ["eda", "missing_values", "duplicates", "outliers", "correlation"],
    },
    4: {
      title: "Step 4: Select Prediction Objective",
      goal: "Specify whether you are predicting categories or numbers, and choose the target column.",
      why: "The target column tells the algorithm exactly what question it is solving.",
      tips: [
        "Choose Classification if your answer is discrete (Yes/No, Category).",
        "Choose Regression if your answer is a continuous number (Price, Count, Salary).",
      ],
      relevantTerms: ["target_column", "classification", "regression"],
    },
    5: {
      title: "Step 5: Clean and Prepare Data",
      goal: "Handle missing values, encode text categories, and scale numbers.",
      why: "Algorithms require clean numbers; median imputation and one-hot encoding convert raw data into algorithm-ready matrices.",
      tips: [
        "Use median imputation for skewed numerical data with outliers.",
        "One-hot encoding is ideal for categories with fewer than 10 unique values.",
      ],
      relevantTerms: ["imputation", "encoding", "scaling", "missing_values"],
    },
    6: {
      title: "Step 6: Engineer Features",
      goal: "Create new signals from dates, interactions, or mathematical transforms.",
      why: "High-quality engineered features often boost accuracy more than tuning complex algorithms.",
      tips: [
        "Extract day-of-week and month from raw dates to capture seasonal trends.",
        "Multiply two related features to test interaction effects.",
      ],
      relevantTerms: ["feature_engineering", "feature_selection", "multicollinearity"],
    },
    7: {
      title: "Step 7: Train Candidate Models",
      goal: "Train multiple diverse machine learning algorithms on your data.",
      why: "No single algorithm is best for every problem (the 'No Free Lunch' theorem). Comparing multiple models finds the true winner.",
      tips: [
        "Random Forest and Gradient Boosting typically lead on tabular data.",
        "Logistic Regression provides an unbeatable baseline for fast explainability.",
      ],
      relevantTerms: ["training_set", "validation_set", "cross_validation", "hyperparameter"],
    },
    8: {
      title: "Step 8: Compare Results & Generalization",
      goal: "Benchmark candidate models on validation and holdout test data.",
      why: "We compare Cross-Validation with unseen Holdout scores to verify honest generalization and prevent overfitting.",
      tips: [
        "Look for a small Generalization Gap (<0.05).",
        "Prioritize F1 score over raw Accuracy when classes are imbalanced.",
      ],
      relevantTerms: ["accuracy", "f1_score", "r2", "mae", "rmse", "overfitting"],
    },
    9: {
      title: "Step 9: Explain Predictions (SHAP)",
      goal: "Understand which features drive model decisions.",
      why: "Building trust and auditing for fairness ensures your model doesn't rely on deceptive artifacts.",
      tips: [
        "Features at the top of the SHAP summary chart have the highest impact.",
        "Remember: Statistical influence does not necessarily imply real-world causation.",
      ],
      relevantTerms: ["shap", "permutation_importance"],
    },
    10: {
      title: "Step 10: Final Project Report & Export",
      goal: "Generate audit-ready PDF, HTML, JSON, and ZIP project packages.",
      why: "Documenting all 15 technical steps ensures total reproducibility for university, enterprise, or regulatory review.",
      tips: [
        "Choose the Academic template for university submissions or Professional for enterprise.",
        "Enable original dataset inclusion in the ZIP bundle for 100% reproducibility.",
      ],
      relevantTerms: ["dataset", "target_column", "cross_validation"],
    },
  };

  const currentGuide = STEP_GUIDES[currentStep] || STEP_GUIDES[1];

  return (
    <aside
      aria-label="AI-Guided Assistance"
      className="rounded-2xl bg-[#0D0D1B] border border-white/[0.08] shadow-xl text-[#F7F7FC] overflow-hidden flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.08] bg-[#111126]">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-[#8265FF] to-[#A855F7] flex items-center justify-center text-white shadow-sm shadow-[#8265FF]/20">
            <Sparkles className="w-3.5 h-3.5" />
          </div>
          <div>
            <span className="font-bold text-xs text-white block">AutoVeyra Assistant</span>
            <span className="text-[10px] text-[#A4A4B8]">Intelligent Guidance Engine</span>
          </div>
        </div>

        <button
          onClick={onOpenGlossary}
          className="flex items-center gap-1 px-2 py-1 rounded-lg bg-white/[0.06] hover:bg-white/[0.1] text-[11px] text-[#8265FF] hover:text-white transition-all font-medium"
          title="Open Searchable ML Glossary"
        >
          <BookOpen className="w-3 h-3" />
          <span>Glossary</span>
        </button>
      </div>

      {/* Tabs */}
      <div className="flex items-center border-b border-white/[0.08] bg-[#0A0A16] px-2 pt-2 gap-1">
        <button
          onClick={() => setActiveTab("next_action")}
          className={`flex-1 py-1.5 px-2 rounded-t-lg text-xs font-semibold transition-all ${
            activeTab === "next_action"
              ? "bg-[#111126] text-white border-t border-x border-[#8265FF]/40 text-[#8265FF]"
              : "text-[#A4A4B8] hover:text-white"
          }`}
        >
          Next Action
        </button>
        <button
          onClick={() => setActiveTab("step_guide")}
          className={`flex-1 py-1.5 px-2 rounded-t-lg text-xs font-semibold transition-all ${
            activeTab === "step_guide"
              ? "bg-[#111126] text-white border-t border-x border-[#8265FF]/40 text-[#8265FF]"
              : "text-[#A4A4B8] hover:text-white"
          }`}
        >
          Step Guide
        </button>
        <button
          onClick={() => setActiveTab("preflight")}
          className={`flex-1 py-1.5 px-2 rounded-t-lg text-xs font-semibold transition-all ${
            activeTab === "preflight"
              ? "bg-[#111126] text-white border-t border-x border-[#8265FF]/40 text-[#8265FF]"
              : "text-[#A4A4B8] hover:text-white"
          }`}
        >
          Checklist
        </button>
      </div>

      {/* Body Content */}
      <div className="p-4 space-y-3 text-xs flex-1 overflow-y-auto max-h-[460px]">
        {/* Tab 1: Next Action */}
        {activeTab === "next_action" && (
          <div className="space-y-3">
            {recommendation ? (
              <div className="p-3.5 rounded-xl bg-[#171630] border border-[#8265FF]/30 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-[#8265FF]/20 text-[#8265FF]">
                    Recommended Action
                  </span>
                  <span className="text-[10px] text-[#A4A4B8]">
                    Target Step {recommendation.step}
                  </span>
                </div>

                <h4 className="font-bold text-white text-sm">
                  {recommendation.title}
                </h4>

                <p className="text-[#A4A4B8] text-xs leading-relaxed">
                  {recommendation.reason}
                </p>

                {recommendation.cta_action && onApplyRecommendation && (
                  <button
                    onClick={() => onApplyRecommendation(recommendation.step)}
                    className="w-full mt-2 py-2 px-3 rounded-lg text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-md shadow-[#8265FF]/20 transition-all flex items-center justify-center gap-1.5"
                  >
                    <span>{recommendation.cta_label || "Proceed to Step"}</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            ) : (
              <div className="p-3 rounded-xl bg-[#111126] border border-white/[0.08] text-center text-[#A4A4B8]">
                <Info className="w-4 h-4 mx-auto mb-1 text-[#8265FF]" />
                <span>Upload a dataset to generate recommendations.</span>
              </div>
            )}

            {/* Context Terms on this step */}
            {currentGuide.relevantTerms?.length > 0 && (
              <div className="p-3 rounded-xl bg-[#111126] border border-white/[0.06] space-y-2">
                <span className="font-semibold text-white block text-[11px]">
                  Concepts on this step:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {currentGuide.relevantTerms.map((termKey) => (
                    <HelpTooltip key={termKey} term={termKey} inline>
                      <span className="text-[11px] text-[#8265FF] hover:text-[#A855F7]">
                        {termKey.replace(/_/g, " ")}
                      </span>
                    </HelpTooltip>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Step Guide */}
        {activeTab === "step_guide" && (
          <div className="space-y-3">
            <div className="p-3 rounded-xl bg-[#111126] border border-white/[0.08] space-y-1.5">
              <h4 className="font-bold text-white text-xs">{currentGuide.title}</h4>
              <p className="text-white/90 text-xs">{currentGuide.goal}</p>
            </div>

            <div className="p-3 rounded-xl bg-[#171630] border border-[#8265FF]/20 space-y-1">
              <span className="font-semibold text-[#8265FF] text-[10px] uppercase tracking-wider block">
                Why this matters:
              </span>
              <p className="text-[#A4A4B8] text-xs leading-relaxed">
                {currentGuide.why}
              </p>
            </div>

            <div className="p-3 rounded-xl bg-[#0A0A16] border border-white/[0.06] space-y-2">
              <span className="font-semibold text-white text-[11px] block">
                Best practices:
              </span>
              <ul className="space-y-1.5 text-xs text-[#A4A4B8]">
                {currentGuide.tips.map((tip, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-[#8265FF] font-bold">•</span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Tab 3: Preflight Checklist */}
        {activeTab === "preflight" && (
          <div className="space-y-3">
            <div className="space-y-2">
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#111126] border border-white/[0.06]">
                <span className="text-xs text-[#A4A4B8]">Dataset Uploaded</span>
                {datasetInfo?.filename ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <span className="text-[10px] text-amber-400 font-medium">Pending</span>
                )}
              </div>

              <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#111126] border border-white/[0.06]">
                <span className="text-xs text-[#A4A4B8]">Target Column Selected</span>
                {datasetInfo?.target_column ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <span className="text-[10px] text-amber-400 font-medium">Pending</span>
                )}
              </div>

              <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#111126] border border-white/[0.06]">
                <span className="text-xs text-[#A4A4B8]">Data Quality Checked</span>
                {currentStep >= 4 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <span className="text-[10px] text-gray-500">Not started</span>
                )}
              </div>

              <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#111126] border border-white/[0.06]">
                <span className="text-xs text-[#A4A4B8]">Model Trained & Evaluated</span>
                {currentStep >= 8 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <span className="text-[10px] text-gray-500">Not started</span>
                )}
              </div>
            </div>

            {preflight?.errors?.length > 0 && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs space-y-1">
                <span className="font-bold flex items-center gap-1">
                  <ShieldAlert className="w-3.5 h-3.5" /> Issues to resolve:
                </span>
                <ul className="list-disc pl-4 space-y-0.5">
                  {preflight.errors.map((e, idx) => (
                    <li key={idx}>{e}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
