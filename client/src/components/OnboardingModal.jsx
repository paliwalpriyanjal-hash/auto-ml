// client/src/components/OnboardingModal.jsx
import React, { useState } from "react";
import { Sparkles, ArrowRight, ArrowLeft, CheckCircle2, Shield, Cpu, BarChart2, FileText, X } from "lucide-react";
import { setOnboardingCompleted } from "../utils/workspaceContext";

const ONBOARDING_SLIDES = [
  {
    step: 1,
    title: "Welcome to AutoVeyra",
    subtitle: "Your Intelligent Machine Learning Workspace",
    icon: Sparkles,
    badge: "Phase 6 Guided Experience",
    content: (
      <div className="space-y-3 text-sm text-[#A4A4B8] leading-relaxed">
        <p>
          AutoVeyra is designed so that <strong className="text-white">anyone can build reliable machine learning models</strong>—even with zero programming or data science background.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
          <div className="p-3 rounded-xl bg-[#111126] border border-[#8265FF]/20">
            <span className="font-bold text-white text-xs block mb-1">🧭 Guided Mode</span>
            <span className="text-xs text-[#A4A4B8]">Step-by-step walkthrough with sensible defaults and plain-English explanations.</span>
          </div>
          <div className="p-3 rounded-xl bg-[#111126] border border-white/[0.08]">
            <span className="font-bold text-white text-xs block mb-1">⚡ Auto Mode</span>
            <span className="text-xs text-[#A4A4B8]">One-click autonomous pipeline execution with pre-flight approval.</span>
          </div>
          <div className="p-3 rounded-xl bg-[#111126] border border-white/[0.08]">
            <span className="font-bold text-white text-xs block mb-1">🛠️ Advanced Mode</span>
            <span className="text-xs text-[#A4A4B8]">Granular control over tuning, feature engineering, and independent modules.</span>
          </div>
        </div>
      </div>
    ),
  },
  {
    step: 2,
    title: "1. Upload & Instant Profiling",
    subtitle: "Understand your data before making decisions",
    icon: BarChart2,
    badge: "Steps 1–3",
    content: (
      <div className="space-y-3 text-sm text-[#A4A4B8] leading-relaxed">
        <p>
          Simply drag and drop your tabular CSV dataset. AutoVeyra immediately analyzes:
        </p>
        <ul className="space-y-2 text-xs">
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong className="text-white">Dimensions & Types:</strong> Automatically categorizes numerical columns, text categories, and dates.</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong className="text-white">Data Quality Audit:</strong> Highlights missing values, duplicate records, and potential outlier values in plain language.</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong className="text-white">Cryptographic Integrity:</strong> Verifies SHA-256 file hashes to guarantee your original data is never altered.</span>
          </li>
        </ul>
      </div>
    ),
  },
  {
    step: 3,
    title: "2. Objective, Cleaning & Features",
    subtitle: "Prepare your data without technical confusion",
    icon: Shield,
    badge: "Steps 4–6",
    content: (
      <div className="space-y-3 text-sm text-[#A4A4B8] leading-relaxed">
        <p>
          Choose what you want to predict:
        </p>
        <ul className="space-y-2 text-xs">
          <li className="flex items-start gap-2">
            <span className="w-2 h-2 rounded-full bg-[#8265FF] shrink-0 mt-1.5" />
            <span><strong className="text-white">Classification:</strong> Predict categories (e.g. Yes/No, Churn/Retain, Spam/Ham).</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="w-2 h-2 rounded-full bg-[#A855F7] shrink-0 mt-1.5" />
            <span><strong className="text-white">Regression:</strong> Predict continuous numerical values (e.g. Prices, Revenue, Scores).</span>
          </li>
        </ul>
        <p className="text-xs pt-1">
          The <strong className="text-white">Data Cleaning Assistant</strong> and <strong className="text-white">Feature Engineering Engine</strong> recommend the best transformations, explaining the benefits and risks of each choice.
        </p>
      </div>
    ),
  },
  {
    step: 4,
    title: "3. Training & Honest Generalization",
    subtitle: "Real benchmarks without inflated scores",
    icon: Cpu,
    badge: "Steps 7–8",
    content: (
      <div className="space-y-3 text-sm text-[#A4A4B8] leading-relaxed">
        <p>
          AutoVeyra trains and benchmarks multiple state-of-the-art algorithms (Logistic Regression, Random Forest, Gradient Boosting, etc.) on your data.
        </p>
        <div className="p-3 rounded-xl bg-[#171630] border border-[#8265FF]/20 text-xs space-y-1.5">
          <span className="font-semibold text-[#8265FF] block">🛡️ Honest Generalization Guarantee</span>
          <p className="text-[#A4A4B8]">
            We rigorously compare cross-validation scores with holdout test scores to detect overfitting early. If a model scores 98% in training but drops on unseen data, we alert you immediately.
          </p>
        </div>
      </div>
    ),
  },
  {
    step: 5,
    title: "4. Explainability & Final Report",
    subtitle: "Understand why models decide, then export your work",
    icon: FileText,
    badge: "Steps 9–10",
    content: (
      <div className="space-y-3 text-sm text-[#A4A4B8] leading-relaxed">
        <p>
          Never settle for a black box. In the final stages:
        </p>
        <ul className="space-y-2 text-xs">
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong className="text-white">SHAP & Permutation Explanations:</strong> Discover exactly which features drive predictions.</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong className="text-white">Unified Project Report:</strong> Compile a 15-section technical report in PDF, HTML, JSON, or an all-inclusive ZIP bundle with your original dataset.</span>
          </li>
        </ul>
      </div>
    ),
  },
];

export default function OnboardingModal({ isOpen, onClose }) {
  const [currentSlide, setCurrentSlide] = useState(0);

  if (!isOpen) return null;

  const slide = ONBOARDING_SLIDES[currentSlide];
  const Icon = slide.icon;
  const isLast = currentSlide === ONBOARDING_SLIDES.length - 1;

  const handleFinish = () => {
    setOnboardingCompleted(true);
    onClose();
  };

  const handleNext = () => {
    if (isLast) {
      handleFinish();
    } else {
      setCurrentSlide((prev) => prev + 1);
    }
  };

  const handlePrev = () => {
    setCurrentSlide((prev) => Math.max(0, prev - 1));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-xl rounded-2xl bg-[#0D0D1B] border border-white/[0.12] shadow-2xl shadow-[#8265FF]/20 text-[#F7F7FC] overflow-hidden flex flex-col">
        
        {/* Header bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.08] bg-[#111126]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#8265FF] to-[#A855F7] flex items-center justify-center text-white">
              <Icon className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-bold tracking-wider text-[#8265FF] block">
                {slide.badge}
              </span>
              <h2 className="text-base font-bold text-white">{slide.title}</h2>
            </div>
          </div>
          <button
            onClick={handleFinish}
            className="p-1.5 rounded-lg text-[#A4A4B8] hover:text-white hover:bg-white/[0.06] transition-colors"
            aria-label="Skip tour"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Slide Content */}
        <div className="p-6 space-y-4">
          <p className="text-xs text-[#8265FF] font-medium">{slide.subtitle}</p>
          {slide.content}
        </div>

        {/* Footer controls & Progress dots */}
        <div className="px-6 py-4 border-t border-white/[0.08] bg-[#0A0A16] flex items-center justify-between">
          {/* Progress dots */}
          <div className="flex items-center gap-1.5">
            {ONBOARDING_SLIDES.map((s, idx) => (
              <button
                key={s.step}
                onClick={() => setCurrentSlide(idx)}
                aria-label={`Go to slide ${idx + 1}`}
                className={`h-2 rounded-full transition-all ${
                  currentSlide === idx
                    ? "w-6 bg-[#8265FF]"
                    : "w-2 bg-white/[0.15] hover:bg-white/[0.3]"
                }`}
              />
            ))}
          </div>

          {/* Navigation buttons */}
          <div className="flex items-center gap-2">
            {currentSlide > 0 && (
              <button
                onClick={handlePrev}
                className="px-3 py-1.5 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white bg-[#111126] border border-white/[0.08] transition-all flex items-center gap-1"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                Back
              </button>
            )}

            <button
              onClick={handleFinish}
              className="px-3 py-1.5 rounded-xl text-xs font-medium text-[#A4A4B8] hover:text-white transition-all"
            >
              Skip
            </button>

            <button
              onClick={handleNext}
              className="px-4 py-1.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-md shadow-[#8265FF]/20 transition-all flex items-center gap-1.5"
            >
              <span>{isLast ? "Get Started" : "Next"}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
