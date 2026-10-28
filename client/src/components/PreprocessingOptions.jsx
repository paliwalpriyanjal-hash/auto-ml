import React from "react";
import { FiSettings, FiHelpCircle } from "react-icons/fi";

const PreprocessingOptions = ({
  missingStrategy = "drop",
  setMissingStrategy,
  encodingStrategy = "label",
  setEncodingStrategy,
  scaler = "standard",
  setScaler,
  splitRatio = "0.8",
  setSplitRatio,
}) => (
  <section className="bg-[#111126] p-6 rounded-2xl border border-white/[0.08] shadow-xl mb-6">
    <div className="flex items-center mb-5 border-b border-white/[0.06] pb-3">
      <div className="bg-[#8265FF]/10 p-2.5 rounded-xl mr-3 border border-[#8265FF]/20">
        <FiSettings className="text-[#8265FF]" size={18} />
      </div>
      <div>
        <h3 className="text-base font-bold text-white">Pipeline Preprocessing Setup</h3>
        <p className="text-xs text-[#A4A4B8]">Deterministic data transformation steps fitted on training folds</p>
      </div>
    </div>

    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Missing Values */}
      <div className="bg-[#171630] p-4 rounded-xl border border-white/[0.06] space-y-2">
        <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 flex items-center justify-between">
          <span>Missing Values</span>
          <span className="text-[#A4A4B8]" title="Strategy for handling missing cells in dataset">
            <FiHelpCircle size={14} />
          </span>
        </label>
        <select
          value={missingStrategy}
          onChange={(e) => setMissingStrategy(e.target.value)}
          className="w-full px-3 py-2 bg-[#0D0D1B] border border-white/[0.1] rounded-lg text-xs text-white focus:outline-none focus:border-[#8265FF]"
        >
          <option value="drop">Drop Rows</option>
          <option value="mean">Fill with Mean (Numeric)</option>
          <option value="median">Fill with Median (Numeric)</option>
        </select>
        <p className="text-[10px] text-[#A4A4B8]">Applied automatically per column</p>
      </div>

      {/* Categorical Encoding */}
      <div className="bg-[#171630] p-4 rounded-xl border border-white/[0.06] space-y-2">
        <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 flex items-center justify-between">
          <span>Categorical Encoding</span>
          <span className="text-[#A4A4B8]" title="How string & categorical columns are mapped to numbers">
            <FiHelpCircle size={14} />
          </span>
        </label>
        <select
          value={encodingStrategy}
          onChange={(e) => setEncodingStrategy(e.target.value)}
          className="w-full px-3 py-2 bg-[#0D0D1B] border border-white/[0.1] rounded-lg text-xs text-white focus:outline-none focus:border-[#8265FF]"
        >
          <option value="onehot">One-Hot Encoding (Recommended)</option>
          <option value="label">Label Encoding</option>
        </select>
        <p className="text-[10px] text-[#A4A4B8]">Handles unseen categories safely</p>
      </div>

      {/* Feature Scaling */}
      <div className="bg-[#171630] p-4 rounded-xl border border-white/[0.06] space-y-2">
        <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 flex items-center justify-between">
          <span>Feature Scaling</span>
          <span className="text-[#A4A4B8]" title="Numeric normalization & scaling method">
            <FiHelpCircle size={14} />
          </span>
        </label>
        <select
          value={scaler}
          onChange={(e) => setScaler(e.target.value)}
          className="w-full px-3 py-2 bg-[#0D0D1B] border border-white/[0.1] rounded-lg text-xs text-white focus:outline-none focus:border-[#8265FF]"
        >
          <option value="standard">StandardScaler (Zero mean, unit var)</option>
          <option value="minmax">MinMaxScaler [0, 1]</option>
        </select>
        <p className="text-[10px] text-[#A4A4B8]">Prevents feature scale dominance</p>
      </div>

      {/* Train/Test Split */}
      <div className="bg-[#171630] p-4 rounded-xl border border-white/[0.06] space-y-2">
        <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 flex items-center justify-between">
          <span>Train / Test Split</span>
          <span className="text-[#A4A4B8]" title="Partition ratio for model holdout validation">
            <FiHelpCircle size={14} />
          </span>
        </label>
        <select
          value={splitRatio}
          onChange={(e) => setSplitRatio(e.target.value)}
          className="w-full px-3 py-2 bg-[#0D0D1B] border border-white/[0.1] rounded-lg text-xs text-white focus:outline-none focus:border-[#8265FF]"
        >
          <option value="0.8">80% Train / 20% Holdout Test</option>
          <option value="0.7">70% Train / 30% Holdout Test</option>
        </select>
        <p className="text-[10px] text-[#A4A4B8]">Stratified for classification</p>
      </div>
    </div>
  </section>
);

export default PreprocessingOptions;
