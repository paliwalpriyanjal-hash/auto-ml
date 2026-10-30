// client/src/components/ModeSwitcher.jsx
import React from "react";
import { Compass, Zap, Sliders, Check } from "lucide-react";
import { setExperienceMode } from "../utils/workspaceContext";

const MODES = [
  {
    id: "guided",
    label: "Guided Mode",
    shortLabel: "Guided",
    icon: Compass,
    badge: "Recommended",
    description: "Step-by-step assistant for beginners with automated defaults, plain-English advice, and mistake guardrails.",
  },
  {
    id: "auto",
    label: "Auto Mode",
    shortLabel: "Auto",
    icon: Zap,
    badge: "Fast",
    description: "One-click automated execution. Analyzes data, trains candidate models, and prepares reports with minimal interaction.",
  },
  {
    id: "advanced",
    label: "Advanced Mode",
    shortLabel: "Advanced",
    icon: Sliders,
    badge: "Expert",
    description: "Full experimental control. Direct access to algorithms, custom hyperparameters, and standalone modules.",
  },
];

export default function ModeSwitcher({ currentMode, onChange, compact = false }) {
  const handleSelect = (modeId) => {
    setExperienceMode(modeId);
    if (onChange) {
      onChange(modeId);
    }
  };

  if (compact) {
    return (
      <div className="inline-flex items-center p-1 rounded-xl bg-[#0D0D1B] border border-white/[0.08] shadow-inner">
        {MODES.map((m) => {
          const Icon = m.icon;
          const isActive = currentMode === m.id;
          return (
            <button
              key={m.id}
              onClick={() => handleSelect(m.id)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all duration-200 ${
                isActive
                  ? "bg-[#171630] text-white shadow-sm border border-[#8265FF]/40 text-[#8265FF]"
                  : "text-[#A4A4B8] hover:text-white hover:bg-white/[0.04]"
              }`}
              title={m.description}
            >
              <Icon className={`w-3.5 h-3.5 ${isActive ? "text-[#8265FF]" : "text-[#A4A4B8]"}`} />
              <span>{m.shortLabel}</span>
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
      {MODES.map((m) => {
        const Icon = m.icon;
        const isActive = currentMode === m.id;
        return (
          <button
            key={m.id}
            type="button"
            onClick={() => handleSelect(m.id)}
            className={`text-left p-4 rounded-xl border transition-all duration-200 relative group flex flex-col justify-between ${
              isActive
                ? "bg-[#171630] border-[#8265FF] shadow-lg shadow-[#8265FF]/15"
                : "bg-[#111126] border-white/[0.08] hover:border-white/[0.2] hover:bg-[#14142d]"
            }`}
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div
                    className={`w-7 h-7 rounded-lg flex items-center justify-center ${
                      isActive
                        ? "bg-[#8265FF] text-white shadow-md shadow-[#8265FF]/30"
                        : "bg-white/[0.06] text-[#A4A4B8] group-hover:text-white"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="font-bold text-sm text-white">{m.label}</span>
                </div>
                {m.badge && (
                  <span
                    className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full ${
                      isActive
                        ? "bg-[#8265FF]/20 text-[#8265FF] border border-[#8265FF]/30"
                        : "bg-white/[0.06] text-[#A4A4B8]"
                    }`}
                  >
                    {m.badge}
                  </span>
                )}
              </div>
              <p className="text-xs text-[#A4A4B8] leading-relaxed">
                {m.description}
              </p>
            </div>

            {isActive && (
              <div className="mt-3 pt-2 border-t border-[#8265FF]/20 flex items-center gap-1.5 text-[11px] font-semibold text-[#8265FF]">
                <Check className="w-3.5 h-3.5" />
                <span>Active Mode</span>
              </div>
            )}
          </button>
        );
      })}
    </div>
  );
}
