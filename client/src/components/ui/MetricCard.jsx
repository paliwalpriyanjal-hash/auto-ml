// client/src/components/ui/MetricCard.jsx
import React from "react";

export default function MetricCard({
  title,
  value,
  subtitle,
  change,
  changeType = "neutral", // 'positive' | 'negative' | 'neutral'
  icon: Icon,
  className = "",
  valueClassName = "",
  ...props
}) {
  const changeStyles = {
    positive: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    negative: "text-rose-400 bg-rose-500/10 border-rose-500/20",
    neutral: "text-[#9FA2B6] bg-white/[0.04] border-white/[0.08]",
  };

  return (
    <div
      className={`p-5 rounded-2xl bg-[#0F0F22] border border-white/[0.08] hover:border-[#8265FF]/30 transition-all space-y-2 ${className}`}
      {...props}
    >
      <div className="flex items-center justify-between text-[#9FA2B6]">
        <span className="text-xs font-semibold tracking-wide uppercase">{title}</span>
        {Icon && (
          <div className="w-7 h-7 rounded-lg bg-white/[0.04] text-[#8265FF] flex items-center justify-center">
            <Icon className="w-3.5 h-3.5" />
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-2">
        <span
          className={`text-2xl sm:text-3xl font-extrabold text-white tracking-tight ${valueClassName}`}
        >
          {value}
        </span>
        {change && (
          <span
            className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${
              changeStyles[changeType] || changeStyles.neutral
            }`}
          >
            {change}
          </span>
        )}
      </div>

      {subtitle && <p className="text-xs text-[#9FA2B6] leading-relaxed">{subtitle}</p>}
    </div>
  );
}
