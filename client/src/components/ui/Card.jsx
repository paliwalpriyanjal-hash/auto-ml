// client/src/components/ui/Card.jsx
import React from "react";

export default function Card({
  children,
  title,
  subtitle,
  badge,
  action,
  icon: Icon,
  variant = "default", // 'default' | 'elevated' | 'gradient' | 'dashed'
  className = "",
  headerClassName = "",
  bodyClassName = "",
  ...props
}) {
  const variantStyles = {
    default: "bg-[#0F0F22] border border-white/[0.08]",
    elevated: "bg-[#16152F] border border-white/[0.12] shadow-xl shadow-black/40",
    gradient: "card-gradient",
    dashed: "bg-[#0B0B16] border-2 border-dashed border-white/[0.12]",
  };

  const hasHeader = title || subtitle || badge || action || Icon;

  return (
    <div
      className={`rounded-2xl transition-all ${variantStyles[variant] || variantStyles.default} ${className}`}
      {...props}
    >
      {hasHeader && (
        <div
          className={`flex items-center justify-between gap-4 p-5 sm:p-6 border-b border-white/[0.06] ${headerClassName}`}
        >
          <div className="flex items-start gap-3">
            {Icon && (
              <div className="w-8 h-8 rounded-xl bg-[#8265FF]/15 text-[#8265FF] flex items-center justify-center border border-[#8265FF]/25 shrink-0 mt-0.5">
                <Icon className="w-4 h-4" />
              </div>
            )}
            <div>
              <div className="flex items-center gap-2">
                {title && (
                  <h3 className="font-bold text-white text-base tracking-tight">
                    {title}
                  </h3>
                )}
                {badge}
              </div>
              {subtitle && (
                <p className="text-xs text-[#9FA2B6] mt-0.5 leading-relaxed">
                  {subtitle}
                </p>
              )}
            </div>
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}

      <div className={`p-5 sm:p-6 ${bodyClassName}`}>{children}</div>
    </div>
  );
}
