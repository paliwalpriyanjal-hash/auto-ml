// client/src/components/ui/EmptyState.jsx
import React from "react";
import Button from "./Button";

export default function EmptyState({
  title = "No Data Found",
  description = "No records or results have been generated yet.",
  icon: Icon,
  actionLabel,
  onAction,
  actionIcon,
  className = "",
  children,
}) {
  return (
    <div
      className={`p-10 sm:p-14 rounded-2xl bg-[#0F0F22]/60 border border-white/[0.06] text-center flex flex-col items-center justify-center space-y-3.5 ${className}`}
    >
      {Icon && (
        <div className="w-12 h-12 rounded-2xl bg-[#8265FF]/10 text-[#8265FF] flex items-center justify-center mb-1">
          <Icon className="w-6 h-6" />
        </div>
      )}

      <div className="space-y-1 max-w-md">
        <h3 className="font-bold text-white text-base tracking-tight">{title}</h3>
        <p className="text-xs sm:text-sm text-[#9FA2B6] leading-relaxed">
          {description}
        </p>
      </div>

      {actionLabel && onAction && (
        <div className="pt-2">
          <Button
            variant="primary"
            size="sm"
            onClick={onAction}
            icon={actionIcon}
          >
            {actionLabel}
          </Button>
        </div>
      )}

      {children}
    </div>
  );
}
