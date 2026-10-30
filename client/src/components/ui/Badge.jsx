// client/src/components/ui/Badge.jsx
import React from "react";

export default function Badge({
  children,
  variant = "violet", // 'violet' | 'success' | 'warning' | 'danger' | 'info' | 'neutral'
  size = "md", // 'sm' | 'md'
  icon: Icon,
  className = "",
  ...props
}) {
  const baseStyles =
    "inline-flex items-center font-semibold rounded-full tracking-wide uppercase select-none";

  const sizeStyles = {
    sm: "px-2 py-0.5 text-[9px] gap-1",
    md: "px-2.5 py-1 text-[10px] sm:text-[11px] gap-1.5",
  };

  const variantStyles = {
    violet:
      "bg-[#8265FF]/15 text-[#C084FC] border border-[#8265FF]/30",
    success:
      "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30",
    warning:
      "bg-amber-500/15 text-amber-300 border border-amber-500/30",
    danger:
      "bg-rose-500/15 text-rose-300 border border-rose-500/30",
    info:
      "bg-blue-500/15 text-blue-300 border border-blue-500/30",
    neutral:
      "bg-white/[0.06] text-[#9FA2B6] border border-white/[0.1]",
  };

  return (
    <span
      className={`${baseStyles} ${sizeStyles[size] || sizeStyles.md} ${
        variantStyles[variant] || variantStyles.violet
      } ${className}`}
      {...props}
    >
      {Icon && <Icon className="w-3 h-3 shrink-0" />}
      <span>{children}</span>
    </span>
  );
}
