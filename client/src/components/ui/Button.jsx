// client/src/components/ui/Button.jsx
import React from "react";

export default function Button({
  children,
  variant = "primary", // 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger'
  size = "md", // 'sm' | 'md' | 'lg'
  icon: Icon,
  iconPosition = "left",
  isLoading = false,
  disabled = false,
  className = "",
  type = "button",
  onClick,
  ...props
}) {
  const baseStyles =
    "inline-flex items-center justify-center font-semibold rounded-xl transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#8265FF] focus-visible:ring-offset-2 focus-visible:ring-offset-[#070711] disabled:opacity-50 disabled:cursor-not-allowed select-none active:scale-[0.98]";

  const sizeStyles = {
    sm: "px-3 py-1.5 text-xs gap-1.5",
    md: "px-4 py-2 text-xs sm:text-sm gap-2",
    lg: "px-6 py-3 text-sm sm:text-base gap-2.5",
  };

  const variantStyles = {
    primary:
      "bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-md shadow-[#8265FF]/20 hover:shadow-[#8265FF]/35",
    secondary:
      "bg-[#16152F] hover:bg-[#1D1C3C] text-white border border-white/[0.1] hover:border-[#8265FF]/40",
    outline:
      "bg-transparent hover:bg-white/[0.05] text-[#F8F9FD] border border-white/[0.12] hover:border-white/[0.25]",
    ghost:
      "bg-transparent hover:bg-white/[0.06] text-[#9FA2B6] hover:text-white",
    danger:
      "bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/30 hover:border-rose-500/50",
  };

  return (
    <button
      type={type}
      disabled={disabled || isLoading}
      onClick={onClick}
      className={`${baseStyles} ${sizeStyles[size] || sizeStyles.md} ${
        variantStyles[variant] || variantStyles.primary
      } ${className}`}
      {...props}
    >
      {isLoading ? (
        <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin shrink-0" />
      ) : Icon && iconPosition === "left" ? (
        <Icon className="w-4 h-4 shrink-0" />
      ) : null}

      <span>{children}</span>

      {!isLoading && Icon && iconPosition === "right" && (
        <Icon className="w-4 h-4 shrink-0" />
      )}
    </button>
  );
}
