// client/src/components/ui/PageHeader.jsx
import React from "react";
import Badge from "./Badge";

export default function PageHeader({
  title,
  subtitle,
  category,
  badge,
  actions,
  breadcrumbs = [],
  className = "",
}) {
  return (
    <div
      className={`flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-white/[0.08] ${className}`}
    >
      <div className="space-y-1.5">
        {/* Category or breadcrumb */}
        <div className="flex items-center gap-2">
          {category && (
            <Badge variant="violet" size="sm">
              {category}
            </Badge>
          )}
          {badge}
        </div>

        {/* Title */}
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          {title}
        </h1>

        {/* Subtitle */}
        {subtitle && (
          <p className="text-xs sm:text-sm text-[#9FA2B6] max-w-3xl leading-relaxed">
            {subtitle}
          </p>
        )}
      </div>

      {/* Action CTA buttons */}
      {actions && (
        <div className="flex flex-wrap items-center gap-2.5 shrink-0 pt-1 md:pt-0">
          {actions}
        </div>
      )}
    </div>
  );
}
