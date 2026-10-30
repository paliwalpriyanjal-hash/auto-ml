// client/src/components/ui/LoadingSkeleton.jsx
import React from "react";

export default function LoadingSkeleton({
  lines = 3,
  className = "",
  card = false,
}) {
  if (card) {
    return (
      <div
        className={`p-6 rounded-2xl bg-[#0F0F22] border border-white/[0.08] space-y-4 animate-pulse ${className}`}
      >
        <div className="h-4 w-1/3 bg-white/[0.06] rounded-md" />
        <div className="space-y-2">
          <div className="h-3 w-full bg-white/[0.04] rounded-md" />
          <div className="h-3 w-5/6 bg-white/[0.04] rounded-md" />
        </div>
        <div className="h-8 w-1/4 bg-white/[0.06] rounded-lg pt-2" />
      </div>
    );
  }

  return (
    <div className={`space-y-2.5 animate-pulse ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-3.5 bg-white/[0.06] rounded-md"
          style={{ width: `${Math.max(45, 100 - i * 15)}%` }}
        />
      ))}
    </div>
  );
}
