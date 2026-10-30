// client/src/components/HelpTooltip.jsx
import React, { useState, useRef, useEffect } from "react";
import { HelpCircle, Info, X, ExternalLink } from "lucide-react";
import { ML_GLOSSARY } from "../utils/glossary";

export default function HelpTooltip({
  term,
  title,
  text,
  why,
  example,
  className = "",
  inline = false,
  children,
}) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  // Lookup glossary entry if a term key was provided
  const glossaryKey = term ? term.toLowerCase().replace(/[\s-]/g, "_") : null;
  const glossaryEntry = glossaryKey ? ML_GLOSSARY[glossaryKey] : null;

  const displayTitle = title || glossaryEntry?.term || term || "Concept Help";
  const displayDefinition = text || glossaryEntry?.simple_definition || "";
  const displayWhy = why || glossaryEntry?.why_it_matters || "";
  const displayExample = example || glossaryEntry?.example || "";
  const displayLocation = glossaryEntry?.where_in_autoveyra || "";

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  return (
    <div
      ref={containerRef}
      className={`relative inline-flex items-center ${className}`}
    >
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setIsOpen(!isOpen);
          }
        }}
        aria-label={`Help about ${displayTitle}`}
        title={`Click for explanation: ${displayTitle}`}
        className="inline-flex items-center gap-1 text-[#8265FF] hover:text-[#A855F7] transition-colors focus:outline-none focus:ring-1 focus:ring-[#8265FF] rounded"
      >
        {children ? (
          <span className="underline decoration-dotted decoration-[#8265FF]/60 underline-offset-2">
            {children}
          </span>
        ) : (
          <HelpCircle className="w-3.5 h-3.5 opacity-80 hover:opacity-100" />
        )}
      </button>

      {isOpen && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 sm:w-80 p-3.5 rounded-xl bg-[#0D0D1B] border border-[#8265FF]/40 shadow-2xl shadow-[#8265FF]/20 text-left text-xs text-[#F7F7FC] backdrop-blur-xl animate-in fade-in zoom-in-95 duration-150">
          <div className="flex items-start justify-between gap-2 border-b border-white/[0.08] pb-2 mb-2">
            <div className="flex items-center gap-1.5 font-bold text-white text-sm">
              <Info className="w-4 h-4 text-[#8265FF]" />
              <span>{displayTitle}</span>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-[#A4A4B8] hover:text-white p-0.5"
              aria-label="Close tooltip"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-2">
            <div>
              <p className="text-[#E2E2F0] leading-relaxed">{displayDefinition}</p>
            </div>

            {displayWhy && (
              <div className="p-2 rounded-lg bg-[#171630] border border-[#8265FF]/20">
                <span className="font-semibold text-[#8265FF] block text-[11px] uppercase tracking-wider mb-0.5">
                  Why this matters
                </span>
                <p className="text-[#A4A4B8] text-[11px] leading-relaxed">
                  {displayWhy}
                </p>
              </div>
            )}

            {displayExample && (
              <div className="text-[11px] text-[#A4A4B8]">
                <strong className="text-white">Example: </strong>
                <span>{displayExample}</span>
              </div>
            )}

            {displayLocation && (
              <div className="pt-1.5 border-t border-white/[0.06] flex items-center justify-between text-[10px] text-[#A4A4B8]">
                <span>Appears in: {displayLocation}</span>
              </div>
            )}
          </div>

          {/* Pointing triangle */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-[#0D0D1B]" />
        </div>
      )}
    </div>
  );
}
