// client/src/components/GlossaryModal.jsx
import React, { useState, useMemo } from "react";
import { BookOpen, Search, X, Tag, ExternalLink, HelpCircle } from "lucide-react";
import { ML_GLOSSARY, GLOSSARY_CATEGORIES } from "../utils/glossary";

export default function GlossaryModal({ isOpen, onClose, initialSearch = "" }) {
  const [searchTerm, setSearchTerm] = useState(initialSearch);
  const [selectedCategory, setSelectedCategory] = useState("All");

  const termsList = useMemo(() => Object.values(ML_GLOSSARY), []);

  const filteredTerms = useMemo(() => {
    return termsList.filter((item) => {
      const matchesCat =
        selectedCategory === "All" || item.category === selectedCategory;
      const q = searchTerm.toLowerCase().trim();
      const matchesSearch =
        !q ||
        item.term.toLowerCase().includes(q) ||
        item.simple_definition.toLowerCase().includes(q) ||
        item.example.toLowerCase().includes(q);
      return matchesCat && matchesSearch;
    });
  }, [termsList, selectedCategory, searchTerm]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-4xl max-h-[85vh] flex flex-col rounded-2xl bg-[#0D0D1B] border border-white/[0.12] shadow-2xl shadow-[#8265FF]/20 text-[#F7F7FC] overflow-hidden">
        
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.08] bg-[#111126]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-[#8265FF]/20 text-[#8265FF] flex items-center justify-center border border-[#8265FF]/30">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                Machine Learning Glossary
                <span className="text-xs px-2 py-0.5 rounded-full bg-[#8265FF]/10 text-[#8265FF] border border-[#8265FF]/20">
                  {filteredTerms.length} Concepts
                </span>
              </h2>
              <p className="text-xs text-[#A4A4B8]">
                Plain-English explanations for complex data science and machine learning terminology.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-[#A4A4B8] hover:text-white hover:bg-white/[0.06] transition-colors"
            aria-label="Close glossary"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Category Filter Controls */}
        <div className="p-6 border-b border-white/[0.08] space-y-4 bg-[#0A0A16]">
          {/* Search bar */}
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-[#A4A4B8]" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search concepts, algorithms, metrics (e.g. Overfitting, SHAP, F1 Score)..."
              className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-[#111126] border border-white/[0.12] focus:border-[#8265FF] text-sm text-white placeholder-[#71718A] focus:outline-none focus:ring-1 focus:ring-[#8265FF] transition-all"
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-[#A4A4B8] hover:text-white"
              >
                Clear
              </button>
            )}
          </div>

          {/* Category Tabs */}
          <div className="flex flex-wrap gap-1.5">
            {GLOSSARY_CATEGORIES.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  selectedCategory === cat
                    ? "bg-[#8265FF] text-white shadow-md shadow-[#8265FF]/20"
                    : "bg-[#111126] text-[#A4A4B8] hover:text-white hover:bg-[#171630] border border-white/[0.06]"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Content list */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {filteredTerms.length === 0 ? (
            <div className="text-center py-12 text-[#A4A4B8] space-y-2">
              <HelpCircle className="w-8 h-8 mx-auto opacity-50 text-[#8265FF]" />
              <p className="text-sm">No machine learning terms matched "{searchTerm}".</p>
              <button
                onClick={() => { setSearchTerm(""); setSelectedCategory("All"); }}
                className="text-xs text-[#8265FF] hover:underline"
              >
                Reset search filters
              </button>
            </div>
          ) : (
            filteredTerms.map((item) => (
              <div
                key={item.term}
                className="p-4 rounded-xl bg-[#111126]/80 border border-white/[0.08] hover:border-[#8265FF]/40 transition-all space-y-2.5"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-bold text-white text-base tracking-tight">
                    {item.term}
                  </h3>
                  <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-[#8265FF]/10 text-[#8265FF] border border-[#8265FF]/20">
                    {item.category}
                  </span>
                </div>

                <p className="text-sm text-[#E2E2F0] leading-relaxed">
                  {item.simple_definition}
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                  <div className="p-2.5 rounded-lg bg-[#171630] border border-[#8265FF]/15">
                    <span className="text-[10px] uppercase font-bold text-[#8265FF] block tracking-wider mb-0.5">
                      Why It Matters
                    </span>
                    <p className="text-xs text-[#A4A4B8] leading-relaxed">
                      {item.why_it_matters}
                    </p>
                  </div>

                  <div className="p-2.5 rounded-lg bg-[#070711] border border-white/[0.06]">
                    <span className="text-[10px] uppercase font-bold text-gray-400 block tracking-wider mb-0.5">
                      Practical Example
                    </span>
                    <p className="text-xs text-[#A4A4B8] leading-relaxed">
                      {item.example}
                    </p>
                  </div>
                </div>

                {item.where_in_autoveyra && (
                  <div className="text-[11px] text-[#71718A] pt-1">
                    <span className="text-[#A4A4B8] font-medium">In AutoVeyra: </span>
                    {item.where_in_autoveyra}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-white/[0.08] bg-[#0A0A16] flex items-center justify-between text-xs text-[#A4A4B8]">
          <span>Tip: Hover over underlined concepts throughout AutoVeyra for instant definitions.</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-white/[0.08] hover:bg-white/[0.12] text-white font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
