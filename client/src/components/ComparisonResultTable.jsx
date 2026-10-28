import React from "react";
import { FiAward, FiAlertCircle, FiCheckCircle } from "react-icons/fi";

const ComparisonResultTable = ({
  modelResults = [],
  rankingMetric = "",
  higherIsBetter = true,
}) => {
  if (!modelResults || modelResults.length === 0) return null;

  // Sort: ranked models first (ascending by rank), then failed models
  const sorted = [...modelResults].sort((a, b) => {
    if (a.rank != null && b.rank != null) return a.rank - b.rank;
    if (a.rank != null) return -1;
    if (b.rank != null) return 1;
    return 0;
  });

  const metricLabel = rankingMetric
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <section className="bg-[#111126] rounded-2xl border border-white/[0.08] shadow-2xl overflow-hidden mb-8">
      <div className="p-6 border-b border-white/[0.06] flex items-center justify-between">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <FiAward className="text-[#8265FF]" size={20} />
          <span>Cross-Validation Leaderboard — Ranked by {metricLabel}</span>
          <span className="text-xs text-[#A4A4B8] font-normal ml-2">
            ({higherIsBetter ? "↑ higher is better" : "↓ lower is better"})
          </span>
        </h3>
        <span className="text-xs font-mono text-[#A4A4B8] bg-[#171630] px-3 py-1 rounded-full border border-white/[0.06]">
          {sorted.length} Models Tested
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-white/[0.06]">
          <thead className="bg-[#0D0D1B]">
            <tr>
              <th className="px-6 py-3.5 text-left text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider">
                Rank
              </th>
              <th className="px-6 py-3.5 text-left text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider">
                Algorithm
              </th>
              <th className="px-6 py-3.5 text-left text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider">
                CV Mean Score
              </th>
              <th className="px-6 py-3.5 text-left text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider">
                CV Variance (Std)
              </th>
              <th className="px-6 py-3.5 text-left text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3.5 text-left text-[11px] font-semibold text-[#A4A4B8] uppercase tracking-wider">
                Diagnostics
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/[0.04]">
            {sorted.map((result) => {
              const isWinner = result.rank === 1;
              const isFailed = result.status === "failed";

              return (
                <tr
                  key={result.model}
                  className={`transition-colors ${
                    isWinner
                      ? "bg-[#171630]/90 border-l-4 border-l-[#8265FF]"
                      : isFailed
                      ? "bg-rose-950/20"
                      : "hover:bg-white/[0.02]"
                  }`}
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
                    {result.rank != null ? (
                      <span className={isWinner ? "text-[#8265FF] flex items-center gap-1.5" : "text-gray-400"}>
                        {isWinner && <FiAward className="text-[#8265FF] w-4 h-4" />}
                        #{result.rank}
                      </span>
                    ) : (
                      <span className="text-gray-500">—</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-white">
                    {result.model}
                    {isWinner && (
                      <span className="ml-2.5 text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#8265FF]/20 text-[#8265FF] border border-[#8265FF]/30">
                        Top Performer
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {result.cv_mean != null ? (
                      <span className="font-mono font-bold text-emerald-400">
                        {result.cv_mean.toFixed(4)}
                      </span>
                    ) : (
                      <span className="text-gray-500">—</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-[#A4A4B8]">
                    {result.cv_std != null ? `± ${result.cv_std.toFixed(4)}` : "—"}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-xs">
                    {isFailed ? (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full font-medium bg-rose-500/15 text-rose-300 border border-rose-500/30">
                        <FiAlertCircle className="mr-1" size={12} /> Failed
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                        <FiCheckCircle className="mr-1" size={12} /> Success
                      </span>
                    )}
                  </td>
                  <td
                    className="px-6 py-4 text-xs text-rose-400 max-w-xs truncate"
                    title={result.error || ""}
                  >
                    {result.error || <span className="text-[#A4A4B8]">—</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default ComparisonResultTable;
