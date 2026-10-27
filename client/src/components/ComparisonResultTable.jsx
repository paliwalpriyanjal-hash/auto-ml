import React from 'react';
import { FiAward, FiAlertCircle } from 'react-icons/fi';

/**
 * Renders the cross-validation comparison table for all candidate models.
 *
 * Props:
 *   modelResults  – array of { model, cv_mean, cv_std, rank, status, error }
 *   rankingMetric – string label of the metric used to rank (e.g. "f1_macro")
 *   higherIsBetter – boolean, controls the arrow indicator
 */
const ComparisonResultTable = ({ modelResults = [], rankingMetric = '', higherIsBetter = true }) => {
  if (!modelResults || modelResults.length === 0) return null;

  // Sort: ranked models first (ascending by rank), then failed models
  const sorted = [...modelResults].sort((a, b) => {
    if (a.rank != null && b.rank != null) return a.rank - b.rank;
    if (a.rank != null) return -1;
    if (b.rank != null) return 1;
    return 0;
  });

  const metricLabel = rankingMetric.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <section className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
      <div className="p-6 border-b border-gray-200">
        <h3 className="text-xl font-semibold text-gray-800 flex items-center">
          <FiAward className="text-yellow-500 mr-2" size={20} />
          Model Comparison — Ranked by {metricLabel}
          <span className="ml-2 text-xs text-gray-500 font-normal">
            ({higherIsBetter ? '↑ higher is better' : '↓ lower is better'})
          </span>
        </h3>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">CV Mean</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">CV Std</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Error</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sorted.map((result, idx) => {
              const isWinner = result.rank === 1;
              const isFailed = result.status === 'failed';
              const rowBg = isWinner
                ? 'bg-yellow-50'
                : isFailed
                ? 'bg-red-50'
                : idx % 2 === 0
                ? 'bg-white'
                : 'bg-gray-50';

              return (
                <tr key={result.model} className={rowBg}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
                    {result.rank != null ? (
                      <span className={isWinner ? 'text-yellow-600' : 'text-gray-700'}>
                        {isWinner && '🏆 '}#{result.rank}
                      </span>
                    ) : (
                      <span className="text-gray-400">–</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-800">
                    {result.model}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {result.cv_mean != null ? (
                      <span className="font-mono text-gray-800">{result.cv_mean.toFixed(4)}</span>
                    ) : (
                      <span className="text-gray-400">–</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {result.cv_std != null ? (
                      <span className="font-mono text-gray-600">± {result.cv_std.toFixed(4)}</span>
                    ) : (
                      <span className="text-gray-400">–</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {isFailed ? (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                        <FiAlertCircle className="mr-1" size={12} /> Failed
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        Success
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-sm text-red-600 max-w-xs truncate" title={result.error || ''}>
                    {result.error || '–'}
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
