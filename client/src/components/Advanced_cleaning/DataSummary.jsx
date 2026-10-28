import React from 'react';
import { FiAlertTriangle, FiCheckCircle, FiDatabase, FiList, FiBarChart2, FiType } from 'react-icons/fi';

export default function DataSummary({ stats, columns, className = '' }) {
  if (!stats) return null;

  // Count data types for visualization
  const typeCounts = columns.reduce((acc, col) => {
    const type = stats.columns[col.name]?.type || 'unknown';
    acc[type] = (acc[type] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className={`bg-[#111126] p-6 rounded-2xl border border-white/[0.08] shadow-xl ${className}`}>
      <div className="flex items-center justify-between mb-6 border-b border-white/[0.06] pb-4">
        <div>
          <h3 className="text-base font-bold text-white">Dataset Summary & Health</h3>
          <p className="text-xs text-[#A4A4B8]">Automated structural inspection of uploaded tabular file</p>
        </div>
        <div className="flex items-center space-x-2 bg-[#171630] border border-[#8265FF]/30 px-3.5 py-1.5 rounded-full text-xs font-mono">
          <FiDatabase className="text-[#8265FF]" />
          <span className="font-semibold text-white">
            {stats.totalRows.toLocaleString()} rows × {columns.length} columns
          </span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Overview Card */}
        <div className="border border-white/[0.08] rounded-xl p-5 bg-[#0D0D1B] space-y-3">
          <div className="flex items-center mb-3">
            <div className="p-2 rounded-lg bg-[#8265FF]/15 text-[#8265FF] mr-3">
              <FiBarChart2 />
            </div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-white">Volume Metrics</h4>
          </div>
          
          <div className="space-y-2 text-xs">
            <div className="flex justify-between items-center py-1.5 border-b border-white/[0.04]">
              <span className="text-[#A4A4B8] flex items-center">
                <FiDatabase className="mr-2 text-[#8265FF]" /> Total Rows
              </span>
              <span className="font-mono font-bold text-white">{stats.totalRows.toLocaleString()}</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-white/[0.04]">
              <span className="text-[#A4A4B8] flex items-center">
                <FiList className="mr-2 text-[#A855F7]" /> Features
              </span>
              <span className="font-mono font-bold text-white">{columns.length}</span>
            </div>
            <div className="flex justify-between items-center py-1.5">
              <span className="text-[#A4A4B8] flex items-center">
                <FiType className="mr-2 text-amber-400" /> Missing Cells
              </span>
              <span className="font-mono font-bold text-amber-400">
                {Object.values(stats.columns).reduce((sum, col) => sum + (col.emptyCount || 0), 0)}
              </span>
            </div>
          </div>
        </div>

        {/* Data Types Card */}
        <div className="border border-white/[0.08] rounded-xl p-5 bg-[#0D0D1B] space-y-3">
          <div className="flex items-center mb-3">
            <div className="p-2 rounded-lg bg-[#A855F7]/15 text-[#A855F7] mr-3">
              <FiType />
            </div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-white">Inferred Schema</h4>
          </div>
          
          <div className="flex flex-wrap gap-1.5 mb-2">
            {Object.entries(typeCounts).map(([type, count]) => (
              <span 
                key={type} 
                className="px-2.5 py-0.5 text-[10px] font-mono font-semibold rounded-md uppercase bg-[#171630] border border-white/[0.08] text-[#8265FF]"
              >
                {type}: {count}
              </span>
            ))}
          </div>
          
          <div className="space-y-1.5 max-h-32 overflow-y-auto pr-1 text-xs">
            {columns.slice(0, 6).map((col) => {
              const colInfo = stats.columns[col.name] || {};
              return (
                <div key={col.name} className="flex justify-between items-center py-1 border-b border-white/[0.04] text-[11px]">
                  <span className="text-[#A4A4B8] truncate max-w-[120px]" title={col.name}>
                    {col.name}
                  </span>
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-[#8265FF] text-[10px]">
                      {colInfo.type}
                    </span>
                    {colInfo.emptyCount > 0 && (
                      <span className="text-[10px] bg-rose-500/20 text-rose-300 px-1.5 py-0.5 rounded font-mono">
                        {colInfo.emptyCount} null
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Issues Card */}
        <div className="border border-white/[0.08] rounded-xl p-5 bg-[#0D0D1B] space-y-3">
          <div className="flex items-center mb-3">
            <div className="p-2 rounded-lg bg-amber-500/15 text-amber-400 mr-3">
              <FiAlertTriangle />
            </div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-white">Quality Diagnostics</h4>
          </div>
          
          {stats.potentialIssues.length > 0 ? (
            <div className="space-y-2 max-h-36 overflow-y-auto pr-1 text-xs">
              {stats.potentialIssues.slice(0, 4).map((issue, idx) => (
                <div key={idx} className="flex items-start gap-2 p-2 rounded-lg bg-[#171630]/60 border border-white/[0.04]">
                  <span className="text-amber-400 text-xs shrink-0">•</span>
                  <div className="truncate">
                    <span className="text-xs font-semibold text-white block truncate">
                      {issue.column}
                    </span>
                    <span className="text-[10px] text-[#A4A4B8]">
                      {formatIssue(issue.issue)}: {issue.count} rows
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex items-center py-4 px-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-300 text-xs gap-2">
              <FiCheckCircle className="text-emerald-400 shrink-0" />
              <span>Zero missing values detected</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Helper functions and styles
const typeColors = {
  number: { bg: '#e0f2fe', text: '#0369a1' },
  string: { bg: '#f0fdf4', text: '#15803d' },
  boolean: { bg: '#fef9c3', text: '#854d0e' },
  date: { bg: '#f3e8ff', text: '#7e22ce' },
  datetime: { bg: '#ede9fe', text: '#6d28d9' },
  object: { bg: '#fee2e2', text: '#b91c1c' }
};

function formatIssue(issue) {
  return issue
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}