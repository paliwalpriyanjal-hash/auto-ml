import React, { useState, useEffect } from 'react';
import { getApiBase } from '../../utils/api';
import { FiDownload, FiCheckCircle, FiAlertTriangle, FiLoader } from 'react-icons/fi';

export default function AdvancedResult({ csv, preprocessing = null, columns = [] }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [cleanedCSV, setCleanedCSV] = useState('');
  const [previewData, setPreviewData] = useState([]);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    if (!csv) return;

    const sendToBackend = async () => {
      setLoading(true);
      setError('');
      setCleanedCSV('');
      setStats(null);

      const apiBase = getApiBase();

      try {
        const formData = new FormData();
        formData.append('file', csv);
        
        // Send preprocessing options as JSON if they exist
        if (preprocessing) {
          const availableCols = columns.map(c => (typeof c === 'object' && c !== null ? c.name : c));
          formData.append('preprocessing', JSON.stringify({
            ...preprocessing,
            available_columns: availableCols
          }));
        }

        const res = await fetch(`${apiBase}/api/clean`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.detail || errorData.message || `Server responded with ${res.status}`);
        }

        const data = await res.json();
        setCleanedCSV(data.csv);
        setStats(data.stats);
        
        // Create preview of cleaned data
        if (data.csv) {
          const lines = data.csv.split('\n').filter(Boolean).slice(0, 6);
          if (lines.length > 0) {
            const headers = lines[0].split(',');
            const previewRows = lines.slice(1).map(line => {
              const cells = line.split(',');
              const obj = {};
              headers.forEach((header, idx) => {
                obj[header] = cells[idx] ?? '';
              });
              return obj;
            });
            setPreviewData(previewRows);
          }
        }
      } catch (err) {
        console.error(err);
        setError(err.message || 'Failed to clean dataset.');
      } finally {
        setLoading(false);
      }
    };

    sendToBackend();
  }, [csv, preprocessing, columns]);

  const handleDownload = () => {
    if (!cleanedCSV) return;
    const blob = new Blob([cleanedCSV], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cleaned_data.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {loading && (
        <div className="bg-[#111126] border border-white/[0.08] rounded-2xl p-16 flex flex-col items-center justify-center text-center space-y-4">
          <FiLoader className="h-10 w-10 animate-spin text-[#8265FF]" />
          <h3 className="text-xl font-bold text-white">Cleaning & Transforming Dataset...</h3>
          <p className="text-xs text-[#A4A4B8] max-w-sm">
            Applying imputation, scaling, outlier filters, and selected encoding matrices.
          </p>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
          <FiAlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {cleanedCSV && !loading && (
        <div className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl shadow-xl space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.06] pb-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <FiCheckCircle className="text-emerald-400 w-4 h-4" />
                <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400">
                  Data Cleaning Succeeded
                </span>
              </div>
              <h3 className="text-xl font-bold text-white">Cleaned Dataset Ready</h3>
              {stats && (
                <p className="text-xs text-[#A4A4B8] mt-1 font-mono">
                  {stats.rows_processed} rows processed • {stats.rows_removed || 0} removed • {stats.columns_processed} features
                </p>
              )}
            </div>
            <button
              onClick={handleDownload}
              className="px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 text-white shadow-lg shadow-emerald-500/25 flex items-center gap-2 transition-all self-start sm:self-auto"
            >
              <FiDownload className="w-4 h-4" />
              Download Cleaned CSV
            </button>
          </div>

          {stats && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-[#0D0D1B] border border-white/[0.06] p-4 rounded-xl space-y-2">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-[#8265FF]">Missing Value Treatment</h4>
                <div className="space-y-1 text-xs">
                  {Object.keys(stats.missing_values_imputed || {}).length > 0 ? (
                    Object.entries(stats.missing_values_imputed).map(([col, count]) => (
                      <div key={col} className="flex justify-between text-white font-mono text-[11px]">
                        <span className="text-[#A4A4B8] truncate">{col}</span>
                        <span>{count} imputed</span>
                      </div>
                    ))
                  ) : (
                    <p className="text-[#A4A4B8] text-xs">No missing values required imputation</p>
                  )}
                </div>
              </div>

              <div className="bg-[#0D0D1B] border border-white/[0.06] p-4 rounded-xl space-y-2">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-[#A855F7]">Encodings & Normalization</h4>
                <div className="space-y-1 text-xs text-[#A4A4B8]">
                  {stats.columns_normalized?.length > 0 && (
                    <p className="text-[11px]"><b className="text-white">Normalized:</b> {stats.columns_normalized.join(', ')}</p>
                  )}
                  {stats.columns_encoded?.length > 0 && (
                    <p className="text-[11px]"><b className="text-white">Encoded:</b> {stats.columns_encoded.join(', ')}</p>
                  )}
                  {(!stats.columns_normalized?.length && !stats.columns_encoded?.length) && (
                    <p className="text-xs">No transforms applied</p>
                  )}
                </div>
              </div>

              <div className="bg-[#0D0D1B] border border-white/[0.06] p-4 rounded-xl space-y-2">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-emerald-400">Quality Metrics</h4>
                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-[#A4A4B8]">Duplicates Pruned:</span>
                    <span className="font-mono font-bold text-white">{stats.duplicates_removed || 0}</span>
                  </div>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-[#A4A4B8]">Outliers Removed:</span>
                    <span className="font-mono font-bold text-white">{stats.outliers_removed || 0}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {previewData.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-[#A4A4B8]">Cleaned Dataset Preview (First 5 Rows)</h4>
              <div className="overflow-x-auto border border-white/[0.06] rounded-xl">
                <table className="min-w-full divide-y divide-white/[0.06] text-xs text-left">
                  <thead className="bg-[#0D0D1B]">
                    <tr>
                      {Object.keys(previewData[0] || {}).map((col) => (
                        <th key={col} className="px-4 py-3 font-mono text-[11px] font-semibold text-[#A4A4B8] uppercase">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04]">
                    {previewData.map((row, idx) => (
                      <tr key={idx} className="hover:bg-white/[0.02]">
                        {Object.values(row).map((val, i) => (
                          <td key={i} className="px-4 py-2 font-mono text-[#F7F7FC]">
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}