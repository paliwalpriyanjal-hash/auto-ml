// client/src/components/ui/DataTable.jsx
import React, { useState, useMemo } from "react";
import { Search, ChevronLeft, ChevronRight, ArrowUpDown } from "lucide-react";
import EmptyState from "./EmptyState";

export default function DataTable({
  columns = [], // [{ key, label, align: 'left'|'right'|'center', render, sortable }]
  data = [],
  searchKey = "",
  searchPlaceholder = "Search records...",
  pageSize = 10,
  emptyTitle = "No records found",
  emptyDescription = "No data rows matched the query.",
  className = "",
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const [sortKey, setSortKey] = useState("");
  const [sortOrder, setSortOrder] = useState("asc"); // 'asc' | 'desc'
  const [currentPage, setCurrentPage] = useState(1);

  // Filter
  const filteredData = useMemo(() => {
    if (!searchQuery.trim()) return data;
    const q = searchQuery.toLowerCase().trim();
    return data.filter((row) => {
      if (searchKey && row[searchKey] !== undefined) {
        return String(row[searchKey]).toLowerCase().includes(q);
      }
      return Object.values(row).some((val) =>
        String(val || "").toLowerCase().includes(q)
      );
    });
  }, [data, searchQuery, searchKey]);

  // Sort
  const sortedData = useMemo(() => {
    if (!sortKey) return filteredData;
    return [...filteredData].sort((a, b) => {
      const valA = a[sortKey];
      const valB = b[sortKey];
      if (valA === valB) return 0;
      if (valA === undefined || valA === null) return 1;
      if (valB === undefined || valB === null) return -1;
      const cmp = valA < valB ? -1 : 1;
      return sortOrder === "asc" ? cmp : -cmp;
    });
  }, [filteredData, sortKey, sortOrder]);

  // Pagination
  const totalPages = Math.ceil(sortedData.length / pageSize) || 1;
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return sortedData.slice(start, start + pageSize);
  }, [sortedData, currentPage, pageSize]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortOrder("asc");
    }
  };

  return (
    <div
      className={`rounded-2xl bg-[#0F0F22] border border-white/[0.08] overflow-hidden space-y-3 ${className}`}
    >
      {/* Search Header */}
      {searchPlaceholder && data.length > 0 && (
        <div className="p-4 border-b border-white/[0.06] flex items-center justify-between gap-4">
          <div className="relative max-w-sm w-full">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[#9FA2B6]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              placeholder={searchPlaceholder}
              className="w-full pl-9 pr-3 py-1.5 rounded-xl bg-[#070711] border border-white/[0.08] focus:border-[#8265FF] text-xs text-white placeholder-[#64677E] focus:outline-none transition-all"
            />
          </div>
          <span className="text-xs text-[#9FA2B6]">
            Showing {sortedData.length} entries
          </span>
        </div>
      )}

      {/* Table Container */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-[#16152F] text-[#9FA2B6] border-b border-white/[0.08] select-none">
            <tr>
              {columns.map((col) => {
                const alignClass =
                  col.align === "right"
                    ? "text-right"
                    : col.align === "center"
                    ? "text-center"
                    : "text-left";
                return (
                  <th
                    key={col.key}
                    onClick={() => col.sortable && handleSort(col.key)}
                    className={`p-3.5 font-semibold ${alignClass} ${
                      col.sortable ? "cursor-pointer hover:text-white" : ""
                    }`}
                  >
                    <div
                      className={`inline-flex items-center gap-1 ${
                        col.align === "right" ? "justify-end" : ""
                      }`}
                    >
                      <span>{col.label}</span>
                      {col.sortable && (
                        <ArrowUpDown className="w-3 h-3 opacity-60" />
                      )}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>

          <tbody className="divide-y divide-white/[0.04]">
            {paginatedData.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="p-8 text-center">
                  <EmptyState
                    title={emptyTitle}
                    description={emptyDescription}
                    className="bg-transparent border-0 p-4"
                  />
                </td>
              </tr>
            ) : (
              paginatedData.map((row, idx) => (
                <tr
                  key={row.id || idx}
                  className="hover:bg-white/[0.02] transition-colors"
                >
                  {columns.map((col) => {
                    const alignClass =
                      col.align === "right"
                        ? "text-right"
                        : col.align === "center"
                        ? "text-center"
                        : "text-left";
                    return (
                      <td
                        key={col.key}
                        className={`p-3.5 text-white/90 ${alignClass}`}
                      >
                        {col.render
                          ? col.render(row[col.key], row, idx)
                          : row[col.key] !== undefined
                          ? String(row[col.key])
                          : "-"}
                      </td>
                    );
                  })}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      {totalPages > 1 && (
        <div className="p-4 border-t border-white/[0.06] flex items-center justify-between text-xs text-[#9FA2B6]">
          <span>
            Page {currentPage} of {totalPages}
          </span>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-1.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] disabled:opacity-40 disabled:cursor-not-allowed text-white"
              aria-label="Previous Page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="p-1.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] disabled:opacity-40 disabled:cursor-not-allowed text-white"
              aria-label="Next Page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
