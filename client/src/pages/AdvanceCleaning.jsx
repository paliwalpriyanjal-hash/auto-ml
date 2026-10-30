import React, { useState, useEffect } from 'react';
import AdvancedResult from '../components/Advanced_cleaning/AdvancedResult';
import PreprocessingOption from '../components/Advanced_cleaning/PreprocessingOption';
import DataSummary from '../components/Advanced_cleaning/DataSummary';
import Papa from 'papaparse';
import PageHeader from '../components/ui/PageHeader';
import Badge from '../components/ui/Badge';
import Card from '../components/ui/Card';

export default function AutoMLStudio() {
    const [currentStep, setCurrentStep] = useState(1);
    const [file, setFile] = useState(null);
    const [previewData, setPreviewData] = useState([]);
    const [columns, setColumns] = useState([]);
    const [preprocessing, setPreprocessing] = useState(null);
    const [dataStats, setDataStats] = useState(null);
    const [fileError, setFileError] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [parsingConfig, setParsingConfig] = useState({
        delimiter: ',',
        header: true,
        skipEmptyLines: true,
    });

    // Step navigation handlers
    const nextStep = () => setCurrentStep(prev => Math.min(prev + 1, 4));
    const prevStep = () => setCurrentStep(prev => Math.max(prev - 1, 1));
    const goToStep = (step) => setCurrentStep(step);

    // Parse CSV file using Papa Parse
    const parseCSV = (text, config) => {
        return new Promise((resolve, reject) => {
            Papa.parse(text, {
                ...config,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(results.errors);
                    } else {
                        resolve(results);
                    }
                },
                error: (error) => reject(error)
            });
        });
    };

    // Handle file upload and process to step 2
    const handleFileUpload = async (e) => {
        const selected = e.target.files[0];
        if (!selected) return;

        // File validation
        if (!selected.name.match(/\.(csv|txt|tsv|xlsx?)$/i)) {
            setFileError('Please upload a CSV, Excel, or text file');
            return;
        }

        if (selected.size > 10 * 1024 * 1024) {
            setFileError('File size exceeds 10MB limit');
            return;
        }

        setFileError(null);
        setFile(selected);
        setIsAnalyzing(true);

        try {
            const text = await readFileAsText(selected);
            const results = await parseCSV(text, parsingConfig);

            if (results.data.length === 0) {
                setPreviewData([]);
                setColumns([]);
                setDataStats(null);
                return;
            }

            const headers = results.meta.fields || Object.keys(results.data[0] || []);
            const stats = analyzeData(headers, results.data.slice(0, 100));

            // Transform plain header strings into structured column objects.
            // PreprocessingOption.jsx and DataSummary.jsx both require { name, type, isText }.
            //
            // analyzeData() produces types: 'numeric' | 'text' | 'boolean' | 'date'
            // PreprocessingOption filters on:
            //   col.type === 'number'    → numeric columns
            //   col.type === 'string'    → categorical columns
            //   col.isText               → text columns (high cardinality free-text)
            //   col.type === 'datetime'  → datetime columns
            const columnObjects = headers.map(header => {
                const analysedType = stats?.columns[header]?.type || 'text';
                let mappedType;
                switch (analysedType) {
                    case 'numeric':  mappedType = 'number';   break;
                    case 'date':     mappedType = 'datetime'; break;
                    case 'boolean':  mappedType = 'string';   break;  // booleans → categorical
                    case 'text':
                    default:         mappedType = 'string';   break;
                }
                // Mark free-text columns: 'text' type with multiple words or very high cardinality
                const colInfo = stats?.columns[header];
                const sampleValues = colInfo?.sampleValues || [];
                const hasMultiWordText = sampleValues.some(v => typeof v === 'string' && v.trim().includes(' '));
                const isText = analysedType === 'text' && (
                    hasMultiWordText ||
                    (colInfo != null && colInfo.uniqueCount > 20 && colInfo.uniquePercentage > 70)
                );
                return { name: header, type: mappedType, isText };
            });

            setColumns(columnObjects);
            setPreviewData(results.data.slice(0, 5));
            setDataStats(stats);
            nextStep(); // Move to preview step after successful upload
        } catch (error) {
            console.error('Error parsing file:', error);
            setFileError('Error parsing file. Please check the format.');
        } finally {
            setIsAnalyzing(false);
        }
    };

    // Helper to read file as text
    const readFileAsText = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (ev) => resolve(ev.target.result);
            reader.onerror = () => reject(new Error('Error reading file'));
            reader.readAsText(file);
        });
    };

    // Analyze data structure and content
    const analyzeData = (headers, rows) => {
        if (rows.length === 0) return null;

        const stats = {
            totalRows: rows.length,
            columns: {},
            potentialIssues: [],
            dataTypes: {
                numeric: 0,
                text: 0,
                date: 0,
                boolean: 0,
                other: 0
            }
        };

        headers.forEach(header => {
            const colData = rows.map(row => row[header]);
            const nonEmpty = colData.filter(val => val !== '' && val !== null && val !== undefined);
            const emptyCount = colData.length - nonEmpty.length;

            let type = 'text';
            let numericValues = [];
            let booleanValues = [];

            if (nonEmpty.length > 0) {
                const isBoolean = nonEmpty.every(val =>
                    ['true', 'false', 'yes', 'no', '1', '0'].includes(val.toLowerCase())
                );

                if (isBoolean) {
                    type = 'boolean';
                    booleanValues = nonEmpty.map(val =>
                        ['true', 'yes', '1'].includes(val.toLowerCase())
                    );
                }
                else if (nonEmpty.every(val => !isNaN(val) && val.trim() !== '')) {
                    type = 'numeric';
                    numericValues = nonEmpty.map(Number);
                }
                else if (nonEmpty.some(val => isPotentialDate(val))) {
                    type = 'date';
                }
            }

            stats.columns[header] = {
                type,
                emptyCount,
                emptyPercentage: Math.round((emptyCount / colData.length) * 100),
                uniqueCount: new Set(nonEmpty).size,
                uniquePercentage: Math.round((new Set(nonEmpty).size / nonEmpty.length) * 100),
                sampleValues: Array.from(new Set(nonEmpty.slice(0, 5))),
            };

            stats.dataTypes[type]++;

            if (type === 'numeric') {
                stats.columns[header].stats = {
                    min: Math.min(...numericValues),
                    max: Math.max(...numericValues),
                    avg: numericValues.reduce((a, b) => a + b, 0) / numericValues.length,
                    stdDev: calculateStdDev(numericValues),
                    median: calculateMedian(numericValues)
                };
            } else if (type === 'boolean') {
                stats.columns[header].stats = {
                    trueCount: booleanValues.filter(Boolean).length,
                    falseCount: booleanValues.length - booleanValues.filter(Boolean).length
                };
            }

            if (emptyCount > 0) {
                stats.potentialIssues.push({
                    column: header,
                    issue: 'missing_values',
                    count: emptyCount,
                    severity: emptyCount / colData.length > 0.3 ? 'high' :
                        emptyCount / colData.length > 0.1 ? 'medium' : 'low'
                });
            }
        });

        return stats;
    };

    // Statistical helper functions
    const calculateStdDev = (values) => {
        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        const squareDiffs = values.map(val => Math.pow(val - avg, 2));
        const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / squareDiffs.length;
        return Math.sqrt(avgSquareDiff);
    };

    const calculateMedian = (values) => {
        const sorted = [...values].sort((a, b) => a - b);
        const middle = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle];
    };

    // Date detection helper
    const isPotentialDate = (value) => {
        if (typeof value !== 'string') return false;
        const date = new Date(value);
        return !isNaN(date.getTime());
    };

    // Handle configuration changes
    const handleConfigChange = (e) => {
        const { name, value, type, checked } = e.target;
        setParsingConfig(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    // Handle preprocessing options
    const handlePreprocessingChange = (options) => {
        setPreprocessing(options);
    };

    // Reset the entire process
    const handleReset = () => {
        setFile(null);
        setPreviewData([]);
        setColumns([]);
        setPreprocessing(null);
        setDataStats(null);
        setFileError(null);
        setCurrentStep(1);
    };

    return (
        <div className="min-h-screen pt-24 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-8">
            {/* Universal Page Header */}
            <PageHeader
                category="Data Studio"
                title="Advanced Data Cleaning & Transformation"
                subtitle="KNN/iterative imputation, IQR outlier treatment, SMOTE class balancing, and TF-IDF extraction powered by AutoVeyra cleaner."
                badge={
                    <Badge variant="violet" size="sm">
                        Intelligent Preprocessing
                    </Badge>
                }
            />

            {/* Step progress bar */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                    { step: 1, title: '1. Ingestion' },
                    { step: 2, title: '2. Inspection' },
                    { step: 3, title: '3. Pipeline Setup' },
                    { step: 4, title: '4. Cleaned Data' },
                ].map(({ step, title }) => (
                    <div
                        key={step}
                        onClick={() => currentStep > step && goToStep(step)}
                        className={`p-3.5 rounded-xl border flex items-center justify-between text-xs font-semibold transition-all ${
                            currentStep >= step
                                ? 'bg-[#171630] border-[#8265FF]/40 text-white shadow-sm'
                                : 'bg-[#111126] border-white/[0.06] text-[#A4A4B8]'
                        } ${currentStep > step ? 'cursor-pointer hover:border-[#8265FF]/70' : ''}`}
                    >
                        <span>{title}</span>
                        <span className={`w-5 h-5 rounded-md flex items-center justify-center text-[10px] font-bold ${
                            currentStep >= step ? 'bg-[#8265FF] text-white' : 'bg-white/[0.06] text-[#A4A4B8]'
                        }`}>
                            {step}
                        </span>
                    </div>
                ))}
            </div>

            {/* Step 1: Upload Data */}
            {currentStep === 1 && (
                <section className="bg-[#111126] p-8 rounded-2xl border border-white/[0.08] shadow-2xl space-y-4">
                    <div className={`border-2 border-dashed ${fileError ? 'border-rose-500/40 bg-rose-500/5' : 'border-white/[0.12] hover:border-[#8265FF]/50 bg-[#0D0D1B]'} p-12 rounded-2xl text-center transition-all`}>
                        <div className="flex flex-col items-center justify-center">
                            <div className="w-16 h-16 rounded-2xl bg-[#8265FF]/10 text-[#8265FF] flex items-center justify-center mb-4 border border-[#8265FF]/20 shadow-lg shadow-[#8265FF]/10">
                                <svg
                                    className="w-8 h-8"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                                    />
                                </svg>
                            </div>
                            <h2 className="text-xl font-bold text-white mb-1">Upload Tabular Dataset</h2>
                            <p className="text-xs text-[#A4A4B8] mb-5">Select or drop a CSV file to inspect schema and clean</p>

                            <input
                                type="file"
                                accept=".csv,.txt,.tsv"
                                id="upload"
                                className="hidden"
                                onChange={handleFileUpload}
                                disabled={isAnalyzing}
                            />
                            <label
                                htmlFor="upload"
                                className="px-6 py-3 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/25 cursor-pointer transition-all inline-flex items-center gap-2"
                            >
                                {isAnalyzing ? 'Analyzing schema...' : 'Select CSV File'}
                            </label>
                            <p className="text-[11px] text-[#A4A4B8]/60 mt-3 font-mono">
                                Supported format: CSV (Max 10MB)
                            </p>
                        </div>

                        {fileError && (
                            <div className="mt-4 p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-xs font-medium">
                                {fileError}
                            </div>
                        )}
                    </div>
                </section>
            )}

            {/* Step 2: Data Preview & Summary */}
            {currentStep === 2 && dataStats && (
                <div className="space-y-6">
                    <DataSummary
                        stats={dataStats}
                        columns={columns}
                    />

                    <div className="bg-[#111126] border border-white/[0.08] p-6 rounded-2xl shadow-xl space-y-4">
                        <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                            <h3 className="text-sm font-semibold uppercase tracking-wider text-white">
                                First 5 Raw Data Samples
                            </h3>
                            <span className="text-xs text-[#A4A4B8] font-mono">
                                {columns.length} Total Columns Detected
                            </span>
                        </div>
                        <div className="overflow-x-auto border border-white/[0.06] rounded-xl">
                            <table className="min-w-full divide-y divide-white/[0.06] text-xs text-left">
                                <thead className="bg-[#0D0D1B]">
                                    <tr>
                                        {columns.map((col) => {
                                            const colName = typeof col === 'object' ? col.name : col;
                                            return (
                                                <th
                                                    key={colName}
                                                    className="px-4 py-3 text-[#A4A4B8] font-medium uppercase tracking-wider font-mono text-[11px]"
                                                >
                                                    {colName}
                                                </th>
                                            );
                                        })}
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-white/[0.04]">
                                    {previewData.map((row, idx) => (
                                        <tr key={idx} className="hover:bg-white/[0.02]">
                                            {columns.map((col) => {
                                                const colName = typeof col === 'object' ? col.name : col;
                                                return (
                                                    <td
                                                        key={`${idx}-${colName}`}
                                                        className="px-4 py-2.5 font-mono text-[#F7F7FC]"
                                                    >
                                                        {row[colName] || <span className="text-[#A4A4B8]/40 italic">empty</span>}
                                                    </td>
                                                );
                                            })}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div className="flex justify-between items-center pt-2">
                        <button
                            onClick={handleReset}
                            className="px-4 py-2.5 rounded-xl border border-white/[0.08] bg-[#171630] hover:bg-[#1f1e40] text-xs font-semibold text-white transition-all"
                        >
                            Start Over
                        </button>
                        <button
                            onClick={nextStep}
                            className="px-6 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/25 transition-all"
                        >
                            Configure Cleaning Pipeline →
                        </button>
                    </div>
                </div>
            )}

            {/* Step 3: Configuration */}
            {currentStep === 3 && (
                <div className="space-y-6">
                    <PreprocessingOption
                        onChange={handlePreprocessingChange}
                        columns={columns}
                        dataStats={dataStats}
                    />

                    <div className="flex justify-between items-center pt-2">
                        <button
                            onClick={prevStep}
                            className="px-4 py-2.5 rounded-xl border border-white/[0.08] bg-[#171630] hover:bg-[#1f1e40] text-xs font-semibold text-white transition-all"
                        >
                            ← Back to Preview
                        </button>
                        <button
                            onClick={nextStep}
                            className="px-6 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/25 transition-all"
                        >
                            Execute Cleaning Pipeline →
                        </button>
                    </div>
                </div>
            )}

            {/* Step 4: Results */}
            {currentStep === 4 && (
                <div className="space-y-6">
                    <AdvancedResult
                        csv={file}
                        preprocessing={preprocessing}
                        columns={columns}
                        dataStats={dataStats}
                    />

                    <div className="flex justify-between items-center pt-2">
                        <button
                            onClick={prevStep}
                            className="px-4 py-2.5 rounded-xl border border-white/[0.08] bg-[#171630] hover:bg-[#1f1e40] text-xs font-semibold text-white transition-all"
                        >
                            ← Adjust Parameters
                        </button>
                        <button
                            onClick={handleReset}
                            className="px-6 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/25 transition-all"
                        >
                            Clean Another Dataset
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}