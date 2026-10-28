import React, { useState, useEffect } from 'react';
import { Switch } from '@headlessui/react';

export default function PreprocessingOption({ columns = [], onChange }) {
  const [selectedOptions, setSelectedOptions] = useState({
    // Missing data handling
    impute_missing: false,
    imputation_method: 'knn',
    imputation_n_neighbors: 5,
    
    // Outlier handling
    remove_outliers: false,
    outlier_method: 'iqr',
    outlier_threshold: 1.5,
    
    // Feature scaling
    feature_scaling: false,
    scaling_method: 'standard',
    
    // Encoding
    encode_categorical: false,
    encoding_method: 'onehot',
    
    // Feature engineering
    feature_engineering: false,
    create_interactions: false,
    polynomial_features: false,
    polynomial_degree: 2,
    
    // Text processing
    text_processing: false,
    remove_stopwords: false,
    lemmatize: false,
    text_vectorization: 'tfidf',
    
    // Dimensionality reduction
    dimensionality_reduction: false,
    reduction_method: 'pca',
    n_components: 10,
    
    // Class imbalance
    handle_imbalance: false,
    imbalance_method: 'smote',
    
    // Column selections
    numeric_columns: [],
    categorical_columns: [],
    text_columns: [],
    target_column: '',
    datetime_columns: []
  });

  // Update parent when options change
  useEffect(() => {
    onChange(selectedOptions);
  }, [selectedOptions, onChange]);

  const handleCheckboxChange = (e) => {
    const { name, checked } = e.target;
    setSelectedOptions(prev => ({ ...prev, [name]: checked }));
  };

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setSelectedOptions(prev => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value
    }));
  };

  const handleColumnSelection = (e) => {
    const { name, options } = e.target;
    const selected = Array.from(options)
      .filter(option => option.selected)
      .map(option => option.value);
    setSelectedOptions(prev => ({ ...prev, [name]: selected }));
  };

  // Toggle for better UX
  const toggleOption = (name) => {
    setSelectedOptions(prev => ({ ...prev, [name]: !prev[name] }));
  };

  return (
    <div className="space-y-6 mb-6 p-6 border border-white/[0.08] rounded-2xl bg-[#111126] shadow-xl">
      <div className="border-b border-white/[0.06] pb-4">
        <h3 className="text-base font-bold text-white">Data Cleaning & Transformation Engine</h3>
        <p className="text-xs text-[#A4A4B8]">Activate intelligent transformation algorithms for missing values, outliers, and text columns</p>
      </div>
      
      <div className="space-y-6">
        {/* Missing Data Handling */}
        <div className="border-b border-white/[0.06] pb-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h4 className="text-sm font-semibold text-white">Missing Data Handling</h4>
              <p className="text-[11px] text-[#A4A4B8]">Impute null cells using KNN or iterative multivariable estimation</p>
            </div>
            <Switch
              checked={selectedOptions.impute_missing}
              onChange={() => toggleOption('impute_missing')}
              className={`${selectedOptions.impute_missing ? 'bg-[#8265FF]' : 'bg-[#171630] border border-white/[0.1]'} relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer`}
            >
              <span className="sr-only">Enable missing data imputation</span>
              <span
                className={`${selectedOptions.impute_missing ? 'translate-x-6' : 'translate-x-1'} inline-block h-4 w-4 transform rounded-full bg-white transition`}
              />
            </Switch>
          </div>
          
          {selectedOptions.impute_missing && (
            <div className="mt-3 p-4 bg-[#0D0D1B] rounded-xl border border-white/[0.06] space-y-3">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-1.5">Imputation Method</label>
                <select
                  name="imputation_method"
                  value={selectedOptions.imputation_method}
                  onChange={handleInputChange}
                  className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                >
                  <option value="knn">KNN Imputation (K-Nearest Neighbors)</option>
                  <option value="iterative">Iterative Imputer (Multivariate MICE)</option>
                  <option value="mean">Mean (Univariate Numeric)</option>
                  <option value="median">Median (Robust Numeric)</option>
                  <option value="most_frequent">Most Frequent (Mode)</option>
                </select>
              </div>
              
              {selectedOptions.imputation_method === 'knn' && (
                <div>
                  <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-1.5">Number of Neighbors (K)</label>
                  <input
                    type="number"
                    name="imputation_n_neighbors"
                    min="1"
                    max="20"
                    value={selectedOptions.imputation_n_neighbors}
                    onChange={handleInputChange}
                    className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Outlier Handling */}
        <div className="border-b border-white/[0.06] pb-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h4 className="text-sm font-semibold text-white">Outlier Detection & Treatment</h4>
              <p className="text-[11px] text-[#A4A4B8]">Truncate or isolate abnormal extreme value points</p>
            </div>
            <Switch
              checked={selectedOptions.remove_outliers}
              onChange={() => toggleOption('remove_outliers')}
              className={`${selectedOptions.remove_outliers ? 'bg-[#8265FF]' : 'bg-[#171630] border border-white/[0.1]'} relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer`}
            >
              <span className="sr-only">Enable outlier handling</span>
              <span
                className={`${selectedOptions.remove_outliers ? 'translate-x-6' : 'translate-x-1'} inline-block h-4 w-4 transform rounded-full bg-white transition`}
              />
            </Switch>
          </div>
          
          {selectedOptions.remove_outliers && (
            <div className="mt-3 p-4 bg-[#0D0D1B] rounded-xl border border-white/[0.06] space-y-3">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-1.5">Detection Method</label>
                <select
                  name="outlier_method"
                  value={selectedOptions.outlier_method}
                  onChange={handleInputChange}
                  className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                >
                  <option value="iqr">IQR (Interquartile Range 1.5x)</option>
                  <option value="zscore">Z-Score (Standard deviation 3σ)</option>
                  <option value="isolation">Isolation Forest</option>
                  <option value="dbscan">DBSCAN Clustering</option>
                </select>
              </div>
              
              {selectedOptions.outlier_method === 'iqr' && (
                <div>
                  <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-1.5">IQR Threshold</label>
                  <input
                    type="number"
                    name="outlier_threshold"
                    step="0.1"
                    min="1"
                    max="3"
                    value={selectedOptions.outlier_threshold}
                    onChange={handleInputChange}
                    className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Feature Scaling */}
        <div className="border-b border-white/[0.06] pb-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h4 className="text-sm font-semibold text-white">Feature Scaling & Normalization</h4>
              <p className="text-[11px] text-[#A4A4B8]">Scale numeric columns for uniform gradient convergence</p>
            </div>
            <Switch
              checked={selectedOptions.feature_scaling}
              onChange={() => toggleOption('feature_scaling')}
              className={`${selectedOptions.feature_scaling ? 'bg-[#8265FF]' : 'bg-[#171630] border border-white/[0.1]'} relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer`}
            >
              <span className="sr-only">Enable feature scaling</span>
              <span
                className={`${selectedOptions.feature_scaling ? 'translate-x-6' : 'translate-x-1'} inline-block h-4 w-4 transform rounded-full bg-white transition`}
              />
            </Switch>
          </div>
          
          {selectedOptions.feature_scaling && (
            <div className="mt-3 p-4 bg-[#0D0D1B] rounded-xl border border-white/[0.06]">
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-1.5">Scaling Method</label>
              <select
                name="scaling_method"
                value={selectedOptions.scaling_method}
                onChange={handleInputChange}
                className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                <option value="standard">StandardScaler (Z-score mean=0, std=1)</option>
                <option value="minmax">MinMaxScaler [0, 1]</option>
                <option value="robust">RobustScaler (Median / IQR)</option>
                <option value="power">PowerTransformer (Yeo-Johnson)</option>
              </select>
            </div>
          )}
        </div>

        {/* Categorical Encoding */}
        <div className="border-b border-white/[0.06] pb-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h4 className="text-sm font-semibold text-white">Categorical Encoding</h4>
              <p className="text-[11px] text-[#A4A4B8]">Convert discrete string features to numeric tensors</p>
            </div>
            <Switch
              checked={selectedOptions.encode_categorical}
              onChange={() => toggleOption('encode_categorical')}
              className={`${selectedOptions.encode_categorical ? 'bg-[#8265FF]' : 'bg-[#171630] border border-white/[0.1]'} relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer`}
            >
              <span className="sr-only">Enable categorical encoding</span>
              <span
                className={`${selectedOptions.encode_categorical ? 'translate-x-6' : 'translate-x-1'} inline-block h-4 w-4 transform rounded-full bg-white transition`}
              />
            </Switch>
          </div>
          
          {selectedOptions.encode_categorical && (
            <div className="mt-3 p-4 bg-[#0D0D1B] rounded-xl border border-white/[0.06]">
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-1.5">Encoding Method</label>
              <select
                name="encoding_method"
                value={selectedOptions.encoding_method}
                onChange={handleInputChange}
                className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                <option value="onehot">One-Hot Encoding</option>
                <option value="target">Target Encoding</option>
                <option value="frequency">Frequency Encoding</option>
                <option value="leaveoneout">Leave-One-Out</option>
              </select>
            </div>
          )}
        </div>

        {/* Text Processing */}
        <div className="border-b border-white/[0.06] pb-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h4 className="text-sm font-semibold text-white">Free-Text NLP Processing</h4>
              <p className="text-[11px] text-[#A4A4B8]">Vectorize text columns using TF-IDF or Count vectors</p>
            </div>
            <Switch
              checked={selectedOptions.text_processing}
              onChange={() => toggleOption('text_processing')}
              className={`${selectedOptions.text_processing ? 'bg-[#8265FF]' : 'bg-[#171630] border border-white/[0.1]'} relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer`}
            >
              <span className="sr-only">Enable text processing</span>
              <span
                className={`${selectedOptions.text_processing ? 'translate-x-6' : 'translate-x-1'} inline-block h-4 w-4 transform rounded-full bg-white transition`}
              />
            </Switch>
          </div>
          
          {selectedOptions.text_processing && (
            <div className="mt-3 p-4 bg-[#0D0D1B] rounded-xl border border-white/[0.06] space-y-3">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-1.5">Vectorization Algorithm</label>
                <select
                  name="text_vectorization"
                  value={selectedOptions.text_vectorization}
                  onChange={handleInputChange}
                  className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
                >
                  <option value="tfidf">TF-IDF Vectorizer (Term Frequency)</option>
                  <option value="count">Count Vectorizer (Bag of Words)</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Class Imbalance */}
        <div className="border-b border-white/[0.06] pb-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h4 className="text-sm font-semibold text-white">Class Imbalance Handling</h4>
              <p className="text-[11px] text-[#A4A4B8]">Resample rare classes using SMOTE synthetic oversampling</p>
            </div>
            <Switch
              checked={selectedOptions.handle_imbalance}
              onChange={() => toggleOption('handle_imbalance')}
              className={`${selectedOptions.handle_imbalance ? 'bg-[#8265FF]' : 'bg-[#171630] border border-white/[0.1]'} relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer`}
            >
              <span className="sr-only">Enable class imbalance handling</span>
              <span
                className={`${selectedOptions.handle_imbalance ? 'translate-x-6' : 'translate-x-1'} inline-block h-4 w-4 transform rounded-full bg-white transition`}
              />
            </Switch>
          </div>
          
          {selectedOptions.handle_imbalance && (
            <div className="mt-3 p-4 bg-[#0D0D1B] rounded-xl border border-white/[0.06]">
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-1.5">Imbalance Strategy</label>
              <select
                name="imbalance_method"
                value={selectedOptions.imbalance_method}
                onChange={handleInputChange}
                className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
              >
                <option value="smote">SMOTE (Synthetic Minority Oversampling)</option>
                <option value="adasyn">ADASYN</option>
                <option value="class_weight">Class Weights</option>
              </select>
            </div>
          )}
        </div>

        {/* Target Column Selection */}
        <div className="p-4 bg-[#0D0D1B] rounded-xl border border-white/[0.06]">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-white mb-2">Target Variable (For Supervised Imbalance / Target Encoding)</h4>
          <select
            name="target_column"
            value={selectedOptions.target_column}
            onChange={handleInputChange}
            className="w-full bg-[#171630] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-[#8265FF]"
          >
            <option value="">Select target column (optional)...</option>
            {columns.map(col => {
              const name = typeof col === 'object' ? col.name : col;
              return <option key={name} value={name}>{name}</option>;
            })}
          </select>
        </div>
      </div>
    </div>
  );
}