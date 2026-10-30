# AUTOVEYRA PHASE 3: Advanced Feature Engineering, Intelligent Feature Selection & Dataset-Adaptive Transformation Engine

## 1. Executive Audit of Existing Implementation

### 1.1 Existing Capabilities
- **Feature Profiling (`profile_dataset`)**: Computes dataset dimensions, missingness, cardinality, duplicate rows, constant columns, near-constant columns, identifier heuristic candidates, and Pearson collinear pairs ($|r| \ge 0.85$).
- **Tabular Transformation (`TabularFeatureEngineer`)**: Basic calendar component extraction (`year`, `month`, `day`, `dayofweek`), safe `log1p` on non-negative columns, percentile-based outlier clipping learned on training data, and rare category grouping ($< 2\%$).
- **Feature Selection**: `VarianceThreshold`, `CorrelationFilter` (Pearson), `SelectFromModel` (Random Forest), `SelectKBest` (ANOVA $F$-test).
- **Strategy Evaluation (`evaluate_feature_strategies`)**: Compares 4 fixed configurations (`baseline`, `feature_engineered`, `feature_selected`, `combined`) using training-only Cross-Validation, evaluates the winner on the holdout test set, and registers the winning pipeline in the Model Registry.
- **Persistence & API**: SQLite database `feature_history.db` storing run history; endpoints for `/profile`, `/preview`, `/evaluate`, `/history`.

### 1.2 Missing Capabilities & Limitations Identified
1. **Missing Transformations**:
   - *Numerical*: No Power transformations (Yeo-Johnson, Box-Cox), Quantile transforms, Square-root transforms, arithmetic pairwise interactions (ratios with division-by-zero protection, products, sums, differences), or bounded scaling (RobustScaler, MinMaxScaler).
   - *Categorical*: No out-of-fold target encoding with smoothing, frequency/count encoding, or ordinal encoding.
   - *Datetime & Time-Series*: Missing cyclical sine/cosine encodings (for month, day of week, hour), weekend flags, quarter, elapsed time, and past-only lag/rolling features without future leakage.
   - *Text*: Missing lightweight text metrics (char count, word count, avg word length, emptiness) and bounded sparse TF-IDF with training-only fitting.
2. **Missing Feature Selection Methods**:
   - No Mutual Information (`mutual_info_classif`, `mutual_info_regression`).
   - No Recursive Feature Elimination (RFE / RFECV).
   - No L1 regularized models (Lasso / LogisticRegression L1) or ExtraTrees embedded selectors.
   - No Spearman rank non-linear redundancy filtering.
3. **Integration Gaps with Phase 2 EDA**:
   - Currently, `/features` does not consume precomputed EDA results from Phase 2 (`get_eda_analysis_from_db`).
   - Does not adapt strategy presets based on the 11 dataset profiles detected by the EDA engine.
4. **Strategy Presets & Multi-Strategy Comparison**:
   - Missing automated presets: `CONSERVATIVE / AUTO`, `BALANCED`, `ADVANCED`.
   - Missing side-by-side comparison across presets with CV mean/std, feature counts, and honest statistical verdicts.
5. **Feature Lineage & Explainability**:
   - No tracking of parent columns, applied transformations, parameters, selection reasons, or mapping of engineered features back to raw inputs.

---

## 2. Phase 3 Architecture & Design

```
+----------------------------------------------------------------------------------------------------+
|                                    AUTOVEYRA PHASE 3 ARCHITECTURE                                  |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Phase 2 EDA Intelligence] ---------> [Intelligent Strategy Selector]                            |
|  (Dataset Profile, Semantic Types,      (11 Dataset Profiles -> Conservative, Balanced, Advanced)   |
|   Skewness, Missingness, Leakage)                            |                                     |
|                                                              v                                     |
|                                         [Advanced Feature Engineering Engine]                      |
|                                         +-- Distribution Transforms (Yeo-Johnson, Log1p, Sqrt)     |
|                                         +-- Bounded Pairwise Interactions (Ratios, Products, Sums) |
|                                         +-- Categorical (OOF Target Encoding, Frequency, Rare)     |
|                                         +-- Datetime (Cyclical Sin/Cos, Calendar, Elapsed Time)   |
|                                         +-- Time-Series (Past-Only Lags & Rolling Windows)         |
|                                         +-- Lightweight Text (Word Count, Length, Sparse TF-IDF)  |
|                                                              |                                     |
|                                                              v                                     |
|                                         [Intelligent Feature Selection Engine]                     |
|                                         +-- Filter (VarianceThreshold, Mutual Info, Spearman)      |
|                                         +-- Embedded (L1 Lasso/Logistic, ExtraTrees, RandomForest) |
|                                         +-- Wrapper (Bounded RFE / RFECV)                          |
|                                         +-- Redundancy Reduction (Collinear Pairs, VIF)            |
|                                                              |                                     |
|                                                              v                                     |
|                                         [Strategy Comparison & Lineage Tracker]                    |
|                                         +-- Identical CV Evaluation (Mean, Std, Duration)          |
|                                         +-- Feature Lineage Graph (Raw -> Engineered -> Selected)  |
|                                         +-- Untouched Holdout Evaluation for Winner Only           |
|                                         +-- Model Registry & Inference Pipeline Persistence        |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Implementation Plan

1. **Service Layer Upgrades (`server/app/services/feature_service.py`)**:
   - `AdvancedFeatureTransformer`: Scikit-Learn transformer supporting Yeo-Johnson, Log1p, Sqrt, Outlier clipping, Interaction features (products, safe ratios $A / (B + \epsilon)$), Cyclical temporal features, Time-series past-only lags, Lightweight text statistics, and Out-of-fold target encoding.
   - `EnhancedFeatureSelector`: Scikit-Learn transformer supporting Mutual Information, VarianceThreshold, CorrelationFilter (Pearson & Spearman), Model-based (RandomForest, ExtraTrees, L1 Lasso), and RFE.
   - `FeatureLineageTracker`: Records parent columns, operations, parameters, selection status, and removal reasons for all columns.
   - `get_eda_feature_recommendations`: Bridges Phase 2 EDA results with strategy selection across 11 dataset profiles.
   - `compare_feature_strategies`: Evaluates Baseline vs Conservative vs Balanced vs Advanced under identical CV conditions with strict holdout isolation.
   - Update SQLite database schema to store lineage and preset metadata.

2. **API Layer Upgrades (`server/app/routes/feature_routes.py`)**:
   - Extend `POST /features/profile` to support optional `eda_analysis_id`.
   - Extend `POST /features/preview` to support new transformations.
   - Add `POST /features/recommendations` to provide dataset-adaptive transformation suggestions.
   - Add `POST /features/compare-strategies` for multi-preset benchmarking.
   - Extend `POST /features/evaluate` to support mutual info, RFE, and advanced settings.
   - Add `GET /features/lineage/{run_id}` to retrieve lineage graph data.

3. **Frontend Dashboard Upgrades (`client/src/pages/FeatureEngineering.jsx`)**:
   - Guided SaaS 11-step workflow with Auto vs. Advanced mode.
   - EDA integration banner displaying dataset profile badges and recommendations.
   - Multi-Strategy Comparison Dashboard (Baseline vs Conservative vs Balanced vs Advanced).
   - Searchable, interactive Feature Lineage Table.
   - Direct handoffs to Model Training (`/automl`) and Saved Models (`/models`).

4. **Verification & Testing (`server/tests/test_feature_engineering.py`)**:
   - Unit tests for all new transformers (Yeo-Johnson, ratios, cyclical, text, lags, target encoding, mutual info, RFE).
   - Integration tests for all endpoints and strategy comparisons.
   - Verify all 145 existing tests continue to pass.
   - Verify `npm run build --prefix client`.
