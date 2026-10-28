# AutoVeyra — Automated Feature Engineering & Selection Technical Design

**Document Version:** 1.0  
**Phase:** Phase 4 — Production-Grade Automated Feature Engineering & Feature Selection  
**Author:** ML Engineering & Systems Architecture  
**Status:** Approved & Implemented  

---

## 1. Architectural Overview

AutoVeyra's Feature Engineering & Selection Studio provides an automated, modular, and leakage-free workflow for tabular feature transformation and feature selection.

Built natively on top of scikit-learn `Pipeline`, `ColumnTransformer`, and custom estimators, all learned parameters (imputation statistics, scaling parameters, outlier clipping percentiles, rare category groupings, correlation filters, and model-based feature selection masks) are fitted strictly on training slices during cross-validation, guaranteeing zero data leakage.

```mermaid
flowchart TD
    A[Tabular CSV Upload] --> B[Statistical Profiling & Collinearity Detection]
    B --> C[Zero-Leakage Train / Holdout Split]

    subgraph Multi-Strategy Evaluation [Training Partition Only]
        C -- X_train, y_train --> D[Strategy 1: Baseline Pipeline]
        C -- X_train, y_train --> E[Strategy 2: Feature-Engineered Pipeline]
        C -- X_train, y_train --> F[Strategy 3: Feature-Selected Pipeline]
        C -- X_train, y_train --> G[Strategy 4: Combined Pipeline]
        
        D & E & F & G --> H[K-Fold / StratifiedKFold CV Scoring]
        H --> I[Winning Strategy Selection via Training CV Mean]
    end

    subgraph Unbiased Holdout Verification
        C -- Untouched X_test, y_test --> J[Baseline Pipeline Holdout Evaluation]
        C -- Untouched X_test, y_test --> K[Winning Pipeline Holdout Evaluation]
        J & K --> L[Delta Performance & Significance Analysis]
    end

    subgraph Persistence & Ecosystem Integration
        I --> M[Refit Winning Pipeline on Dataset]
        M --> N[(Model Registry DB + Joblib)]
        L & I --> O[(Feature History SQLite DB)]
    end

    N --> P[REST Predictions: /models/{id}/predict]
    N --> Q[Explainability: /models/{id}/explainability/*]
    N --> R[Optuna Hyperparameter Tuning: /tune]
```

---

## 2. Statistical Profiling & Diagnostics

The profiling engine (`POST /features/profile`) operates without using the target column for feature filtering, ensuring unbiased exploratory insights:

1. **Column-Level Characteristics:**
   - Inferred types: Numeric, Categorical, Datetime candidates.
   - Missing counts and percentages.
   - Distinct values count and cardinality ratios.
   - Non-negativity checks for logarithmic eligibility.
2. **Data Quality Alerts:**
   - **Constant Columns:** Features with $\le 1$ unique value flagged for elimination.
   - **Near-Constant Columns:** Features where a single dominant value exceeds $95\%$ of samples.
   - **High-Cardinality Categoricals:** Features with $> 50$ distinct values flagged for grouping.
   - **Identifier-Like Candidates:** Features exhibiting $\ge 98\%$ distinctness and $> 20$ unique values (or matching ID naming patterns) alerted with an advisory warning (never dropped automatically).
3. **Collinearity & Redundancy Diagnostics:**
   - Pearson correlation matrix computed across non-constant numeric features.
   - Redundant feature pairs exceeding the correlation threshold ($|r| \ge \text{threshold}$, default $0.85$) identified for selective filtering.
4. **Computational Bounding:**
   - Datasets exceeding $10,000$ rows are deterministically sampled for profiling, reporting `is_sampled=True` to prevent memory contention.

---

## 3. Supported Transformations

All transformations are implemented in `TabularFeatureEngineer` (`BaseEstimator`, `TransformerMixin`):

| Transformation | Type | Mechanism | Parameter Fitting |
| :--- | :--- | :--- | :--- |
| **Calendar Feature Extraction** | Date/Time | Extracts `_year`, `_month`, `_day`, `_dayofweek`. Drops original raw string. | Deterministic extraction during `transform`. |
| **Logarithmic Transform (`log1p`)** | Numeric | Computes $\ln(1 + \max(0, x))$ to compress right-skewed positive distributions. | Pure function transformer applied to non-negative columns. |
| **Outlier Clipping** | Numeric | Clips values to learned empirical percentiles ($1^{\text{st}}$ and $99^{\text{th}}$ percentiles). | Percentile bounds learned strictly during `fit` on training slice. |
| **Rare Category Grouping** | Categorical | Groups categorical levels with frequency $< 2\%$ into `'__other__'`. | Frequent category vocabulary learned strictly during `fit` on training slice. |

---

## 4. Supported Feature Selectors

| Selector | Category | Operation | Mechanism |
| :--- | :--- | :--- | :--- |
| **Variance Threshold** | Unsupervised | `VarianceThreshold(threshold)` | Drops features with empirical variance below configured threshold on training slice. |
| **Correlation Filter** | Unsupervised | `CorrelationFilter(threshold)` | Computes pairwise Pearson correlation on training numeric slice and drops redundant collinear columns. |
| **Model-Based Importance** | Supervised | `SelectFromModel(RandomForest)` | Trains shallow Random Forest on training fold and retains features meeting importance threshold. |
| **SelectKBest** | Supervised | `SelectKBest(f_classif / f_regression, k)` | Computes ANOVA F-value on training fold and retains top $K$ features. |

---

## 5. Zero Data-Leakage Evaluation Protocol

1. **Initial Split:** Raw dataset is split into `X_train, y_train` ($80\%$) and `X_test, y_test` ($20\%$). Holdout test data is sealed.
2. **Strategy CV Selection:** All 4 candidate pipelines (`baseline`, `feature_engineered`, `feature_selected`, `combined`) are cross-validated on `X_train, y_train` using identical CV splits (`StratifiedKFold` for classification, `KFold` for regression).
3. **Winning Strategy:** The strategy with the highest training CV mean score (or lowest RMSE/MAE) is declared the winner.
4. **Holdout Evaluation:** Baseline and winning pipelines are fit on `X_train, y_train` and evaluated once on `X_test, y_test` to measure true generalization.
5. **Registration:** Winning pipeline is fit on the full dataset and persisted to `model_registry.db` + `.joblib`.

---

## 6. API Endpoints

- `POST /features/profile`: Profile dataset for distributions, quality alerts, and collinearity.
- `POST /features/preview`: Preview transformations on sample rows.
- `POST /features/evaluate`: Multi-strategy training CV comparison, holdout evaluation, and winning pipeline registration.
- `GET /features/history`: Paginated list of historical feature runs.
- `GET /features/history/{run_id}`: Detail of a specific feature run.
- `DELETE /features/history/{run_id}`: Delete feature run record.
