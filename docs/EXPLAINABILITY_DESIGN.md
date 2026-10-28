# AutoVeyra — Model Explainability Engine Design Document (Phase 2)

**Document Version:** 1.0  
**Phase:** Phase 2 — Production-Grade Model Explainability Engine  
**Author:** Explainable AI & ML Architecture Team  
**Date:** October 28, 2026  
**Status:** Implemented & Verified  

---

## 1. System Overview

AutoVeyra's Model Explainability Engine provides interpretable, game-theoretic, and empirical insights into trained machine learning pipelines. It addresses both:
1. **Global Feature Importance**: Quantifying which original dataset variables drive overall model performance.
2. **Local Prediction Attribution**: Explaining individual row predictions via additive Shapley values (SHAP waterfalls).

The engine strictly maintains:
- **Zero Pipeline Mutation**: Never refits or alters weights of registered pipelines during explanation.
- **Leakage Prevention**: Evaluates out-of-sample data when supplied, and explicitly warns when fallback sample data is utilized.
- **Original Feature Alignment**: Automatically aggregates one-hot encoded dummy columns back to original input column names via the Shapley efficiency axiom.
- **Resource Bounds**: Restricts evaluation samples, background sizes, and repeat counts to ensure high responsiveness in local or serverless deployments.

---

## 2. Explainability Methods Architecture

```
                                    +------------------------------+
                                    | Registered Scikit-Learn Model|
                                    | Pipeline (Preprocessor + Est)|
                                    +--------------+---------------+
                                                   |
                         +-------------------------+-------------------------+
                         |                                                   |
             [Permutation Importance]                                 [SHAP Attributions]
                         |                                                   |
               +---------v---------+                               +---------v---------+
               |  Model-Agnostic   |                               | Estimator-Specific|
               | Baseline Analysis |                               |  Attributions     |
               +---------+---------+                               +---------+---------+
                         |                                                   |
               +---------v---------+                               +---------v---------+
               | Shuffles Feature  |                               | TreeExplainer or  |
               | Columns in Raw X  |                               |  LinearExplainer  |
               +---------+---------+                               +---------+---------+
                         |                                                   |
               +---------v---------+                               +---------v---------+
               | Evaluates Metric  |                               | Aggregates OHE    |
               | Degradation (F1,  |                               | Dummies to Orig   |
               | R2, RMSE, Acc)    |                               | Features          |
               +---------+---------+                               +---------+---------+
                         |                                                   |
               +---------v---------+                               +---------v---------+
               | Global Importance |                               | Local Waterfall & |
               | Mean +/- Std Dev  |                               | Global Mean |SHAP||
               +-------------------+                               +-------------------+
```

### 2.1. Permutation Feature Importance (Baseline)
- **Algorithm**: `sklearn.inspection.permutation_importance`.
- **Target Audience**: Applicable to all estimators (tree ensembles, linear models, nonlinear models).
- **Evaluation Mechanism**: Shuffles each original column $j$ in $X$ $B$ times (default $B=5$, bounded $1 \le B \le 10$) and computes score decrease:
  $$I(f_j) = s_{\text{baseline}} - \frac{1}{B} \sum_{b=1}^B s_{\text{permuted}(j, b)}$$
- **Feature Preservation**: Because shuffling occurs on the original dataframe prior to `preprocessor.transform()`, multi-category one-hot representations are permuted together, measuring the genuine full impact of categorical variables without collinear dilution.
- **Negative Importance Interpretation**: Shuffling noise features can occasionally increase out-of-sample scores due to minor random variation. The engine explicitly flags negative importances with educational diagnostics.
- **Collinearity Notice**: Discloses that correlated features may share or split attribution.

### 2.2. SHAP (SHapley Additive exPlanations)
- **Supported Estimators**:
  - `TreeExplainer`: `RandomForestClassifier`, `DecisionTreeClassifier`, `ExtraTreesClassifier`, `RandomForestRegressor`, `DecisionTreeRegressor`.
  - `LinearExplainer`: `LogisticRegression`, `LinearRegression`.
- **Mathematical Guarantee**:
  - Local attributions satisfy the efficiency axiom:
    $$\sum_{j=1}^M \phi_j(x) = f(x) - E[f(X)]$$
    where $E[f(X)]$ is the expected base value and $f(x)$ is the model prediction.
- **Categorical Feature Re-aggregation**:
  - One-hot encoded dummy categories (`cat__city_NY`, `cat__city_SF`) are mapped to the parent feature `city`. By the additivity property of Shapley values:
    $$\phi_{\text{city}} = \sum_{k \in \text{city\_dummies}} \phi_k$$
  - The UI presents both the aggregated original attribution and detailed underlying category contributions.
- **Graceful Fallback**: If an estimator is unsupported, the endpoint returns a clear `HTTP 400` diagnostic recommending Permutation Importance.

---

## 3. API Contract Specifications

### 3.1. Capabilities Endpoint
`GET /models/{model_id}/explainability/capabilities`
- **Response**:
  ```json
  {
    "model_id": 1,
    "model_name": "CreditRisk - RandomForestClassifier",
    "task_type": "classification",
    "model_type": "RandomForestClassifier",
    "target_column": "approved",
    "feature_columns": ["age", "income", "credit_score"],
    "expected_columns_count": 3,
    "supported_methods": [
      {
        "method": "permutation",
        "name": "Permutation Feature Importance",
        "scope": "global",
        "supported": true,
        "requires_target": true
      },
      {
        "method": "shap",
        "name": "SHAP (SHapley Additive exPlanations)",
        "scope": "local_and_global",
        "supported": true,
        "explainer_type": "TreeExplainer",
        "requires_target": false
      }
    ],
    "default_scoring": "f1_macro",
    "available_scorers": ["f1_macro", "accuracy", "roc_auc", "balanced_accuracy"]
  }
  ```

### 3.2. Permutation Importance Endpoint
`POST /models/{model_id}/explainability/permutation`
- **Supported Ingest**: Multipart CSV upload (`file`) OR JSON body (`{"rows": [...], "target_column": "..."}`).
- **Parameters**: `n_repeats` (1-10, default 5), `scoring` (metric string), `target_column`, `dataset_source`.
- **Response**:
  ```json
  {
    "model_id": 1,
    "method": "permutation",
    "dataset_source": "evaluation_data",
    "rows_evaluated": 250,
    "target_column": "approved",
    "scoring_metric": "f1_macro",
    "n_repeats": 5,
    "duration_seconds": 0.42,
    "features": [
      {
        "feature": "credit_score",
        "importance_mean": 0.1852,
        "importance_std": 0.0124,
        "raw_repeats": [0.182, 0.191, 0.179, 0.188, 0.186],
        "rank": 1
      }
    ],
    "top_features": ["credit_score", "income"],
    "warnings": [
      "Evaluated on external holdout dataset. Permutation importance reflects impact on out-of-sample generalization.",
      "Notice: Permutation importance describes model attribution, not real-world causality."
    ]
  }
  ```

### 3.3. SHAP Attributions Endpoint
`POST /models/{model_id}/explainability/shap`
- **Supported Ingest**: Multipart CSV upload (`file`) OR JSON body (`{"rows": [...]}`).
- **Parameters**: `max_rows` (1-100, default 50), `row_indices` (list of integers), `selected_class` (class index/label).
- **Response**:
  ```json
  {
    "model_id": 1,
    "method": "shap",
    "explainer_type": "TreeExplainer",
    "dataset_source": "evaluation_data",
    "rows_evaluated": 50,
    "classes": ["0", "1"],
    "explained_class": "1",
    "duration_seconds": 0.31,
    "global_importance": [
      { "feature": "credit_score", "mean_abs_shap": 0.2451, "rank": 1 }
    ],
    "local_explanations": [
      {
        "row_index": 0,
        "prediction": "1",
        "prediction_probability": 0.88,
        "actual_target": "1",
        "base_value": 0.52,
        "sum_shap_values": 0.36,
        "contributions": [
          {
            "feature": "credit_score",
            "original_value": 780,
            "shap_value": 0.28,
            "direction": "positive"
          }
        ]
      }
    ]
  }
  ```

---

## 4. UI Architecture & Design Tokens

Integrated in `client/src/pages/Explainability.jsx`:
- **Theme**: Dark navy background `#070711`, container surfaces `#111126`, secondary badges `#171630`.
- **Brand Accents**: Violet `#8265FF` and purple `#A855F7`.
- **Status Accents**: Emerald `#10B981` (positive pushes / success), Amber `#F59E0B` (negative pulls / warnings), Rose `#F43F5E` (errors).
- **Responsive Navigation**: One-click transition between models, evaluation datasets, and explanation methods.
- **Export Utility**: Instant client-side generation of `JSON` and `CSV` download blobs.

---

## 5. Security & Resource Controls

1. **Memory & Runtime Bounds**: Evaluation datasets capped at 5,000 rows for permutation and 100 rows for SHAP; repeat count bounded between 1 and 10.
2. **Deterministic Single-Threading**: Runs with `n_jobs=1` to eliminate process locks on Windows environments.
3. **Strict Feature Schema Validation**: Input rows must contain all expected features. Extra unrecognized features trigger `HTTP 400`.
4. **JSON Sanitization**: All numbers are sanitized via `_json_safe` to convert non-finite numeric values (`NaN`, `Inf`) to `null`.
