# AutoVeyra Phase 1: ML Engine Technical Audit & Generalization Architecture

## 1. Executive Summary

This audit assesses the predictive accuracy, model-selection capabilities, preprocessing pipelines, evaluation methodology, and data integrity of the AutoVeyra AutoML platform. Prior to Phase 1, the platform suffered from critical data leakage in regression workflows, a severely restricted candidate model suite (only 4 classification models and 3 regression algorithms), evaluation that relied on single held-out splits without cross-validation or generalization tracking, and the inability to register, explain, or serve trained regression models.

In Phase 1, the platform was upgraded into a reliable, dataset-adaptive AutoML system with leak-free `ColumnTransformer` pipelines fitted strictly on training folds, stratified cross-validation, 19 total estimator architectures, automated pre-training dataset intelligence profiling, class imbalance mitigation, and full Model Registry integration for both classification and regression tasks.

---

## 2. Technical Audit Findings

### 2.1 Why Legacy Predictive Accuracy Was Suboptimal
1. **Severe Data Leakage in Regression**: In legacy `train_service.py`, `scale_features(X, scaler)` and `handle_missing(df, missing)` were executed on the entire dataset *prior* to `train_test_split`. The scaler learned test-set means and standard deviations, while imputers learned global distributions. This caused overly optimistic training scores that degraded upon real-world inference.
2. **Lack of Modern Tree-Boosting Algorithms**: Neither classification nor regression supported modern gradient boosting (such as `HistGradientBoosting` or `GradientBoosting`), which consistently achieve state-of-the-art results on tabular benchmarks.
3. **Class Imbalance Blindness**: While classification used `class_weight='balanced'`, there was no automated imbalance detection, ratio severity calculation, or diagnostics explaining when standard accuracy is misleading.
4. **Single-Split Variance**: Models were evaluated on a single test split. On smaller tabular datasets (<1000 rows), test metrics fluctuate heavily based on the random seed.

### 2.2 Model Selection Limitations
- **Classification was restricted to 4 estimators**: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `ExtraTreesClassifier`. Modern algorithms such as `HistGradientBoostingClassifier`, `GradientBoostingClassifier`, `KNeighborsClassifier`, and `SVC` were absent.
- **Regression was restricted to 3 estimators**: `LinearRegression`, `DecisionTreeRegressor`, and `RandomForestRegressor`. Regularized linear models (`Ridge`, `Lasso`, `ElasticNet`), boosting models (`HistGradientBoostingRegressor`, `GradientBoostingRegressor`), `ExtraTreesRegressor`, `KNeighborsRegressor`, and `SVR` were completely absent.
- **No dataset-aware adaptation**: Candidate models were evaluated without consideration of dataset dimensions, sample size, or feature counts.

### 2.3 Preprocessing Weaknesses
- **No Encapsulated Pipeline in Regression**: Regressors were trained on numpy arrays directly instead of an sklearn `Pipeline` containing `ColumnTransformer`. As a result, the fitted transformation state was not serializable with the model.
- **Categorical Failure Modes**: Categorical encodings did not robustly handle unseen categories on holdout sets during inference.
- **Outlier Vulnerability**: Standard scaling was applied indiscriminately without offering robust scaling for heavy-tailed or outlier-prone numerical distributions.

### 2.4 Evaluation Methodology Problems
- **Absence of Generalization Gap Tracking**: Systems reported either a holdout score or a single CV score. There was no tracking of the gap between training fit and cross-validation performance ($Score_{train} - Score_{cv}$), leaving users unaware of overfitting.
- **No Hierarchical Metric Reporting**: Training, cross-validation, and holdout scores were not reported together in a clear, beginner-friendly hierarchy.

### 2.5 Data Leakage Analysis
```
[LEGACY FLAWED FLOW - REGRESSION]
Dataset (All Rows) ──> handle_missing() [LEAK!] ──> scale_features() [LEAK!] ──> train_test_split()
                         (Global mean/median)         (Global mean/std)

[PHASE 1 LEAK-FREE PIPELINE]
Raw Dataset ──> train_test_split() ──┬──> Training Split ──> Pipeline.fit() [Imputer + Scaler + Model]
                                     │                         │
                                     │                         ├──> K-Fold CV on Training Folds Only
                                     │                         │
                                     └──> Untouched Holdout ───┴──> Pipeline.predict() [Strictly Unseen]
```

### 2.6 Integration & Serving Bottlenecks
- **Regression Registration Failure**: In legacy `POST /train`, regression runs returned metrics and visualization plots, but never returned a fitted `Pipeline` and never called `register_model()`. Consequently, regression models could not be saved to `model_registry.db`, could not be tested on `/predict`, could not generate SHAP explanations on `/explain`, and could not generate PDF evaluation reports on `/reports`.

---

## 3. Implemented Upgrades & Architecture

### 3.1 Pre-Training Dataset Intelligence (`analyze_dataset_intelligence`)
Located in `server/app/services/ml_intelligence_service.py` and exposed via `POST /train/analyze` and `POST /api/analyze-dataset`:
- **Dimensions & Dtypes**: Automatically counts rows, columns, duplicate rows, duplicate columns, numeric features, categorical features, and datetime candidates.
- **Task Inference**: Automatically determines whether the task is `binary_classification`, `multiclass_classification`, or `regression` based on cardinality, target dtype, and value distribution.
- **Class Imbalance Intelligence**: Computes class counts, majority-to-minority ratio, and classifies severity (`balanced` < 3:1, `moderate` 3:1 to 10:1, `severe` > 10:1).
- **Data Quality Alerts**: Scans for high cardinality (>50 categories), constant columns, potential ID columns (unique string per row), potential target leakage ($r > 0.98$), and collinear feature pairs ($r > 0.88$).
- **Actionable Recommendations**: Produces deterministic recommendations for optimal scaler, imputer, encoding, metric, and candidate models.

### 3.2 Leak-Free Preprocessing Engine
- Preprocessor is built using `ColumnTransformer` containing:
  - Numeric pipeline: `SimpleImputer(strategy='median' | 'mean')` + `StandardScaler()` / `RobustScaler()` / `MinMaxScaler()` / `Passthrough`.
  - Categorical pipeline: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False/True)`.
- Preprocessing is fitted strictly inside `Pipeline.fit(X_train, y_train)` and inside each cross-validation fold. The test split is never touched during fitting.

### 3.3 Dataset-Adaptive Model Suites
#### Classification (8 models):
1. `LogisticRegression` (L2 penalty, balanced class weights)
2. `DecisionTreeClassifier` (Gini criterion, balanced class weights)
3. `RandomForestClassifier` (100 balanced decision trees)
4. `ExtraTreesClassifier` (Extremely randomized trees ensemble)
5. `HistGradientBoostingClassifier` (Histogram-based gradient boosting)
6. `GradientBoostingClassifier` (Sequential additive boosting)
7. `KNeighborsClassifier` (Instance-based k-nearest neighbors)
8. `SVC` (Support Vector Classifier with RBF kernel and probability calibration)

#### Regression (11 models):
1. `LinearRegression` (Ordinary Least Squares)
2. `Ridge` (L2 Tikhonov regularization)
3. `Lasso` (L1 sparse feature selection)
4. `ElasticNet` (Convex combination of L1 and L2 penalties)
5. `DecisionTreeRegressor` (Piecewise constant trees)
6. `RandomForestRegressor` (Bagged ensemble of regression trees)
7. `ExtraTreesRegressor` (Extremely randomized regression trees)
8. `HistGradientBoostingRegressor` (Histogram gradient boosting)
9. `GradientBoostingRegressor` (Stage-wise additive regression)
10. `KNeighborsRegressor` (Distance-weighted neighbor regression)
11. `SVR` (Support Vector Regressor with epsilon-insensitive loss)

### 3.4 Rigorous Evaluation & Overfitting Detection
- **Training Cross-Validation**: Evaluates 5-fold (or class-constrained) Stratified K-Fold for classification and K-Fold for regression strictly on the training partition.
- **Untouched Holdout**: Evaluates holdout metrics on the unseen test split ($1 - split\_ratio$).
- **Generalization Gap**: Computes $Overfitting\_Gap = Score_{train} - Score_{cv}$. Gaps $> 0.18$ trigger `overfitting` warnings; gaps $< -0.05$ trigger `unstable_cv` variance warnings.
- **Failure Diagnostics**: Plain-language explanations detailing sample size adequacy, imbalance caveats, and actionable corrective steps.

### 3.5 Full Model Registry & Serving Integration
- `POST /train` now refits the full pipeline after evaluation and registers both classification AND regression models in `model_registry.db`.
- Registered regression models receive an ID, artifact persistence (`.joblib`), schema metadata, and immediate serving compatibility on `/predict`, `/explain`, and `/reports`.

### 3.6 Frontend Experience Enhancements
- **Auto Mode vs. Advanced Mode**:
  - *Auto Mode*: Dataset-adaptive defaults automatically configured based on `/train/analyze` profiling.
  - *Advanced Mode*: Fine-grained control over algorithm selection, scalers, imputers, and split ratios.
- **Validation & Generalization Hierarchy**: Displays Training Metric, Mean K-Fold Cross-Validation Score ($\pm$ std), and Untouched Holdout Score side-by-side.
- **Diagnostics Panel**: Surfaces generalization status, class imbalance severity, diagnostic findings, and actionable recommendations.
- **Quick Links**: One-click navigation to `/predict`, `/explain`, and `/reports`.
