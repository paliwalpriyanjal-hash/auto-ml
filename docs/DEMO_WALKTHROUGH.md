# AutoVeyra — Quickstart Demo Walkthrough & Evaluation Guide

## 1. Overview

AutoVeyra is an enterprise tabular AutoML platform. This walkthrough demonstrates an end-to-end customer churn prediction workflow using the included synthetic demo dataset:

- **Dataset**: [`demo_data/customer_churn_synthetic.csv`](file:///c:/Users/Priyanjal%20Paliwal/Downloads/AutoML-Studio-main/AutoML-Studio-main/demo_data/customer_churn_synthetic.csv)
- **Problem Type**: Binary Classification
- **Target Column**: `Churn` (0 = Retained, 1 = Churned)
- **Sample Count**: 150 records (instant training and cross-validation)
- **Features**:
  - `Age` (Integer: 18–74)
  - `TenureMonths` (Integer: 1–71)
  - `MonthlyCharges` (Float: \$20.46–\$109.10)
  - `TotalCharges` (Float: \$58.02–\$7323.96)
  - `SupportCalls` (Integer: 0–8)
  - `ContractType` (Categorical: `Month-to-Month`, `One-Year`, `Two-Year`)
  - `PaymentMethod` (Categorical: `ElectronicCheck`, `MailedCheck`, `BankTransfer`)
  - `InternetService` (Categorical: `Fiber`, `DSL`, `No`)

---

## 2. Step-by-Step Walkthrough

### Step 1: Start Services

```bash
# Terminal 1: Launch Backend (FastAPI on http://127.0.0.1:8000)
cd server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch Frontend (Vite on http://localhost:5173)
cd client
npm run dev
```

Open `http://localhost:5173` in your browser. Verify the green **"API Live"** badge in the navbar.

---

### Step 2: Automated Data Profiling & Feature Inspection

1. Navigate to **Features** (`/features`).
2. Upload `demo_data/customer_churn_synthetic.csv`.
3. Select `Churn` as the Target Column and click **"Generate Profile"**.
4. Observe:
   - Summary statistics (150 rows, 9 columns, 0 missing rows).
   - Cardinality and distribution of numerical features (`TenureMonths`, `MonthlyCharges`).
   - Feature correlation alerts flagging collinearity or potential identifier columns.

---

### Step 3: Multi-Algorithm Model Comparison

1. Navigate to **Compare Models** (`/compare`).
2. Upload `demo_data/customer_churn_synthetic.csv`.
3. Set configuration:
   - **Task Type**: `Classification`
   - **Target Column**: `Churn`
   - **Scaler**: `Standard`
   - **Split Ratio**: `0.8` (80% Train, 20% Holdout Test)
   - **Imputation**: `Median`
   - **Encoding**: `One-Hot`
4. Click **"Run Model Comparison"**.
5. Observe side-by-side performance rankings across 4 models (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`):
   - $F_1$-Macro, ROC-AUC, Accuracy, Precision, Recall, and Log Loss.
   - The winning algorithm is automatically serialized and registered in the **Model Registry**!

---

### Step 4: Hyperparameter Optimization (Optuna)

1. Navigate to **Tuning** (`/tune`).
2. Upload `demo_data/customer_churn_synthetic.csv`.
3. Set configuration:
   - **Model Algorithm**: `RandomForestClassifier`
   - **Trials**: `10`
   - **Timeout**: `60` seconds
4. Click **"Start Hyperparameter Tuning"**.
5. View real-time trial progression, parameter convergence, and delta comparison comparing the tuned pipeline against the baseline.
6. The improved tuned model is saved to the Model Registry.

---

### Step 5: Model Explainability (SHAP & Permutation Importance)

1. Navigate to **Explainability** (`/explain`).
2. Select your registered `RandomForestClassifier` from the model selector dropdown.
3. Upload `demo_data/customer_churn_synthetic.csv` as the evaluation dataset.
4. Click **"Compute Permutation Importance"**:
   - Inspect global feature rankings showing `MonthlyCharges` and `TenureMonths` as top contributors.
5. Click **"Compute SHAP Explanations"**:
   - Inspect TreeExplainer global feature importances.
   - Review row-level waterfall attributions showing how individual customer attributes pushed the prediction towards Churn vs. Retain.

---

### Step 6: Model Evaluation Report & Model Card Export

1. Navigate to **Reports** (`/reports`).
2. Click **"Create Evaluation Report"**.
3. Select the registered model, assign title `"Customer Churn Production Baseline"`, and specify intended operational use in the Model Card editor.
4. Click **"Generate Report"**.
5. View the interactive report snapshot showing model provenance, metrics, and risk assessment.
6. Click **"Download PDF"**, **"Download HTML"**, or **"Download JSON"** to export compliance documentation.

---

### Step 7: Real-Time Served Prediction

1. Navigate to **Predictions** (`/predict`).
2. Select the registered model.
3. Input custom customer values:
   ```json
   {
     "Age": 45,
     "TenureMonths": 12,
     "MonthlyCharges": 85.50,
     "TotalCharges": 1026.00,
     "SupportCalls": 4,
     "ContractType": "Month-to-Month",
     "PaymentMethod": "ElectronicCheck",
     "InternetService": "Fiber"
   }
   ```
4. Click **"Run Prediction"**.
5. Observe:
   - Cold load: `cached: false`, ~20ms latency.
   - Subsequent calls: `cached: true`, **< 1ms latency** served directly from the in-memory LRU model cache!

---

### Step 8: Operational Telemetry & Feature Drift

1. Navigate to **Monitoring** (`/monitoring`).
2. Observe:
   - Live KPI cards: Total served predictions, error rate, $p50$, $p90$, $p99$ latencies.
   - **Active Serving Cache Inventory**: Shows in-memory model pipelines.
   - **Recent Audit Logs**: Real-time log of inference events with correlation IDs.
   - **Covariate Drift Inspector**: Select the model and observe feature distribution comparisons comparing incoming inference statistics against baseline training distributions.
