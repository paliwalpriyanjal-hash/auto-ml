# server/app/services/assistant_service.py
"""
AutoVeyra Phase 6: Beginner-Friendly AI-Guided Machine Learning Workspace
Intelligent User Assistance & Contextual Guidance Engine.
"""

from datetime import datetime, timezone
import json
import sqlite3
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "experiments.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_workspace_projects_db():
    conn = _connect()
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dataset_id TEXT,
                    dataset_name TEXT,
                    target_column TEXT,
                    task_type TEXT DEFAULT 'classification',
                    experience_mode TEXT DEFAULT 'guided',
                    current_step INTEGER DEFAULT 1,
                    completed_steps TEXT DEFAULT '[]',
                    active_runs TEXT DEFAULT '{}',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_proj_updated ON workspace_projects(updated_at DESC)")
    finally:
        conn.close()


# ======================================================================
# 1. COMPREHENSIVE ML GLOSSARY (32+ Core Concepts)
# ======================================================================

ML_GLOSSARY: Dict[str, Dict[str, str]] = {
    "dataset": {
        "term": "Dataset",
        "category": "Data & Features",
        "simple_definition": "A structured collection of data organized in rows (records or examples) and columns (variables or attributes).",
        "why_it_matters": "The dataset is the foundation of all machine learning. A model can only learn patterns that exist in the provided data.",
        "example": "A table where each row is a customer and columns show their age, monthly bill, and whether they canceled their subscription.",
        "where_in_autoveyra": "Step 1 & Step 2 of Guided Workspace, and the Overview page.",
    },
    "feature": {
        "term": "Feature",
        "category": "Data & Features",
        "simple_definition": "An individual measurable property, characteristic, or column used as input to help the model make predictions.",
        "why_it_matters": "Choosing relevant features makes it easier for the algorithm to discover reliable mathematical relationships.",
        "example": "'Customer Age', 'Account Tenure (Months)', and 'Monthly Spending' are features used to predict churn.",
        "where_in_autoveyra": "Step 2 (Inspect Dataset) and Step 6 (Engineer Features).",
    },
    "target column": {
        "term": "Target Column",
        "category": "Problem Formulation",
        "simple_definition": "The specific column or outcome variable that you want your machine learning model to learn how to predict.",
        "why_it_matters": "Without a designated target column, the computer does not know what question it is answering.",
        "example": "'Churned' (Yes/No) for a subscription service, or 'House Sale Price' ($) for real estate.",
        "where_in_autoveyra": "Step 4 (Select Prediction Objective) in the Guided Workspace.",
    },
    "classification": {
        "term": "Classification",
        "category": "Problem Formulation",
        "simple_definition": "A machine learning task where the target outcome is a discrete category, label, or class.",
        "why_it_matters": "Used when the answer is an either/or decision, a diagnosis, or a grouping.",
        "example": "Predicting whether an email is 'Spam' or 'Not Spam', or whether a loan applicant is 'Approved' or 'Denied'.",
        "where_in_autoveyra": "Step 4 (Objective Selection), Model Training, and Comparison benchmarks.",
    },
    "regression": {
        "term": "Regression",
        "category": "Problem Formulation",
        "simple_definition": "A machine learning task where the target outcome is a continuous numerical quantity.",
        "why_it_matters": "Used when you need to estimate 'how much' or 'how many' rather than a category.",
        "example": "Predicting next month's company revenue in dollars or a patient's systolic blood pressure.",
        "where_in_autoveyra": "Step 4 (Objective Selection), Model Training, and Metric benchmarks (R², MAE, RMSE).",
    },
    "eda": {
        "term": "EDA (Exploratory Data Analysis)",
        "category": "Data Quality & Cleaning",
        "simple_definition": "The initial investigation of a dataset to discover patterns, spot anomalies, test hypotheses, and verify assumptions.",
        "why_it_matters": "Rushing into training without EDA leads to garbage-in, garbage-out models that fail in the real world.",
        "example": "Checking which features have missing values and visualizing how customer age correlates with purchase frequency.",
        "where_in_autoveyra": "Step 3 (Understand Dataset Quality) and the dedicated /eda module.",
    },
    "missing values": {
        "term": "Missing Values",
        "category": "Data Quality & Cleaning",
        "simple_definition": "Empty cells or missing observations in a column where information was not recorded.",
        "why_it_matters": "Most standard machine learning algorithms will crash or fail if fed empty numerical or text values.",
        "example": "A survey where 20% of respondents left the 'Annual Income' field blank.",
        "where_in_autoveyra": "Step 3 (Dataset Quality) and Step 5 (Data Cleaning Assistant).",
    },
    "duplicates": {
        "term": "Duplicates",
        "category": "Data Quality & Cleaning",
        "simple_definition": "Identical or repeated rows representing the exact same record or entity.",
        "why_it_matters": "Duplicate rows artificially inflate validation scores and bias the model toward over-represented examples.",
        "example": "Accidentally recording the same transaction twice due to a network glitch.",
        "where_in_autoveyra": "Step 3 (Dataset Quality) and Step 5 (Data Cleaning).",
    },
    "outliers": {
        "term": "Outliers",
        "category": "Data Quality & Cleaning",
        "simple_definition": "Data points that differ substantially from the rest of the observations in a feature.",
        "why_it_matters": "Outliers can severely distort models like Linear Regression by skewing slope coefficients.",
        "example": "A house price listed as $10,000,000 in a neighborhood where all other homes are $300,000, or an age entered as 999.",
        "where_in_autoveyra": "Step 3 (Dataset Quality) and Step 5 (Data Cleaning).",
    },
    "correlation": {
        "term": "Correlation",
        "category": "Data Quality & Cleaning",
        "simple_definition": "A statistical measure indicating the extent to which two variables fluctuate together (-1.0 to +1.0).",
        "why_it_matters": "Helps spot features that have a strong relationship with your target, as well as redundant features.",
        "example": "Years of experience usually has a positive correlation with salary.",
        "where_in_autoveyra": "Step 3 (Dataset Quality) and EDA Correlation Matrix.",
    },
    "multicollinearity": {
        "term": "Multicollinearity",
        "category": "Data Quality & Cleaning",
        "simple_definition": "A condition where two or more input features are so strongly correlated that they provide duplicate information.",
        "why_it_matters": "Makes linear model coefficients unstable and inflates feature importance artificially.",
        "example": "Including both 'Temperature in Fahrenheit' and 'Temperature in Celsius' as separate features.",
        "where_in_autoveyra": "Step 3 (Quality Audit) and Step 6 (Feature Selection).",
    },
    "encoding": {
        "term": "Encoding",
        "category": "Data Quality & Cleaning",
        "simple_definition": "Converting text categories (e.g. 'Red', 'Blue', 'Green') into numbers that algorithms can compute.",
        "why_it_matters": "Mathematical algorithms only understand numbers; encoding translates text attributes into numeric form.",
        "example": "One-Hot Encoding turns a 'Color' column with 3 colors into 3 binary 0/1 columns.",
        "where_in_autoveyra": "Step 5 (Clean & Prepare Data).",
    },
    "scaling": {
        "term": "Scaling",
        "category": "Data Quality & Cleaning",
        "simple_definition": "Adjusting the range of numeric features so they share comparable magnitudes (e.g. mean=0, std=1, or 0 to 1).",
        "why_it_matters": "Prevents features with large raw numbers (like Salary in thousands) from dominating features with small values (like Age).",
        "example": "Transforming Salary ($20k-$100k) and Age (18-70) so both span from 0.0 to 1.0.",
        "where_in_autoveyra": "Step 5 (Clean & Prepare Data) and AutoML settings.",
    },
    "imputation": {
        "term": "Imputation",
        "category": "Data Quality & Cleaning",
        "simple_definition": "The process of replacing missing data with substituted values, such as the column mean, median, or mode.",
        "why_it_matters": "Allows you to preserve rows with partially missing data instead of discarding entire records.",
        "example": "Filling missing customer ages with the median age of 38.",
        "where_in_autoveyra": "Step 5 (Clean & Prepare Data).",
    },
    "feature engineering": {
        "term": "Feature Engineering",
        "category": "Data & Features",
        "simple_definition": "Creating new, more informative input variables by transforming, combining, or decomposing raw dataset columns.",
        "why_it_matters": "High-quality engineered features often boost model accuracy more than tuning complex algorithm hyperparameters.",
        "example": "Extracting 'Day of Week' and 'Is_Weekend' from a raw timestamp column like '2026-10-30'.",
        "where_in_autoveyra": "Step 6 (Engineer Features) and the /features module.",
    },
    "feature selection": {
        "term": "Feature Selection",
        "category": "Data & Features",
        "simple_definition": "Automatically filtering out noisy, redundant, or irrelevant features to retain only the most predictive ones.",
        "why_it_matters": "Simplifies models, reduces training time, prevents overfitting, and improves model explainability.",
        "example": "Removing customer ID numbers and phone numbers before training a credit risk model.",
        "where_in_autoveyra": "Step 6 (Engineer Features).",
    },
    "training set": {
        "term": "Training Set",
        "category": "Training & Validation",
        "simple_definition": "The majority portion of your dataset (typically 70-80%) used by the algorithm to learn mathematical patterns.",
        "why_it_matters": "This is the study material the model uses during the learning process.",
        "example": "If you have 1,000 records, using 800 of them to train the model.",
        "where_in_autoveyra": "Step 7 (Train Models) split ratio configuration.",
    },
    "validation set": {
        "term": "Validation Set",
        "category": "Training & Validation",
        "simple_definition": "A sample of data held back during training to tune hyperparameters and provide an unbiased evaluation while fine-tuning.",
        "why_it_matters": "Helps select the best model configuration without cheating on the final test set.",
        "example": "Comparing a 5-tree vs 100-tree Random Forest on the validation split.",
        "where_in_autoveyra": "Step 7 (Train Models) and Step 8 (Compare Results).",
    },
    "test set": {
        "term": "Test Set (Holdout)",
        "category": "Training & Validation",
        "simple_definition": "A completely unseen portion of the dataset set aside at the beginning and used only once to evaluate final real-world performance.",
        "why_it_matters": "Gives an honest, unbiased estimate of how well the model will generalize to new future data.",
        "example": "Testing the trained model on 200 records it has never seen before.",
        "where_in_autoveyra": "Step 8 (Compare Results) Holdout Evaluation.",
    },
    "cross-validation": {
        "term": "Cross-Validation",
        "category": "Training & Validation",
        "simple_definition": "A technique where the training dataset is split into multiple folds (e.g. 5 folds), rotating which fold is used for testing.",
        "why_it_matters": "Provides a far more reliable performance estimate than a single train/test split, especially on smaller datasets.",
        "example": "5-Fold Cross-Validation trains 5 models on 80% and evaluates each on 20%, averaging the score.",
        "where_in_autoveyra": "Step 7 (Train Models) and Step 8 (Compare Results).",
    },
    "overfitting": {
        "term": "Overfitting",
        "category": "Training & Validation",
        "simple_definition": "When a model memorizes the training data (including its random noise) so thoroughly that it performs poorly on new data.",
        "why_it_matters": "A model that scores 99% on training data but 60% on unseen data is useless in production.",
        "example": "A student memorizing practice test answer letters (A, B, C) instead of understanding the underlying math.",
        "where_in_autoveyra": "Step 8 (Compare Results) Generalization Gap warning.",
    },
    "underfitting": {
        "term": "Underfitting",
        "category": "Training & Validation",
        "simple_definition": "When a model is too simplistic to capture the underlying patterns in the dataset, resulting in low accuracy everywhere.",
        "why_it_matters": "The model fails to learn useful patterns on both training and test data.",
        "example": "Trying to fit a straight linear line through data that follows a complex U-shaped curve.",
        "where_in_autoveyra": "Step 8 (Compare Results).",
    },
    "accuracy": {
        "term": "Accuracy",
        "category": "Evaluation Metrics",
        "simple_definition": "The percentage of all predictions that the model got exactly right out of total predictions.",
        "why_it_matters": "Good intuitive benchmark for balanced datasets, but dangerously misleading if classes are imbalanced.",
        "example": "Getting 92 out of 100 predictions correct gives an accuracy of 92%.",
        "where_in_autoveyra": "Step 8 (Compare Results) and Model Evaluation.",
    },
    "precision": {
        "term": "Precision",
        "category": "Evaluation Metrics",
        "simple_definition": "Out of all instances where the model predicted 'Positive', what fraction was actually positive?",
        "why_it_matters": "Crucial when false alarms (False Positives) are costly, such as spam filters or fraud flags.",
        "example": "If the model flags 10 transactions as fraud and 8 actually were fraud, Precision is 80%.",
        "where_in_autoveyra": "Step 8 (Compare Results).",
    },
    "recall": {
        "term": "Recall (Sensitivity)",
        "category": "Evaluation Metrics",
        "simple_definition": "Out of all actual positive cases that existed in reality, what fraction did the model successfully find?",
        "why_it_matters": "Crucial when missing a positive case (False Negative) is dangerous, such as medical disease detection.",
        "example": "If 10 patients have a disease and the model catches 9 of them, Recall is 90%.",
        "where_in_autoveyra": "Step 8 (Compare Results).",
    },
    "f1 score": {
        "term": "F1 Score",
        "category": "Evaluation Metrics",
        "simple_definition": "The harmonic mean of Precision and Recall, providing a single balanced metric between 0.0 and 1.0.",
        "why_it_matters": "The gold standard metric for classification on real-world datasets with uneven class distributions.",
        "example": "If Precision is 0.80 and Recall is 0.80, the F1 Score is 0.80.",
        "where_in_autoveyra": "Step 8 (Compare Results) and Model Leaderboards.",
    },
    "balanced accuracy": {
        "term": "Balanced Accuracy",
        "category": "Evaluation Metrics",
        "simple_definition": "The average of recall obtained on each class, treating minority and majority classes with equal weight.",
        "why_it_matters": "Prevents majority classes from hiding poor performance on minority target classes.",
        "example": "A fraud dataset where non-fraud accuracy is 99% and fraud recall is 50% yields a Balanced Accuracy of 74.5%.",
        "where_in_autoveyra": "Step 8 (Compare Results).",
    },
    "rmse": {
        "term": "RMSE (Root Mean Squared Error)",
        "category": "Evaluation Metrics",
        "simple_definition": "The square root of the average squared difference between predicted values and actual values.",
        "why_it_matters": "Expressed in the exact same units as your target column; heavily penalizes large mistake errors.",
        "example": "If predicting house prices, an RMSE of $15,000 means typical errors are about $15,000.",
        "where_in_autoveyra": "Step 8 (Compare Results) for Regression tasks.",
    },
    "mae": {
        "term": "MAE (Mean Absolute Error)",
        "category": "Evaluation Metrics",
        "simple_definition": "The average absolute difference between predicted numbers and actual numbers.",
        "why_it_matters": "Very easy to interpret and less sensitive to a few extreme outlier predictions than RMSE.",
        "example": "An MAE of 3.2 days when predicting project delivery duration.",
        "where_in_autoveyra": "Step 8 (Compare Results) for Regression tasks.",
    },
    "r²": {
        "term": "R² (R-Squared / Coefficient of Determination)",
        "category": "Evaluation Metrics",
        "simple_definition": "The proportion of variance in the target variable that is explained by the model's input features (0.0 to 1.0).",
        "why_it_matters": "An R² of 0.85 means the model explains 85% of the variation in the outcome.",
        "example": "An R² of 0.0 means the model is no better than guessing the average value; 1.0 is perfect prediction.",
        "where_in_autoveyra": "Step 8 (Compare Results) for Regression tasks.",
    },
    "hyperparameter": {
        "term": "Hyperparameter",
        "category": "Training & Validation",
        "simple_definition": "A configuration setting defined by the engineer before training begins that controls how the algorithm learns.",
        "why_it_matters": "Tuning hyperparameters can significantly improve performance or prevent overfitting.",
        "example": "The maximum depth of a decision tree (e.g. max_depth=5) or the number of trees in a Random Forest (n_estimators=100).",
        "where_in_autoveyra": "Step 7 (Train Models) and the /tune module.",
    },
    "shap": {
        "term": "SHAP (SHapley Additive exPlanations)",
        "category": "Explainability & Governance",
        "simple_definition": "A game-theory method that calculates the exact contribution of each feature to a model's final prediction.",
        "why_it_matters": "Explains 'why' a model made a specific prediction, building trust and satisfying regulatory audit requirements.",
        "example": "SHAP shows that a loan was denied primarily because the applicant's Debt-to-Income ratio was +0.35 above threshold.",
        "where_in_autoveyra": "Step 9 (Explain Model Predictions) and /explain module.",
    },
    "permutation importance": {
        "term": "Permutation Importance",
        "category": "Explainability & Governance",
        "simple_definition": "A technique that measures feature importance by shuffling values of one column and observing how much the model's accuracy drops.",
        "why_it_matters": "Model-agnostic, reliable, and prevents tree-based biases toward high-cardinality numbers.",
        "example": "If shuffling the 'Salary' column drops model accuracy from 90% down to 65%, 'Salary' is critically important.",
        "where_in_autoveyra": "Step 9 (Explain Model Predictions).",
    },
}


# ======================================================================
# 2. NEXT-ACTION RECOMMENDATION ENGINE
# ======================================================================

def recommend_next_action(project_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates current project context and returns a prioritized recommendation.
    """
    dataset_id = project_state.get("dataset_id")
    dataset_name = project_state.get("dataset_name")
    target_column = project_state.get("target_column")
    active_runs = project_state.get("active_runs", {})
    completed_steps = project_state.get("completed_steps", [])

    # Case 1: No dataset uploaded yet
    if not dataset_id and not dataset_name:
        return {
            "step": 1,
            "title": "Upload Your Dataset",
            "reason": "Machine learning starts with data. Upload a CSV file to begin your project.",
            "cta_label": "Upload CSV Dataset",
            "cta_action": "open_step_1",
            "priority": "high",
            "icon": "upload",
        }

    # Case 2: Dataset uploaded, but objective/target not selected
    if not target_column:
        return {
            "step": 4,
            "title": "Define Prediction Objective",
            "reason": "We need to know what you want to predict (e.g. classification vs regression, and the target column).",
            "cta_label": "Select Target Column",
            "cta_action": "open_step_4",
            "priority": "high",
            "icon": "target",
        }

    # Case 3: Dataset & Target ready, but EDA not performed
    has_eda = 3 in completed_steps or "eda" in active_runs
    if not has_eda:
        return {
            "step": 3,
            "title": "Inspect Dataset Quality & Run EDA",
            "reason": "Analyzing data quality now prevents surprises. Check missingness, duplicates, and distributions.",
            "cta_label": "Review Data Quality",
            "cta_action": "open_step_3",
            "priority": "medium",
            "icon": "search",
        }

    # Case 4: Quality checked, but cleaning not performed
    has_clean = 5 in completed_steps or "clean" in active_runs
    has_missing_or_issues = project_state.get("has_data_issues", False)
    if has_missing_or_issues and not has_clean:
        return {
            "step": 5,
            "title": "Review Data Cleaning Recommendations",
            "reason": "Your dataset contains missing values or duplicates that can degrade model training accuracy.",
            "cta_label": "Clean & Prepare Data",
            "cta_action": "open_step_5",
            "priority": "high",
            "icon": "wand",
        }

    # Case 5: Cleaning done/skipped, but features not checked
    has_features = 6 in completed_steps or "feature_engineering" in active_runs
    if not has_features:
        return {
            "step": 6,
            "title": "Explore Feature Engineering",
            "reason": "Transforming dates or numerical interactions can help algorithms learn subtle relationships.",
            "cta_label": "Inspect Features",
            "cta_action": "open_step_6",
            "priority": "medium",
            "icon": "filter",
        }

    # Case 6: Data prepared, but models not trained
    has_models = 7 in completed_steps or "training" in active_runs or "model_id" in active_runs
    if not has_models:
        return {
            "step": 7,
            "title": "Train Candidate Models",
            "reason": "Your data is clean and prepared. Train and compare multiple suitable algorithms.",
            "cta_label": "Start Training",
            "cta_action": "open_step_7",
            "priority": "high",
            "icon": "cpu",
        }

    # Case 7: Models trained, but results not compared
    has_compared = 8 in completed_steps or "compare" in active_runs
    if not has_compared:
        return {
            "step": 8,
            "title": "Compare Model Benchmarks",
            "reason": "Evaluate accuracy, F1 score, and holdout generalization to select your champion model.",
            "cta_label": "Review Comparison",
            "cta_action": "open_step_8",
            "priority": "high",
            "icon": "git-compare",
        }

    # Case 8: Champion selected, but explainability not run
    has_explain = 9 in completed_steps or "explain" in active_runs
    if not has_explain:
        return {
            "step": 9,
            "title": "Explain Model Predictions (SHAP)",
            "reason": "Understand which features drive predictions to verify that your model is fair and trustworthy.",
            "cta_label": "Run Explainability",
            "cta_action": "open_step_9",
            "priority": "medium",
            "icon": "shield-check",
        }

    # Case 9: Everything done, report not generated
    has_report = 10 in completed_steps or "report" in active_runs
    if not has_report:
        return {
            "step": 10,
            "title": "Generate Final Project Report",
            "reason": "Your project is complete! Compile all 15 technical sections and download your PDF and ZIP package.",
            "cta_label": "Create Project Report",
            "cta_action": "open_step_10",
            "priority": "high",
            "icon": "file-text",
        }

    # Default: Completed
    return {
        "step": 10,
        "title": "Project Fully Compiled",
        "reason": "All 10 steps of your machine learning workflow are complete and documented.",
        "cta_label": "Download Report & Bundle",
        "cta_action": "download_bundle",
        "priority": "info",
        "icon": "check-circle",
    }


# ======================================================================
# 3. CONTEXT-AWARE EXPLANATIONS & METRIC INTERPRETER
# ======================================================================

ALGORITHM_PROFILES = {
    "logisticregression": {
        "name": "Logistic Regression",
        "category": "Classification",
        "simple_analogy": "Draws a smooth boundary separating two groups based on weighted evidence.",
        "strengths": "Fast, highly interpretable, outputs clear probabilities, works well on linearly separable data.",
        "limitations": "Cannot capture complex non-linear interactions without manual feature engineering.",
        "best_for": "Baseline classification, credit scoring, healthcare risk when explainability is mandatory.",
    },
    "randomforestclassifier": {
        "name": "Random Forest Classifier",
        "category": "Classification",
        "simple_analogy": "A committee of 100 decision trees voting on the outcome; majority vote wins.",
        "strengths": "Robust against overfitting, handles non-linear relationships, requires little data scaling.",
        "limitations": "Can be slower to train on huge datasets; trees can be large memory artifacts.",
        "best_for": "General-purpose tabular classification with mixed data types.",
    },
    "randomforest": {
        "name": "Random Forest Regressor",
        "category": "Regression",
        "simple_analogy": "A committee of decision trees predicting numbers; final answer is the average prediction.",
        "strengths": "Captures non-linear curves, robust against outlier distortion, excellent baseline.",
        "limitations": "Cannot extrapolate outside the minimum and maximum numeric values seen in training.",
        "best_for": "Predicting prices, demand, quantities from tabular features.",
    },
    "gradientboostingclassifier": {
        "name": "Gradient Boosting Classifier",
        "category": "Classification",
        "simple_analogy": "A sequence of decision trees where each tree specifically focuses on correcting the errors of previous trees.",
        "strengths": "Often produces state-of-the-art accuracy on structured tabular datasets.",
        "limitations": "More prone to overfitting if learning rate is too high; sequential training takes longer.",
        "best_for": "High-accuracy tabular competitions and customer conversion modeling.",
    },
    "decisiontreeclassifier": {
        "name": "Decision Tree Classifier",
        "category": "Classification",
        "simple_analogy": "A flowchart of if/then rules based on feature thresholds.",
        "strengths": "Completely transparent, easy to explain visually to non-technical stakeholders.",
        "limitations": "Prone to severe overfitting if depth is unconstrained.",
        "best_for": "Simple rule-based business logic and educational exploration.",
    },
    "linearregression": {
        "name": "Linear Regression",
        "category": "Regression",
        "simple_analogy": "Draws the best-fitting straight line through your data points.",
        "strengths": "Instant training, mathematically transparent, clear positive/negative coefficients.",
        "limitations": "Fails when relationships are curved or non-linear; very sensitive to outliers.",
        "best_for": "Simple continuous estimation and trend analysis.",
    },
    "ridge": {
        "name": "Ridge Regression (L2)",
        "category": "Regression",
        "simple_analogy": "Linear regression with a penalty that shrinks overly large coefficients.",
        "strengths": "Handles correlated features (multicollinearity) without coefficients exploding.",
        "limitations": "Still assumes linear relationships.",
        "best_for": "Regression datasets with many correlated numerical features.",
    },
    "lasso": {
        "name": "Lasso Regression (L1)",
        "category": "Regression",
        "simple_analogy": "Linear regression that shrinks useless coefficients all the way to zero, acting like automatic feature selection.",
        "strengths": "Identifies the truly important linear features automatically.",
        "limitations": "Can drop useful features if two columns are nearly identical.",
        "best_for": "Sparse datasets where only a few features matter.",
    },
}


def explain_metric_result(metric_name: str, value: float, task_type: str = "classification") -> Dict[str, Any]:
    """
    Translates a raw metric number into plain-English feedback for beginners.
    """
    metric_key = metric_name.lower().replace(" ", "_")

    if metric_key in ("accuracy", "acc"):
        pct = value * 100.0 if value <= 1.0 else value
        if pct >= 90:
            quality = "Excellent"
            detail = f"The model correctly predicted {pct:.1f}% of all test cases. Verify class balance to ensure this isn't biased."
        elif pct >= 75:
            quality = "Good"
            detail = f"The model correctly predicted {pct:.1f}% of cases, performing well above random baseline."
        else:
            quality = "Moderate"
            detail = f"The model achieved {pct:.1f}% accuracy. Consider cleaning noisy features or testing non-linear algorithms."
        return {
            "metric": "Accuracy",
            "value": f"{pct:.1f}%",
            "quality": quality,
            "plain_explanation": detail,
            "benchmark_tip": "High accuracy can be deceptive if one class makes up 90% of your data.",
        }

    if metric_key in ("f1", "f1_score", "f1_macro", "f1_weighted"):
        val = value if value <= 1.0 else value / 100.0
        if val >= 0.85:
            quality = "Strong"
            detail = f"An F1 score of {val:.3f} indicates excellent harmony between Precision (low false alarms) and Recall (catching true cases)."
        elif val >= 0.70:
            quality = "Good"
            detail = f"An F1 score of {val:.3f} shows a balanced, dependable model for real-world deployment."
        else:
            quality = "Needs Improvement"
            detail = f"An F1 score of {val:.3f} suggests the model struggles with false positives or false negatives."
        return {
            "metric": "F1 Score",
            "value": f"{val:.3f}",
            "quality": quality,
            "plain_explanation": detail,
            "benchmark_tip": "F1 is the best metric when classes are unevenly distributed.",
        }

    if metric_key in ("r2", "r_squared", "r2_score"):
        if value >= 0.80:
            quality = "Excellent"
            detail = f"The model explains {value * 100:.1f}% of the variance in the target variable."
        elif value >= 0.50:
            quality = "Moderate"
            detail = f"The model explains {value * 100:.1f}% of the outcome variation. Some underlying factors remain unexplained."
        else:
            quality = "Weak"
            detail = f"An R² of {value:.3f} means the features only explain a small fraction of the changes in the target."
        return {
            "metric": "R² (Coefficient of Determination)",
            "value": f"{value:.3f}",
            "quality": quality,
            "plain_explanation": detail,
            "benchmark_tip": "1.0 is a perfect score; 0.0 means the model is no better than guessing the average.",
        }

    if metric_key in ("mae", "mean_absolute_error"):
        return {
            "metric": "MAE (Mean Absolute Error)",
            "value": f"{value:.2f}",
            "quality": "Informative",
            "plain_explanation": f"On average, the model's predictions differ from actual reality by {value:.2f} units.",
            "benchmark_tip": "Lower MAE is always better. Compare this number to the overall range of your target column.",
        }

    if metric_key in ("rmse", "root_mean_squared_error"):
        return {
            "metric": "RMSE",
            "value": f"{value:.2f}",
            "quality": "Informative",
            "plain_explanation": f"Typical error magnitude is approximately {value:.2f} units, with larger penalties for extreme misses.",
            "benchmark_tip": "If RMSE is much higher than MAE, it means the model makes occasional very large errors.",
        }

    # Default fallback
    return {
        "metric": metric_name,
        "value": f"{value:.4f}",
        "quality": "Recorded",
        "plain_explanation": f"The model recorded {metric_name} = {value:.4f} during holdout evaluation.",
        "benchmark_tip": "Refer to the ML Glossary for full metric definitions.",
    }


def explain_generalization_gap(cv_score: float, holdout_score: float, metric_name: str = "Accuracy") -> Dict[str, Any]:
    """
    Evaluates potential overfitting or distribution shifts between CV and Holdout.
    """
    gap = cv_score - holdout_score
    if gap > 0.08:
        status = "Possible Overfitting"
        explanation = (
            f"The model scored {cv_score:.3f} during cross-validation, but dropped to {holdout_score:.3f} "
            f"on unseen holdout data (gap of {gap:.3f}). This suggests the model may have memorized training quirks "
            "rather than learning universal patterns."
        )
        recommendation = "Consider pruning tree depth, increasing regularization, or adding more diverse training examples."
    elif gap < -0.05:
        status = "Favorable Holdout"
        explanation = (
            f"Holdout performance ({holdout_score:.3f}) was slightly higher than cross-validation ({cv_score:.3f}). "
            "The model generalizes well to the held-out sample."
        )
        recommendation = "The model is stable and ready for final evaluation."
    else:
        status = "Well Generalized"
        explanation = (
            f"Cross-validation ({cv_score:.3f}) and holdout ({holdout_score:.3f}) align closely (gap: {abs(gap):.3f}). "
            "The model demonstrates honest, reproducible generalization on unseen data."
        )
        recommendation = "No sign of overfitting detected."

    return {
        "status": status,
        "cv_score": cv_score,
        "holdout_score": holdout_score,
        "gap": round(gap, 4),
        "explanation": explanation,
        "recommendation": recommendation,
    }


# ======================================================================
# 4. PRE-FLIGHT VALIDATION & MISTAKE PREVENTION
# ======================================================================

def validate_preflight(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs comprehensive validations before expensive operations to protect beginners from mistakes.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if action == "upload":
        filename = payload.get("filename", "")
        file_size = payload.get("file_size", 0)
        if not filename.lower().endswith(".csv"):
            errors.append("Only CSV (.csv) files are supported. Please convert your file to CSV format.")
        if file_size <= 0:
            errors.append("The uploaded file appears to be empty (0 bytes).")
        if file_size > 50 * 1024 * 1024:
            errors.append("File size exceeds 50MB limit. Consider sampling rows before uploading.")

    elif action in ("train", "compare", "automl"):
        target_column = payload.get("target_column")
        task_type = payload.get("task_type", "classification")
        columns = payload.get("columns", [])
        row_count = payload.get("row_count", 0)
        class_distribution = payload.get("class_distribution", {})

        if not target_column:
            errors.append("No target column selected. Please select which column you want to predict.")
        elif columns and target_column not in columns:
            errors.append(f"Target column '{target_column}' does not exist in the dataset columns.")

        if row_count > 0 and row_count < 15:
            errors.append(f"Dataset has only {row_count} rows. Machine learning requires at least 15 rows to form train/test splits.")

        if task_type == "classification" and class_distribution:
            num_classes = len(class_distribution)
            if num_classes < 2:
                errors.append("The selected target column only contains 1 unique class. Classification requires at least 2 classes.")
            min_samples = min(class_distribution.values()) if class_distribution else 0
            if min_samples < 2:
                errors.append(f"The minority class has only {min_samples} sample(s). At least 2 samples per class are needed for stratified splitting.")
            elif min_samples < 5:
                warnings.append(f"The minority class has only {min_samples} samples. Cross-validation folds may experience imbalance.")

    elif action == "clean":
        row_count = payload.get("row_count", 0)
        missing_count = payload.get("missing_count", 0)
        if row_count > 0 and missing_count == 0:
            warnings.append("No missing values were detected in your dataset. Imputation may be skipped.")

    elif action == "report":
        dataset_id = payload.get("dataset_id")
        selected_runs = payload.get("selected_run_ids", [])
        if not dataset_id:
            errors.append("Please select a target dataset for the project report.")
        if not selected_runs:
            warnings.append("No experiment runs selected. The report will document the raw dataset without model training benchmarks.")

    return {
        "action": action,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ======================================================================
# 5. WORKSPACE PROJECTS PERSISTENCE
# ======================================================================

def create_or_update_workspace_project(
    name: str,
    dataset_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    target_column: Optional[str] = None,
    task_type: str = "classification",
    experience_mode: str = "guided",
    current_step: int = 1,
    completed_steps: Optional[List[int]] = None,
    active_runs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    init_workspace_projects_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    pid = project_id or f"proj_{uuid.uuid4().hex[:10]}"
    completed_steps_json = json.dumps(completed_steps or [])
    active_runs_json = json.dumps(active_runs or {})
    meta_json = json.dumps(metadata or {})

    conn = _connect()
    try:
        with conn:
            existing = conn.execute(
                "SELECT project_id, created_at FROM workspace_projects WHERE project_id = ?",
                (pid,),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE workspace_projects
                    SET name = ?, dataset_id = ?, dataset_name = ?, target_column = ?,
                        task_type = ?, experience_mode = ?, current_step = ?,
                        completed_steps = ?, active_runs = ?, metadata_json = ?,
                        updated_at = ?
                    WHERE project_id = ?
                    """,
                    (
                        name,
                        dataset_id,
                        dataset_name,
                        target_column,
                        task_type,
                        experience_mode,
                        current_step,
                        completed_steps_json,
                        active_runs_json,
                        meta_json,
                        now_iso,
                        pid,
                    ),
                )
                created_at = existing["created_at"]
            else:
                created_at = now_iso
                conn.execute(
                    """
                    INSERT INTO workspace_projects (
                        project_id, name, dataset_id, dataset_name, target_column,
                        task_type, experience_mode, current_step, completed_steps,
                        active_runs, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pid,
                        name,
                        dataset_id,
                        dataset_name,
                        target_column,
                        task_type,
                        experience_mode,
                        current_step,
                        completed_steps_json,
                        active_runs_json,
                        meta_json,
                        created_at,
                        now_iso,
                    ),
                )
    finally:
        conn.close()

    return get_workspace_project(pid)


def get_workspace_project(project_id: str) -> Optional[Dict[str, Any]]:
    init_workspace_projects_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM workspace_projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["completed_steps"] = json.loads(d["completed_steps"]) if d["completed_steps"] else []
        d["active_runs"] = json.loads(d["active_runs"]) if d["active_runs"] else {}
        d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
        d.pop("metadata_json", None)
        return d
    finally:
        conn.close()


def list_workspace_projects(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    init_workspace_projects_db()
    conn = _connect()
    try:
        total = conn.execute("SELECT COUNT(*) FROM workspace_projects").fetchone()[0]
        rows = conn.execute(
            """
            SELECT project_id, name, dataset_id, dataset_name, target_column,
                   task_type, experience_mode, current_step, completed_steps,
                   created_at, updated_at
            FROM workspace_projects
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        projects = []
        for r in rows:
            p = dict(r)
            p["completed_steps"] = json.loads(p["completed_steps"]) if p["completed_steps"] else []
            projects.append(p)

        return {"total": total, "projects": projects, "limit": limit, "offset": offset}
    finally:
        conn.close()


def delete_workspace_project(project_id: str) -> bool:
    init_workspace_projects_db()
    conn = _connect()
    try:
        with conn:
            res = conn.execute("DELETE FROM workspace_projects WHERE project_id = ?", (project_id,))
            return res.rowcount > 0
    finally:
        conn.close()
