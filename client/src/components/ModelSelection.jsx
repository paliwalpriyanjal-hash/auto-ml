import React from "react";
import { FiCpu } from "react-icons/fi";

const ModelSelection = ({ model, setModel, taskType = "regression" }) => (
  <section className="bg-[#111126] p-6 rounded-2xl border border-white/[0.08] shadow-xl mb-6">
    <div className="flex items-center mb-5 border-b border-white/[0.06] pb-3">
      <div className="bg-[#A855F7]/10 p-2.5 rounded-xl mr-3 border border-[#A855F7]/20">
        <FiCpu className="text-[#A855F7]" size={18} />
      </div>
      <div>
        <h3 className="text-base font-bold text-white">Algorithm Selection</h3>
        <p className="text-xs text-[#A4A4B8]">
          Choose candidate estimator for {taskType} task
        </p>
      </div>
    </div>

    <div className="space-y-4">
      <div>
        <label className="block text-xs font-semibold uppercase tracking-wider text-gray-300 mb-2">
          Machine Learning Model
        </label>
        {taskType === "classification" ? (
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full px-4 py-2.5 bg-[#171630] border border-white/[0.1] rounded-xl text-sm text-white focus:outline-none focus:border-[#8265FF] transition duration-200"
          >
            <option value="RandomForestClassifier">Random Forest (Balanced Ensemble)</option>
            <option value="HistGradientBoostingClassifier">Hist Gradient Boosting (Fast, Tabular-Optimized)</option>
            <option value="GradientBoostingClassifier">Gradient Boosting (Sequential Trees)</option>
            <option value="ExtraTreesClassifier">Extra Trees (Extremely Randomized)</option>
            <option value="LogisticRegression">Logistic Regression (Linear Probabilistic)</option>
            <option value="DecisionTreeClassifier">Decision Tree (Single Interpretable Tree)</option>
            <option value="KNeighborsClassifier">K-Nearest Neighbors (Instance-Based)</option>
            <option value="SVC">Support Vector Machine (Kernelized Margin)</option>
          </select>
        ) : (
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full px-4 py-2.5 bg-[#171630] border border-white/[0.1] rounded-xl text-sm text-white focus:outline-none focus:border-[#8265FF] transition duration-200"
          >
            <option value="RandomForest">Random Forest Regressor (Ensemble)</option>
            <option value="HistGradientBoosting">Hist Gradient Boosting Regressor (Tabular-Optimized)</option>
            <option value="GradientBoosting">Gradient Boosting Regressor (Sequential Trees)</option>
            <option value="ExtraTrees">Extra Trees Regressor (Randomized Ensemble)</option>
            <option value="LinearRegression">Linear Regression (Ordinary Least Squares)</option>
            <option value="Ridge">Ridge Regression (L2 Regularized)</option>
            <option value="Lasso">Lasso Regression (L1 Sparse Feature Selection)</option>
            <option value="ElasticNet">ElasticNet (L1 + L2 Regularization)</option>
            <option value="DecisionTree">Decision Tree Regressor (Piecewise Constant)</option>
            <option value="KNeighborsRegressor">K-Nearest Neighbors Regressor</option>
            <option value="SVR">Support Vector Regressor (Kernel Epsilon-Tube)</option>
          </select>
        )}
      </div>

      <div className="bg-[#171630]/70 p-4 rounded-xl border border-white/[0.06]">
        <h4 className="text-xs font-semibold text-[#8265FF] uppercase tracking-wider mb-1">
          Algorithm Architecture Notes
        </h4>
        <p className="text-xs text-[#A4A4B8] leading-relaxed">
          {taskType === "classification" ? (
            <>
              {(model === "LogisticRegression" || model === "logistic") &&
                "Linear classifier with balanced class weights, L2 regularization, and calibrated probability estimates."}
              {(model === "DecisionTreeClassifier" || model === "DecisionTree") &&
                "Non-linear decision tree splitting samples on Gini impurity. Highly interpretable."}
              {(model === "RandomForestClassifier" || model === "RandomForest") &&
                "Ensemble of 100 balanced decision trees. Reduces variance and prevents overfitting on high-dimensional data."}
              {(model === "ExtraTreesClassifier" || model === "ExtraTrees") &&
                "Extremely randomized trees ensemble for fast variance reduction and robust generalization."}
              {model === "HistGradientBoostingClassifier" &&
                "Modern histogram-based gradient boosting algorithm optimized for tabular data with native numerical binning."}
              {model === "GradientBoostingClassifier" &&
                "Sequential boosting ensemble minimizing pseudo-residuals with stage-wise additive decision trees."}
              {model === "KNeighborsClassifier" &&
                "Non-parametric distance-weighted classification based on Euclidean neighborhood voting."}
              {model === "SVC" &&
                "Support Vector Classifier constructing maximum-margin separating hyperplanes with RBF kernel and probability calibration."}
            </>
          ) : (
            <>
              {model === "LinearRegression" &&
                "Linear model minimizing residual sum of squares between observed targets and predicted values."}
              {model === "Ridge" &&
                "Linear regression with L2 ridge penalty, preventing multicollinearity instability and weight inflation."}
              {model === "Lasso" &&
                "Linear regression with L1 lasso penalty, performing automatic feature shrinkage and sparse selection."}
              {model === "ElasticNet" &&
                "Hybrid linear regularizer balancing L1 sparsity and L2 collinearity stabilization."}
              {model === "DecisionTree" &&
                "Non-linear regressor mapping continuous piecewise constant approximations."}
              {model === "RandomForest" &&
                "Ensemble of averaging regression trees, mitigating variance and outlier vulnerability."}
              {model === "ExtraTrees" &&
                "Extremely randomized regression trees providing higher diversity and lower variance."}
              {model === "HistGradientBoosting" &&
                "High-speed histogram gradient boosted regressor with bin-based tree construction."}
              {model === "GradientBoosting" &&
                "Sequential gradient boosting regressor fitting successive trees to negative gradients."}
              {model === "KNeighborsRegressor" &&
                "Local non-parametric regressor predicting target as distance-weighted mean of closest neighbors."}
              {model === "SVR" &&
                "Support Vector Regressor fitting predictions inside an epsilon-insensitive tube with margin penalties."}
            </>
          )}
        </p>
      </div>
    </div>
  </section>
);

export default ModelSelection;
