# DMML — Data Mining & Machine Learning

Lecture animations for the Master's-level DMML course, built with [Manim Community](https://www.manim.community/).

---

## Week 02 — Time Series & ARIMA

### Time Series Decomposition
![Decomposition](weeks/w02_time_series_arima/Decomposition.gif)

### Stationarity & Differencing
![Stationarity](weeks/w02_time_series_arima/Stationarity.gif)

### AR(1) Process
![AR1](weeks/w02_time_series_arima/AR1.gif)

### Forecast Cone
![ForecastCone](weeks/w02_time_series_arima/ForecastCone.gif)

---

## Week 03 — Regression & Classification

### Bias–Variance Tradeoff
![BiasVariance](weeks/w03_regression_classification/BiasVariance.gif)

### Decision Boundary (Logistic Regression vs kNN)
![DecisionBoundary](weeks/w03_regression_classification/DecisionBoundary.gif)

---

## Week 04 — SVMs & Model Tuning

### SVM Max-Margin
![SVMMargin](weeks/w04_svm_tuning/SVMMargin.gif)

### Kernel Trick
![KernelTrick](weeks/w04_svm_tuning/KernelTrick.gif)

### Cross-Validation
![CrossValidation](weeks/w04_svm_tuning/CrossValidation.gif)

---

## Week 05 — Trees & Ensembles

### Decision Tree (Splits & Depth)
![DecisionTree](weeks/w05_trees_ensembles/DecisionTree.gif)

### Bagging & Bootstrap
![Bagging](weeks/w05_trees_ensembles/Bagging.gif)

### Feature Importance (Permutation)
![FeatureImportance](weeks/w05_trees_ensembles/FeatureImportance.gif)

---

## Week 06 — Boosting

### Gradient Boosting
![GradientBoosting](weeks/w06_boosting_shap/GradientBoosting.gif)

---

## Running the animations yourself

```bash
# clone and create the environment
git clone <repo-url>
cd dmml
python -m venv env && source env/bin/activate
pip install manim

# render any scene (example)
cd weeks/w02_time_series_arima
../../env/bin/manim -pql decomposition.py Decomposition
```

> **Note:** No LaTeX required.
