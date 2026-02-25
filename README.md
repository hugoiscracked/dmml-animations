# DMML — Data Mining & Machine Learning

Lecture animations for the Master's-level DMML course, built with [Manim Community](https://www.manim.community/).

---

## Week 02 — Time Series & ARIMA

### Time Series Decomposition
<video src="weeks/w02_time_series_arima/Decomposition.mp4" controls width="100%"></video>

### Stationarity & Differencing
<video src="weeks/w02_time_series_arima/Stationarity.mp4" controls width="100%"></video>

### AR(1) Process
<video src="weeks/w02_time_series_arima/AR1.mp4" controls width="100%"></video>

### Forecast Cone
<video src="weeks/w02_time_series_arima/ForecastCone.mp4" controls width="100%"></video>

---

## Week 03 — Regression & Classification

### Bias–Variance Tradeoff
<video src="weeks/w03_regression_classification/BiasVariance.mp4" controls width="100%"></video>

### Decision Boundary (Logistic Regression vs kNN)
<video src="weeks/w03_regression_classification/DecisionBoundary.mp4" controls width="100%"></video>

---

## Week 04 — SVMs & Model Tuning

### SVM Max-Margin
<video src="weeks/w04_svm_tuning/SVMMargin.mp4" controls width="100%"></video>

### Kernel Trick
<video src="weeks/w04_svm_tuning/KernelTrick.mp4" controls width="100%"></video>

### Cross-Validation
<video src="weeks/w04_svm_tuning/CrossValidation.mp4" controls width="100%"></video>

---

## Week 05 — Trees & Ensembles

### Decision Tree (Splits & Depth)
<video src="weeks/w05_trees_ensembles/DecisionTree.mp4" controls width="100%"></video>

### Bagging & Bootstrap
<video src="weeks/w05_trees_ensembles/Bagging.mp4" controls width="100%"></video>

### Feature Importance (Permutation)
<video src="weeks/w05_trees_ensembles/FeatureImportance.mp4" controls width="100%"></video>

---

## Week 06 — Boosting

### Gradient Boosting
<video src="weeks/w06_boosting_shap/GradientBoosting.mp4" controls width="100%"></video>

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
