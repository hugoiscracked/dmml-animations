# DMML — Data Mining & Machine Learning

Lecture animations for the Master's-level DMML course, built with [Manim Community](https://www.manim.community/).

---

## Week 02 — Time Series & ARIMA

### Time Series Decomposition
![Decomposition](animations/w02_time_series_arima/Decomposition.gif)

### Stationarity & Differencing
![Stationarity](animations/w02_time_series_arima/Stationarity.gif)

### AR(1) Process
![AR1](animations/w02_time_series_arima/AR1.gif)

### Forecast Cone
![ForecastCone](animations/w02_time_series_arima/ForecastCone.gif)

---

## Week 03 — Regression & Classification

### Bias–Variance Tradeoff
![BiasVariance](animations/w03_regression_classification/BiasVariance.gif)

### Decision Boundary (Logistic Regression vs kNN)
![DecisionBoundary](animations/w03_regression_classification/DecisionBoundary.gif)

---

## Week 04 — SVMs & Model Tuning

### SVM Max-Margin
![SVMMargin](animations/w04_svm_tuning/SVMMargin.gif)

### Kernel Trick
![KernelTrick](animations/w04_svm_tuning/KernelTrick.gif)

### Cross-Validation
![CrossValidation](animations/w04_svm_tuning/CrossValidation.gif)

---

## Week 05 — Trees & Ensembles

### Decision Tree (Splits & Depth)
![DecisionTree](animations/w05_trees_ensembles/DecisionTree.gif)

### Bagging & Bootstrap
![Bagging](animations/w05_trees_ensembles/Bagging.gif)

### Feature Importance (Permutation)
![FeatureImportance](animations/w05_trees_ensembles/FeatureImportance.gif)

---

## Week 06 — Boosting

### Gradient Boosting
![GradientBoosting](animations/w06_boosting_shap/GradientBoosting.gif)

### SHAP Values
![SHAPValues](animations/w06_boosting_shap/SHAPValues.gif)

---

## Week 07 — Clustering & Dimensionality Reduction

### K-Means (Lloyd's Algorithm + Elbow Method)
![KMeans](animations/w07_clustering_pca/KMeans.gif)

### PCA (Principal Components + Projection + Explained Variance)
![PCA](animations/w07_clustering_pca/PCA.gif)

### DBSCAN (ε-ball Expansion + Core/Border/Noise)
![DBSCAN](animations/w07_clustering_pca/DBSCAN.gif)

---

## Week 08 — Neural Networks Intro

### MLP Forward Pass
![MLPForward](animations/w08_neural_networks_intro/MLPForward.gif)

### Gradient Descent
![GradientDescent](animations/w08_neural_networks_intro/GradientDescent.gif)

### Activation Functions & Vanishing Gradient
![ActivationFunctions](animations/w08_neural_networks_intro/ActivationFunctions.gif)

---

## Week 09 — Training Techniques & CNNs

### Convolution (Sliding Filter + Feature Maps + Max Pooling)
![Convolution](animations/w09_training_cnns/Convolution.gif)

### Dropout & Batch Normalization
![DropoutBatchNorm](animations/w09_training_cnns/DropoutBatchNorm.gif)

### CNN Architecture (LeNet Pipeline + Hierarchical Features + Transfer Learning)
![CNNArchitecture](animations/w09_training_cnns/CNNArchitecture.gif)

---

## Week 10 — Sequences & Transformers

### RNN vs LSTM (Unrolled RNN + Vanishing Gradient + LSTM Gates)
![RNNvsLSTM](animations/w10_sequences_transformers/RNNvsLSTM.gif)

### Attention Mechanism (Seq2seq Bottleneck + Attention Weights + Self-Attention Matrix)
![Attention](animations/w10_sequences_transformers/Attention.gif)

### Transformer Architecture (Positional Encoding + Encoder Layer + Classification Head)
![Transformer](animations/w10_sequences_transformers/Transformer.gif)

### Practical Fine-Tuning (Pre-train Cost + Tokenization + Strategies + LR Sensitivity)
![FineTuning](animations/w10_sequences_transformers/FineTuning.gif)

---

## Running the animations yourself

```bash
# clone and create the environment
git clone <repo-url>
cd dmml
python -m venv env && source env/bin/activate
pip install manim

# render any scene (example)
cd animations/w02_time_series_arima
../../env/bin/manim -pql decomposition.py Decomposition
```

> **Note:** No LaTeX required.
