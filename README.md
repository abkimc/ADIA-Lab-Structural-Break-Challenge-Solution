# ADIA-Lab Structural Break Challenge — Solution

<img width="1320" height="683" alt="image" src="https://github.com/user-attachments/assets/a6b4ae25-f217-4bae-8055-b4810f50f03e" />

## Overview

This repository contains a complete solution for the **ADIA-Lab Structural Break Challenge**, which focuses on detecting structural breakpoints in short univariate time series.  
The notebook implements a reproducible end-to-end pipeline — from feature extraction to model training, inference, and evaluation.

Despite spending most of the competition period on reserve duty, I still managed to develop and refine a complete end-to-end pipeline, combining feature engineering, model ensembling, and robust validation strategies.

Out of 490 competitors, my solution ranked 52nd, placing in the top 11% of participants.

<img width="740" height="233" alt="image" src="https://github.com/user-attachments/assets/08a00a36-0b89-4593-8864-836a7c7c18ea" />

---
## Methods & Components (Detailed)

<img width="1412" height="824" alt="image" src="https://github.com/user-attachments/assets/9baaf376-2f54-450c-a554-c1af19350fb6" />


### 1. Data Loading
- Data is loaded using the `crunch.load_data()` helper provided by the competition.
- Each sample is a short **univariate time series** with consistent formatting for training and testing.

---

### 2. Feature Engineering

A rich set of features is extracted from each time-series segment.  
Main feature families include:

#### Time-Domain Statistics
- Mean, median, standard deviation, variance  
- Skewness, kurtosis  
- Minimum, maximum, range, percentiles  
- First and second differences (derivatives)  
- Counts of sign changes and zero crossings  

#### Rolling / Local Features
- Rolling mean, standard deviation, variance  
- Local maxima and minima counts, density statistics  

#### Spectral Features
- Fourier Transform (FFT) magnitude peaks  
- Energy in specific frequency bands  
- Power Spectral Density (PSD) summaries  

#### Wavelet Features
- Wavelet decomposition coefficients and statistics (via **PyWavelets**)  

#### Distributional / Divergence Measures
- Jensen–Shannon distance  
- Kolmogorov–Smirnov style comparisons  

#### Simulated DGP Features
- Optional data-generating-process features (enabled with `include_dgp=True`)  
- Capture synthetic dynamics fitted to each segment  

> All feature extractors are integrated into a unified function returning a `pandas.DataFrame` per sample.

---

### 3. Feature Selection & Dimensionality Reduction

A layered selection pipeline is applied to remove redundant and uninformative features:

- **VarianceThreshold** — removes near-constant features  
- **SelectKBest** — keeps top features by univariate score (mutual information or similar)  
- **RFE (Recursive Feature Elimination)** — iterative wrapper-based feature pruning  
- **SelectFromModel** — model-based feature importance filtering (e.g., using tree models)

These transformations are stored in a pipeline to ensure consistency between training and inference.

---

### 4. Modeling & Ensembling

Multiple models were trained and compared:

- **XGBoost Classifier** (main model)  
- **LightGBM** and **RandomForest** (alternative/ensemble models)  
- **StackingClassifier** with a **Logistic Regression meta-learner**

Predictions use **class probabilities** rather than hard labels to enable AUC-based evaluation.

---

### 5. Training Procedure

The `train()` function performs the full training workflow:

1. Extracts features from all training samples  
2. Applies the same feature-selection pipeline as used at inference  
3. Computes **class weights** (via `compute_class_weight`) to handle class imbalance  
4. Fits the model pipeline  
5. Computes **AUC-ROC** on a holdout or validation set  
6. Saves:
   - Trained model → `model.joblib`
   - Training metrics → `train_metrics.csv`

> No explicit hyperparameter search is included — default parameters are used unless manually set.  
> For optimization, you can extend this with `GridSearchCV`, `RandomizedSearchCV`, or **Optuna**.

---

### 6. Inference

The `infer()` function handles prediction on unseen data:

1. Accepts an iterable of test samples (`Iterable[pd.DataFrame]`)  
2. Reconstructs the feature extraction and selection pipeline  
3. Outputs **predicted probabilities** as a `pandas.Series`  
4. Saves results in the required Crunch format:  
   `data/prediction.parquet`

---

### 7. Evaluation

- **Primary metric:** AUC-ROC (`sklearn.metrics.roc_auc_score`)  
- **Additional outputs:** Feature importances and training diagnostics, stored in DataFrames for analysis



## If I had more time i would try to: 

- **Hyperparameter tuning** — integrate `Optuna` or `GridSearchCV`  
- **Cross-validation** — apply stratified or time-based CV  
- **Feature importance** — add SHAP/Permutation Importance analysis  
- **Performance optimization** — vectorize heavy operations  


---
