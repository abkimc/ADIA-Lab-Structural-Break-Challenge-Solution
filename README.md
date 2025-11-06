# ADIA-Lab-Structural-Break-Challenge-Solution

Methods & components (detailed)
Data loading

Uses the Crunch-provided crunch.load_data() helper to obtain lists of dataframes for training and testing.

Assumes each sample is a short univariate time series (same format expected by the competition).

Feature engineering

A comprehensive set of features is extracted per time-series segment. Main families include:

Time-domain statistics

Mean, median, standard deviation, variance

Skewness, kurtosis

Min / max / range / percentiles

First and second differences (derivatives) statistics

Counts of sign-changes, zero-crossings

Rolling / local features

Rolling statistics (windowed means/std/var)

Local maxima/minima counts and densities

Spectral features

Fourier transform (FFT) magnitude peaks / energy in frequency bands

Power spectral density summaries

Wavelet features

Wavelet decomposition coefficients/statistics (via PyWavelets)

Distributional / divergence measures

Jensen–Shannon distance, Kolmogorov–Smirnov style comparisons between segments (where applicable)

Simulated DGP features

Optional features created by estimating / fitting simple data-generating-process characteristics (added as “DGP” features in the code when include_dgp is enabled).

Note: the notebook bundles these feature extractors into a features function that outputs a pandas.DataFrame of features for each training sample.

Feature selection / dimensionality reduction

Multiple selection layers are used to reduce redundancy and remove non-informative features:

VarianceThreshold — remove near-constant features.

SelectKBest (univariate scoring — mutual information or other) — pick top features by score.

RFE (Recursive Feature Elimination) — wrapper-based selection using an estimator.

SelectFromModel / model-based selection (e.g., based on tree-model importances).

The pipeline builds these steps so the same selection is applied at training and inference.

Models / ensembling

The notebook shows training and evaluation of the following:

XGBoost classifier (primary gradient-boosted tree model)

LightGBM and RandomForest (used in experiments and possible stacking)

StackingClassifier (meta model combining base learners — LogisticRegression used as meta learner in stacking)

Model probabilities are used as final outputs (not raw class predictions) so scoring can be done with AUC-ROC.

Training procedure

The train() function:

Extracts features for each training sample.

Applies the feature-selection pipeline.

Computes class weights (via compute_class_weight) to handle class imbalance.

Fits the model pipeline (feature selector + model).

Computes AUC-ROC on a holdout or cross-validated split (not strictly specified — notebook computes AUC on an available test/validation sample).

Saves the fitted pipeline as model.joblib and writes basic training metrics to train_metrics.csv.

Note: explicit cross-validation or hyperparameter search (GridSearchCV / RandomizedSearchCV) is not shown in the notebook; default model hyperparameters appear to be used unless user-specified elsewhere. If you want automatic hyperparameter tuning, adding GridSearchCV or Optuna is recommended.

Inference

The infer() function:

Accepts test set iterator (Iterable[pd.DataFrame]) and the path to the saved model directory.

Reconstructs the same features and selection pipeline used in training.

Returns predicted probabilities (1D pd.Series) for each sample.

Writes data/prediction.parquet (the Crunch expected output).

Evaluation

Primary metric used: AUC-ROC (sklearn.metrics.roc_auc_score).

Additional diagnostics include feature importances, saved as a DataFrame for inspection.


flowchart TD
  A[Raw time-series samples] --> B[Feature engineering]
  B --> B1[Time-domain stats]
  B --> B2[Spectral (FFT, PSD)]
  B --> B3[Wavelet coefficients]
  B --> B4[DGP / synthetic features]
  B --> C[Feature cleaning & scaling]
  C --> D[Feature selection]
  D --> E[Model training]
  E --> E1[XGBoost / LightGBM / RandomForest]
  E1 --> F[Optional stacking (meta-learner)]
  F --> G[Saved pipeline/model.joblib]
  G --> H[Inference: infer() -> probabilities]
  H --> I[data/prediction.parquet]
  I --> J[Evaluation (AUC-ROC)]
