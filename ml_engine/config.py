from pathlib import Path

# Base directories

BASE_DIR = Path(__file__).resolve().parent

SHARED_DIR = Path("/shared")

ARTIFACT_DIR = SHARED_DIR / "artifacts"

MODEL_DIR = SHARED_DIR / "saved_models"

REPORT_DIR = SHARED_DIR / "reports"

LOG_DIR = SHARED_DIR / "logs"

# Dataset

DEFAULT_TEST_SIZE = 0.2

DEFAULT_RANDOM_STATE = 42

# Classification metrics

CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

# Regression metrics

REGRESSION_METRICS = [
    "mae",
    "mse",
    "rmse",
    "r2"
]

# Forecasting metrics

FORECASTING_METRICS = [
    "mae",
    "rmse",
    "mape"
]

# Supported problem types

SUPPORTED_PROBLEM_TYPES = [
    "classification",
    "regression",
    "forecasting",
    "clustering"
]

# Supported file extensions

SUPPORTED_DATASETS = [
    ".csv",
    ".xlsx"
]

# Figure settings

FIGURE_DPI = 200

DEFAULT_FIGSIZE = (8, 5)

LARGE_FIGSIZE = (10, 6)

SQUARE_FIGSIZE = (6, 6)

# Report settings

REPORT_NAME = "AutoML_Report.html"

METRICS_JSON = "metrics.json"

SUMMARY_JSON = "summary.json"

# Saved model

MODEL_FILENAME = "best_model.pkl"

# Artifact filenames

MODEL_COMPARISON_PLOT = "model_comparison.png"

CONFUSION_MATRIX_PLOT = "confusion_matrix.png"

ROC_CURVE_PLOT = "roc_curve.png"

PRECISION_RECALL_PLOT = "precision_recall_curve.png"

FEATURE_IMPORTANCE_PLOT = "feature_importance.png"

ACTUAL_VS_PREDICTED_PLOT = "actual_vs_predicted.png"

RESIDUAL_PLOT = "residual_plot.png"

ERROR_DISTRIBUTION_PLOT = "error_distribution.png"

CORRELATION_HEATMAP = "correlation_heatmap.png"

MISSING_VALUE_HEATMAP = "missing_values.png"

FORECAST_PLOT = "forecast.png"

TREND_PLOT = "trend.png"

SEASONALITY_PLOT = "seasonality.png"

ELBOW_CURVE = "elbow_curve.png"

SILHOUETTE_PLOT = "silhouette.png"

PCA_CLUSTER_PLOT = "cluster_pca.png"