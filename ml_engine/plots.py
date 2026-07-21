import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def generate_model_comparison(artifact_dir, results, **kwargs):
    _ensure_dir(artifact_dir)

    models = [r["model"] for r in results]
    scores = [r["accuracy"] for r in results]

    plt.figure(figsize=(8,5))
    plt.bar(models, scores)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.tight_layout()

    output = os.path.join(artifact_dir, "model_comparison.png")
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def generate_confusion_matrix(artifact_dir, model, X_test, y_test, **kwargs):
    _ensure_dir(artifact_dir)

    plt.figure(figsize=(6,6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title("Confusion Matrix")

    output = os.path.join(artifact_dir, "confusion_matrix.png")
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def generate_roc_curve(artifact_dir, model, X_test, y_test, **kwargs):
    _ensure_dir(artifact_dir)

    plt.figure(figsize=(6,6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")

    output = os.path.join(artifact_dir, "roc_curve.png")
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def generate_precision_recall_curve(artifact_dir, model, X_test, y_test, **kwargs):
    _ensure_dir(artifact_dir)

    plt.figure(figsize=(6,6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision Recall Curve")

    output = os.path.join(artifact_dir, "precision_recall_curve.png")
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def generate_feature_importance(
    artifact_dir,
    model,
    model_name,
    X_test,
    options=None,
    **kwargs
):
    _ensure_dir(artifact_dir)

    options = options or {}
    top_n = options.get("top_n", 15)

    estimator = model.named_steps["classifier"]

    if not hasattr(estimator, "feature_importances_"):
        return None

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = estimator.feature_importances_

    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10,6))
    plt.barh(
        np.array(feature_names)[idx][::-1],
        importances[idx][::-1]
    )
    plt.title(f"{model_name} Feature Importance")
    plt.tight_layout()

    output = os.path.join(artifact_dir, "feature_importance.png")
    plt.savefig(output, dpi=200)
    plt.close()
    return output


PLOT_REGISTRY = {
    "model_comparison": generate_model_comparison,
    "confusion_matrix": generate_confusion_matrix,
    "roc_curve": generate_roc_curve,
    "precision_recall": generate_precision_recall_curve,
    "feature_importance": generate_feature_importance,
}
