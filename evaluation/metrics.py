import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def evaluate_model(model, x_test, y_test):
    """
    Evaluate trained 3D CNN model.

    Parameters:
        model: trained keras model
        x_test: test volumes (N, H, W, D, 1)
        y_test: one-hot encoded labels (N, num_classes)

    Returns:
        dict of metrics
    """

    # -----------------------------
    # Predict probabilities
    # -----------------------------
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = np.argmax(y_test, axis=1)

    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # -----------------------------
    # ROC-AUC (multiclass safe)
    # -----------------------------
    try:
        auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr")
    except Exception:
        auc = None

    # -----------------------------
    # Print results
    # -----------------------------
    print("\n===== EVALUATION RESULTS =====")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    print(f"ROC-AUC: {auc}\n")

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": auc
    }
