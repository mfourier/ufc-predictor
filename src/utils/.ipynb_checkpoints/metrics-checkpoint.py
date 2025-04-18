# /utils/metrics.py

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

def evaluate_model(model, X_test, y_test, verbose=True, plot=False):
    """
    Evaluates a classification model and optionally prints and plots results.
    Returns a dictionary with all key metrics.
    """
    preds, probs = get_predictions(model, X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1_score': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probs)
    }

    if verbose:
        print("🔍 Model Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

    if plot:
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()

        RocCurveDisplay.from_predictions(y_test, probs)
        plt.title("ROC Curve")
        plt.show()

    return metrics


