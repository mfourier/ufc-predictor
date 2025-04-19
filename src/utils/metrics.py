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
from .helpers import *

def evaluate_model(model, data_test, verbose=True, plot=True, metrics_to_compute=None):
    """
    Evaluates a classification model and optionally prints and plots results.
    Returns a dictionary with all key metrics.
    
    Args:
        model: The trained model to evaluate.
        data_test: A DataFrame containing the input features and labels for testing.
        verbose: Whether to print the metrics (default True).
        plot: Whether to plot confusion matrix and ROC curve (default True).
        metrics_to_compute: List of specific metrics to compute, if None computes all.
        
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Prepare the test data
    X_test, y_test = prepare_data(data_test)

    # Default to compute all metrics if none are specified
    if metrics_to_compute is None:
        metrics_to_compute = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    # Get model predictions and probabilities
    preds, probs = get_predictions(model, X_test)

    # Compute requested metrics
    metrics = compute_metrics(y_test, preds, probs, metrics_to_compute)

    # Optionally print the metrics
    if verbose:
        print_metrics(metrics)

    # Optionally plot confusion matrix and ROC curve
    if plot:
        plot_confusion_matrix(y_test, preds)
        plot_roc_curve(y_test, probs)

    return metrics



