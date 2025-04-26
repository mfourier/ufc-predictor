import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
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
    
    model_name = get_pretty_model_name(model)
    print_box(f"📊 Starting Evaluation for: {model_name} ✅")
    
    # Prepare the test data (X_train, y_train) to evaluate 'model'
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']

    default_metrics_to_compute = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Default to compute all metrics if none are specified
    if metrics_to_compute is None:
        metrics_to_compute = default_metrics_to_compute

    # Get model predictions and probabilities
    preds, probs = get_predictions(model, X_test)

    # Compute requested metrics
    metrics = compute_metrics(y_test, preds, probs, metrics_to_compute)

    # Optionally print the metrics
    if verbose:
        # Print the best parameters if using GridSearch
        print_box(f"🚀 Best Parameters Found with GridSearch: {model.best_params_}")
        print_metrics(metrics)

    # Optionally plot confusion matrix 
    if plot:
        plot_confusion_matrix(y_test, preds)
    return metrics

def compare_parameters(models_dict, data):
    params = []
    
    for name, model in models_dict.items():
        print_box(f"🚀 Best Parameters Found with GridSearch for {name}: {model.best_params_}")
        params.append(model.best_params_)
    return params
    
def compare_metrics(models_dict, data, metrics_to_compute=None):
    """
    Compares multiple models on the same dataset and returns a DataFrame with metrics.

    Args:
        models_dict (dict): Dictionary with model_name as key and trained model as value.
        data (DataFrame): DataFrame with features and 'label' column.
        metrics_to_compute (list): Optional list of metrics.

    Returns:
        pd.DataFrame: DataFrame with one row per model and evaluation metrics.
    """
    results = []
    
    for name, model in models_dict.items():
        metrics = evaluate_model(model, data, verbose=False, plot=False, metrics_to_compute=metrics_to_compute)
        metrics['Model'] = name
        results.append(metrics)

    return pd.DataFrame(results).set_index('Model')

