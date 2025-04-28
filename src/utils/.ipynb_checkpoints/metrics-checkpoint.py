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

# Logging config
logging.basicConfig(level=logging.INFO)

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
    
    # Prepare the test data (X_train, y_train) to evaluate 'model'
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']

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
        print_header(f"Starting Evaluation for: [{model_name}]", color = 'bright_green')
        # Print the best parameters if using GridSearch
        print_header(f"Best Parameters Found with GridSearch: {model.best_params_}", color = 'bright_magenta')
        metrics_str = "🔍 Model Evaluation Metrics 🔍:\n"
        for k, v in metrics.items():
            metrics_str += f"{k.capitalize()}: {v:.4f}\n"
        print(metrics_str)
        
    # Optionally plot confusion matrix 
    if plot:
        plot_confusion_matrix(y_test, preds)
    return metrics

def compute_metrics(y_test, preds, probs, metrics_to_compute):
    """
    Computes the specified metrics for the model evaluation.
    
    Args:
        y_test: The ground truth labels.
        preds: The model predictions.
        probs: The predicted probabilities.
        metrics_to_compute: List of metrics to compute.
        
    Returns:
        dict: Calculated metrics.
    """
    metrics = {}

    if 'Accuracy' in metrics_to_compute:
        metrics['Accuracy'] = accuracy_score(y_test, preds)

    if 'Precision' in metrics_to_compute:
        metrics['Precision'] = precision_score(y_test, preds, zero_division=1)

    if 'Recall' in metrics_to_compute:
        metrics['Recall'] = recall_score(y_test, preds, zero_division=1)

    if 'F1 Score' in metrics_to_compute:
        metrics['F1 Score'] = f1_score(y_test, preds, zero_division=1)
        
    return metrics
    
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
    print_header('Model Performance Metrics Computed', color = 'bright_green')
    return pd.DataFrame(results).set_index('Model')

def best_model_per_metric(metrics_df):
    """
    Finds the best model for each metric in the evaluation DataFrame.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing models' evaluation metrics.
    
    Returns:
        dict: A dictionary where each key is a metric and the value is the model name
              with the best score for that metric.
    """
    best_models = {}
    
    # Iterate over each metric (column) in the DataFrame
    for metric in metrics_df.columns:
        best_model = metrics_df[metric].idxmax()  # Find the model with the highest score
        best_models[metric] = best_model
    
    return best_models

def plot_confusion_matrix(y_test, preds):
    """
    Plots the confusion matrix.
    
    Args:
        y_test: The ground truth labels.
        preds: The model predictions.
    """
    print_header('Confusion Matrix', color = 'bright_cyan')
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
    
