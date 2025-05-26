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
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from .helpers import *

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Brier Score']

def evaluate_model(model, data_test, verbose=True, plot=True, metrics_to_compute=None):
    if 'label' not in data_test.columns:
        raise ValueError("The test set must include a 'label' column.")

    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']
    model_name = get_pretty_model_name(model)

    metrics_to_compute = metrics_to_compute or DEFAULT_METRICS

    try:
        preds, probs = get_predictions(model, X_test)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

    results = compute_metrics(y_test, preds, probs, metrics_to_compute)

    if verbose:
        print_header(f"Evaluation for: [{model_name}]", color='bright_green')
        if hasattr(model, "best_params_"):
            print_header(f"Best Parameters: {model.best_params_}", color='bright_magenta')
        for k, v in results.items():
            print(f"{k:>12}: {v:.4f}")

    if plot:
        plot_confusion_matrix(y_test, preds)

    return results

def compute_metrics(y_true, y_pred, y_proba, metrics_to_compute):
    results = {}

    for metric in metrics_to_compute:
        if metric == 'Accuracy':
            results['Accuracy'] = accuracy_score(y_true, y_pred)
        elif metric == 'Precision':
            results['Precision'] = precision_score(y_true, y_pred, zero_division=1)
        elif metric == 'Recall':
            results['Recall'] = recall_score(y_true, y_pred, zero_division=1)
        elif metric == 'F1 Score':
            results['F1 Score'] = f1_score(y_true, y_pred, zero_division=1)
        elif metric == 'ROC AUC':
            if y_proba is not None:
                results['ROC AUC'] = roc_auc_score(y_true, y_proba)
        elif metric == 'Brier Score':
            if y_proba is not None:
                results['Brier Score'] = brier_score_loss(y_true, y_proba)
        else:
            logger.warning(f"Unsupported metric requested: {metric}")

    results = {k: round(v, 4) for k, v in results.items()}
    return results

def compare_metrics(models_dict, data, metrics_to_compute=None):
    logger.info("Starting comparison of models...")
    results = []

    for name, model in models_dict.items():
        logger.info(f"Evaluating: {name}")
        metrics = evaluate_model(model, data, verbose=False, plot=False, metrics_to_compute=metrics_to_compute)
        metrics['Model'] = name
        results.append(metrics)

    df = pd.DataFrame(results).set_index('Model')
    print_header("Comparison Completed", color='bright_green')
    return df

def best_model_per_metric(metrics_df):
    best = []
    for metric in metrics_df.columns:
        best_model = metrics_df[metric].idxmax()
        best_value = metrics_df[metric].max()
        best.append({"Metric": metric, "Best Model": best_model, "Value": best_value})
        logger.info(f"Best model for {metric}: {best_model} ({best_value:.4f})")

    return pd.DataFrame(best)

def plot_confusion_matrix(y_true, y_pred):
    print_header("Confusion Matrix", color='bright_cyan')
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
