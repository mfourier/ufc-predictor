import logging
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

from .helpers import get_predictions, print_header

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Brier Score']


def evaluate_metrics(
        model: object,
        data_test: pd.DataFrame,
        verbose: bool = False,
        metrics_to_compute: Optional[Sequence[str]] = None
    ) -> dict[str, float]:
    """
    Evaluate a trained model using various metrics and optionally plot the confusion matrix.

    Args:
        model (UFCModel): A trained UFCModel.
        data_test (pd.DataFrame): Test dataset with a 'label' column.
        verbose (bool): Whether to print evaluation results.
        plot (bool): Whether to plot the confusion matrix.
        metrics_to_compute (list, optional): List of metrics to compute.

    Returns:
        dict: Dictionary with computed metric names and values.

    Raises:
        ValueError: If 'label' column is missing in the test data.
    """
    if 'label' not in data_test.columns:
        raise ValueError("The test set must include a 'label' column.")

    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']

    metrics_to_compute = metrics_to_compute or DEFAULT_METRICS

    try:
        preds, probs = get_predictions(model, X_test)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

    metrics = compute_metrics(y_test, preds, probs, metrics_to_compute)
    
    if verbose:
        print_header(f"Evaluation for: [{model.name}]", color='bright_green')
        if hasattr(model, "best_params_"):
            print_header(f"Best Parameters: {model.best_params_}", color='bright_magenta')
        for k, v in results.items():
            print(f"{k:>12}: {v:.4f}")

    return metrics

def evaluate_cm(
        model: object,
        data_test: pd.DataFrame,
    ) -> dict[str, float]:
    """
    Compute the confusion matrix for a given trained model and test dataset.

    Args:
        model (object): A trained classifier with a `.predict()` method.
        data_test (pd.DataFrame): Test dataset containing feature columns and a 'label' column.

    Returns:
        np.ndarray: Confusion matrix as a 2D array.

    Raises:
        ValueError: If 'label' column is missing in the test data.
    """
    if 'label' not in data_test.columns:
        raise ValueError("The test set must include a 'label' column.")

    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return cm
    
def compute_metrics(
        y_test: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_proba: Optional[Union[np.ndarray, list]],
        metrics_to_compute: Sequence[str]
) -> dict[str, float]:
    """
    Compute performance metrics for classification tasks.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like, optional): Probabilities or decision scores.
        metrics_to_compute (list): List of metrics to calculate.

    Returns:
        dict: Dictionary of computed metrics.
    """
    results: dict[str, float] = {}

    for metric in metrics_to_compute:
        if metric == 'Accuracy':
            results['Accuracy'] = accuracy_score(y_test, y_pred)
        elif metric == 'Precision':
            results['Precision'] = precision_score(y_test, y_pred, zero_division=1)
        elif metric == 'Recall':
            results['Recall'] = recall_score(y_test, y_pred, zero_division=1)
        elif metric == 'F1 Score':
            results['F1 Score'] = f1_score(y_test, y_pred, zero_division=1)
        elif metric == 'ROC AUC' and y_proba is not None:
            results['ROC AUC'] = roc_auc_score(y_test, y_proba)
        elif metric == 'Brier Score' and y_proba is not None:
            results['Brier Score'] = brier_score_loss(y_test, y_proba)
        else:
            logger.warning(f"Unsupported or unavailable metric: {metric}")

    return {k: round(v, 4) for k, v in results.items()}


def compare_metrics(
        models_list: list[object],
    ) -> pd.DataFrame:
    """
    Compare multiple models using specified metrics.

    Args:
        models_dict (dict): Dictionary of trained models keyed by name.
        metrics_to_compute (list, optional): List of metrics to compute.

    Returns:
        pd.DataFrame: DataFrame with models as rows and metrics as columns.
    """
    logger.info("Starting comparison of models...")
    results = []

    for model in models_list:
        logger.info(f"Evaluating: {model.name}")
        metrics = model.metrics
        metrics['Model'] = model.name
        results.append(metrics)

    df = pd.DataFrame(results).set_index('Model')
    print_header("Comparison Completed", color='bright_green')
    return df

def best_model_per_metric(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the best-performing model for each metric.

    Args:
        metrics_df (pd.DataFrame): DataFrame with models and metrics.

    Returns:
        pd.DataFrame: DataFrame listing the best model per metric.
    """
    best = []
    for metric in metrics_df.columns:
        best_model = metrics_df[metric].idxmax()
        best_value = metrics_df[metric].max()
        best.append({"Metric": metric, "Best Model": best_model, "Value": best_value})
        logger.info(f"Best model for {metric}: {best_model} ({best_value:.4f})")

    return pd.DataFrame(best)
