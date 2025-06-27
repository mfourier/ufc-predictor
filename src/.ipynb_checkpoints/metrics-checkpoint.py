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
from .ufc_data import UFCData
from models.ufc_model import UFCModel

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Brier Score']


def evaluate_metrics(
        model: object,
        UFCData: UFCData,
        verbose: bool = False,
        metrics_to_compute: Optional[Sequence[str]] = None
    ) -> dict[str, float]:
    """
    Evaluate a trained UFCModel on test data stored in a UFCData object.

    Args:
        model (UFCModel): A trained model wrapper.
        ufc_data (UFCData): Dataset handler with standardized test data.
        verbose (bool): Whether to print detailed results.
        metrics_to_compute (list, optional): Metrics to evaluate.

    Returns:
        dict[str, float]: Computed metric results.
    """
    X_test, y_test = UFCData.get_processed_test()
    metrics_to_compute = metrics_to_compute or DEFAULT_METRICS

    try:
        preds, probs = get_predictions(model, X_test)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

    results = compute_metrics(y_test, preds, probs, metrics_to_compute)

    if verbose:
        print_header(f"Evaluation for: [{model.name}]", color='bright_green')
        if model.best_params_:
            print_header(f"Best Parameters: {model.best_params_}", color='bright_magenta')
        for k, v in results.items():
            print(f"{k:>12}: {v:.4f}")

    return results


def evaluate_cm(
        model: UFCModel,
        ufc_data: UFCData,
    ) -> np.ndarray:
    """
    Compute and store the confusion matrix for a UFCModel using UFCData.

    Args:
        model (UFCModel): A trained model.
        ufc_data (UFCData): Dataset handler with standardized test data.

    Returns:
        np.ndarray: Confusion matrix.
    """
    X_test, y_test = ufc_data.get_processed_test()
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
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like, optional): Probabilities or decision scores.
        metrics_to_compute (list): List of metrics to calculate.

    Returns:
        dict[str, float]: Computed metric results.
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
        models_list: list[UFCModel],
    ) -> pd.DataFrame:
    """
    Compare multiple UFCModel objects using stored metrics.

    Args:
        models_list (list): List of trained UFCModel instances.

    Returns:
        pd.DataFrame: Table comparing model performance.
    """
    logger.info("Starting comparison of models...")
    results = []

    for model in models_list:
        logger.info(f"Evaluating: {model.name}")
        if model.metrics is None:
            logger.warning(f"Model {model.name} has no stored metrics.")
            continue
        row = model.metrics.copy()
        row['Model'] = model.name
        results.append(row)

    df = pd.DataFrame(results).set_index('Model')
    print_header("Comparison Completed", color='bright_green')
    return df


def best_model_per_metric(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the best-performing model per metric.

    Args:
        metrics_df (pd.DataFrame): DataFrame with models and metrics.

    Returns:
        pd.DataFrame: Best model and score for each metric.
    """
    best = []
    for metric in metrics_df.columns:
        best_model = metrics_df[metric].idxmax()
        best_value = metrics_df[metric].max()
        best.append({"Metric": metric, "Best Model": best_model, "Value": round(best_value, 4)})
        logger.info(f"Best model for {metric}: {best_model} ({best_value:.4f})")

    return pd.DataFrame(best)
