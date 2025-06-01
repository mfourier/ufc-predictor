import pandas as pd
import numpy as np
from typing import Optional, Sequence
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from .helpers import get_pretty_model_name, get_predictions, print_header
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

logger = logging.getLogger(__name__)

class UFCModel:
    """
    Encapsulates a trained UFC fight prediction model, allowing prediction, evaluation, and reporting.

    Attributes:
        model (BaseEstimator): Trained scikit-learn model or GridSearchCV.
        name (str): Display-friendly model name.
        best_params_ (dict | None): GridSearchCV best parameters, if available.
        score (float | None): Best cross-validation score, if available.
        metrics (dict[str, float] | None): Last computed evaluation metrics.
        cm (np.ndarray | None): Stored confusion matrix.
    """

    def __init__(
        self,
        model: BaseEstimator
        ):
        """
        Initialize the UFCModel with a trained scikit-learn model.
        
        Args:
            model (BaseEstimator): A trained classifier or GridSearchCV object.
        """
        
        self.model = model
        self.name = get_pretty_model_name(model)
        self.best_params_ = getattr(model, "best_params_", None)
        self.score = getattr(model, "best_score_", None)
        self.metrics: Optional[dict[str, float]] = None
        self.cm = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the internal model.

        Args:
            X (pd.DataFrame): Feature set.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate prediction probabilities using the internal model.

        Args:
            X (pd.DataFrame): Feature set.

        Returns:
            np.ndarray: Class probabilities.

        Raises:
            AttributeError: If the model does not support probability prediction.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("This model does not support predict_proba.")

    def plot_cm(self) -> None:
        """
        Plot the stored confusion matrix using matplotlib.
    
        Raises:
            ValueError: If the confusion matrix has not been computed.
        """
        if self.cm is None:
            raise ValueError("Confusion matrix is not available. Please compute it before plotting.")
    
        print_header(f"Confusion Matrix: {self.name}", color='bright_cyan')
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def summary(self) -> None:
        """
        Print a formatted summary of the model, including best hyperparameters and last evaluation metrics.
        
        This method does not return anything; it displays information in the console.
        """
        print_header(f"Model: {self.name}", color='bright_blue')
        if self.best_params_:
            print(f"Best Params: {self.best_params_}")
        if self.metrics:
            print("Last evaluation results:")
            for k, v in self.metrics.items():
                print(f"{k:>12}: {v:.4f}")

