import pandas as pd
import numpy as np
from typing import Optional, Sequence
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from helpers import get_pretty_model_name, print_header
import logging
import os
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

save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../img/")
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

    def plot_feature_importance(self, feature_names: Optional[Sequence[str]] = None, max_display: int = 20) -> None:
        """
        Plot feature importance for the underlying model.
    
        Supports models with:
            - feature_importances_ attribute (e.g., RandomForest, AdaBoost, etc.)
            - coef_ attribute (e.g., LogisticRegression, LinearSVC, etc.)
    
        Args:
            feature_names (Sequence[str] | None): List of feature names. If None, features will be named numerically.
            max_display (int): Maximum number of top features to display.
    
        Raises:
            AttributeError: If the underlying model does not provide feature importances.
            ValueError: If feature_names length doesn't match the number of features.
        """
        model = self.model
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_
    
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title = f"Feature Importances - {self.name}"
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_.ravel())
            title = f"Absolute Coefficient Importances - {self.name}"
        else:
            raise AttributeError(
                f"Model '{self.name}' does not provide feature importances (feature_importances_ or coef_)."
            )
    
        # Default feature names if none provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        else:
            if len(feature_names) != len(importances):
                raise ValueError(
                    f"Length of feature_names ({len(feature_names)}) does not match number of features ({len(importances)})."
                )
    
        # Sort and select top features
        sorted_idx = np.argsort(importances)[::-1][:max_display]
        top_features = [feature_names[i] for i in sorted_idx]
        top_importances = importances[sorted_idx]
    
        plt.figure(figsize=(10, max(4, len(top_features) // 2)))
        plt.barh(top_features[::-1], top_importances[::-1])
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_feature_importances_grid(
        models: list,
        feature_names: Optional[Sequence[str]] = None,
        max_display: int = 20,
        save_file: bool = False,
        filename: str = "model_feature_importances_grid.png"
    ) -> None:
        """
        Static method to plot feature importances or coefficients for a list of UFCModel instances.
    
        Args:
            models (list): List of UFCModel instances.
            feature_names (list | None): Feature names to display on the x-axis. Taken from first model if None.
            max_display (int): Maximum number of top features to display per model.
            save_file (bool): Whether to save the figure to disk.
            filename (str): Filename to use if saving. Saved under /img/.
        """
        import os
    
        # Filter models that support importance
        filtered = [
            m for m in models
            if hasattr(m.model, "best_estimator_")
            and (hasattr(m.model.best_estimator_, "feature_importances_") or hasattr(m.model.best_estimator_, "coef_"))
        ]
    
        if not filtered:
            print("❌ No models with feature importances or coefficients.")
            return
    
        n_models = len(filtered)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        axes = axes.flatten()
    
        for i, model in enumerate(filtered):
            ax = axes[i]
            mdl = model.model.best_estimator_
    
            if hasattr(mdl, "feature_importances_"):
                importances = mdl.feature_importances_
                title = f"{model.name} (Importances)"
            else:
                importances = np.abs(mdl.coef_.ravel())
                title = f"{model.name} (Coefficients)"
    
            if feature_names is None:
                fnames = [f"Feature {j}" for j in range(len(importances))]
            else:
                fnames = feature_names
    
            if len(fnames) != len(importances):
                raise ValueError(f"Length mismatch between feature_names and importances in model {model.name}.")
    
            sorted_idx = np.argsort(importances)[::-1][:max_display]
            top_features = [fnames[j] for j in sorted_idx]
            top_importances = importances[sorted_idx]
    
            ax.barh(top_features[::-1], top_importances[::-1])
            ax.set_title(title)
            ax.set_xlabel("Importance")
    
        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
    
        plt.tight_layout()
    
        if save_file:
            img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../img/"))
            os.makedirs(img_dir, exist_ok=True)
            save_path = os.path.join(img_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
    
        plt.show()

    def __repr__(self) -> str:
        summary = [f"<UFCModel: {self.name}>"]
        if self.score is not None:
            summary.append(f"  - Best CV Score : {self.score:.4f}")
        if self.best_params_:
            summary.append(f"  - Best Params   : {self.best_params_}")
        if self.metrics:
            summary.append("  - Last Evaluation:")
            for k, v in self.metrics.items():
                summary.append(f"      {k:<12}: {v:.4f}")
        return "\n".join(summary)

    def summary(self) -> None:
        print_header(f"Model: {self.name}", color='bright_blue')
        print(self)


