import torch
import numpy as np
from sklearn.model_selection import train_test_split

def is_pytorch_model(model):
    """
    Checks if the given model is a PyTorch neural network model.

    Args:
        model (object): The model to check.

    Returns:
        bool: True if the model is a PyTorch neural network model, otherwise False.
    """
    return isinstance(model, torch.nn.Module)

def get_predictions(model, X_test):
    """
    Returns model predictions and probabilities based on the input model type.
    
    Args:
        model (object): The trained model (either PyTorch or scikit-learn).
        X_test (numpy.ndarray): The input features for testing.

    Returns:
        tuple: A tuple containing the predictions and probabilities.
    """
    if is_pytorch_model(model):
        # For PyTorch models, set to evaluation mode and compute predictions
        model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            logits = model(X_tensor).view(-1)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)
        return preds, probs
    else:
        # For scikit-learn models, check if the model has predict_proba or decision_function
        if hasattr(model, "predict_proba"):
            # For models with a predict_proba method (like RandomForest, AdaBoost, etc.)
            probs = model.predict_proba(X_test)[:, 1]
        else:
            # For models like SVM or Logistic Regression, use decision_function
            probs = model.decision_function(X_test)
            # Apply sigmoid for SVMs or linear models to convert to probabilities
            probs = 1 / (1 + np.exp(-probs))
        
        preds = model.predict(X_test)
        return preds, probs
    
def prepare_data(data, test_size = 0.25):
    """
    Prepares the test data by separating features and labels.
    
    Args:
        data_test: A DataFrame containing the input features and labels for testing.
        
    Returns:
        tuple: X_test (features), y_test (labels)
    """
    X = data.drop(columns=['label']).values
    y = data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state=42)

    return X_train, X_test, y_train, y_test

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

    if 'accuracy' in metrics_to_compute:
        metrics['accuracy'] = accuracy_score(y_test, preds)

    if 'precision' in metrics_to_compute:
        metrics['precision'] = precision_score(y_test, preds, zero_division=1)

    if 'recall' in metrics_to_compute:
        metrics['recall'] = recall_score(y_test, preds, zero_division=1)

    if 'f1_score' in metrics_to_compute:
        metrics['f1_score'] = f1_score(y_test, preds, zero_division=1)

    if 'roc_auc' in metrics_to_compute:
        metrics['roc_auc'] = roc_auc_score(y_test, probs[:, 1])  # Use probabilities for class 1

    return metrics

def print_metrics(metrics):
    """
    Prints the evaluation metrics.
    
    Args:
        metrics: Dictionary containing the calculated metrics.
    """
    print("🔍 Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

def plot_confusion_matrix(y_test, preds):
    """
    Plots the confusion matrix.
    
    Args:
        y_test: The ground truth labels.
        preds: The model predictions.
    """
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(y_test, probs):
    """
    Plots the ROC curve.
    
    Args:
        y_test: The ground truth labels.
        probs: The predicted probabilities.
    """
    RocCurveDisplay.from_predictions(y_test, probs[:, 1])  # Assuming probs contains probabilities for class 1
    plt.title("ROC Curve")
    plt.show()