import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

def split_and_standardize(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the input DataFrame into training and testing sets, and standardizes
    the feature columns using statistics computed from the training set.
    
    This function assumes the DataFrame contains a 'label' column as the target.
    The rest of the columns are treated as features. Standardization is performed
    via sklearn's StandardScaler (zero mean, unit variance).
    Stratification ensures that the class distribution is preserved in both train and test sets.

    Args:
        data (pd.DataFrame): Input DataFrame with features and a 'label' column.
        test_size (float, optional): Proportion of the dataset to include in the test split (default is 0.2).
        random_state (int, optional): Random seed for reproducibility (default is 42).

    Returns:
        tuple:
            - pd.DataFrame: data_train with standardized features and original labels.
            - pd.DataFrame: data_test with standardized features and original labels.

    Raises:
        ValueError: If 'label' column is missing, data is empty, or contains null values.
    """
    # Check for empty DataFrame
    if data.empty:
        raise ValueError("Input DataFrame is empty.")
    
    # Check for presence of 'label' column
    if 'label' not in data.columns:
        raise ValueError("Input DataFrame must contain a 'label' column.")
    
    # Check for missing values in features and label
    if data.isnull().sum().any():
        raise ValueError("DataFrame contains missing values. Please clean the data before proceeding.")

    # Separate features and target
    X = data.drop(columns='label')
    y = data['label']

    # Split into train and test sets (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features using training set statistics
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Concatenate standardized features with targets
    data_train = pd.concat([X_train_scaled, y_train], axis=1).reset_index(drop=True)
    data_test = pd.concat([X_test_scaled, y_test], axis=1).reset_index(drop=True)

    return data_train, data_test
    
def get_predictions(model, X_test):
    """
    Returns model predictions and probabilities based on the input model type.
    
    Args:
        model (object): The trained model (either PyTorch or scikit-learn).
        X_test (numpy.ndarray): The input features for testing.

    Returns:
        tuple: A tuple containing the predictions and probabilities.
    """
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

def print_metrics(metrics):
    """
    Prints the evaluation metrics.
    
    Args:
        metrics: Dictionary containing the calculated metrics.
    """
    print("🔍 Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

def plot_confusion_matrix(data_test, preds):
    """
    Plots the confusion matrix.
    
    Args:
        y_test: The ground truth labels.
        preds: The model predictions.
    """
    print('📊 Confusion Matrix:')
    y_test = data_test['label']
    
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
