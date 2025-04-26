import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def split_and_standardize(data: pd.DataFrame, categorical_columns: list, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the input DataFrame into training and testing sets, and standardizes
    only the numerical feature columns (deduced from excluding categorical_columns and 'label').

    Args:
        data (pd.DataFrame): Input DataFrame with features and a 'label' column.
        categorical_columns (list): List of column names considered categorical (not to be standardized).
        test_size (float, optional): Proportion of the dataset to include in the test split.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple:
            - pd.DataFrame: data_train with standardized numerical features and original labels.
            - pd.DataFrame: data_test with standardized numerical features and original labels.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty.")
    if 'label' not in data.columns:
        raise ValueError("Input DataFrame must contain a 'label' column.")
    if data.isnull().sum().any():
        raise ValueError("DataFrame contains missing values. Please clean the data before proceeding.")

    # Deduce numerical columns (excluding 'label' and categorical columns)
    excluded_columns = set(categorical_columns + ['label'])
    numerical_columns = [col for col in data.columns if col not in excluded_columns]

    # Separate features and label
    X = data.drop(columns='label')
    y = data['label']

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize only numerical columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Combine scaled features with target
    data_train = pd.concat([X_train_scaled, y_train], axis=1).reset_index(drop=True)
    data_test = pd.concat([X_test_scaled, y_test], axis=1).reset_index(drop=True)

    print_box('Numerical Data has been standardized and the dataset has been split')
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
    Prints the evaluation metrics inside a decorated box.
    
    Args:
        metrics: Dictionary containing the calculated metrics.
    """
    # Construir el mensaje con las métricas
    metrics_str = "🔍 Model Evaluation Metrics:\n"
    for k, v in metrics.items():
        metrics_str += f"{k.capitalize()}: {v:.4f}\n"

    # Usar print_box para mostrar todo dentro de una caja
    print_box(metrics_str)

def plot_confusion_matrix(data_test, preds):
    """
    Plots the confusion matrix.
    
    Args:
        y_test: The ground truth labels.
        preds: The model predictions.
    """
    print_header('📊 Confusion Matrix:📊')
    y_test = data_test['label']
    
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

def print_header(text: str) -> None:
    """
    Prints a beautified string enclosed in an ASCII-style box.

    Example:
    >>> print_header("Training UFC Fight Predictor Model")
    ╔════════════════════════════════════════════╗
    ║  Training UFC Fight Predictor Model        ║
    ╚════════════════════════════════════════════╝
    """
    padded_text = f"  {text}  "
    box_width = len(padded_text)

    top_border = f"╔{'═' * box_width}╗"
    middle = f"║{padded_text}║"
    bottom_border = f"╚{'═' * box_width}╝"

    print(top_border)
    print(middle)
    print(bottom_border)

def print_box(text: str) -> None:
    """
    Prints text inside a simple ASCII box.
    
    Example:
    >>> print_box("Some info goes here")
    +-----------------------------+
    |     Some info goes here     |
    +-----------------------------+
    """
    box_width = len(text) + 6
    print(f"+{'-' * (box_width - 2)}+")
    print(f"|  {text}  |")
    print(f"+{'-' * (box_width - 2)}+")

# Function to print info message in blue
def print_info(text: str) -> None:
    """
    Prints text in blue for informational messages.
    """
    print(f"\033[94m{text}\033[0m")

# Function to print error message in red
def print_error(text: str) -> None:
    """
    Prints text in red for error messages.
    """
    print(f"\033[91m{text}\033[0m")
    
def get_pretty_model_name(model) -> str:
    """
    Args:
    model (object): A scikit-learn model object (e.g., RandomForestClassifier, SVC).

    Returns a prettified name for a scikit-learn model object.
    For example:
        RandomForestClassifier -> "Random Forest Classifier"
        LogisticRegression     -> "Logistic Regression"
        SVC                    -> "Support Vector Classifier"
    """
    raw_name = type(model).__name__

    # Manual mapping for known abbreviations
    replacements = {
        "SVC": "Support Vector Classifier",
        "SVR": "Support Vector Regressor",
        "KNeighborsClassifier": "K-Nearest Neighbors",
        "KMeans": "K-Means",
        "MLPClassifier": "Neural Network (MLP)",
        "GaussianNB": "Gaussian Naive Bayes",
        "MultinomialNB": "Multinomial Naive Bayes",
        "DecisionTreeClassifier": "Decision Tree",
        "RandomForestClassifier": "Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting",
        "AdaBoostClassifier": "AdaBoost",
        "BaggingClassifier": "Bagging",
        "LogisticRegression": "Logistic Regression"
    }

    if raw_name in replacements:
        return replacements[raw_name]

    # Default: convert CamelCase to spaced words
    pretty = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_name)
    return pretty.strip()