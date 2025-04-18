import os
import pickle

def get_models_dir():
    """
    Returns the absolute path to the 'models' folder within the project.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))  # ufc-predictor
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)  # Create the folder if it doesn't exist
    return models_dir


def save_model(model, name):
    """
    Saves a model in pickle (.pkl) format inside the 'models' folder.

    Parameters:
    - model: the trained model (e.g., a scikit-learn estimator)
    - name: name of the file without extension (e.g., 'knn_best')
    """
    path = os.path.join(get_models_dir(), f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'✅ Model saved as {path}')


def load_model(name):
    """
    Loads a model from the 'models' folder given its name.

    Parameters:
    - name: name of the file without extension (e.g., 'knn_best')

    Returns:
    - The loaded model (Python object)
    """
    path = os.path.join(get_models_dir(), f'{name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'❌ File {path} does not exist.')
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f'📦 Model loaded from {path}')
    return model
