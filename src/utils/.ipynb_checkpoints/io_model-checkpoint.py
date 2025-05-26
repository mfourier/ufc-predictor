import os
import pickle
from models.config import *
from .helpers import *

def get_models_dir():
    """
    Returns the absolute path to the 'models' directory at the project root.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))  # ufc-predictor/
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def save_model(model, name, overwrite=True):
    """
    Saves a model in .pkl format in the 'models' folder.

    Parameters:
    - model: Trained model object.
    - name (str): Filename without extension.
    - overwrite (bool): If False and file exists, raises an error.
    """
    path = os.path.join(get_models_dir(), f"{name}.pkl")

    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"❌ File '{path}' already exists. Use overwrite=True to replace.")

    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model {get_pretty_model_name(model)} saved to: {path}")


def load_model(name, verbose=True):
    """
    Loads a model from the 'models' folder.

    Parameters:
    - name (str): Filename without extension.
    - verbose (bool): Whether to print the loading path.

    Returns:
    - model: Loaded model object.
    """
    path = os.path.join(get_models_dir(), f"{name}.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found at: {path}")

    with open(path, 'rb') as f:
        model = pickle.load(f)

    if verbose:
        print(f"📦 Model {pretty_model_name[name]} loaded from: {path}")

    return model