import os
import pickle

from models.config import pretty_model_name
from .helpers import get_pretty_model_name


def get_models_dir() -> str:
    """
    Get the absolute path to the 'models' directory at the project root.
    Creates the directory if it does not exist.

    Returns:
        str: Absolute path to the models directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))  # ufc-predictor/
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def save_model(model: object, name: str, overwrite: bool = True) -> None:
    """
    Save a trained model to a .pkl file in the 'models' directory.

    Args:
        model (object): The trained model object.
        name (str): Filename (without extension) to save the model as.
        overwrite (bool): Whether to overwrite the file if it already exists.

    Raises:
        FileExistsError: If the file exists and overwrite is False.
    """
    path = os.path.join(get_models_dir(), f"{name}.pkl")

    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"❌ File '{path}' already exists. Use overwrite=True to replace.")

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model {get_pretty_model_name(model)} saved to: {path}")


def load_model(name: str, verbose: bool = True) -> object:
    """
    Load a model from the 'models' directory.

    Args:
        name (str): Filename (without extension) of the model to load.
        verbose (bool): Whether to print the loading message.

    Returns:
        object: The loaded model object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = os.path.join(get_models_dir(), f"{name}.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found at: {path}")

    with open(path, 'rb') as f:
        model = pickle.load(f)

    if verbose:
        print(f"📦 Model {pretty_model_name[name]} loaded from: {path}")

    return model
