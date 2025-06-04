import os
import pickle

from models.config import pretty_model_name
from .helpers import get_pretty_model_name, print_header


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

def get_data_dir() -> str:
    """
    Get the absolute path to the 'data' directory at the project root.
    Creates the directory if it does not exist.

    Returns:
        str: Absolute path to the data directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))  # ufc-predictor/
    models_dir = os.path.join(project_root, 'data/processed')
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

def save_data(data: object, name: str = 'ufc_data', overwrite: bool = True) -> None:
    """
    Save a UFCData object to a .pkl file in the 'data' directory.

    Args:
        data (UFCData): The UFCData object to serialize.
        name (str): Filename (without extension) to save the object as.
        overwrite (bool): Whether to overwrite the file if it already exists.
    """
    path = os.path.join(get_data_dir(), f"{name}.pkl")
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"❌ File '{path}' already exists. Use overwrite=True to replace.")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ UFCData object saved to: {path}")


def load_data(name: str = 'ufc_data', verbose: bool = True) -> object:
    """
    Load a UFCData object from a .pkl file.

    Args:
        name (str): Filename (without extension) of the UFCData object to load.
        verbose (bool): Whether to print the loading message.

    Returns:
        UFCData: The loaded UFCData object.
    """
    path = os.path.join(get_data_dir(), f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ UFCData file not found at: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if verbose:
        print(f"📦 UFCData object loaded from: {path}")
    return data

def save_ufc_datasets(UFCData, project_root):
    """
    Save raw and processed UFC train/test splits as CSV files.

    Args:
        UFCData (UFCData): Instance of UFCData with all preprocessing completed.
        project_root (str or Path): Root directory of the project.

    The following files are saved in 'data/processed/':
        - ufc_train.csv: Raw training data (features + label)
        - ufc_test.csv: Raw testing data (features + label)
        - ufc_train_processed.csv: Processed (standardized + encoded) training data (features + label)
        - ufc_test_processed.csv: Processed (standardized + encoded) testing data (features + label)
    """
    # Collect datasets
    ufc_train = UFCData.get_df_train()
    ufc_test = UFCData.get_df_test()
    ufc_processed_train = UFCData.get_df_processed_train()
    ufc_processed_test = UFCData.get_df_processed_test()

    # Output file mapping
    output_paths = {
        "ufc_train.csv": ufc_train,
        "ufc_test.csv": ufc_test,
        "ufc_processed_train.csv": ufc_processed_train,
        "ufc_processed_test.csv": ufc_processed_test,
    }

    # Save each dataset to CSV
    for fname, df in output_paths.items():
        df.to_csv(f"{project_root}/data/processed/{fname}", index=False)

    print(f"✅ UFCData object saved to: {output_paths.keys()}")
