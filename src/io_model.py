import os
import pickle
import logging
from src.config import pretty_model_name
from src.helpers import get_pretty_model_name
from src.model import UFCModel

# Logger setup
logger = logging.getLogger(__name__)

def get_models_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils
    project_root = os.path.abspath(os.path.join(current_dir, '..'))  # ufc-predictor/
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def get_data_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    project_root = os.path.abspath(os.path.join(current_dir, '..'))  # ufc-predictor/
    data_dir = os.path.join(project_root, 'data/processed')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def save_model(model: object, name: str, overwrite: bool = True) -> None:
    path = os.path.join(get_models_dir(), f"{name}.pkl")

    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"❌ File '{path}' already exists. Use overwrite=True to replace.")

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"✅ Model '{get_pretty_model_name(model)}' saved to: {path}")

def load_model(name: str, verbose: bool = True) -> object:
    path = os.path.join(get_models_dir(), f"{name}.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found at: {path}")

    with open(path, 'rb') as f:
        model = pickle.load(f)

    if verbose:
        logger.info(f"📦 Model '{pretty_model_name[name]}' loaded from: {path}")

    return model

def save_data(data: object, name: str = 'ufc_data', overwrite: bool = True) -> None:
    path = os.path.join(get_data_dir(), f"{name}.pkl")
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"❌ File '{path}' already exists. Use overwrite=True to replace.")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"✅ UFCData object saved to: {path}")

def load_data(name: str = 'ufc_data', verbose: bool = True) -> object:
    path = os.path.join(get_data_dir(), f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ UFCData file not found at: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if verbose:
        logger.info(f"📦 UFCData object loaded from: {path}")
    return data

def save_ufc_datasets(UFCData, project_root, name=None):
    suffix = f"_{name}" if name else ""

    ufc_train = UFCData.get_df_train()
    ufc_test = UFCData.get_df_test()
    ufc_processed_train = UFCData.get_df_processed_train()
    ufc_processed_test = UFCData.get_df_processed_test()

    output_paths = {
        f"ufc_train{suffix}.csv": ufc_train,
        f"ufc_test{suffix}.csv": ufc_test,
        f"ufc_processed_train{suffix}.csv": ufc_processed_train,
        f"ufc_processed_test{suffix}.csv": ufc_processed_test,
    }

    for fname, df in output_paths.items():
        df.to_csv(f"{project_root}/data/processed/{fname}", index=False)

    logger.info(f"✅ UFCData CSV files saved: {list(output_paths.keys())}")

def list_models():
    return [f.replace('.pkl', '') for f in os.listdir(get_models_dir()) if f.endswith('.pkl')]

def load_all_models(include_no_odds: bool = True, verbose: bool = True) -> list:
    """
    Load all trained models, including '_no_odds' versions if available.

    Args:
        include_no_odds (bool): Whether to attempt loading '_no_odds' models.
        verbose (bool): Whether to show verbose logs when loading each model.

    Returns:
        list: A list of UFCModel instances.
    """
    model_list = []

    # Only use base models (skip already '_no_odds' keys)
    base_names = [name for name in pretty_model_name if not name.endswith('_no_odds')]

    for name in base_names:
        # Load regular model
        try:
            model_normal = UFCModel(model=load_model(name, verbose=verbose))
            model_normal.is_no_odds = False  # ✅ mark as with-odds
            model_list.append(model_normal)
            logger.info(f"✅ Loaded model: {model_normal.name}")
        except Exception as e:
            logger.error(f"❌ Failed to load model '{name}': {e}")

        # Optionally load _no_odds model
        if include_no_odds:
            no_odds_name = f"{name}_no_odds"
            try:
                model_no_odds = UFCModel(model=load_model(no_odds_name, verbose=verbose))
                model_no_odds.name += " (no_odds)"
                model_no_odds.is_no_odds = True  # ✅ mark as no-odds
                model_list.append(model_no_odds)
                logger.info(f"✅ Loaded model: {model_no_odds.name}")
            except FileNotFoundError:
                logger.warning(f"⚠️ No '_no_odds' model found for '{name}', skipping.")
            except Exception as e:
                logger.error(f"❌ Failed to load '_no_odds' model '{no_odds_name}': {e}")

    return model_list

