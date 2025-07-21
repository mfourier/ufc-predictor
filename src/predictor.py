# Author: Maximiliano Lioi | License: MIT

import pandas as pd
import numpy as np
from src.config import pretty_model_name
from src.io_model import load_model, load_all_models
from src.model import UFCModel
from src.metrics import evaluate_metrics, evaluate_cm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UFCPredictor:
    """
    Predictor class to handle UFC fight predictions using trained models and fighter stats.
    """

    def __init__(self, fighters_df, ufc_data_with_odds, 
                ufc_data_no_odds, verbose=False):

        self.fighters_df = fighters_df
        self.ufc_data_with_odds = ufc_data_with_odds
        self.ufc_data_no_odds = ufc_data_no_odds
        self.ufc_data = None # dynamically set in predict() for encode()

        models_list = load_all_models(include_no_odds=True, verbose=verbose)
        self.models = {m.name: m for m in models_list}
        self.model_keys_with_odds = [m.name for m in models_list if not m.is_no_odds]
        self.model_keys_no_odds = [m.name for m in models_list if m.is_no_odds]
        self.default_model_with_odds = 'nn_best'
        self.default_model_no_odds = 'xgb_best_no_odds'
        self.evaluate_all_models(verbose=verbose)

    def evaluate_all_models(self, verbose=False):
        """
        Evaluate all loaded models and store their metrics and confusion matrices.
        """
        for model in self.models.values():
            data = self.ufc_data_no_odds if model.is_no_odds else self.ufc_data_with_odds
            try:
                model.metrics = evaluate_metrics(model, data, verbose=verbose)
                model.cm = evaluate_cm(model, data)
                if verbose:
                    logger.info(f"✅ Evaluated model: {model.name}")
            except Exception as e:
                logger.error(f"❌ Failed to evaluate model {model.name}: {e}")

    def get_available_models(self):
        return list(self.models.keys())
    
    def get_available_weightclasses(self):
        return sorted(self.fighters_df['WeightClass'].unique())

    def get_fighters_by_weightclass(self, weightclass):
        return sorted(self.fighters_df[self.fighters_df['WeightClass'] == weightclass]['Fighter'].unique())

    def get_fighter_stats(self, name, year=None):
        """
        Retrieve the stats of a fighter for a specific year.

        Args:
            name (str): Fighter's name.
            year (int, optional): Year to retrieve stats for. If None, raises an error listing available years.

        Returns:
            pd.Series: Row of the fighter for the specified year.

        Raises:
            ValueError: If the fighter is not found or the year is invalid.
        """
        df = self.fighters_df

        # Filter by fighter name
        fighter_rows = df[df['Fighter'] == name]

        if fighter_rows.empty:
            raise ValueError(f"❌ Fighter '{name}' not found in the dataset.")

        # Ensure 'Date' is datetime
        if not np.issubdtype(fighter_rows['Date'].dtype, np.datetime64):
            fighter_rows = fighter_rows.copy()
            fighter_rows['Date'] = pd.to_datetime(fighter_rows['Date'], errors='coerce')

        # Check available years
        available_years = sorted(fighter_rows['Year'].unique())

        if year is None:
            raise ValueError(
                f"⚠️ Please specify a year for fighter '{name}'. "
                f"Available years: {available_years}"
            )

        if year not in available_years:
            raise ValueError(
                f"❌ Fighter '{name}' does not have stats for year {year}. "
                f"Available years: {available_years}"
            )

        # Get the row for the specified year (should be unique)
        selected_row = fighter_rows[fighter_rows['Year'] == year].iloc[0]

        return selected_row

    def set_active_ufc_data(self, ufc_data):
        """
        Set the active UFCData context (used temporarily during predict).

        Args:
            ufc_data (UFCData): The UFCData object to activate.
        """
        self.ufc_data = ufc_data
        self.scaler = ufc_data.get_scaler()
        self.numerical_columns = ufc_data.numerical_columns
        self.categorical_columns = ufc_data.categorical_columns

    def compute_feature_vector(self, red, blue, is_five_round_fight, include_odds, red_odds=None, blue_odds=None):
        """
        Compute engineered features between two fighters, optionally including odds.

        Args:
            red (pd.Series): Red fighter stats.
            blue (pd.Series): Blue fighter stats.
            red_odds (float): Simulated odds for Red.
            blue_odds (float): Simulated odds for Blue.
            include_odds (bool): Whether to include the OddsDif feature.

        Returns:
            pd.DataFrame: One-row DataFrame with feature differences, ready for model input.
        """
        feature_vector = {
            'BlueTotalTitleBouts': blue['TotalTitleBouts'], 
            'RedTotalTitleBouts': red['TotalTitleBouts'],
            'LoseStreakDif': blue['CurrentLoseStreak'] - red['CurrentLoseStreak'],
            'WinStreakDif': blue['CurrentWinStreak'] - red['CurrentWinStreak'],
            'LongestWinStreakDif': blue['LongestWinStreak'] - red['LongestWinStreak'],
            'KODif': blue['WinsByKO'] - red['WinsByKO'],
            'SubDif': blue['WinsBySubmission'] - red['WinsBySubmission'],
            'HeightDif': blue['HeightCms'] - red['HeightCms'],
            'ReachDif': blue['ReachCms'] - red['ReachCms'],
            'AgeDif': blue['Age'] - red['Age'],
            'SigStrDif': blue['AvgSigStrLanded'] - red['AvgSigStrLanded'],
            'AvgSubAttDif': blue['AvgSubAtt'] - red['AvgSubAtt'],
            'AvgTDDif': blue['AvgTDLanded'] - red['AvgTDLanded'],
            'RedTotalFights': red['TotalFights'], 
            'BlueTotalFights': blue['TotalFights'],
            'FightStance': 'Closed Stance' if blue['Stance'] == red['Stance'] else 'Open Stance',
            'WeightGroup': blue['WeightClassMap'],
            'BlueFinishRate': blue['FinishRate'],
            'RedFinishRate': red['FinishRate'],
            'BlueWinRatio': blue['WinRatio'],
            'RedWinRatio': red['WinRatio'],
            'HeightReachRatioDif': blue['HeightReachRatio'] - red['HeightReachRatio'],
            'RedKOPerFight': red['KOPerFight'],
            'BlueKOPerFight':blue['KOPerFight'],
            'RedSubPerFight': red['SubPerFight'],
            'BlueSubPerFight': blue['SubPerFight'],
            'IsFiveRoundFight': is_five_round_fight
        }

        if include_odds:
            if red_odds is None or blue_odds is None:
                raise ValueError("❌ Odds are required but were not provided.")
            feature_vector['OddsDif'] = blue_odds - red_odds

        return pd.DataFrame([feature_vector])


    def standardize(self, features_df):
        num_cols_present = [col for col in self.numerical_columns if col in features_df.columns]
        if self.scaler is not None and num_cols_present:
            features_df[num_cols_present] = self.scaler.transform(features_df[num_cols_present])
        return features_df

    def encode(self, features_df):
        bin_cols_present = [col for col in self.ufc_data.binary_columns if col in features_df.columns]
        multi_cols_present = [col for col in self.ufc_data.multiclass_columns if col in features_df.columns]

        # Binary encoding
        if bin_cols_present:
            bin_encoded = pd.get_dummies(features_df[bin_cols_present], drop_first=True).astype(int)
        else:
            bin_encoded = pd.DataFrame(index=features_df.index)

        # Multiclass encoding
        if multi_cols_present:
            multi_encoded = pd.get_dummies(features_df[multi_cols_present], drop_first=False).astype(int)
        else:
            multi_encoded = pd.DataFrame(index=features_df.index)

        # Numerical (already standardized)
        num_encoded = features_df[[col for col in self.numerical_columns if col in features_df.columns]]

        # Combine all
        X_final = pd.concat([bin_encoded, multi_encoded, num_encoded], axis=1)
        return X_final

    def predict(self, red_id=None, blue_id=None, is_five_round_fight=0, model_name=None, red_odds=None, blue_odds=None, red_series=None, blue_series=None):
        model = self.models[model_name]
        include_odds = not model.is_no_odds
        ufc_data = self.ufc_data_no_odds if model.is_no_odds else self.ufc_data_with_odds
        self.set_active_ufc_data(ufc_data)
            
        # Decide data source
        if red_series is not None and blue_series is not None:
            red = red_series
            blue = blue_series
        else:
            if red_id == blue_id:
                raise ValueError("❌ Red and Blue fighters must be different.")
            red_name, red_year = red_id
            blue_name, blue_year = blue_id
            red = self.get_fighter_stats(red_name, red_year)
            blue = self.get_fighter_stats(blue_name, blue_year)

            if red['WeightClass'] != blue['WeightClass']:
                raise ValueError(
                    f"❌ Fighters must be in the same weight class. "
                    f"Red: {red['WeightClass']}, Blue: {blue['WeightClass']}"
                )

        # Compute feature vector
        features_df = self.compute_feature_vector(red, blue, is_five_round_fight, include_odds, red_odds, blue_odds)
        features_df_raw = features_df.copy()

        # Use correct scaler and columns
        self.scaler = ufc_data.get_scaler()
        self.numerical_columns = ufc_data.numerical_columns
        self.categorical_columns = ufc_data.categorical_columns

        features_df = self.standardize(features_df)
        X_final = self.encode(features_df)

        # Align with model features
        if hasattr(model.estimator, "feature_names_in_"):
            model_features = model.estimator.feature_names_in_
            for col in model_features:
                if col not in X_final.columns:
                    X_final[col] = 0
            X_final = X_final[model_features]

        # Prediction
        pred = model.predict(X_final)
        try:
            prob_array = model.predict_proba(X_final)[0]
            prob_red, prob_blue = prob_array[0], prob_array[1]
        except AttributeError:
            prob_red, prob_blue = None, None

        result = {
            'prediction': 'Blue' if pred[0] == 1 else 'Red',
            'probability_red': prob_red,
            'probability_blue': prob_blue,
            'feature_vector': features_df_raw.to_dict(orient='records')[0],
            'red_summary': red.to_dict(),
            'blue_summary': blue.to_dict(),
        }

        if include_odds:
            result['odds'] = (red_odds, blue_odds)

        return result
    
    def __repr__(self):
        num_fighters = self.fighters_df['Fighter'].nunique()
        num_weightclasses = self.fighters_df['WeightClass'].nunique()
        scaler_name = type(self.ufc_data_with_odds.get_scaler()).__name__

        models_with_odds = [m.name for m in self.models.values() if not m.is_no_odds]
        models_no_odds = [m.name for m in self.models.values() if m.is_no_odds]

        return (
            f"🥋<UFCPredictor>🥋\n"
            f"  📊 Fighters loaded       : {num_fighters}\n"
            f"  🏋️‍♂️ Weight classes       : {num_weightclasses}\n"
            f"  🧠 Models with odds      : {', '.join(sorted(models_with_odds))}\n"
            f"  🚫 Models no odds        : {', '.join(sorted(models_no_odds))}\n"
            f"  ⭐ Default with odds     : {self.default_model_with_odds}\n"
            f"  ⭐ Default no odds       : {self.default_model_no_odds}\n"
            f"  🔢 Numerical features    : {len(self.ufc_data_with_odds.numerical_columns)} (with odds), "
            f"{len(self.ufc_data_no_odds.numerical_columns)} (no odds)\n"
            f"{len(self.ufc_data_with_odds.categorical_columns)} (with odds), {len(self.ufc_data_no_odds.categorical_columns)} (no odds)\n"
            f"  🛠️  Scaler               : {scaler_name}\n"
        )

