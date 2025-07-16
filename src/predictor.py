import pandas as pd
import numpy as np
from config import pretty_model_name
from io_model import load_model
from model import UFCModel

class UFCPredictor:
    """
    Predictor class to handle UFC fight predictions using trained models and fighter stats.
    """

    def __init__(self, fighters_df, ufc_data):
        self.fighters_df = fighters_df
        self.models = {
            name: UFCModel(load_model(name, verbose=False))
            for name in pretty_model_name
        }

        self.default_model = 'nn_best'  # or set to your preferred default
        self.ufc_data = ufc_data
        self.scaler = ufc_data.get_scaler()
        self.numerical_columns = ufc_data.numerical_columns
        
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


    def compute_feature_vector(self, red, blue, red_odds, blue_odds):
        """
        Compute engineered features between two fighters, including odds.

        Args:
            red (pd.Series): Red fighter stats.
            blue (pd.Series): Blue fighter stats.
            red_odds (float): Simulated odds for Red.
            blue_odds (float): Simulated odds for Blue.

        Returns:
            pd.DataFrame: One-row DataFrame with feature differences, ready for model input.
        """

        feature_vector = {
            'LoseStreakDif': blue['CurrentLoseStreak'] - red['CurrentLoseStreak'],
            'WinStreakDif': blue['CurrentWinStreak'] - red['CurrentWinStreak'],
            'KODif': blue['WinsByKO'] - red['WinsByKO'],
            'SubDif': blue['WinsBySubmission']- red['WinsBySubmission'],
            'HeightDif': blue['HeightCms'] - red['HeightCms'],
            'AgeDif': blue['Age'] - red['Age'],
            'SigStrDif': blue['AvgSigStrLanded'] - red['AvgSigStrLanded'],
            'AvgSubAttDif': blue['AvgSubAtt'] - red['AvgSubAtt'],
            'AvgTDDif': blue['AvgTDLanded'] - red['AvgTDLanded'],
            'FightStance': 'Closed Stance' if blue['Stance'] == red['Stance'] else 'Open Stance',
            'WeightGroup': blue['WeightGroupMap'],
            'FinishRateDif': blue['FinishRate'] - red['FinishRate'],
            'WinRatioDif': blue['WinRatio'] - red['WinRatio'],
            'ExpPerAgeDif': blue['ExpPerAge'] - red['ExpPerAge'],
            'ReachAdvantageRatioDif': blue['ReachCms'] / red['ReachCms'],
            'HeightReachRatioDif': blue['HeightReachRatio'] - red['HeightReachRatio'],
            'DecisionRateDif': blue['DecisionRate'] - red['DecisionRate'],
            'OddsDif': blue_odds - red_odds
        }
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

    def predict(self, red_id, blue_id, red_odds, blue_odds, model_name):
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
        features_df = self.compute_feature_vector(red, blue, red_odds, blue_odds)

        # Standardize numerical
        features_df = self.standardize(features_df)

        # Encode categorical
        X_final = self.encode(features_df)

        # Align with model features
        model = self.models[model_name]
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
            'feature_vector': features_df.to_dict(orient='records')[0],
            'red_summary': red[['Fighter', 'Record', 'WeightClass', 'Stance']].to_dict(),
            'blue_summary': blue[['Fighter', 'Record', 'WeightClass', 'Stance']].to_dict()
        }
        return result





