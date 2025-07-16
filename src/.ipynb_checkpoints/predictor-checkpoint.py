import pandas as pd
import numpy as np
from io_model import load_model

class Predictor:
    """
    Predictor class to handle UFC fight predictions using trained models and fighter stats.
    """

    def __init__(self, fighters_df, model_names):
        self.fighters_df = fighters_df
        self.model_paths = {name: f'{name}.pkl' for name in model_names}
        self.models = {}

    def get_available_models(self):
        return list(self.model_paths.keys())

    def get_available_weightclasses(self):
        return sorted(self.fighters_df['WeightClass'].unique())

    def get_fighters_by_weightclass(self, weightclass):
        return sorted(self.fighters_df[self.fighters_df['WeightClass'] == weightclass]['Fighter'].unique())

    def get_fighter_stats(self, name, weightclass):
        df = self.fighters_df
        row = df[(df['Fighter'] == name) & (df['WeightClass'] == weightclass)]
        if row.empty:
            raise ValueError(f"Fighter '{name}' not found in weightclass '{weightclass}'.")
        return row.iloc[0]

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
        red_wins = max(red['Wins'], 1)
        blue_wins = max(blue['Wins'], 1)
        red_losses = max(red['Losses'], 1)
        blue_losses = max(blue['Losses'], 1)

        red_finish = (red['WinsByKO'] + red['WinsBySubmission'] + red['WinsByTKODoctorStoppage']) / red_wins
        blue_finish = (blue['WinsByKO'] + blue['WinsBySubmission'] + blue['WinsByTKODoctorStoppage']) / blue_wins

        red_win_ratio = red['Wins'] / (red_wins + red_losses)
        blue_win_ratio = blue['Wins'] / (blue_wins + blue_losses)

        red_exp_age = red['TotalRoundsFought'] / red['Age']
        blue_exp_age = blue['TotalRoundsFought'] / blue['Age']

        red_dec_wins = red[['WinsByDecisionMajority', 'WinsByDecisionSplit', 'WinsByDecisionUnanimous']].sum()
        blue_dec_wins = blue[['WinsByDecisionMajority', 'WinsByDecisionSplit', 'WinsByDecisionUnanimous']].sum()

        red_dec_rate = red_dec_wins / red_wins
        blue_dec_rate = blue_dec_wins / blue_wins

        feature_vector = {
            'FinishRateDif': blue_finish - red_finish,
            'WinRatioDif': blue_win_ratio - red_win_ratio,
            'ExpPerAgeDif': blue_exp_age - red_exp_age,
            'ReachAdvantageRatio': blue['ReachCms'] / red['ReachCms'],
            'HeightReachRatioDif': (blue['HeightCms'] / blue['ReachCms']) - (red['HeightCms'] / red['ReachCms']),
            'WinsByDecisionDif': blue_dec_wins - red_dec_wins,
            'DecisionRateDif': blue_dec_rate - red_dec_rate,
            'OddsDiff': blue_odds / red_odds
        }
        return pd.DataFrame([feature_vector])

    def load_model_on_demand(self, model_name):
        if model_name not in self.models:
            self.models[model_name] = load_model(model_name)
        return self.models[model_name]

    def predict(self, red_name, blue_name, red_odds, blue_odds, weightclass, model_name):
        if red_name == blue_name:
            raise ValueError("Red and Blue fighters must be different.")

        red = self.get_fighter_stats(red_name, weightclass)
        blue = self.get_fighter_stats(blue_name, weightclass)

        features_df = self.compute_feature_vector(red, blue, red_odds, blue_odds)
        model = self.load_model_on_demand(model_name)

        pred = model.predict(features_df)
        prob = model.predict_proba(features_df) if hasattr(model, 'predict_proba') else None

        result = {
            'prediction': 'Blue' if pred[0] == 1 else 'Red',
            'probability': prob[0].tolist() if prob is not None else None,
            'feature_vector': features_df.to_dict(orient='records')[0],
            'red_summary': red[['Fighter', 'Record', 'WinRate', 'Stance']].to_dict(),
            'blue_summary': blue[['Fighter', 'Record', 'WinRate', 'Stance']].to_dict()
        }
        return result
