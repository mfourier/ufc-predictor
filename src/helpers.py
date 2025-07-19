import pandas as pd
import numpy as np
import os
from src.config import colors, pretty_names, default_params
from datetime import datetime
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.table import Table
from rich.columns import Columns
from rich.box import ROUNDED
from rich.text import Text
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

def get_predictions(model: object, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and probabilities using the input model.

    Args:
        model (UFCModel): A trained UFCModel.
        X_test (np.ndarray): Test feature matrix.

    Returns:
        tuple: (predictions, probabilities)
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model doesn't have a 'predict_proba' atributte.")
    preds = model.predict(X_test)
    return preds, probs

def display_model_params_table(model_params):
    rows = []
    for model_name, (_, params) in model_params.items():
        param_str = "; ".join([f"{key}: {value}" for key, value in params.items()])
        rows.append({
            "Model": model_name,
            "Hyperparameters": param_str
        })
    df = pd.DataFrame(rows)
    display(df)

def get_pretty_model_name(model: object) -> str:
    """
    Return the display-friendly name of a trained model.

    If the model is a GridSearchCV wrapper, extract the base estimator.

    Args:
        model (UFCModel): A trained UFCModel.

    Returns:
        str: Human-readable model name defined in `pretty_names`.

    Raises:
        ValueError: If the model's class is not mapped in `pretty_names`.
    """
    base_model = model.best_estimator_
    model_name = type(base_model).__name__

    if model_name not in pretty_names:
        raise ValueError(
            f"Model '{model_name}' does not have a predefined pretty name in the mapping."
        )

    return pretty_names[model_name]


def get_supported_models() -> list[str]:
    """
    Retrieve all supported model identifiers defined in `default_params`.

    Returns:
        list[str]: Sorted list of model names.
    """
    return sorted(default_params.keys())

def log_training_result(
    model_name: str,
    best_params: dict,
    metrics: dict,
    duration: float,
    log_path: str = "../data/results/training_log.csv"
) -> None:
    """
    Log the training results of a model into a cumulative CSV file.

    Args:
        model_name (str): Name of the model used in training.
        best_params (dict): Dictionary of hyperparameters found by GridSearchCV.
        metrics (dict): Dictionary containing evaluation metrics (accuracy, F1, etc.).
        duration (float): Duration of training in seconds.
        log_path (str): Path where the CSV log will be stored.

    Returns:
        None
    """
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'duration_sec': round(duration, 2),
        **metrics,
        **{f'param_{k}': v for k, v in best_params.items()}
    }

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(log_path, index=False)
    logger.info(f"✅ Training logged to {log_path}")

def print_corner_summary(corner, label, color, odds=None):
    """
    Pretty-print summary stats for a fighter corner using rich.
    """

    lines = [
        f"[bold]Record[/]         : {corner.get('Record', 'N/A')}",
        f"[bold]Weight Class[/]  : {corner.get('WeightClass', 'N/A')} | Stance: {corner.get('Stance', 'N/A')}",
    ]

    if odds is not None:
        lines.append(f"[bold]Odds[/]           : {odds}")

    lines.extend([
        f"[bold]Height[/]         : {corner.get('HeightCms', 'N/A')} cm | Reach: {corner.get('ReachCms', 'N/A')} cm",
        f"[bold]Age[/]           : {corner.get('Age', 'N/A')}",
        f"[bold]Win Rate[/]      : {corner.get('WinRate', 0):.3f} | Finish Rate: {corner.get('FinishRate', 0):.3f}",
        f"[bold]KO Wins[/]       : {corner.get('WinsByKO', 'N/A')} | Sub Wins: {corner.get('WinsBySubmission', 'N/A')}",
        f"[bold]Decision Rate[/] : {corner.get('DecisionRate', 0):.3f} | Avg. Sig. Strike Landed: {corner.get('AvgSigStrLanded', 0):.3f}",
        f"[bold]Avg Sub Att[/]   : {corner.get('AvgSubAtt', 0):.3f} | Avg TD Landed: {corner.get('AvgTDLanded', 0):.3f}"
    ])

    content = "\n".join(lines)

    panel = Panel(
        content,
        title=f"{label}",
        title_align="center",
        border_style=color  
    )

    console.print(panel)


def print_prediction_result_pipeline(result):
    """
    Pretty-print the result dictionary from UFCPredictor.predict() with detailed fighter stats and CLI colors.
    """
    red = result['red_summary']
    blue = result['blue_summary']
    pred = result['prediction']
    prob_red = result['probability_red']
    prob_blue = result['probability_blue']
    features = result['feature_vector']
    red_odds, blue_odds = result['odds']
    line_sep = "-" * 70
    c = colors  # alias for shorter

    # Header
    print(f"\n{c['bright_yellow']}{'🏆 UFC FIGHT PREDICTION RESULT':^70}{c['default']}")
    print(f"{c['bright_yellow']}{line_sep}{c['default']}")

    # Red corner
    print_corner_summary(red, f"🔴 RED CORNER (Favorite): {red['Fighter']} ({red['Year']})", c['bright_red'], red_odds)

    # Blue corner
    print_corner_summary(blue, f"🔵 BLUE CORNER (Underdog): {blue['Fighter']} ({blue['Year']})", c['bright_blue'], blue_odds)

    # Prediction result
    winner_color = c['bright_blue'] if pred == 'Blue' else c['bright_red']
    print(f"🏅 Predicted Winner: {winner_color}{'🔵 BLUE' if pred == 'Blue' else '🔴 RED'}{c['default']}")
    if prob_red is not None and prob_blue is not None:
        print(f" → {c['bright_red']}Red Win Probability : {prob_red*100:.1f}%{c['default']}")
        print(f" → {c['bright_blue']}Blue Win Probability: {prob_blue*100:.1f}%{c['default']}")
    print(f"{c['bright_yellow']}{line_sep}{c['default']}")

    # Feature differences
    print(f"{c['bright_cyan']}📊 MODEL INPUT VECTOR:{c['default']}")
    for k, v in features.items():
        if k == 'IsFiveRoundFight':
            value = 'Yes' if v == 1 else 'No'
            print(f"   {k:25}: {value}")
        elif isinstance(v, (int, float)):
            print(f"   {k:25}: {v: .3f}")
        else:
            print(f"   {k:25}: {v}")
    print(f"{c['bright_yellow']}{line_sep}{c['default']}" + "\n")

def print_prediction_result(result):
    """
    Pretty-print the result dictionary from UFCPredictor.predict() with detailed fighter stats using rich.
    """
    red = result['red_summary']
    blue = result['blue_summary']
    pred = result['prediction']
    prob_red = result['probability_red']
    prob_blue = result['probability_blue']
    features = result['feature_vector']
    red_odds, blue_odds = result['odds']

    header_text = Text("🏆 UFC FIGHT PREDICTION RESULT", style="bold yellow", justify="center")
    console.print(Panel(header_text, expand=True, border_style="magenta", box=ROUNDED))

    # Prediction result
    winner_color = "blue" if pred == 'Blue' else "red"
    winner_text = f"🏅 Predicted Winner: [bold {winner_color}]{'🔵 BLUE' if pred == 'Blue' else '🔴 RED'}[/]"

    prob_text = ""
    if prob_red is not None and prob_blue is not None:
        prob_text = f"\n→ [red]Red Win Probability[/]: {prob_red*100:.1f}%\n→ [blue]Blue Win Probability[/]: {prob_blue*100:.1f}%"

    console.print(
        Panel(winner_text + prob_text, border_style=winner_color, title="Prediction", expand=True),
        justify="center"
    )

    # Red corner summary
    print_corner_summary(
        corner=red,
        label=f"🔴 RED CORNER (Favorite): {red['Fighter']} ({red['Year']})",
        color="red",
        odds=red_odds
    )

    # Blue corner summary
    print_corner_summary(
        corner=blue,
        label=f"🔵 BLUE CORNER (Underdog): {blue['Fighter']} ({blue['Year']})",
        color="blue",
        odds=blue_odds
    )

    # Feature differences as a table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="dim", width=30)
    table.add_column("Value", justify="right")

    for k, v in features.items():
        if k == 'IsFiveRoundFight':
            value = 'Yes' if v == 1 else 'No'
        elif isinstance(v, (int, float)):
            value = f"{v:.3f}"
        else:
            value = str(v)
        table.add_row(k, value)

    console.print(
        Panel(table, border_style="bright_cyan", title="📊 MODEL INPUT VECTOR", expand=False),
        justify="center"
    )


