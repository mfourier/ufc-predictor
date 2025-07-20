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
import builtins
plain_print = builtins.print
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

# ANSI color codes
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_corner_summary_notebooks(corner, label, odds=None, color=RESET):
    """
    Pretty-print summary stats for a fighter corner in notebook (with ANSI colors).
    """
    lines = [
        f"{color}{BOLD}{label}{RESET}",
        f"{color}" + "-" * 70 + f"{RESET}",
        f"Record               : {corner.get('Record', 'N/A')}",
        f"Weight Class        : {corner.get('WeightClass', 'N/A')} | Stance: {corner.get('Stance', 'N/A')}",
    ]

    if odds is not None:
        lines.append(f"Odds                 : {odds}")

    lines.extend([
        f"Total Fights        : {corner.get('TotalFights', 'N/A')} | Title Bouts: {corner.get('TotalTitleBouts', 'N/A')}",
        f"Age                : {corner.get('Age', 'N/A')}",
        f"Height             : {corner.get('HeightCms', 'N/A')} cm | Reach: {corner.get('ReachCms', 'N/A')} cm",
        f"Height/Reach Ratio : {corner.get('HeightReachRatio', 0):.3f}",
        f"Win Ratio          : {corner.get('WinRatio', 0):.3f} | Finish Rate: {corner.get('FinishRate', 0):.3f}",
        f"KO per Fight       : {corner.get('KOPerFight', 0):.3f} | Sub per Fight: {corner.get('SubPerFight', 0):.3f}",
        f"Avg Sig Str Landed : {corner.get('AvgSigStrLanded', 0):.3f}",
        f"Avg Sub Att        : {corner.get('AvgSubAtt', 0):.3f} | Avg TD Landed: {corner.get('AvgTDLanded', 0):.3f}",
    ])

    for line in lines:
        plain_print(line)
    plain_print("")

def print_prediction_result_notebooks(result):
    """
    Pretty-print the result dictionary from UFCPredictor.predict() in notebook (with ANSI colors).
    """
    red = result['red_summary']
    blue = result['blue_summary']
    pred = result['prediction']
    prob_red = result['probability_red']
    prob_blue = result['probability_blue']
    features = result['feature_vector']

    red_odds, blue_odds = result.get('odds', (None, None))
    line_sep = "-" * 70

    # Header
    plain_print(f"\n{YELLOW}{BOLD}🏆 UFC FIGHT PREDICTION RESULT{RESET}")
    plain_print(f"{YELLOW}{line_sep}{RESET}")

    print_corner_summary_notebooks(red, f"🔴 RED CORNER (Favorite): {red['Fighter']} ({red['Year']})", red_odds, RED)
    print_corner_summary_notebooks(blue, f"🔵 BLUE CORNER (Underdog): {blue['Fighter']} ({blue['Year']})", blue_odds, BLUE)

    # Prediction result
    winner_color = BLUE if pred == 'Blue' else RED
    winner_text = f"{winner_color}{BOLD}{'🔵 BLUE' if pred == 'Blue' else '🔴 RED'}{RESET}"
    plain_print(f"🏅 Predicted Winner: {winner_text}")

    if prob_red is not None and prob_blue is not None:
        plain_print(f"{RED}→ Red Win Probability : {prob_red*100:.1f}%{RESET}")
        plain_print(f"{BLUE}→ Blue Win Probability: {prob_blue*100:.1f}%{RESET}")

    plain_print(f"{YELLOW}{line_sep}{RESET}")

    # Feature differences
    plain_print(f"{CYAN}📊 MODEL INPUT VECTOR:{RESET}")
    for k, v in features.items():
        if k == 'IsFiveRoundFight':
            value = 'Yes' if v == 1 else 'No'
        elif isinstance(v, (int, float)):
            value = f"{v:.3f}"
        else:
            value = str(v)
        plain_print(f"   {k:25}: {value}")
    plain_print(f"{YELLOW}{line_sep}{RESET}\n")


def print_corner_summary(corner, label, color, odds=None):
    """
    Pretty-print summary stats for a fighter corner using rich.
    """

    lines = [
        f"[bold]Record[/]               : {corner.get('Record', 'N/A')}",
        f"[bold]Weight Class[/]        : {corner.get('WeightClass', 'N/A')} | Stance: {corner.get('Stance', 'N/A')}",
    ]

    if odds is not None:
        lines.append(f"[bold]Odds[/]                 : {odds}")

    lines.extend([
        f"[bold]Total Fights[/]         : {corner.get('TotalFights', 'N/A')} | Title Bouts: {corner.get('TotalTitleBouts', 'N/A')}",
        f"[bold]Age[/]                 : {corner.get('Age', 'N/A')}",
        f"[bold]Height[/]             : {corner.get('HeightCms', 'N/A')} cm | Reach: {corner.get('ReachCms', 'N/A')} cm",
        f"[bold]Height/Reach Ratio[/] : {corner.get('HeightReachRatio', 0):.3f}",
        f"[bold]Win Ratio[/]          : {corner.get('WinRatio', 0):.3f} | Finish Rate: {corner.get('FinishRate', 0):.3f}",
        f"[bold]KO per Fight[/]       : {corner.get('KOPerFight', 0):.3f} | Sub per Fight: {corner.get('SubPerFight', 0):.3f}",
        f"[bold]Avg Sig Str Landed[/] : {corner.get('AvgSigStrLanded', 0):.3f}",
        f"[bold]Avg Sub Att[/]        : {corner.get('AvgSubAtt', 0):.3f} | Avg TD Landed: {corner.get('AvgTDLanded', 0):.3f}"
    ])

    content = "\n".join(lines)

    panel = Panel(
        content,
        title=f"{label}",
        title_align="center",
        border_style=color
    )

    console.print(panel)


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

    # ✅ Handle case when 'odds' might not be present
    red_odds, blue_odds = result.get('odds', (None, None))

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

    # Build table with 4 columns: Feature, Value
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="dim", width=20)
    table.add_column("Value", justify="right", width=10)
    table.add_column("Feature", style="dim", width=20)
    table.add_column("Value", justify="right", width=10)

    items = list(features.items())

    for i in range(0, len(items), 2):
        # Left pair
        left_key, left_val = items[i]
        left_val_str = 'Yes' if left_key == 'IsFiveRoundFight' and left_val == 1 else (
            f"{left_val:.3f}" if isinstance(left_val, (int, float)) else str(left_val)
        )

        # Right pair (if exists)
        if i + 1 < len(items):
            right_key, right_val = items[i + 1]
            right_val_str = 'Yes' if right_key == 'IsFiveRoundFight' and right_val == 1 else (
                f"{right_val:.3f}" if isinstance(right_val, (int, float)) else str(right_val)
            )
        else:
            right_key, right_val_str = "", ""

        table.add_row(left_key, left_val_str, right_key, right_val_str)

    console.print(
        Panel(table, border_style="bright_cyan", title="📊 MODEL INPUT VECTOR", expand=False),
        justify="center"
    )
