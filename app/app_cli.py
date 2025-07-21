# Author: Maximiliano Lioi | License: MIT

import os
import sys
import logging
import pandas as pd
from rich import print, box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import IntPrompt, FloatPrompt, Confirm
from rich.columns import Columns
from src.io_model import load_data
from src.predictor import UFCPredictor
from src.helpers import print_prediction_result, print_corner_summary
from src.config import pretty_model_name
from src.metrics import evaluate_metrics, evaluate_cm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# ============================
# SIMULATE UFC FIGHT
# ============================

def simulate_ufc_fight(predictor):
    weightclasses = predictor.get_available_weightclasses()
    console.rule("[bold green]Select Weight Class[/]")
    weightclass = select_from_list(weightclasses, "👉 Select weight class")
    if weightclass is None:
        return  
    
    red_name, red_year = select_fighter(predictor, weightclass, "🔴 Red")
    if red_name is None:
        return

    blue_name, blue_year = select_fighter(predictor, weightclass, "🔵 Blue")
    if blue_name is None:
        return

    if red_name == blue_name and red_year == blue_year:
        console.print("[bold red]❌ Red and Blue fighters must be different.[/]")
        return

    is_five_round = int(Confirm.ask("👉 Is this a five-round fight?"))
    include_odds = Confirm.ask("👉 Do you want to include betting odds in the prediction?")

    show_model_performance_summary(predictor, include_odds)
    model_name = select_prediction_model(predictor, include_odds)
    if model_name is None:
        return

    red_odds = blue_odds = None
    if include_odds:
        red_odds = get_float_input("👉 Enter Red odds (e.g., -100)")
        blue_odds = get_float_input("👉 Enter Blue odds (e.g., 200)")

    console.print("\n[bold cyan]🔮 Making prediction...[/]")
    result = predictor.predict(
        red_id=(red_name, red_year),
        blue_id=(blue_name, blue_year),
        is_five_round_fight=is_five_round,
        model_name=model_name,
        red_odds=red_odds,
        blue_odds=blue_odds,
    )
    print_prediction_result(result)

    console.print(Panel(
        Text("🎉 Thank you for using UFC Fight Predictor! 💥🥋\n\nPress Enter to start a new prediction.", style="bold green", justify="center"),
        border_style="bright_green",
        box=box.DOUBLE,
        expand=True
    ))
    input()  # wait for Enter to continue



# ============================
# SIMULATE CUSTOM FIGHT
# ============================

def collect_fighter_input(corner_name, weight_class, weight_class_map):
    console.rule(f"[bold cyan]{corner_name} Fighter Input[/]")
    fighter = {}

    fighter['Fighter'] = input(f"👉 Enter {corner_name} fighter name (or type 'b' to go back): ").strip()
    if fighter['Fighter'].lower() in ['b', 'back']:
        return None
    fighter['Year'] = 2025
    fighter['WeightClass'] = weight_class
    fighter['WeightClassMap'] = weight_class_map[weight_class]

    fighter['Wins'] = get_int_input("Wins", default=0)
    if fighter['Wins'] is None: return None
    fighter['Losses'] = get_int_input("Losses", default=0)
    if fighter['Losses'] is None: return None
    fighter['Draws'] = get_int_input("Draws", default=0)
    if fighter['Draws'] is None: return None
    fighter['WinsByKO'] = get_int_input("Wins by KO/TKO", default=0)
    if fighter['WinsByKO'] is None: return None
    fighter['WinsBySubmission'] = get_int_input("Wins by Submission", default=0)
    if fighter['WinsBySubmission'] is None: return None

    fighter['Age'] = get_float_input("Age (years)")
    if fighter['Age'] is None: return None
    fighter['HeightCms'] = get_float_input("Height (cm)")
    if fighter['HeightCms'] is None: return None
    fighter['ReachCms'] = get_float_input("Reach (cm)")
    if fighter['ReachCms'] is None: return None

    fighter['TotalTitleBouts'] = get_int_input("Total Title Bouts", default=0)
    if fighter['TotalTitleBouts'] is None: return None
    fighter['CurrentWinStreak'] = get_int_input("Current Win Streak", default=0)
    if fighter['CurrentWinStreak'] is None: return None
    fighter['CurrentLoseStreak'] = get_int_input("Current Lose Streak", default=0)
    if fighter['CurrentLoseStreak'] is None: return None
    fighter['LongestWinStreak'] = get_int_input("Longest Win Streak", default=0)
    if fighter['LongestWinStreak'] is None: return None
    fighter['AvgSigStrLanded'] = get_float_input("Avg Significant Strikes Landed", )
    if fighter['AvgSigStrLanded'] is None: return None
    fighter['AvgSubAtt'] = get_float_input("Avg Submission Attempts", )
    if fighter['AvgSubAtt'] is None: return None
    fighter['AvgTDLanded'] = get_float_input("Avg Takedowns Landed", )
    if fighter['AvgTDLanded'] is None: return None

    stance_options = ['Orthodox', 'Southpaw', 'Switch']
    stance = select_from_list(stance_options, f"{corner_name} Stance")
    if stance is None: return None
    fighter['Stance'] = stance

    fighter['TotalFights'] = fighter['Wins'] + fighter['Losses'] + fighter['Draws']
    total_fights_safe = max(fighter['TotalFights'], 1)
    fighter['WinRatio'] = fighter['Wins'] / total_fights_safe
    fighter['FinishRate'] = (fighter['WinsByKO'] + fighter['WinsBySubmission']) / max(fighter['Wins'], 1)
    fighter['HeightReachRatio'] = fighter['HeightCms'] / max(fighter['ReachCms'], 1)
    fighter['KOPerFight'] = fighter['WinsByKO'] / total_fights_safe
    fighter['SubPerFight'] = fighter['WinsBySubmission'] / total_fights_safe

    return pd.Series(fighter)


def simulate_custom_fight(predictor):
    console.rule("[bold green]Simulate Custom Fight[/]")

    weight_class_map = {
        'Flyweight': 'Light',
        'Bantamweight': 'Light',
        'Featherweight': 'Light',
        'Lightweight': 'Light',
        'Welterweight': 'Medium',
        'Middleweight': 'Medium',
        'Light Heavyweight': 'Heavy',
        'Heavyweight': 'Heavy',
        "Women's Flyweight": 'Women',
        "Women's Strawweight": 'Women',
        "Women's Bantamweight": 'Women',
        "Women's Featherweight": 'Women',
    }

    weight_classes = list(weight_class_map.keys())
    weight_class = select_from_list(weight_classes, "👉 Select Weight Class")
    if weight_class is None:
        return

    red = collect_fighter_input("🔴 Red", weight_class, weight_class_map)
    if red is None:
        return

    blue = collect_fighter_input("🔵 Blue", weight_class, weight_class_map)
    if blue is None:
        return

    fight_stance = 'Closed Stance' if red['Stance'] == blue['Stance'] else 'Open Stance'
    red['Stance'] = blue['Stance'] = fight_stance

    is_five_round = int(Confirm.ask("👉 Is this a five-round fight?"))
    include_odds = Confirm.ask("👉 Do you want to include betting odds in the prediction?")

    show_model_performance_summary(predictor, include_odds)
    model_name = select_prediction_model(predictor, include_odds)
    if model_name is None:
        return

    red_odds = blue_odds = None
    if include_odds:
        red_odds = get_float_input("👉 Enter Red odds (e.g., -100)")
        if red_odds is None:
            return
        blue_odds = get_float_input("👉 Enter Blue odds (e.g., 200)")
        if blue_odds is None:
            return

    console.print("\n[bold cyan]🔮 Making prediction...[/]")
    result = predictor.predict(
        red_series=red,
        blue_series=blue,
        is_five_round_fight=is_five_round,
        model_name=model_name,
        red_odds=red_odds,
        blue_odds=blue_odds,
    )
    print_prediction_result(result)

    console.print(Panel(
        Text("🎉 Thank you for using UFC Fight Predictor! 💥🥋\n\nPress Enter to start a new prediction.", style="bold green", justify="center"),
        border_style="bright_green",
        box=box.DOUBLE,
        expand=True
    ))
    input()  # wait for Enter to continue


# ============================
# HELPER: SELECT PREDICTION MODEL
# ============================

def select_prediction_model(predictor, include_odds):
    unique_pretty_names = sorted(set(pretty_model_name.values()))
    console.rule("[bold green]Select Prediction Model[/]")
    selected_pretty = select_from_list(unique_pretty_names, "👉 Select model")
    if selected_pretty is None:
        return None

    for key, model in predictor.models.items():
        clean_name = model.name.replace(' (no_odds)', '').strip()
        if clean_name == selected_pretty and model.is_no_odds == (not include_odds):
            return key
    console.print(f"[bold red]❌ No model found for selection: {selected_pretty}[/]")
    return None

def select_from_list(options, prompt_text, allow_back=True):
    items = [f"[bold cyan][{idx}] {opt}[/]" for idx, opt in enumerate(options, 1)]
    console.print(Panel(f"[bold green]{prompt_text}[/]"))
    console.print(Columns(items, equal=True, expand=True))

    while True:
        back_text = " or [/][b](b)[/][bold yellow] to go back[/]" if allow_back else ""
        user_input = console.input(
            f"[bold yellow]{prompt_text} (1-{len(options)}){back_text}: "
        ).strip().lower()

        if allow_back and user_input in ['b', 'back']:
            return None
        if user_input.isdigit():
            selection = int(user_input)
            if 1 <= selection <= len(options):
                return options[selection - 1]
        console.print("[bold red]❌ Invalid selection. Please enter a number{}.[/]".format(
            " or 'b' to go back" if allow_back else ""))


def select_fighter(predictor, weightclass, corner_name):
    while True:
        fighters = predictor.get_fighters_by_weightclass(weightclass)
        console.rule(f"[bold cyan]{corner_name} Available Fighters[/]")
        fighter = select_from_list(fighters, f"👉 Select {corner_name} fighter")
        if fighter is None:
            return None, None

        years = sorted(int(y) for y in predictor.fighters_df[
            (predictor.fighters_df['Fighter'] == fighter) &
            (predictor.fighters_df['WeightClass'] == weightclass)
        ]['Year'].unique())

        if not years:
            logger.warning(f"No available years for {fighter} at {weightclass}.")
            console.print(f"[bold red]❌ No available years for {fighter} at {weightclass}.[/]")
            retry = Confirm.ask("🔁 Do you want to select another fighter?")
            if not retry:
                console.print(f"[bold yellow]👋 Cancelled {corner_name} fighter selection. Exiting.[/]")
                return None, None
            continue

        year_str = select_from_list([str(y) for y in years], f"👉 Select {corner_name} year")
        if year_str is None:
            return None, None
        year = int(year_str)

        fighter_stats = predictor.get_fighter_stats(fighter, year)
        print_corner_summary(
            corner=fighter_stats,
            label=f"✅ Selected {corner_name} fighter: {fighter} ({year})",
            color="cyan"
        )

        if Confirm.ask("✅ Confirm this selection?"):
            return fighter, year


def show_model_performance_summary(predictor, include_odds):
    console.rule("[bold green]Model Performance Summary[/]")
    console.print(f"[bold yellow]Showing metrics for models: {'WITH ODDS' if include_odds else 'NO ODDS'}[/]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1 Score", justify="right")
    table.add_column("ROC AUC", justify="right")
    table.add_column("Brier Score", justify="right")

    for key, model in predictor.models.items():
        if model.is_no_odds == (not include_odds):
            clean_name = model.name.replace(' (no_odds)', '').strip()
            metrics = model.metrics or {}
            acc = metrics.get('Accuracy', None)
            precision = metrics.get('Precision', None)
            recall = metrics.get('Recall', None)
            f1 = metrics.get('F1 Score', None)
            roc_auc = metrics.get('ROC AUC', None)
            brier = metrics.get('Brier Score', None)

            acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
            precision_str = f"{precision * 100:.1f}%" if precision is not None else "N/A"
            recall_str = f"{recall * 100:.1f}%" if recall is not None else "N/A"
            f1_str = f"{f1 * 100:.1f}%" if f1 is not None else "N/A"
            roc_auc_str = f"{roc_auc * 100:.1f}%" if roc_auc is not None else "N/A"
            brier_str = f"{brier:.3f}" if brier is not None else "N/A"

            table.add_row(clean_name, acc_str, precision_str, recall_str, f1_str, roc_auc_str, brier_str)

    console.print(table)

    # Add recommendation message
    if include_odds:
        console.print("[bold green]💡 Recommended:[/] Neural Network is recommended for predictions with odds, selected for its accuracy and high F1 score, reducing bias against Blue corner predictions.")
    else:
        console.print("[bold green]💡 Recommended:[/] Logistic Regression is recommended for predictions without odds, selected for its accuracy and high F1 score, reducing bias against Blue corner predictions.")

def get_int_input(prompt_text, default=None):
    while True:
        user_input = console.input(f"[bold yellow]{prompt_text}{' (default ' + str(default) + ')' if default is not None else ''} or [/][b](b)[/][bold yellow] to go back[/]: ").strip().lower()
        if user_input in ['b', 'back']:
            return None
        if user_input == '' and default is not None:
            return default
        if user_input.isdigit():
            return int(user_input)
        console.print("[bold red]❌ Invalid number. Please enter an integer or 'b' to go back.[/]")

def get_float_input(prompt_text):
    while True:
        user_input = console.input(f"[bold yellow]{prompt_text} or [/][b](b)[/][bold yellow] to go back[/]: ").strip().lower()
        if user_input in ['b', 'back']:
            return None
        try:
            return float(user_input)
        except ValueError:
            console.print("[bold red]❌ Invalid number. Please enter a valid float or 'b' to go back.[/]")

def get_project_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main():
    title = Text("🏆 UFC FIGHT PREDICTOR CLI 🏆", style="bold yellow", justify="center")
    subtitle = Text("Predict your fights using ML! 💥🥋", style="italic cyan", justify="center")
    banner_text = title + "\n" + subtitle

    console.print("[bold green]🥊 Welcome to UFC Fight Predictor 🥊[/]\n")
    console.print("[bold green]📦 Loading data and models, please wait...[/]\n")

    try:
        root_dir = get_project_path()
        fighters_path = os.path.join(root_dir, 'data', 'processed', 'fighters_df.csv')
        fighters_df = pd.read_csv(fighters_path)
        ufc_data = load_data("ufc_data")
        ufc_data_no_odds = load_data("ufc_data_no_odds")
        logger.info("✅ Data loaded successfully.")
        console.print("[bold green]✅ Data and models loaded successfully! Ready to go.\n")
    except Exception as e:
        logger.exception("❌ Error loading data")
        console.print(f"[bold red]❌ Error loading data: {e}[/]")
        return

    predictor = UFCPredictor(fighters_df, ufc_data, ufc_data_no_odds)

    try:
        while True:
            console.print(Panel(
                banner_text,
                border_style="magenta",
                box=box.DOUBLE,
                padding=(1, 4),
                expand=True 
            ))

            mode = select_from_list([
            "Simulate UFC Fight",
            "Simulate Custom Fight",
            "Exit"
            ], "👉 Select Mode", allow_back=False)

            if mode == "Simulate UFC Fight":
                simulate_ufc_fight(predictor)
            elif mode == "Simulate Custom Fight":
                simulate_custom_fight(predictor)
            else:
                console.print("\n[bold yellow]👋 Exit requested. Goodbye![/]")
                sys.exit(0)
    except KeyboardInterrupt:
        logger.info("👋 Exit requested by user.")
        console.print("\n[bold yellow]👋 Exit requested. Goodbye![/]")
        sys.exit(0)
    except EOFError:
        logger.info("👋 End of input detected.")
        console.print("\n[bold yellow]👋 End of input detected. Goodbye![/]")
        sys.exit(0)
    except Exception as e:
        logger.exception("❌ Error during prediction")
        console.print(f"[bold red]❌ Error during prediction: {e}[/]")


if __name__ == "__main__":
    main()
