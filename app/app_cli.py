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


def select_from_list(options, prompt_text):
    items = [f"[bold cyan][{idx}] {opt}[/]" for idx, opt in enumerate(options, 1)]
    console.print(Panel(f"[bold green]{prompt_text}[/]"))
    console.print(Columns(items, equal=True, expand=True))

    while True:
        try:
            selection = IntPrompt.ask(f"[bold yellow]{prompt_text} (1-{len(options)})[/]")
            if 1 <= selection <= len(options):
                return options[selection - 1]
            else:
                console.print("[bold red]❌ Invalid selection. Please enter a valid number.[/]")
        except ValueError:
            console.print("[bold red]❌ Please enter a number.[/]")


def select_fighter(predictor, weightclass, corner_name):
    while True:
        fighters = predictor.get_fighters_by_weightclass(weightclass)
        console.rule(f"[bold cyan]{corner_name} Available Fighters[/]")
        fighter = select_from_list(fighters, f"👉 Select {corner_name} fighter")

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

        year = int(select_from_list([str(y) for y in years], f"👉 Select {corner_name} year"))

        fighter_stats = predictor.get_fighter_stats(fighter, year)
        print_corner_summary(
            corner=fighter_stats,
            label=f"✅ Selected {corner_name} fighter: {fighter} ({year})",
            color="cyan"
        )

        if Confirm.ask("✅ Confirm this selection?"):
            return fighter, year


def get_float_input(prompt_text):
    while True:
        try:
            return FloatPrompt.ask(f"[bold yellow]{prompt_text}[/]")
        except ValueError:
            console.print("[bold red]❌ Invalid number. Please enter a valid float.[/]")


def get_project_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    header_text = Text("🏆  UFC FIGHT PREDICTOR CLI  🏆", style="bold yellow", justify="center")
    console.print(Panel(header_text, border_style="magenta", box=box.DOUBLE, expand=True))

    try:
        root_dir = get_project_path()
        fighters_path = os.path.join(root_dir, 'data', 'processed', 'fighters_df.csv')
        fighters_df = pd.read_csv(fighters_path)
        ufc_data = load_data("ufc_data")
        ufc_data_no_odds = load_data("ufc_data_no_odds")
        logger.info("✅ Data loaded successfully.")
    except Exception as e:
        logger.exception("❌ Error loading data")
        console.print(f"[bold red]❌ Error loading data: {e}[/]")
        return

    predictor = UFCPredictor(fighters_df, ufc_data, ufc_data_no_odds)

    try:
        weightclasses = predictor.get_available_weightclasses()
        console.rule("[bold green]Select Weight Class[/]")
        weightclass = select_from_list(weightclasses, "👉 Select weight class")

        red_name, red_year = select_fighter(predictor, weightclass, "🔴 Red")
        if red_name is None:
            return

        blue_name, blue_year = select_fighter(predictor, weightclass, "🔵 Blue")
        if blue_name is None:
            return

        if red_name == blue_name and red_year == blue_year:
            console.print("[bold red]❌ Red and Blue fighters must be different.[/]")
            return

        is_five_round_fight = int(Confirm.ask("[bold cyan]👉 Is this a five-round fight?[/]"))
        include_odds = Confirm.ask("[bold cyan]👉 Do you want to include betting odds in the prediction? (improves model accuracy)[/]")

       # Show Model Performance Summary
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
        
        # Show available models (pretty names only)
        unique_pretty_names = sorted(set(pretty_model_name.values()))
        console.rule("[bold green]Select Prediction Model[/]")

        selected_pretty = select_from_list(unique_pretty_names, "👉 Select model")

        # Find the correct key by matching model.name and is_no_odds
        model_name = None
        for key, model in predictor.models.items():
            clean_name = model.name.replace(' (no_odds)', '').strip()
            if clean_name == selected_pretty and model.is_no_odds == (not include_odds):
                model_name = key
                break

        if model_name is None:
            console.print(f"[bold red]❌ No model found for selection: {selected_pretty}[/]")
            return

        red_odds = blue_odds = None
        if include_odds:
            red_odds = get_float_input("👉 Enter Red odds (e.g., -100)")
            blue_odds = get_float_input("👉 Enter Blue odds (e.g., 200)")

        console.print("\n[bold cyan]🔮 Making prediction...[/]")
        result = predictor.predict(
            (red_name, red_year),
            (blue_name, blue_year),
            is_five_round_fight,
            model_name,
            red_odds,
            blue_odds
        )
        print_prediction_result(result)

        console.print(Panel(
            Text("🎉 Prediction complete! Run again to compare other fights or models.", style="bold green", justify="center"),
            border_style="bold green",
            box=box.DOUBLE,
            expand=True
        ))

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
