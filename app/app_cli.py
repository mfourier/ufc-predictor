import os
import sys
import logging
import pandas as pd
from rich import print, box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.table import Table
from rich.columns import Columns
from src.io_model import load_data
from src.predictor import UFCPredictor
from src.helpers import print_prediction_result, print_corner_summary
from src.config import pretty_model_name, file_model_name

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

        # Get fighter stats
        fighter_stats = predictor.get_fighter_stats(fighter, year)

        # Show summary using rich version
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

    console.print(Panel(
        header_text,
        border_style="magenta",
        box=box.DOUBLE,
        expand=True
    ))

    try:
        root_dir = get_project_path()
        fighters_path = os.path.join(root_dir, 'data', 'processed', 'fighters_df.csv')
        fighters_df = pd.read_csv(fighters_path)
        ufc_data = load_data("ufc_data")
        logger.info("✅ Data loaded successfully.")
    except Exception as e:
        logger.exception("❌ Error loading data")
        console.print(f"[bold red]❌ Error loading data: {e}[/]")
        return

    predictor = UFCPredictor(fighters_df, ufc_data)

    try:
        # Select model
        available_models = predictor.get_available_models()
        pretty_names = [pretty_model_name.get(code, code) for code in available_models]
        console.rule("[bold green]Select Prediction Model[/]")
        selected_pretty = select_from_list(pretty_names, "👉 Select model")
        model_name = file_model_name[selected_pretty]

        # Select weight class
        weightclasses = predictor.get_available_weightclasses()
        console.rule("[bold green]Select Weight Class[/]")
        weightclass = select_from_list(weightclasses, "👉 Select weight class")

        # Select fighters
        red_name, red_year = select_fighter(predictor, weightclass, "🔴 Red")
        if red_name is None:
            return

        blue_name, blue_year = select_fighter(predictor, weightclass, "🔵 Blue")
        if blue_name is None:
            return

        if red_name == blue_name and red_year == blue_year:
            console.print("[bold red]❌ Red and Blue fighters must be different.[/]")
            return

        # Five round fight
        is_five_round_fight = int(Confirm.ask("[bold cyan]👉 Is this a five round fight?[/]"))

        # Odds
        red_odds = get_float_input("👉 Enter Red odds (e.g., -100)")
        blue_odds = get_float_input("👉 Enter Blue odds (e.g., 200)")

        # Predict
        console.print("\n[bold cyan]🔮 Making prediction...[/]")
        result = predictor.predict(
            (red_name, red_year),
            (blue_name, blue_year),
            red_odds,
            blue_odds,
            is_five_round_fight,
            model_name
        )
        print_prediction_result(result)

        console.print(Panel("[bold green]🎉 Prediction complete! Thank you for using UFC Predictor CLI.[/]", expand=False))

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
