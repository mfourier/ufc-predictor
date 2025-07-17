import os
import sys
import logging
import pandas as pd
from src.data import UFCData
from src.io_model import load_data
from src.predictor import UFCPredictor
from src.helpers import print_prediction_result, print_header
from src.config import pretty_model_name, file_model_name, colors

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_from_list(options, prompt_text):
    for idx, option in enumerate(options, 1):
        print(f"  [{idx}] {option}")
    while True:
        user_input = input(f"{prompt_text} (1-{len(options)}): ").strip()
        if user_input.isdigit() and 1 <= int(user_input) <= len(options):
            return options[int(user_input) - 1]
        else:
            print(f"{colors['bright_red']}❌ Invalid selection. Please enter a number between 1 and {len(options)}.{colors['default']}")

def select_fighter(predictor, weightclass, corner_name):
    while True:
        fighters = predictor.get_fighters_by_weightclass(weightclass)
        print(f"\n{corner_name} Available fighters:")
        fighter = select_from_list(fighters, f"👉 Select {corner_name} fighter")

        years = sorted(int(y) for y in predictor.fighters_df[
            (predictor.fighters_df['Fighter'] == fighter) &
            (predictor.fighters_df['WeightClass'] == weightclass)
        ]['Year'].unique())

        if not years:
            logger.warning(f"No available years for {fighter} at weight class {weightclass}.")
            print(f"{colors['bright_red']}❌ No available years for {fighter} at weight class {weightclass}.{colors['default']}")
            retry = input("🔁 Do you want to select another fighter? (y/n): ").strip().lower()
            if retry != 'y':
                print(f"{colors['bright_yellow']}👋 Cancelled {corner_name} fighter selection. Exiting.{colors['default']}")
                return None, None
            continue

        year = int(select_from_list([str(y) for y in years], f"👉 Select {corner_name} year"))
        print(f"{colors['bright_green']}✅ Selected {corner_name} fighter: {fighter} ({year}){colors['default']}")
        return fighter, year

def get_float_input(prompt_text):
    while True:
        value = input(f"{prompt_text}: ").strip()
        try:
            return float(value)
        except ValueError:
            print(f"{colors['bright_red']}❌ Invalid number. Please enter a valid float (e.g., -120, 200).{colors['default']}")

def get_project_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main():
    print_header("UFC FIGHT PREDICTOR CLI", color='bright_yellow')

    try:
        root_dir = get_project_path()
        fighters_path = os.path.join(root_dir, 'data', 'processed', 'fighters_df.csv')
        fighters_df = pd.read_csv(fighters_path)
        ufc_data = load_data("ufc_data")
        logger.info("✅ Data loaded successfully.")
    except Exception as e:
        logger.exception("❌ Error loading data")
        print(f"{colors['bright_red']}❌ Error loading data: {e}{colors['default']}")
        return

    predictor = UFCPredictor(fighters_df, ufc_data)

    try:
        available_models = predictor.get_available_models()
        pretty_names = [pretty_model_name.get(code, code) for code in available_models]
        print("\n📊 Available models:")
        selected_pretty = select_from_list(pretty_names, "👉 Select model")
        model_name = file_model_name[selected_pretty]

        weightclasses = predictor.get_available_weightclasses()
        print("\n🥊 Available weight classes:")
        weightclass = select_from_list(weightclasses, "👉 Select weight class")

        red_name, red_year = select_fighter(predictor, weightclass, "🔴 Red")
        if red_name is None:
            return
        blue_name, blue_year = select_fighter(predictor, weightclass, "🔵 Blue")
        if blue_name is None:
            return

        # Check if Red and Blue are the same
        if red_name == blue_name and red_year == blue_year:
            print(f"{colors['bright_red']}❌ Red and Blue fighters must be different.{colors['default']}")
            return

        red_odds = get_float_input("👉 Enter Red odds (e.g., -100)")
        blue_odds = get_float_input("👉 Enter Blue odds (e.g., 200)")

        result = predictor.predict(
            (red_name, red_year),
            (blue_name, blue_year),
            red_odds,
            blue_odds,
            model_name
        )
        print_prediction_result(result)

        print(f"{colors['bright_green']}🎉 Prediction complete! Thank you for using UFC Predictor CLI.{colors['default']}")

    except KeyboardInterrupt:
        logger.info("👋 Exit requested by user.")
        print(f"\n{colors['bright_yellow']}👋 Exit requested. Goodbye!{colors['default']}")
        sys.exit(0)
    except EOFError:
        logger.info("👋 End of input detected.")
        print(f"\n{colors['bright_yellow']}👋 End of input detected. Goodbye!{colors['default']}")
        sys.exit(0)
    except Exception as e:
        logger.exception("❌ Error during prediction")
        print(f"{colors['bright_red']}❌ Error during prediction: {e}{colors['default']}")

if __name__ == "__main__":
    main()
