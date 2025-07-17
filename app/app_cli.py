import pandas as pd
from src.data import UFCData
from src.io_model import load_data
from src.predictor import UFCPredictor
from src.helpers import print_prediction_result, print_header
from src.config import pretty_model_name, file_model_name

def select_from_list(options, prompt_text):
    """Present options enumerated and get a valid index from user."""
    for idx, option in enumerate(options, 1):
        print(f"  [{idx}] {option}")
    while True:
        user_input = input(f"{prompt_text} (1-{len(options)}): ").strip()
        if user_input.isdigit() and 1 <= int(user_input) <= len(options):
            return options[int(user_input) - 1]
        else:
            print(f"❌ Invalid selection. Please enter a number between 1 and {len(options)}.")

def select_fighter(predictor, weightclass, corner_name):
    """Select fighter and year with validated inputs."""
    fighters = predictor.get_fighters_by_weightclass(weightclass)
    print(f"\n{corner_name} Available fighters:")
    fighter = select_from_list(fighters, f"👉 Select {corner_name} fighter")

    years = sorted(int(y) for y in predictor.fighters_df[
        (predictor.fighters_df['Fighter'] == fighter) & 
        (predictor.fighters_df['WeightClass'] == weightclass)
    ]['Year'].unique())

    if not years:
        raise ValueError(f"❌ No available years for {fighter} at weight class {weightclass}.")

    year = int(select_from_list([str(y) for y in years], f"👉 Select {corner_name} year"))
    return fighter, year

def get_float_input(prompt_text):
    """Get a valid float input from user."""
    while True:
        value = input(f"{prompt_text}: ").strip()
        try:
            return float(value)
        except ValueError:
            print("❌ Invalid number. Please enter a valid float (e.g., -120, 200).")

def main():
    """
    UFC Fight Predictor CLI:
    - Loads models and data.
    - Lets user select model, weightclass, fighters, and odds.
    - Runs and displays prediction.
    """
    print_header("UFC FIGHT PREDICTOR CLI", color='bright_yellow')

    try:
        fighters_df = pd.read_csv("data/processed/fighters_df.csv")
        ufc_data = load_data("ufc_data")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    predictor = UFCPredictor(fighters_df, ufc_data)

    # ─────────────────────────────────────────────
    # Select model
    # ─────────────────────────────────────────────
    available_models = predictor.get_available_models()
    pretty_names = [pretty_model_name.get(code, code) for code in available_models]

    print("\n📊 Available models:")
    selected_pretty = select_from_list(pretty_names, "👉 Select model")
    model_name = file_model_name[selected_pretty]

    # ─────────────────────────────────────────────
    # Select weight class
    # ─────────────────────────────────────────────
    weightclasses = predictor.get_available_weightclasses()
    print("\n🥊 Available weight classes:")
    weightclass = select_from_list(weightclasses, "👉 Select weight class")

    # ─────────────────────────────────────────────
    # Select Red fighter
    # ─────────────────────────────────────────────
    red_name, red_year = select_fighter(predictor, weightclass, "🔴 Red")

    # ─────────────────────────────────────────────
    # Select Blue fighter
    # ─────────────────────────────────────────────
    blue_name, blue_year = select_fighter(predictor, weightclass, "🔵 Blue")

    # ─────────────────────────────────────────────
    # Enter odds
    # ─────────────────────────────────────────────
    red_odds = get_float_input("👉 Enter Red odds (e.g., -100)")
    blue_odds = get_float_input("👉 Enter Blue odds (e.g., 200)")

    # ─────────────────────────────────────────────
    # Run prediction
    # ─────────────────────────────────────────────
    try:
        result = predictor.predict(
            (red_name, red_year),
            (blue_name, blue_year),
            red_odds,
            blue_odds,
            model_name
        )
        print_prediction_result(result)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()
