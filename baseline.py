import pandas as pd
import numpy as np
from src.prepData import (
    load_train_data,
    load_suppl_data,
)
from src.evaluate import evaluate_predictions, predict

pd.set_option("display.max_columns", None)


def generate_predictions_last_known_position(
    test: pd.DataFrame, test_input_std: pd.DataFrame
) -> pd.DataFrame:
    last_positions = (
        test_input_std[test_input_std["player_to_predict"] == True]
        .sort_values("frame_id")
        .groupby(["game_id", "play_id", "nfl_id"])[["x_std", "y_std"]]
        .last()
        .reset_index()
    )

    # Merge to create one prediction per row in test
    predictions = test[["game_id", "play_id", "nfl_id", "frame_id"]].merge(
        last_positions, on=["game_id", "play_id", "nfl_id"], how="left"
    )

    return predictions[["game_id", "play_id", "nfl_id", "x_std", "y_std"]]


def generate_predictions_last_qb_position(
    test: pd.DataFrame, test_input_std: pd.DataFrame
) -> pd.DataFrame:
    last_qb_positions = (
        test_input_std[test_input_std["player_role"] == "Passer"]
        .sort_values("frame_id")
        .groupby(["game_id", "play_id"])[["x_std", "y_std"]]
        .last()
        .reset_index()
    )

    predictions = test[["game_id", "play_id", "nfl_id", "frame_id"]].merge(
        last_qb_positions,
        on=["game_id", "play_id"],
        how="left",
    )

    predictions["x_std"] = predictions["x_std"].fillna(60.0)  # Center of field
    predictions["y_std"] = predictions["y_std"].fillna(26.67)

    return predictions[["game_id", "play_id", "nfl_id", "x_std", "y_std"]]


def generate_predictions_fixed_midfield(
    test: pd.DataFrame, test_input_std: pd.DataFrame
) -> pd.DataFrame:
    last_qb_positions = (
        test_input_std[test_input_std["player_role"] == "Passer"]
        .sort_values("frame_id")
        .groupby(["game_id", "play_id"])[["x_std", "y_std"]]
        .last()
        .reset_index()
    )

    predictions = test[["game_id", "play_id", "nfl_id", "frame_id"]].merge(
        last_qb_positions,
        on=["game_id", "play_id"],
        how="left",
    )

    predictions["x_std"] = 60
    predictions["y_std"] = 26.67

    return predictions[["game_id", "play_id", "nfl_id", "x_std", "y_std"]]


def generate_predictions_standardized_velocity(
    test: pd.DataFrame, test_input_std: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate predictions in STANDARDIZED coordinate space using velocity extrapolation.

    This improves on the baseline by:
    1. Decomposing last known dir + speed into v_x and v_y
    2. Extrapolating position forward in time based on velocity

    Args:
        test: What to predict (game_id, play_id, nfl_id, frame_id, ...)
        test_input_std: Historical data in STANDARDIZED coordinates

    Returns:
        DataFrame with columns: [game_id, play_id, nfl_id, x_std, y_std]
        One row per prediction needed
    """

    # Get last known position + velocity for players to predict
    last_positions = (
        test_input_std[test_input_std["player_to_predict"] == True]
        .sort_values("frame_id")
        .groupby(["game_id", "play_id", "nfl_id"])[
            ["x_std", "y_std", "dir_std", "s", "frame_id"]
        ]
        .last()
        .reset_index()
    )

    # Decompose direction and speed into velocity components
    # NFL convention: 0° = North (+y), 90° = East (+x), clockwise positive
    last_positions["v_x"] = last_positions["s"] * np.sin(
        np.radians(last_positions["dir_std"])
    )
    last_positions["v_y"] = last_positions["s"] * np.cos(
        np.radians(last_positions["dir_std"])
    )

    # Merge to get frame info
    predictions = test[["game_id", "play_id", "nfl_id", "frame_id"]].merge(
        last_positions,
        on=["game_id", "play_id", "nfl_id"],
        how="left",
        suffixes=("_pred", "_last"),
    )

    # Calculate time delta (assuming 10 Hz frame rate = 0.1 sec per frame)
    predictions["dt"] = (
        predictions["frame_id_pred"] - predictions["frame_id_last"]
    ) * 0.1

    # Extrapolate position using constant velocity
    predictions["x_std"] = predictions["x_std"] + predictions["v_x"] * predictions["dt"]
    predictions["x_std"] = predictions["x_std"].clip(0.0, 120.0)  # Cap at field length

    predictions["y_std"] = predictions["y_std"] + predictions["v_y"] * predictions["dt"]
    predictions["y_std"] = predictions["y_std"].clip(0.0, 53.33)  # Cap at field width

    return predictions[["game_id", "play_id", "nfl_id", "x_std", "y_std"]]


if __name__ == "__main__":
    print("Loading data...")
    input_df, output_df = load_train_data(through_week=1)
    suppl_data = load_suppl_data()

    print(f"Loaded {len(input_df)} input rows, {len(output_df)} output rows")
    print(
        f"Unique plays: {input_df[['game_id', 'play_id']].drop_duplicates().shape[0]}"
    )

    # Evaluate on all data
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    rmse, play_details = evaluate_predictions(
        predict_fn=predict,
        generate_predictions_standardized=generate_predictions_last_known_position,
        input_df=input_df,
        output_df=output_df,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"{'='*60}")
