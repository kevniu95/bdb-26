import pandas as pd
import numpy as np
import os
import logging


logger = logging.getLogger(__name__)


def load_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv("test.csv")
    test_input = pd.read_csv("test_input.csv")
    return test, test_input


def load_train_data(
    through_week: int = 18,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_df = load_input_data(through_week)
    output_df = load_output_data(through_week)
    return input_df, output_df


def load_suppl_data() -> pd.DataFrame:
    suppl_data = pd.read_csv("data/supplementary_data.csv")
    return suppl_data


def load_input_data(through_week: int = 18) -> pd.DataFrame:
    if os.path.exists("data/personal/input_df.pkl"):
        logger.info("Loading input_df from pickle")
        input_df = pd.read_pickle("data/personal/input_df.pkl")
        if through_week:
            input_df = input_df[input_df["week"] <= through_week]
        return input_df
    input_dfs = []
    for i in range(1, through_week + 1):
        weekly_df = pd.read_csv(f"data/train/input_2023_w{i:02}.csv")
        weekly_df["week"] = i
        input_dfs.append(weekly_df)
    input_df = pd.concat(input_dfs, ignore_index=True)
    return input_df


def load_output_data(through_week: int = 18) -> pd.DataFrame:
    if os.path.exists("data/personal/output_df.pkl"):
        logger.info("Loading output_df from pickle")
        output_df = pd.read_pickle("data/personal/output_df.pkl")
        if through_week:
            output_df = output_df[output_df["week"] <= through_week]
        return output_df
    output_dfs = []
    for i in range(1, through_week + 1):
        weekly_df = pd.read_csv(f"data/train/output_2023_w{i:02}.csv")
        weekly_df["week"] = i
        output_dfs.append(weekly_df)
    output_df = pd.concat(output_dfs, ignore_index=True)
    return output_df


def normalize_input_fields(input_df: pd.DataFrame) -> pd.DataFrame:

    # Normalize field direction and other stuff

    # 1. Flip so all plays go left to right
    # 2. Shift all plays so LOS is at x = 0

    # Note made: absolute_yardline_number is NOT from perspective of offense
    # Rather it is absolute position on field, and always increases from left to right
    # All x values are relative to scale of 0-120 yards, with middle of field at 60 yards``

    L = 120  # field length in yards
    W = 53.3  # field width in yards

    input_df["absolute_yardline_number_std"] = np.where(
        input_df["play_direction"] == "left",
        L - input_df["absolute_yardline_number"],
        input_df["absolute_yardline_number"],
    )

    input_df["x_std"] = (
        np.where(input_df["play_direction"] == "left", L - input_df["x"], input_df["x"])
        - input_df["absolute_yardline_number_std"]
    )

    input_df["y_std"] = np.where(
        input_df["play_direction"] == "left", W - input_df["y"], input_df["y"]
    )

    input_df["ball_land_x_std"] = (
        np.where(
            input_df["play_direction"] == "left",
            L - input_df["ball_land_x"],
            input_df["ball_land_x"],
        )
        - input_df["absolute_yardline_number_std"]
    )

    input_df["ball_land_y_std"] = np.where(
        input_df["play_direction"] == "left",
        W - input_df["ball_land_y"],
        input_df["ball_land_y"],
    )

    input_df["o_std"] = np.where(
        input_df["play_direction"] == "left", (input_df["o"] + 180) % 360, input_df["o"]
    )

    input_df["dir_std"] = np.where(
        input_df["play_direction"] == "left",
        (input_df["dir"] + 180) % 360,
        input_df["dir"],
    )
    return input_df


def normalize_output_fields(
    output_df: pd.DataFrame, norm_helper: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalize field direction and other stuff

    norm_helper is the original input set data with play direction and absolute_yardline_number
    This is minimal info needed from input data to normalize output data
    """
    output_df = output_df.merge(norm_helper, on=["game_id", "play_id"], how="inner")

    L = 120  # field length in yards
    W = 53.3  # field width in yards

    output_df["absolute_yardline_number_std"] = np.where(
        output_df["play_direction"] == "left",
        L - output_df["absolute_yardline_number"],
        output_df["absolute_yardline_number"],
    )

    output_df["x_std"] = (
        np.where(
            output_df["play_direction"] == "left", L - output_df["x"], output_df["x"]
        )
        - output_df["absolute_yardline_number_std"]
    )

    output_df["y_std"] = np.where(
        output_df["play_direction"] == "left", W - output_df["y"], output_df["y"]
    )
    return output_df


def unnormalize_x_y(
    x_std: pd.Series,
    y_std: pd.Series,
    absolute_yardline_number_std: pd.Series,
    play_direction: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Convert standardized coordinates back to original field coordinates.

    Args:
        x_std: Standardized x coordinates (relative to LOS)
        y_std: Standardized y coordinates (left-to-right normalized)
        absolute_yardline_number_std: Standardized yardline number
        play_direction: 'left' or 'right'

    Returns:
        Tuple of (x_original, y_original)
    """
    L = 120  # field length in yards
    W = 53.3  # field width in yards

    x_unnorm = np.where(
        play_direction == "left",
        L - (x_std + absolute_yardline_number_std),
        x_std + absolute_yardline_number_std,
    )

    y_unnorm = np.where(play_direction == "left", W - y_std, y_std)

    return x_unnorm, y_unnorm


def unnormalize_predictions(
    predictions_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convenience function to unnormalize a full predictions DataFrame.

    Args:
        predictions_df: DataFrame with ['x_std', 'y_std'] columns
        metadata_df: DataFrame with play metadata (must have same length as predictions_df)
                     Must contain: ['absolute_yardline_number_std', 'play_direction']

    Returns:
        DataFrame with ['x', 'y'] columns added (unnormalized coordinates)
    """
    result = predictions_df.copy()

    x_unnorm, y_unnorm = unnormalize_x_y(
        predictions_df["x_std"],
        predictions_df["y_std"],
        metadata_df["absolute_yardline_number_std"],
        metadata_df["play_direction"],
    )

    result["x"] = x_unnorm
    result["y"] = y_unnorm

    return result
