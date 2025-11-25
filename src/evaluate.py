"""
Evaluation utilities for scoring predictions using Kaggle's RMSE metric.
"""

import pandas as pd
import numpy as np
import polars as pl
from typing import Callable, Tuple
from src.prepData import (
    normalize_input_fields,
    unnormalize_predictions,
)


def predict(
    generate_predictions_standardized: Callable,
    test: pl.DataFrame,
    test_input: pl.DataFrame,
) -> pd.DataFrame:
    """
    Template structure:
    1. Convert inputs
    2. Standardize/preprocess
    3. Generate predictions (in standardized space)
    4. Unstandardize/postprocess
    5. Return in original coordinates
    """

    # ===== 1. INPUT CONVERSION =====
    test_pd = test.to_pandas() if isinstance(test, pl.DataFrame) else test
    test_input_pd = (
        test_input.to_pandas() if isinstance(test_input, pl.DataFrame) else test_input
    )

    # ===== 2. STANDARDIZE =====
    test_input_std = normalize_input_fields(test_input_pd)

    # Store metadata needed for unstandardization (IMPORTANT!)
    metadata = test_input_std[
        [
            "game_id",
            "play_id",
            "nfl_id",
            "absolute_yardline_number_std",
            "play_direction",
        ]
    ].drop_duplicates()

    # ===== 3. MODEL PREDICTIONS (standardized space) =====
    predictions_std = generate_predictions_standardized(test_pd, test_input_std)
    # This returns: DataFrame with ['game_id', 'play_id', 'nfl_id', 'x_std', 'y_std']

    # ===== 4. UNSTANDARDIZE =====
    predictions_with_meta = predictions_std.merge(
        metadata, on=["game_id", "play_id", "nfl_id"]
    )
    predictions_original = unnormalize_predictions(
        predictions_std, predictions_with_meta
    )

    # ===== 5. RETURN =====
    return predictions_original[["x", "y"]]


def calculate_rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """
    Calculate RMSE according to Kaggle's formula:

    RMSE = sqrt( (1/2N) * sum((x_true - x_pred)^2 + (y_true - y_pred)^2) )

    Args:
        y_true: DataFrame with columns ['x', 'y'] - ground truth
        y_pred: DataFrame with columns ['x', 'y'] - predictions

    Returns:
        RMSE score
    """
    assert len(y_true) == len(
        y_pred
    ), "Predictions and ground truth must have same length"

    N = len(y_true)

    # Calculate squared errors for x and y
    x_squared_errors = (y_true["x"].values - y_pred["x"].values) ** 2
    y_squared_errors = (y_true["y"].values - y_pred["y"].values) ** 2

    # Kaggle formula: 1/(2N) * sum(x_errors + y_errors)
    rmse = np.sqrt((1.0 / (2 * N)) * (x_squared_errors.sum() + y_squared_errors.sum()))

    return rmse


def evaluate_predictions(
    predict_fn: Callable,
    generate_predictions_standardized: Callable,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate predictions play-by-play and calculate overall RMSE.

    Args:
        predict_fn: Your predict function that takes (test, test_input) and returns predictions
        input_df: Training input data (what the model sees)
        output_df: Ground truth output data (what we're trying to predict)
        verbose: Whether to print progress

    Returns:
        Tuple of (overall_rmse, detailed_results_df)
    """
    # Get unique plays
    plays = (
        output_df[["game_id", "play_id"]]
        .drop_duplicates()
        .sort_values(["game_id", "play_id"])
    )

    all_predictions = []
    all_ground_truth = []
    play_rmses = []

    for idx, (game_id, play_id) in enumerate(plays.itertuples(index=False)):
        if verbose and idx % 10 == 0:
            print(
                f"Processing play {idx+1}/{len(plays)}: Game {game_id}, Play {play_id}"
            )

        # Get test data (what we need to predict)
        test_output = output_df[
            (output_df["game_id"] == game_id) & (output_df["play_id"] == play_id)
        ].copy()

        # Get input data (what the model can see)
        test_input = input_df[
            (input_df["game_id"] == game_id) & (input_df["play_id"] == play_id)
        ].copy()

        if len(test_input) == 0 or len(test_output) == 0:
            if verbose:
                print(f"  Warning: Skipping play {game_id}-{play_id} (no data)")
            continue

        try:
            # Convert to polars (your predict function expects polars)
            test_pl = pl.from_pandas(test_output)
            test_input_pl = pl.from_pandas(test_input)

            # Get predictions
            predictions_pl = predict_fn(
                generate_predictions_standardized, test_pl, test_input_pl
            )

            # Convert back to pandas
            if isinstance(predictions_pl, pl.DataFrame):
                predictions_pd = predictions_pl.to_pandas()
            else:
                predictions_pd = predictions_pl

            # Ensure we have the right columns
            assert (
                "x" in predictions_pd.columns and "y" in predictions_pd.columns
            ), "Predictions must have 'x' and 'y' columns"

            # Calculate RMSE for this play
            ground_truth = test_output[["x", "y"]].reset_index(drop=True)
            preds = predictions_pd[["x", "y"]].reset_index(drop=True)

            play_rmse = calculate_rmse(ground_truth, preds)
            play_rmses.append(
                {
                    "game_id": game_id,
                    "play_id": play_id,
                    "rmse": play_rmse,
                    "n_predictions": len(preds),
                }
            )

            # Store for overall calculation
            all_predictions.append(preds)
            all_ground_truth.append(ground_truth)

        except Exception as e:
            if verbose:
                print(f"  Error on play {game_id}-{play_id}: {e}")
            continue

    # Calculate overall RMSE
    if len(all_predictions) == 0:
        raise ValueError("No valid predictions were made!")

    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_ground_truth = pd.concat(all_ground_truth, ignore_index=True)

    overall_rmse = calculate_rmse(combined_ground_truth, combined_predictions)

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total plays evaluated: {len(play_rmses)}")
        print(f"Total predictions: {len(combined_predictions)}")
        print(f"Overall RMSE: {overall_rmse:.4f}")
        print(f"{'='*60}")

        # Show per-play statistics
        play_results = pd.DataFrame(play_rmses)
        print(f"\nPer-play RMSE statistics:")
        print(f"  Mean: {play_results['rmse'].mean():.4f}")
        print(f"  Median: {play_results['rmse'].median():.4f}")
        print(f"  Std: {play_results['rmse'].std():.4f}")
        print(f"  Min: {play_results['rmse'].min():.4f}")
        print(f"  Max: {play_results['rmse'].max():.4f}")

    return overall_rmse, pd.DataFrame(play_rmses)


def validate_train_test_split(
    predict_fn: Callable,
    train_input: pd.DataFrame,
    train_output: pd.DataFrame,
    test_input: pd.DataFrame,
    test_output: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Validate model on both train and test sets.

    Returns:
        Dictionary with train_rmse, test_rmse, and detailed results
    """
    print("Evaluating on training set...")
    train_rmse, train_details = evaluate_predictions(
        predict_fn, train_input, train_output, verbose
    )

    print("\nEvaluating on test set...")
    test_rmse, test_details = evaluate_predictions(
        predict_fn, test_input, test_output, verbose
    )

    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_details": train_details,
        "test_details": test_details,
    }
