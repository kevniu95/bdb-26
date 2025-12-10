"""
LOCAL INFERENCE SCRIPT
======================
Use this after training on Kaggle to run predictions locally.

SETUP:
1. Train on Kaggle (runs main() in the notebook)
2. Download ./outputs/trained_models/ folder from Kaggle
3. Place in this directory: ./outputs/trained_models/
4. Run this script with test data

FILES NEEDED (from Kaggle training):
- model_fold1.pth, model_fold2.pth, ..., model_fold5.pth
- scaler_fold1.pkl, scaler_fold2.pkl, ..., scaler_fold5.pkl
- route_kmeans.pkl, route_scaler.pkl
- metadata.pkl
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm.auto import tqdm

# ============================================================================
# MODEL DEFINITION (must match training)
# ============================================================================


class JointSeqModel(nn.Module):
    """Your proven architecture - unchanged"""

    def __init__(self, input_dim, horizon):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.pool_ln = nn.LayerNorm(128)
        self.pool_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, 128))

        self.head = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, horizon * 2)
        )

    def forward(self, x):
        h, _ = self.gru(x)
        B = h.size(0)
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        out = self.head(ctx.squeeze(1))
        out = out.view(B, -1, 2)
        return torch.cumsum(out, dim=1)


# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================


def load_trained_models(models_dir="./data/kaggle"):
    """Load all trained models and artifacts from Kaggle training run."""

    models_dir = Path(models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models from: {models_dir}")
    print(f"Device: {device}\n")

    # Load metadata
    with open(models_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"📋 Metadata:")
    print(f"   N Folds: {metadata['n_folds']}")
    print(f"   Input Dim: {metadata['input_dim']}")
    print(f"   Horizon: {metadata['horizon']}")
    print(f"   Window Size: {metadata['window_size']}")
    print(f"   Mean RMSE: {metadata['mean_rmse']:.4f} ± {metadata['std_rmse']:.4f}\n")

    # Load route artifacts
    with open(models_dir / "route_kmeans.pkl", "rb") as f:
        route_kmeans = pickle.load(f)
    with open(models_dir / "route_scaler.pkl", "rb") as f:
        route_scaler = pickle.load(f)

    print("✓ Loaded route_kmeans and route_scaler")

    # Load models and scalers
    models, scalers = [], []
    for fold in range(1, metadata["n_folds"] + 1):
        # Load model
        model = JointSeqModel(metadata["input_dim"], metadata["horizon"])
        model_path = models_dir / f"model_fold{fold}.pth"
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
        model.to(device)
        model.eval()
        models.append(model)

        # Load scaler
        scaler_path = models_dir / f"scaler_fold{fold}.pkl"
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        scalers.append(scaler)

        print(f"✓ Loaded fold {fold}: {model_path.name}, {scaler_path.name}")

    print(f"\n🎉 All {metadata['n_folds']} models loaded successfully!\n")

    return models, scalers, route_kmeans, route_scaler, metadata, device


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================


def load_sample_from_training(n_plays, data_dir="./data/train"):
    """
    Load a sample of n_plays from training data for validation.

    Args:
        n_plays: Number of distinct plays to sample
        data_dir: Directory containing training data

    Returns:
        input_df, output_df: Sampled input and output DataFrames
    """
    data_dir = Path(data_dir)

    # Load a few weeks of training data
    print(f"📂 Loading training data sample ({n_plays} plays)...")
    input_files = sorted(list(data_dir.glob("input_2023_w*.csv")))[:4]  # First 4 weeks
    output_files = sorted(list(data_dir.glob("output_2023_w*.csv")))[:4]

    input_df = pd.concat([pd.read_csv(f) for f in input_files])
    output_df = pd.concat([pd.read_csv(f) for f in output_files])

    # Sample n_plays distinct plays
    plays = (
        input_df[["game_id", "play_id"]]
        .drop_duplicates()
        .sample(n=n_plays, random_state=42)
    )

    input_sample = input_df.merge(plays, on=["game_id", "play_id"])
    output_sample = output_df.merge(plays, on=["game_id", "play_id"])

    print(f"✓ Sampled input: {input_sample.shape}")
    print(f"✓ Sampled output: {output_sample.shape}\n")

    return input_sample, output_sample


def compute_rmse(predictions, actuals, ids_pred, output_df):
    """
    Compute RMSE between predictions and actual positions.

    Args:
        predictions: Numpy array of shape (N, horizon, 2) - ensemble predictions
        actuals: Dictionary mapping (game_id, play_id, nfl_id) to actual trajectories
        ids_pred: List of dicts with game_id, play_id, nfl_id for each prediction
        output_df: DataFrame with actual outputs

    Returns:
        rmse: Root mean squared error
    """
    squared_errors = []
    valid_points = 0

    for i, sid in enumerate(ids_pred):
        # Get actual trajectory
        actual = output_df[
            (output_df["game_id"] == sid["game_id"])
            & (output_df["play_id"] == sid["play_id"])
            & (output_df["nfl_id"] == sid["nfl_id"])
        ].sort_values("frame_id")

        if len(actual) == 0:
            continue

        # Get predicted trajectory (cumulative displacements)
        pred_trajectory = predictions[i]  # (horizon, 2)

        # Match lengths
        n_frames = min(len(actual), pred_trajectory.shape[0])

        # Compute errors
        for t in range(n_frames):
            pred_x = pred_trajectory[t, 0]
            pred_y = pred_trajectory[t, 1]
            actual_x = actual.iloc[t]["x"]
            actual_y = actual.iloc[t]["y"]

            squared_errors.append((pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2)
            valid_points += 1

    if valid_points == 0:
        return float("nan")

    rmse = np.sqrt(np.mean(squared_errors))
    return rmse


# ============================================================================
# RUN INFERENCE
# ============================================================================


def run_inference(
    data_source="test", n_sample=None, models_dir="./data/kaggle", data_dir="./data"
):
    """
    Run inference on test data or training samples using trained models.

    Args:
        data_source: Either "test" or "sample-{N}" (e.g., "sample-1000")
        n_sample: Number of plays to sample (alternative to data_source format)
        models_dir: Directory containing trained models from Kaggle
        data_dir: Directory containing data files

    Returns:
        submission: DataFrame ready for submission (if test data)
        rmse: RMSE score (if training sample)
    """
    from nfl_big_data_bowl_geometric_gnn_lb_58_9302db import prepare_sequences_geometric

    data_dir = Path(data_dir)

    # Parse data source
    is_training_sample = False

    if data_source.startswith("sample-"):
        is_training_sample = True
        n_sample = int(data_source.split("-")[1])
    elif n_sample is not None:
        is_training_sample = True

    # Load trained artifacts
    models, scalers, route_kmeans, route_scaler, metadata, device = load_trained_models(
        models_dir
    )

    # Load data based on source
    if is_training_sample:
        print(f"🎯 Running on TRAINING SAMPLE (n={n_sample} plays)")
        print("=" * 60)

        input_df, output_df = load_sample_from_training(n_sample, data_dir / "train")

        # Use output_df as template
        test_template = output_df[["game_id", "play_id", "nfl_id", "frame_id"]].copy()

    else:
        print("🎯 Running on TEST DATA")
        print("=" * 60)

        print("📂 Loading test data...")
        input_df = pd.read_csv(data_dir / "test_input.csv")
        test_template = pd.read_csv(data_dir / "test.csv")
        output_df = None

        print(f"✓ Test input: {input_df.shape}")
        print(f"✓ Test template: {test_template.shape}\n")
    print("   - prepare_sequences_geometric()")
    print("   - All feature engineering functions it depends on")
    print("   - See the training notebook for these functions\n")

    # This is the structure - you need to fill in with actual function:
    X_test, test_ids, geo_x, geo_y = prepare_sequences_geometric(
        test_input,
        test_template=test_template,
        is_training=False,
        window_size=metadata["window_size"],
        route_kmeans=route_kmeans,
        route_scaler=route_scaler,
    )
    """
    print("🔧 Preparing sequences...")
    
    
    print(f"✓ Created {len(X_test)} sequences\n")
    
    # Get last positions
    x_last = np.array([s[-1, 0] for s in X_test])
    y_last = np.array([s[-1, 1] for s in X_test])
    
    # Ensemble predictions
    print("🎯 Running ensemble predictions...")
    all_preds = []
    
    for fold, (model, scaler) in enumerate(zip(models, scalers), 1):
        # Scale features
        X_scaled = [scaler.transform(s) for s in X_test]
        X_tensor = torch.tensor(np.stack(X_scaled).astype(np.float32)).to(device)
        
        # Predict
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()
        
        all_preds.append(preds)
        print(f"  ✓ Fold {fold} predictions complete")
    
    # Average ensemble
    ens_preds = np.mean(all_preds, axis=0)
    print(f"✓ Ensemble complete (shape: {ens_preds.shape})\n")
    
    # Create submission
    print("📝 Creating submission...")
    rows = []
    H = ens_preds.shape[1]
    
    for i, sid in enumerate(tqdm(test_ids, desc="Building submission")):
        fids = test_template[
            (test_template['game_id'] == sid['game_id']) &
            (test_template['play_id'] == sid['play_id']) &
            (test_template['nfl_id'] == sid['nfl_id'])
        ]['frame_id'].sort_values().tolist()
        
        for t, fid in enumerate(fids):
            tt = min(t, H - 1)
            px = np.clip(x_last[i] + ens_preds[i, tt, 0], 0, 120)
            py = np.clip(y_last[i] + ens_preds[i, tt, 1], 0, 53.3)
            
            rows.append({
                'id': f"{sid['game_id']}_{sid['play_id']}_{sid['nfl_id']}_{fid}",
                'x': px,
                'y': py
            })
    
    submission = pd.DataFrame(rows)
    print(f"\n✓ Submission created: {len(submission)} rows\n")
    
    return submission
    """

    print(
        "💡 Once you copy the feature engineering functions, uncomment the code above"
    )
    print("   and this script will generate submissions locally!\n")

    return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Example usage
    submission = run_inference(
        test_input_path="./data/test_input.csv",
        test_template_path="./data/test.csv",
        models_dir="./data/kaggle",
    )

    if submission is not None:
        submission.to_csv("local_submission.csv", index=False)
        print("🎉 Saved local_submission.csv")
    else:
        print("\n📚 Next steps:")
        print("   1. Copy prepare_sequences_geometric() and dependencies from notebook")
        print("   2. Uncomment the inference code in run_inference()")
        print("   3. Re-run this script")
