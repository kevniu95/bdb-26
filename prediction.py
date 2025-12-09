import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

from src.prepData import (
    load_train_data,
    normalize_input_fields,
    normalize_output_fields,
)

print("Loading data...")
input_df, output_df = load_train_data()
# input_df.to_pickle("data/personal/input_df.pkl")
# output_df.to_pickle("data/personal/output_df.pkl")

print(f"Loaded {len(input_df)} input rows, {len(output_df)} output rows")
print(f"Unique plays: {input_df[['game_id', 'play_id']].drop_duplicates().shape[0]}")

input_df = normalize_input_fields(input_df)
norm_helper = input_df[
    ["game_id", "play_id", "play_direction", "absolute_yardline_number"]
].drop_duplicates()
output_df = normalize_output_fields(output_df, norm_helper)

distinct_plays = input_df[["game_id", "play_id"]].drop_duplicates()
distinct_plays.sort_values(["game_id", "play_id"]).head(3)

# Get max frame_id from input_df for each play (throw_frame_id baseline)
input_max_frames = (
    input_df.groupby(["game_id", "play_id"])[
        ["frame_id", "ball_land_x_std", "ball_land_y_std"]
    ]
    .max()
    .reset_index()
    .rename(columns={"frame_id": "throw_frame_id"})
)

# Get max frame_id from output_df for each play (throw_land_frame_id baseline)
output_max_frames = (
    output_df.groupby(["game_id", "play_id"])[["frame_id"]]
    .max()
    .reset_index()
    .rename(columns={"frame_id": "throw_land_frame_id"})
)

# Combine both into baseline frame info
baseline_frame_info = input_max_frames.merge(
    output_max_frames, on=["game_id", "play_id"], how="outer"
)

print(f"Baseline frame info shape: {baseline_frame_info.shape}")
print(f"Unique plays: {baseline_frame_info.shape[0]}")
baseline_frame_info.head(2)

# Create all play-level features
qb_frame = input_df[input_df["player_role"] == "Passer"]
if qb_frame[["game_id", "play_id"]].drop_duplicates().shape[0] < len(distinct_plays):
    print(
        f"Warning: fewer plays with QB ({qb_frame[['game_id', 'play_id']].drop_duplicates().shape[0]}) than original plays ({len(distinct_plays)})"
    )

# Get QB max frame for plays with a passer
qb_max_frame = (
    qb_frame.groupby(["game_id", "play_id", "nfl_id", "player_role"])["frame_id"]
    .max()
    .reset_index()
)

# Find plays without a passer
plays_with_qb = qb_max_frame[["game_id", "play_id"]].drop_duplicates()
plays_without_qb = (
    distinct_plays.merge(
        plays_with_qb, on=["game_id", "play_id"], how="left", indicator=True
    )
    .query('_merge == "left_only"')
    .drop(columns=["_merge"])
)

# For plays without a passer, use the overall max frame_id
if len(plays_without_qb) > 0:
    print(
        f"Found {len(plays_without_qb)} plays without a Passer. Using overall max frame_id."
    )

    missing_max_frames = (
        input_df.merge(plays_without_qb, on=["game_id", "play_id"])
        .groupby(["game_id", "play_id"])["frame_id"]
        .max()
        .reset_index()
    )

    # Add placeholder columns for nfl_id and player_role
    missing_max_frames["nfl_id"] = None
    missing_max_frames["player_role"] = None

    # Combine with QB frames
    qb_max_frame = pd.concat([qb_max_frame, missing_max_frames], ignore_index=True)

# Join back to input_df to get the full row data
qb_rows = pd.merge(
    input_df,
    qb_max_frame,
    on=["game_id", "play_id", "nfl_id", "frame_id", "player_role"],
    how="inner",
)

# Start with qb_rows
qb_sub = qb_rows.copy()

# Calculate derived features
qb_sub["qb_throw_distance"] = np.sqrt(
    (qb_sub["ball_land_x_std"] - qb_sub["x_std"]) ** 2
    + (qb_sub["ball_land_y_std"] - qb_sub["y_std"]) ** 2
)
qb_sub["qb_ball_dir"] = (
    90
    - np.degrees(
        np.arctan2(
            qb_sub["ball_land_y_std"] - qb_sub["y_std"],
            qb_sub["ball_land_x_std"] - qb_sub["x_std"],
        )
    )
) % 360
qb_sub["qb_direction_diff"] = (
    qb_sub["o_std"] - qb_sub["qb_ball_dir"] + 180
) % 360 - 180  # difference between -180 and 180

# Rename frame_id to be QB-specific
qb_sub.rename(columns={"frame_id": "throw_frame_id"}, inplace=True)

# Drop player_to_predict column (not needed for QB)
qb_sub = qb_sub.drop(columns=["player_to_predict"])

# Rename QB kinematic fields to have qb_ prefix
qb_kinematic_fields_rename = {
    "x_std": "qb_x_std",
    "y_std": "qb_y_std",
    "o_std": "qb_o_std",
    "dir_std": "qb_dir_std",
    "s": "qb_s",
    "a": "qb_a",
}
qb_sub = qb_sub.rename(columns=qb_kinematic_fields_rename)

qb_sub = qb_sub.drop(columns=["ball_land_x_std", "ball_land_y_std"])

qb_sub.head(3)

# Create all play-level features
qb_frame = input_df[input_df["player_role"] == "Passer"]
if qb_frame[["game_id", "play_id"]].drop_duplicates().shape[0] < len(distinct_plays):
    print(
        f"Warning: fewer plays with QB ({qb_frame[['game_id', 'play_id']].drop_duplicates().shape[0]}) than original plays ({len(distinct_plays)})"
    )

# Get QB max frame for plays with a passer
qb_max_frame = (
    qb_frame.groupby(["game_id", "play_id", "nfl_id", "player_role"])["frame_id"]
    .max()
    .reset_index()
)

# Find plays without a passer
plays_with_qb = qb_max_frame[["game_id", "play_id"]].drop_duplicates()
plays_without_qb = (
    distinct_plays.merge(
        plays_with_qb, on=["game_id", "play_id"], how="left", indicator=True
    )
    .query('_merge == "left_only"')
    .drop(columns=["_merge"])
)

# For plays without a passer, use the overall max frame_id
if len(plays_without_qb) > 0:
    print(
        f"Found {len(plays_without_qb)} plays without a Passer. Using overall max frame_id."
    )

    missing_max_frames = (
        input_df.merge(plays_without_qb, on=["game_id", "play_id"])
        .groupby(["game_id", "play_id"])["frame_id"]
        .max()
        .reset_index()
    )

    # Add placeholder columns for nfl_id and player_role
    missing_max_frames["nfl_id"] = None
    missing_max_frames["player_role"] = None

    # Combine with QB frames
    qb_max_frame = pd.concat([qb_max_frame, missing_max_frames], ignore_index=True)

# Join back to input_df to get the full row data
qb_rows = pd.merge(
    input_df,
    qb_max_frame,
    on=["game_id", "play_id", "nfl_id", "frame_id", "player_role"],
    how="inner",
)

# Start with qb_rows
qb_sub = qb_rows.copy()

# Calculate derived features
qb_sub["qb_throw_distance"] = np.sqrt(
    (qb_sub["ball_land_x_std"] - qb_sub["x_std"]) ** 2
    + (qb_sub["ball_land_y_std"] - qb_sub["y_std"]) ** 2
)
qb_sub["qb_ball_dir"] = (
    90
    - np.degrees(
        np.arctan2(
            qb_sub["ball_land_y_std"] - qb_sub["y_std"],
            qb_sub["ball_land_x_std"] - qb_sub["x_std"],
        )
    )
) % 360
qb_sub["qb_direction_diff"] = (
    qb_sub["o_std"] - qb_sub["qb_ball_dir"] + 180
) % 360 - 180  # difference between -180 and 180

# Rename frame_id to be QB-specific
qb_sub.rename(columns={"frame_id": "throw_frame_id"}, inplace=True)

# Drop player_to_predict column (not needed for QB)
qb_sub = qb_sub.drop(columns=["player_to_predict"])

# Rename QB kinematic fields to have qb_ prefix
qb_kinematic_fields_rename = {
    "x_std": "qb_x_std",
    "y_std": "qb_y_std",
    "o_std": "qb_o_std",
    "dir_std": "qb_dir_std",
    "s": "qb_s",
    "a": "qb_a",
}
qb_sub = qb_sub.rename(columns=qb_kinematic_fields_rename)

qb_sub = qb_sub.drop(columns=["ball_land_x_std", "ball_land_y_std"])

qb_sub.head(3)

qb_features = [
    "qb_x_std",
    "qb_y_std",
    "qb_s",
    "qb_a",
    "qb_dir_std",
    "qb_o_std",
    "qb_throw_distance",
    "qb_ball_dir",
]

play_level_features = baseline_frame_info.merge(
    qb_sub[["game_id", "play_id"] + qb_features], how="left", on=["game_id", "play_id"]
)


def impute_qb_features_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing QB features using ball trajectory (always available)
    This is 'safe' because ball_land_x/y are inputs, not targets
    """
    mask = df["qb_x_std"].isnull()

    if mask.sum() > 0:
        # Proxy: assume QB was ~10 yards behind ball landing
        df.loc[mask, "qb_x_std"] = df.loc[mask, "ball_land_x_std"] - 10
        df.loc[mask, "qb_y_std"] = 26.7  # assume center of field

        # Proxy: assume QB was stationary (conservative)
        df.loc[mask, "qb_s"] = 0.0
        df.loc[mask, "qb_a"] = 0.0

        # Throw distance from imputed position
        df.loc[mask, "qb_throw_distance"] = np.sqrt(
            (df.loc[mask, "ball_land_x_std"] - df.loc[mask, "qb_x_std"]) ** 2
            + (df.loc[mask, "ball_land_y_std"] - df.loc[mask, "qb_y_std"]) ** 2
        )

        # Proxy: QB facing ball direction
        df.loc[mask, "qb_o_std"] = (
            90
            - np.degrees(
                np.arctan2(
                    df.loc[mask, "ball_land_y_std"] - df.loc[mask, "qb_y_std"],
                    df.loc[mask, "ball_land_x_std"] - df.loc[mask, "qb_x_std"],
                )
            )
        ) % 360
        df.loc[mask, "qb_dir_std"] = df.loc[mask, "qb_o_std"]

        df.loc[mask, "qb_ball_dir"] = (
            90
            - np.degrees(
                np.arctan2(
                    df.loc[mask, "ball_land_y_std"] - df.loc[mask, "qb_y_std"],
                    df.loc[mask, "ball_land_x_std"] - df.loc[mask, "qb_x_std"],
                )
            )
        ) % 360

    return df


# Apply BEFORE split
play_level_features = impute_qb_features_safe(play_level_features)

x_data = baseline_frame_info[["game_id", "play_id", "throw_frame_id"]].merge(
    input_df[input_df["player_to_predict"] == True],
    left_on=["game_id", "play_id", "throw_frame_id"],
    right_on=["game_id", "play_id", "frame_id"],
    how="inner",
)
player_level_features = [
    "player_height",
    "player_weight",
    "player_birth_date",
    "player_position",
    "player_side",
    "player_role",
    "x_std",
    "y_std",
    "o_std",
    "dir_std",
    "s",
    "a",
]
x_data = x_data[["game_id", "play_id", "nfl_id"] + player_level_features].copy()
x_data = x_data.merge(play_level_features, on=["game_id", "play_id"])


def height_to_inches(col):
    # col: pandas Series of "6-1" strings
    split_vals = col.str.split("-", expand=True)
    feet = split_vals[0].astype(float)
    inches = split_vals[1].astype(float)
    return feet * 12 + inches


x_data["height_in"] = height_to_inches(x_data["player_height"])
# Age in years (super rough)
x_data["birth_year"] = pd.to_datetime(x_data["player_birth_date"]).dt.year


x_data["dx_ball"] = x_data["ball_land_x_std"] - x_data["x_std"]
x_data["dy_ball"] = x_data["ball_land_y_std"] - x_data["y_std"]

x_data["dist_ball"] = (
    np.sqrt(x_data["dx_ball"] ** 2 + x_data["dy_ball"] ** 2) + 1e-6
)  # avoid divide-by-zero
x_data["angle_to_ball"] = (
    90 - np.degrees(np.arctan2(x_data["dy_ball"], x_data["dx_ball"]))
) % 360


def angle_diff(a, b):
    # a, b in degrees
    return ((a - b + 180) % 360) - 180


x_data["angle_to_ball_minus_dir"] = angle_diff(
    x_data["angle_to_ball"], x_data["dir_std"]
)
# similarly for orientation if you want:
x_data["angle_to_ball_minus_o"] = angle_diff(x_data["angle_to_ball"], x_data["o_std"])

# Encode angles as sin/cos
for col in [
    "dir_std",
    "o_std",
    "qb_o_std",
    "qb_dir_std",
    "qb_ball_dir",
    "angle_to_ball",
    "angle_to_ball_minus_dir",
    "angle_to_ball_minus_o",
]:
    rad = np.deg2rad(x_data[col])
    x_data[col + "_sin"] = np.sin(rad)
    x_data[col + "_cos"] = np.cos(rad)


# Calculate speed in directions
dir_rad = np.deg2rad(x_data["dir_std"])
x_data["s_x_std"] = x_data["s"] * np.sin(dir_rad)
x_data["s_y_std"] = x_data["s"] * np.cos(dir_rad)


ux = x_data["dx_ball"] / x_data["dist_ball"]
uy = x_data["dy_ball"] / x_data["dist_ball"]

x_data["ux"] = ux
x_data["uy"] = uy

# v_parallel = vx * ux + vy * uy   # scalar, can be negative
x_data["s_parallel"] = x_data["s_x_std"] * ux + x_data["s_y_std"] * uy
x_data["s_perp"] = x_data["s_x_std"] * (-uy) + x_data["s_y_std"] * (ux)


x_data.sort_values(["game_id", "play_id", "nfl_id"], inplace=True)
# x_data.loc[(x_data['game_id'] == 2023090700) & (x_data['play_id'] == 194),
#            ['game_id','play_id','nfl_id',"dir_std",'x_std','y_std','ball_land_x_std','ball_land_y_std',
#             'dx_ball','dy_ball','dist_ball','angle_to_ball','s','s_x_std','s_y_std','ux','uy','s_parallel','s_perp']]


y_data = output_df.merge(
    baseline_frame_info[["game_id", "play_id"]], on=["game_id", "play_id"]
)

y_data.sort_values(["game_id", "play_id", "nfl_id", "frame_id"], inplace=True)


def hybrid_trajectory_interpolation(x_data, y_data, frame_rate=10, blend_factor=0.5):
    """
    Hybrid: blend velocity projection (early) with ball-directed (late)
    blend_factor: 0 = pure velocity, 1 = pure ball-directed
    """
    results = []

    for idx, row in x_data.iterrows():
        if idx % 10000 == 0:
            print(f"Processing row {idx}/{len(x_data)}")
        gid = row["game_id"]
        pid = row["play_id"]
        nid = row["nfl_id"]

        x_throw = row["x_std"]
        y_throw = row["y_std"]
        vx = row["s_x_std"]
        vy = row["s_y_std"]
        x_land = row["ball_land_x_std"]
        y_land = row["ball_land_y_std"]
        throw_frame = row["throw_frame_id"]

        traj_frames = y_data[
            (y_data["game_id"] == gid)
            & (y_data["play_id"] == pid)
            & (y_data["nfl_id"] == nid)
        ].sort_values("frame_id")

        if traj_frames.empty:
            continue

        frame_ids = traj_frames["frame_id"].values
        n_frames = len(frame_ids)

        for i, fid in enumerate(frame_ids):
            dt = (fid) / frame_rate
            t_norm = i / max(n_frames - 1, 1)  # 0 to 1

            # Velocity projection
            x_vel = x_throw + vx * dt
            y_vel = y_throw + vy * dt

            # Ball-directed interpolation
            x_ball = x_throw + t_norm * (x_land - x_throw)
            y_ball = y_throw + t_norm * (y_land - y_throw)

            # Blend: early frames favor velocity, late frames favor ball
            alpha = t_norm * blend_factor
            x_hybrid = (1 - alpha) * x_vel + alpha * x_ball
            y_hybrid = (1 - alpha) * y_vel + alpha * y_ball

            results.append(
                {
                    "game_id": gid,
                    "play_id": pid,
                    "nfl_id": nid,
                    "frame_id": fid,
                    "x_std_hybrid": x_hybrid,
                    "y_std_hybrid": y_hybrid,
                }
            )

    return pd.DataFrame(results)


# Generate hybrid trajectories
hybrid_traj = hybrid_trajectory_interpolation(x_data, y_data, blend_factor=0.7)
y_with_hybrid = y_data.merge(
    hybrid_traj, on=["game_id", "play_id", "nfl_id", "frame_id"]
)

y_with_hybrid.shape

import numpy as np


def calculate_kaggle_rmse(df):
    """
    Calculate RMSE per Kaggle's formula
    df should have: x_std, y_std (actual), x_std_hybrid, y_std_hybrid (predicted)
    """
    # Calculate squared errors per frame
    squared_errors = (df["x_std"] - df["x_std_hybrid"]) ** 2 + (
        df["y_std"] - df["y_std_hybrid"]
    ) ** 2

    # RMSE = sqrt(mean of squared distances)
    rmse = np.sqrt(squared_errors.mean())

    return rmse


# Calculate overall RMSE
overall_rmse = calculate_kaggle_rmse(y_with_hybrid)
print(f"\n{'='*50}")
print(f"🏈 Hybrid Baseline RMSE: {overall_rmse:.4f} yards")
print(f"{'='*50}\n")

# # Calculate per-frame RMSE (to see if error grows over time)
# frame_rmse = y_with_hybrid.groupby('frame_id').apply(
#     lambda g: np.sqrt(((g['x_std'] - g['x_std_hybrid'])**2 +
#                        (g['y_std'] - g['y_std_hybrid'])**2).mean())
# ).reset_index(name='rmse')

# print("RMSE by frame:")
# print(frame_rmse.head(15))

# # Calculate per-play RMSE (to identify hardest plays)
# play_rmse = y_with_hybrid.groupby(['game_id', 'play_id']).apply(
#     lambda g: np.sqrt(((g['x_std'] - g['x_std_hybrid'])**2 +
#                        (g['y_std'] - g['y_std_hybrid'])**2).mean())
# ).reset_index(name='rmse')

# print(f"\nPlay-level RMSE statistics:")
# print(play_rmse['rmse'].describe())
# print(f"\nWorst 5 plays:")
# print(play_rmse.nlargest(5, 'rmse'))

y_with_hybrid["target_dx"] = y_with_hybrid["x_std_hybrid"] - y_with_hybrid["x_std"]
y_with_hybrid["target_dy"] = y_with_hybrid["y_std_hybrid"] - y_with_hybrid["y_std"]

y_with_hybrid.head(10)
y_data = y_with_hybrid[
    ["game_id", "play_id", "nfl_id", "frame_id", "target_dx", "target_dy"]
].copy()

interaction_features = [
    "x_std",
    "y_std",
    "s_x_std",
    "s_y_std",
    "height_in",
    "dist_ball",
    "s_parallel",
]

inv_numeric_features = [
    # Predicted player features
    "height_in",
    "player_weight",
    "birth_year",
    # Predicted player kinematics
    "x_std",
    "y_std",
    "s_x_std",
    "s_y_std",
    "a",  # if present
    "dir_std_sin",
    "dir_std_cos",
    "o_std_sin",
    "o_std_cos",
    # QB kinematics
    "qb_x_std",
    "qb_y_std",
    "qb_s",
    "qb_a",
    "qb_o_std_sin",
    "qb_o_std_cos",
    "qb_dir_std_sin",
    "qb_dir_std_cos",
    # Throw features - global
    "throw_frame_id",
    "throw_land_frame_id",
    "ball_land_x_std",
    "ball_land_y_std",
    # Time of throw - needs QB kinematics
    "qb_throw_distance",
    "qb_ball_dir_sin",
    "qb_ball_dir_cos",
    # Ball-related features
    "dx_ball",
    "dy_ball",
    "dist_ball",
    "angle_to_ball_sin",
    "angle_to_ball_cos",
    "angle_to_ball_minus_dir_sin",
    "angle_to_ball_minus_dir_cos",
    "angle_to_ball_minus_o_sin",
    "angle_to_ball_minus_o_cos",
    "s_parallel",
    "s_perp",
]

inv_categorical_features = [
    "player_position",
    "player_side",
    "player_role",
]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


preproc_invariant = ColumnTransformer(
    transformers=[
        ("num", "passthrough", inv_numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            inv_categorical_features,
        ),
    ]
)

preproc_invariant.fit(x_data[inv_numeric_features + inv_categorical_features])


import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PlayDataset(Dataset):
    def __init__(
        self,
        x_data,
        y_data,
        interaction_features,
        inv_numeric_features,
        inv_categorical_features,
        preproc_invariant,
        device="cpu",
    ):
        """
        x_data: throw-frame dataframe (one row per (game, play, nfl_id) at throw)
        y_data: output dataframe with (game, play, nfl_id, frame_id, target_dx, target_dy)
        """
        self.device = device
        self.interaction_features = interaction_features
        self.inv_numeric_features = inv_numeric_features
        self.inv_categorical_features = inv_categorical_features
        self.preproc_invariant = preproc_invariant

        # Build list of plays
        self.plays = []
        self.samples = []
        for (gid, pid), play_df_all in tqdm(x_data.groupby(["game_id", "play_id"])):
            play_df = play_df_all.sort_values("nfl_id").reset_index(drop=True)
            nfl_ids = play_df["nfl_id"].tolist()

            # Gather output rows for each player
            frames_per_player = []
            targets_per_player = []
            T_max = 0

            for nid in nfl_ids:
                out_rows = y_data.query(
                    "game_id == @gid and play_id == @pid and nfl_id == @nid"
                ).sort_values("frame_id")
                if out_rows.empty:
                    continue
                frames = out_rows["frame_id"].to_numpy()
                targets = out_rows[["target_dx", "target_dy"]].to_numpy(dtype="float32")
                frames_per_player.append(frames)
                targets_per_player.append(targets)
                T_max = max(T_max, len(frames))

            if len(frames_per_player) == 0:
                continue

            # Normalize time 0..1 using max length in this play
            # Here we just use frame index within each player's sequence
            # (you can also use true time in seconds if you prefer)
            t_norm = torch.linspace(0.0, 1.0, steps=T_max, dtype=torch.float32)

            # We'll pad targets to (N, T_max, 2), with mask
            N = len(targets_per_player)
            targets_tensor = torch.zeros(N, T_max, 2, dtype=torch.float32)
            mask = torch.zeros(N, T_max, dtype=torch.bool)

            for i, targ in enumerate(targets_per_player):
                Ti = targ.shape[0]
                targets_tensor[i, :Ti, :] = torch.from_numpy(targ)
                mask[i, :Ti] = True

            # Store info for this play
            self.plays.append(
                {
                    "gid": gid,
                    "pid": pid,
                    "play_df": play_df,
                    "targets": targets_tensor,
                    "mask": mask,
                    "t_norm": t_norm,
                }
            )

            X_pair, X_inv = self._build_pairwise_and_invariant(play_df)
            self.samples.append((X_pair, X_inv, t_norm, targets_tensor, mask))
            # self.samples.append({"X_pair": X_pair, "X_inv": X_inv, "t_norm": t_norm, "targets": targets_tensor, "mask": mask})

    def __len__(self):
        return len(self.samples)

    def _build_pairwise_and_invariant(
        self, play_df: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- pairwise grid (F_int, N, N) ----
        import numpy as np

        X_int = play_df[self.interaction_features].to_numpy(
            dtype=np.float32
        )  # (N, F_int)
        N, F_int = X_int.shape
        feat_i = X_int[:, None, :]  # (N, 1, F_int)
        feat_j = X_int[None, :, :]  # (1, N, F_int)
        pair_diff = feat_j - feat_i  # (N, N, F_int)
        X_pair = np.transpose(pair_diff, (2, 0, 1))  # (F_int, N, N)

        # ---- invariant features (N, F_inv) ----
        X_inv = self.preproc_invariant.transform(
            play_df[self.inv_numeric_features + self.inv_categorical_features]
        )
        X_inv = X_inv.astype("float32")

        return torch.from_numpy(X_pair), torch.from_numpy(X_inv)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


# Build dataset
full_dataset = PlayDataset(
    x_data=x_data,
    y_data=y_data,  # with proper residual targets!
    interaction_features=interaction_features,
    inv_numeric_features=inv_numeric_features,
    inv_categorical_features=inv_categorical_features,
    preproc_invariant=preproc_invariant,
)


import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseInteractionEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, F_int, N, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # (B, C, N, N)
        x = x.mean(dim=3)  # pool over "other player" j → (B, C, N)
        x = x.permute(0, 2, 1)  # → (B, N, C)
        return x


class TimeConditionedMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: (..., in_dim)
        return self.net(x)


class FullModel(nn.Module):
    def __init__(self, in_channels, inv_dim, hidden_dim=128, enc_hidden=64, enc_out=64):
        super().__init__()
        self.encoder = PairwiseInteractionEncoder(
            in_channels=in_channels,
            hidden_channels=enc_hidden,
            out_channels=enc_out,
        )
        self.mlp = TimeConditionedMLP(
            in_dim=enc_out + inv_dim + 1,  # +1 for time feature
            hidden_dim=hidden_dim,
            out_dim=2,
        )

    def forward(self, X_pair, X_inv, t_norm, mask):
        """
        X_pair: (B, F_int, N, N)
        X_inv:  (B, N, F_inv)
        t_norm: (B, T_max)
        mask:   (B, N, T_max)  (bool) – True where target is valid
        """
        B, F_int, N, _ = X_pair.shape
        _, N_inv, F_inv = X_inv.shape
        _, T_max = t_norm.shape

        assert N == N_inv, "Mismatch in N between pairwise and inv features"

        # --- Encode interactions ---
        z_int = self.encoder(X_pair)  # (B, N, C)

        # --- Prepare features over time ---
        # z_int:     (B, N, C)     → (B, N, T, C)
        # X_inv:     (B, N, F_inv) → (B, N, T, F_inv)
        # t_norm:    (B, T)        → (B, 1, T, 1) broadcast to (B, N, T, 1)
        C = z_int.shape[-1]
        z_int_exp = z_int.unsqueeze(2).expand(B, N, T_max, C)  # (B, N, T, C)
        X_inv_exp = X_inv.unsqueeze(2).expand(B, N, T_max, F_inv)  # (B, N, T, F_inv)
        t_exp = t_norm.unsqueeze(1).unsqueeze(-1).expand(B, N, T_max, 1)  # (B, N, T, 1)

        feat = torch.cat([z_int_exp, X_inv_exp, t_exp], dim=-1)  # (B, N, T, C+F_inv+1)

        # Flatten players and time to feed MLP
        feat_flat = feat.view(B * N * T_max, -1)  # (B*N*T, in_dim)
        out_flat = self.mlp(feat_flat)  # (B*N*T, 2)
        out = out_flat.view(B, N, T_max, 2)  # (B, N, T, 2)

        # Apply mask in loss outside (we return full out)
        return out


from torch.utils.data import DataLoader
import numpy as np

# For now, simple random split by index (you can do group splits by game_id if you like)
# Dataset is already at the play level, so this way of splitting is fine
n = len(full_dataset)
idxs = np.arange(n)
np.random.seed(42)
np.random.shuffle(idxs)

n_train = int(0.7 * n)
n_val = int(0.15 * n)
train_idx = idxs[:n_train]
val_idx = idxs[n_train : n_train + n_val]
test_idx = idxs[n_train + n_val :]

from torch.utils.data import Subset

train_ds = Subset(full_dataset, train_idx)
val_ds = Subset(full_dataset, val_idx)
test_ds = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Infer dims
F_int = len(interaction_features)
# Get one batch to determine inv_dim
X_pair0, X_inv0, t_norm0, targets0, mask0 = next(iter(train_loader))
inv_dim = X_inv0.shape[-1]

model = FullModel(
    in_channels=F_int,
    inv_dim=inv_dim,
    hidden_dim=128,
    enc_hidden=64,
    enc_out=64,
).to(device)

criterion = nn.MSELoss(reduction="sum")  # we'll divide by #valid later
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0  # for backprop (MSE)
    total_squared_distance = 0.0  # for Kaggle metric
    total_samples = 0

    for X_pair, X_inv, t_norm, targets, mask in tqdm(loader):
        X_pair = X_pair.to(device).float()  # (B=1, F_int, N, N)
        X_inv = X_inv.to(device).float()  # (B=1, N, F_inv)
        t_norm = t_norm.to(device).float()  # (B=1, T)
        targets = targets.to(device).float()  # (B=1, N, T, 2)
        mask = mask.to(device)  # (B=1, N, T)

        if train:
            optimizer.zero_grad()

        preds = model(X_pair, X_inv, t_norm, mask)  # (B, N, T, 2)

        # Only count valid frames
        mask_expanded = mask.unsqueeze(-1).expand_as(preds)  # (B, N, T, 2)
        diff = (preds - targets) * mask_expanded
        loss = criterion(diff, torch.zeros_like(diff))
        valid_count = mask.sum().item() * 2  # *2 because x and y

        if valid_count == 0:
            continue

        loss = loss / valid_count  # mean over valid coordinates

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # ---- Kaggle RMSE metric (for monitoring only) ----
        with torch.no_grad():
            diff_x = (preds[..., 0] - targets[..., 0]) * mask
            diff_y = (preds[..., 1] - targets[..., 1]) * mask
            squared_distances = diff_x**2 + diff_y**2
            total_squared_distance += squared_distances.sum().item()
            total_samples += mask.sum().item()
    avg_loss = total_loss / len(loader)  # MSE for logging
    kaggle_rmse = np.sqrt(total_squared_distance / max(total_samples, 1))

    return avg_loss, kaggle_rmse


num_epochs = 90
best_val = float("inf")
best_state = None
current_ts_abbreviated = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss, train_kaggle_rmse = run_epoch(train_loader, train=True)
    val_loss, val_kaggle_rmse = run_epoch(val_loader, train=False)
    print(
        f"Epoch {epoch+1}: train={train_loss:.4f},, val={val_loss:.4f}, Kaggle RMSE val={val_kaggle_rmse:.4f}"
    )
    if val_loss < best_val:
        best_val = val_loss
        best_state = model.state_dict().copy()
        torch.save(best_state, f"best_model_{current_ts_abbreviated}.pth")
        print(f"  New best model saved with val loss {best_val:.4f}")
