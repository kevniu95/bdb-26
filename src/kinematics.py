import pandas as pd
import numpy as np

SAMPLE_RATE_HZ = 10.0
DT_PER_FRAME = 1.0 / SAMPLE_RATE_HZ  # 0.1 s at 10 Hz


def add_kinematics(df: pd.DataFrame, smooth_window: int = 1) -> pd.DataFrame:
    """
    Adds vx, vy, s (yd/s), s_mph, ax, ay, a (yd/s^2), and dir (deg) to the dataframe.
    Expects columns: game_id, play_id, nfl_id, frame_id, x, y
    """
    df = df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]).copy()
    keys = ["game_id", "play_id", "nfl_id"]

    def per_player(g: pd.DataFrame) -> pd.DataFrame:
        # Forward/backward deltas in position and frames
        g["dx_f"] = g["x"].shift(-1) - g["x"]
        g["dy_f"] = g["y"].shift(-1) - g["y"]
        g["dx_b"] = g["x"] - g["x"].shift(1)
        g["dy_b"] = g["y"] - g["y"].shift(1)
        g["df_f"] = (g["frame_id"].shift(-1) - g["frame_id"]).astype(float)
        g["df_b"] = (g["frame_id"] - g["frame_id"].shift(1)).astype(float)

        # Forward/backward velocities (handle occasional dropped frames via df_* scaling)
        g["vx_f"] = g["dx_f"] / (g["df_f"] * DT_PER_FRAME)
        g["vy_f"] = g["dy_f"] / (g["df_f"] * DT_PER_FRAME)
        g["vx_b"] = g["dx_b"] / (g["df_b"] * DT_PER_FRAME)
        g["vy_b"] = g["dy_b"] / (g["df_b"] * DT_PER_FRAME)

        # Central velocity; fall back to available side on edges
        g["vx"] = np.where(
            g["vx_f"].notna() & g["vx_b"].notna(),
            0.5 * (g["vx_f"] + g["vx_b"]),
            g["vx_f"].fillna(g["vx_b"]),
        )
        g["vy"] = np.where(
            g["vy_f"].notna() & g["vy_b"].notna(),
            0.5 * (g["vy_f"] + g["vy_b"]),
            g["vy_f"].fillna(g["vy_b"]),
        )

        # Light symmetric smoothing to mimic NGS filtering (optional)
        if smooth_window and smooth_window > 1:
            g["vx"] = g["vx"].rolling(smooth_window, center=True, min_periods=1).mean()
            g["vy"] = g["vy"].rolling(smooth_window, center=True, min_periods=1).mean()

        # Speed (yards/s) and mph convenience
        g["s"] = np.hypot(g["vx"], g["vy"])
        g["s_mph"] = g["s"] * 2.045  # 1 yd/s ≈ 2.045 mph

        # --- Tangential acceleration (can be negative): a_tan = d(s)/dt ---
        g["ds_f"] = g["s"].shift(-1) - g["s"]
        g["ds_b"] = g["s"] - g["s"].shift(1)
        g["a_tan_f"] = g["ds_f"] / (g["df_f"] * DT_PER_FRAME)
        g["a_tan_b"] = g["ds_b"] / (g["df_b"] * DT_PER_FRAME)
        g["a_tan"] = np.where(
            g["a_tan_f"].notna() & g["a_tan_b"].notna(),
            0.5 * (g["a_tan_f"] + g["a_tan_b"]),
            g["a_tan_f"].fillna(g["a_tan_b"]),
        )
        if smooth_window and smooth_window > 1:
            g["a_tan"] = (
                g["a_tan"].rolling(smooth_window, center=True, min_periods=1).mean()
            )

        # --- Vector acceleration magnitude (always ≥ 0): |a| from ax, ay ---
        g["dvx_f"] = g["vx"].shift(-1) - g["vx"]
        g["dvy_f"] = g["vy"].shift(-1) - g["vy"]
        g["dvx_b"] = g["vx"] - g["vx"].shift(1)
        g["dvy_b"] = g["vy"] - g["vy"].shift(1)

        g["ax_f"] = g["dvx_f"] / (g["df_f"] * DT_PER_FRAME)
        g["ay_f"] = g["dvy_f"] / (g["df_f"] * DT_PER_FRAME)
        g["ax_b"] = g["dvx_b"] / (g["df_b"] * DT_PER_FRAME)
        g["ay_b"] = g["dvy_b"] / (g["df_b"] * DT_PER_FRAME)

        g["ax"] = np.where(
            g["ax_f"].notna() & g["ax_b"].notna(),
            0.5 * (g["ax_f"] + g["ax_b"]),
            g["ax_f"].fillna(g["ax_b"]),
        )
        g["ay"] = np.where(
            g["ay_f"].notna() & g["ay_b"].notna(),
            0.5 * (g["ay_f"] + g["ay_b"]),
            g["ay_f"].fillna(g["ay_b"]),
        )

        if smooth_window and smooth_window > 1:
            g["ax"] = g["ax"].rolling(smooth_window, center=True, min_periods=1).mean()
            g["ay"] = g["ay"].rolling(smooth_window, center=True, min_periods=1).mean()

        g["a_mag"] = np.hypot(g["ax"], g["ay"])

        # --- Choose which one to publish as 'a' ---
        # Your "real a" never goes negative, so use magnitude:
        g["a"] = g["a_mag"]

        if smooth_window and smooth_window > 1:
            g["a"] = g["a"].rolling(smooth_window, center=True, min_periods=1).mean()

        # Direction of travel in degrees [0, 360) using math convention (0°=+x, CCW positive)
        # g["dir"] = np.degrees(np.arctan2(g["vy"], g["vx"])) % 360.0
        g["dir"] = (450.0 - np.degrees(np.arctan2(g["vy"], g["vx"]))) % 360.0
        # g["dir"] = (450.0 - np.degrees(np.arctan2(g["vy"], g["vx"]))) % 360.0

        return g[
            [
                "game_id",
                "play_id",
                "nfl_id",
                "frame_id",
                "x",
                "y",
                "vx",
                "vy",
                "s",
                "s_mph",
                "a",
                "a_tan",
                "a_mag",
                "dir",
            ]
        ]

    out = df.groupby(keys, group_keys=False).apply(per_player)
    return out.reset_index(drop=True)
