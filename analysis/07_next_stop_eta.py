import os
import pandas as pd

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

ROUTE_FILE = os.path.join(PROCESSED_DIR, "route_with_speed.csv")
HISTORY_FILE = os.path.join(PROCESSED_DIR, "stop_to_stop_history.csv")

if not os.path.exists(ROUTE_FILE):
    raise FileNotFoundError(f"Missing file: {ROUTE_FILE}")

if not os.path.exists(HISTORY_FILE):
    raise FileNotFoundError(f"Missing file: {HISTORY_FILE}")

df = pd.read_csv(ROUTE_FILE)
history = pd.read_csv(HISTORY_FILE)

required_cols = [
    "vehicle_id",
    "route_id",
    "trip_id",
    "stop_id",
    "current_stop_sequence",
    "event_time",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in route data: {missing}")

history_required = [
    "route_id",
    "current_stop_sequence",
    "stop_id",
    "next_stop_sequence",
    "next_stop_id",
    "median_travel_time_sec",
    "p75_travel_time_sec",
    "p90_travel_time_sec",
]

missing_hist = [c for c in history_required if c not in history.columns]
if missing_hist:
    raise ValueError(f"Missing required columns in stop history: {missing_hist}")

# -----------------------------
# CLEAN
# -----------------------------
df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
df = df.dropna(subset=["event_time", "vehicle_id", "route_id", "stop_id", "current_stop_sequence"])

df["route_id"] = df["route_id"].astype(str)
df["stop_id"] = df["stop_id"].astype(str)
df["current_stop_sequence"] = pd.to_numeric(df["current_stop_sequence"], errors="coerce")
df = df.dropna(subset=["current_stop_sequence"])
df["current_stop_sequence"] = df["current_stop_sequence"].astype(int)

history["route_id"] = history["route_id"].astype(str)
history["stop_id"] = history["stop_id"].astype(str)
history["next_stop_id"] = history["next_stop_id"].astype(str)
history["current_stop_sequence"] = pd.to_numeric(history["current_stop_sequence"], errors="coerce").astype(int)

# -----------------------------
# LATEST OBSERVATION PER VEHICLE
# -----------------------------
latest = (
    df.sort_values(["vehicle_id", "event_time"])
    .groupby("vehicle_id")
    .tail(1)
    .copy()
)

# -----------------------------
# MERGE HISTORICAL ETA
# -----------------------------
eta = latest.merge(
    history[
        [
            "route_id",
            "current_stop_sequence",
            "stop_id",
            "next_stop_sequence",
            "next_stop_id",
            "observations",
            "median_travel_time_sec",
            "mean_travel_time_sec",
            "p75_travel_time_sec",
            "p90_travel_time_sec",
            "p95_travel_time_sec",
        ]
    ],
    on=["route_id", "current_stop_sequence", "stop_id"],
    how="left"
)

# -----------------------------
# FALLBACKS
# -----------------------------
# If exact stop_id match fails, try route + sequence only.
missing_eta = eta["median_travel_time_sec"].isna()

if missing_eta.any():
    fallback = (
        history.groupby(["route_id", "current_stop_sequence"])
        .agg(
            fallback_next_stop_sequence=("next_stop_sequence", "median"),
            fallback_median_eta=("median_travel_time_sec", "median"),
            fallback_p90_eta=("p90_travel_time_sec", "median"),
            fallback_observations=("observations", "sum"),
        )
        .reset_index()
    )

    eta = eta.merge(
        fallback,
        on=["route_id", "current_stop_sequence"],
        how="left"
    )

    eta.loc[eta["median_travel_time_sec"].isna(), "median_travel_time_sec"] = eta.loc[
        eta["median_travel_time_sec"].isna(), "fallback_median_eta"
    ]

    eta.loc[eta["p90_travel_time_sec"].isna(), "p90_travel_time_sec"] = eta.loc[
        eta["p90_travel_time_sec"].isna(), "fallback_p90_eta"
    ]

    eta.loc[eta["observations"].isna(), "observations"] = eta.loc[
        eta["observations"].isna(), "fallback_observations"
    ]

# -----------------------------
# ETA LABELS
# -----------------------------
eta["eta_min"] = eta["median_travel_time_sec"] / 60.0
eta["eta_p90_min"] = eta["p90_travel_time_sec"] / 60.0

eta["eta_quality"] = "missing"
eta.loc[eta["observations"] >= 30, "eta_quality"] = "strong"
eta.loc[(eta["observations"] >= 10) & (eta["observations"] < 30), "eta_quality"] = "moderate"
eta.loc[(eta["observations"] > 0) & (eta["observations"] < 10), "eta_quality"] = "weak"

# -----------------------------
# OUTPUT
# -----------------------------
output_cols = [
    "vehicle_id",
    "route_id",
    "trip_id",
    "event_time",
    "stop_id",
    "current_stop_sequence",
    "next_stop_sequence",
    "next_stop_id",
    "median_travel_time_sec",
    "eta_min",
    "eta_p90_min",
    "observations",
    "eta_quality",
]

output_cols = [c for c in output_cols if c in eta.columns]

eta_out = os.path.join(PROCESSED_DIR, "next_stop_eta_predictions.csv")
eta[output_cols].to_csv(eta_out, index=False)

summary = {
    "vehicles_with_latest_state": len(latest),
    "vehicles_with_eta": eta["median_travel_time_sec"].notna().sum(),
    "coverage_rate": eta["median_travel_time_sec"].notna().mean(),
    "strong_eta_count": (eta["eta_quality"] == "strong").sum(),
    "moderate_eta_count": (eta["eta_quality"] == "moderate").sum(),
    "weak_eta_count": (eta["eta_quality"] == "weak").sum(),
    "missing_eta_count": (eta["eta_quality"] == "missing").sum(),
}

pd.DataFrame([summary]).to_csv(
    os.path.join(TABLES_DIR, "next_stop_eta_summary.csv"),
    index=False
)

print("Next-stop ETA predictions created successfully.")
print(summary)
print(f"Saved: {eta_out}")