import os
import pandas as pd
import numpy as np

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

ROUTE_FILE = os.path.join(PROCESSED_DIR, "route_with_speed.csv")

if not os.path.exists(ROUTE_FILE):
    raise FileNotFoundError(f"Missing file: {ROUTE_FILE}")

df = pd.read_csv(ROUTE_FILE)

required_cols = [
    "vehicle_id",
    "route_id",
    "trip_id",
    "stop_id",
    "current_stop_sequence",
    "current_status",
    "event_time",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in route_with_speed.csv: {missing}")

# -----------------------------
# CLEAN
# -----------------------------
df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
df = df.dropna(subset=["event_time", "vehicle_id", "trip_id", "stop_id", "current_stop_sequence"])

df["trip_id"] = df["trip_id"].astype(str)
df["route_id"] = df["route_id"].astype(str)
df["stop_id"] = df["stop_id"].astype(str)
df["current_stop_sequence"] = pd.to_numeric(
    df["current_stop_sequence"], errors="coerce"
)

df = df.dropna(subset=["current_stop_sequence"])
df["current_stop_sequence"] = df["current_stop_sequence"].astype(int)

# -----------------------------
# KEEP STOP EVENTS
# -----------------------------
# In GTFS-RT current_status often stores numeric/string values depending on parser.
# We do not rely only on status; stop_id + stop_sequence gives us stop event evidence.
stop_events = (
    df.sort_values(["vehicle_id", "trip_id", "current_stop_sequence", "event_time"])
    .groupby(["vehicle_id", "trip_id", "route_id", "current_stop_sequence", "stop_id"])
    .agg(
        first_seen_time=("event_time", "min"),
        last_seen_time=("event_time", "max"),
        observations=("event_time", "count")
    )
    .reset_index()
)

# -----------------------------
# CONSECUTIVE STOP-TO-STOP TIMES
# -----------------------------
stop_events = stop_events.sort_values(["vehicle_id", "trip_id", "current_stop_sequence"])

stop_events["next_stop_sequence"] = stop_events.groupby(
    ["vehicle_id", "trip_id"]
)["current_stop_sequence"].shift(-1)

stop_events["next_stop_id"] = stop_events.groupby(
    ["vehicle_id", "trip_id"]
)["stop_id"].shift(-1)

stop_events["next_first_seen_time"] = stop_events.groupby(
    ["vehicle_id", "trip_id"]
)["first_seen_time"].shift(-1)

stop_events["travel_time_sec"] = (
    stop_events["next_first_seen_time"] - stop_events["last_seen_time"]
).dt.total_seconds()

# Keep valid consecutive stops only
pairs = stop_events.dropna(subset=["next_stop_sequence", "next_stop_id", "travel_time_sec"]).copy()

pairs = pairs[
    (pairs["next_stop_sequence"] == pairs["current_stop_sequence"] + 1)
    & (pairs["travel_time_sec"] > 0)
    & (pairs["travel_time_sec"] < 3600)
]

# -----------------------------
# HISTORICAL BASELINE TABLE
# -----------------------------
history = (
    pairs.groupby(["route_id", "current_stop_sequence", "stop_id", "next_stop_sequence", "next_stop_id"])
    .agg(
        observations=("travel_time_sec", "count"),
        median_travel_time_sec=("travel_time_sec", "median"),
        mean_travel_time_sec=("travel_time_sec", "mean"),
        p75_travel_time_sec=("travel_time_sec", lambda x: np.quantile(x, 0.75)),
        p90_travel_time_sec=("travel_time_sec", lambda x: np.quantile(x, 0.90)),
        p95_travel_time_sec=("travel_time_sec", lambda x: np.quantile(x, 0.95)),
    )
    .reset_index()
)

# Filter weak history
history = history[history["observations"] >= 5].copy()

# -----------------------------
# SAVE
# -----------------------------
stop_events_out = os.path.join(PROCESSED_DIR, "vehicle_stop_events.csv")
history_out = os.path.join(PROCESSED_DIR, "stop_to_stop_history.csv")

stop_events.to_csv(stop_events_out, index=False)
history.to_csv(history_out, index=False)

summary = {
    "stop_event_rows": len(stop_events),
    "valid_stop_pairs": len(pairs),
    "history_rows": len(history),
    "routes": history["route_id"].nunique() if not history.empty else 0,
}

pd.DataFrame([summary]).to_csv(
    os.path.join(TABLES_DIR, "stop_to_stop_history_summary.csv"),
    index=False
)

print("Stop-to-stop history built successfully.")
print(summary)
print(f"Saved: {stop_events_out}")
print(f"Saved: {history_out}")