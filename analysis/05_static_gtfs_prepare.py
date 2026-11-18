import os
import pandas as pd

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GTFS_DIR = os.path.join(BASE_DIR, "data", "gtfs_static")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# -----------------------------
# REQUIRED FILES
# -----------------------------
required_files = {
    "stops": "stops.txt",
    "stop_times": "stop_times.txt",
    "trips": "trips.txt",
    "routes": "routes.txt",
}

loaded = {}

for key, filename in required_files.items():
    path = os.path.join(GTFS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing GTFS file: {path}")
    loaded[key] = pd.read_csv(path)

stops = loaded["stops"]
stop_times = loaded["stop_times"]
trips = loaded["trips"]
routes = loaded["routes"]

# -----------------------------
# CLEAN STOPS
# -----------------------------
required_stop_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
missing = [c for c in required_stop_cols if c not in stops.columns]
if missing:
    raise ValueError(f"Missing columns in stops.txt: {missing}")

stops_clean = stops[required_stop_cols].copy()
stops_clean["stop_id"] = stops_clean["stop_id"].astype(str)
stops_clean["stop_lat"] = pd.to_numeric(stops_clean["stop_lat"], errors="coerce")
stops_clean["stop_lon"] = pd.to_numeric(stops_clean["stop_lon"], errors="coerce")
stops_clean = stops_clean.dropna(subset=["stop_lat", "stop_lon"])

# -----------------------------
# CLEAN STOP TIMES
# -----------------------------
required_stop_time_cols = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
missing = [c for c in required_stop_time_cols if c not in stop_times.columns]
if missing:
    raise ValueError(f"Missing columns in stop_times.txt: {missing}")

stop_times_clean = stop_times[required_stop_time_cols].copy()
stop_times_clean["trip_id"] = stop_times_clean["trip_id"].astype(str)
stop_times_clean["stop_id"] = stop_times_clean["stop_id"].astype(str)
stop_times_clean["stop_sequence"] = pd.to_numeric(
    stop_times_clean["stop_sequence"], errors="coerce"
)

stop_times_clean = stop_times_clean.dropna(subset=["stop_sequence"])
stop_times_clean["stop_sequence"] = stop_times_clean["stop_sequence"].astype(int)

# -----------------------------
# ADD ROUTE_ID THROUGH TRIPS
# -----------------------------
if "trip_id" not in trips.columns or "route_id" not in trips.columns:
    raise ValueError("trips.txt must contain trip_id and route_id")

trips_small = trips[["trip_id", "route_id"]].copy()
trips_small["trip_id"] = trips_small["trip_id"].astype(str)
trips_small["route_id"] = trips_small["route_id"].astype(str)

stop_times_clean = stop_times_clean.merge(
    trips_small,
    on="trip_id",
    how="left"
)

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
stops_out = os.path.join(PROCESSED_DIR, "gtfs_stops_clean.csv")
stop_times_out = os.path.join(PROCESSED_DIR, "gtfs_stop_times_clean.csv")

stops_clean.to_csv(stops_out, index=False)
stop_times_clean.to_csv(stop_times_out, index=False)

summary = {
    "num_stops": len(stops_clean),
    "num_stop_time_rows": len(stop_times_clean),
    "num_trips": stop_times_clean["trip_id"].nunique(),
    "num_routes": stop_times_clean["route_id"].nunique(),
}

pd.DataFrame([summary]).to_csv(
    os.path.join(TABLES_DIR, "static_gtfs_summary.csv"),
    index=False
)

print("Static GTFS prepared successfully.")
print(summary)
print(f"Saved: {stops_out}")
print(f"Saved: {stop_times_out}")