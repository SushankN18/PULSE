import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
if not files:
    raise FileNotFoundError("No raw files found")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# -----------------------------
# BASIC CLEANING
# -----------------------------
df["event_time"] = pd.to_datetime(df["event_time_utc"], errors="coerce")
df["receive_time"] = pd.to_datetime(df["receive_time_utc"], errors="coerce")

df = df.dropna(subset=["event_time", "receive_time", "vehicle_id", "latitude", "longitude", "route_id"])

# -----------------------------
# LATENCY CLEANING
# -----------------------------
df["latency_sec"] = (df["receive_time"] - df["event_time"]).dt.total_seconds()

df = df[df["latency_sec"] >= 0]          # remove negative latency
df = df[df["latency_sec"] < 300]         # cap extreme latency (5 min)

# -----------------------------
# SORT + GAP COMPUTATION
# -----------------------------
df = df.sort_values(["vehicle_id", "event_time"])

df["gap_sec"] = df.groupby("vehicle_id")["event_time"].diff().dt.total_seconds()

# remove extreme gaps
df = df[(df["gap_sec"].isna()) | (df["gap_sec"] < 600)]

# -----------------------------
# SELECT ROUTE
# -----------------------------
route_counts = df["route_id"].value_counts()
route_counts.to_csv(os.path.join(TABLES_DIR, "route_counts.csv"))

selected_route = route_counts.index[0]
print(f"Selected route: {selected_route}")

route_df = df[df["route_id"] == selected_route].copy()

# -----------------------------
# SORT AGAIN
# -----------------------------
route_df = route_df.sort_values(["vehicle_id", "event_time"])

# -----------------------------
# DISTANCE FUNCTION
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# -----------------------------
# COMPUTE DISTANCE & SPEED
# -----------------------------
route_df["prev_lat"] = route_df.groupby("vehicle_id")["latitude"].shift(1)
route_df["prev_lon"] = route_df.groupby("vehicle_id")["longitude"].shift(1)
route_df["prev_time"] = route_df.groupby("vehicle_id")["event_time"].shift(1)

route_df["distance_m"] = route_df.apply(
    lambda x: haversine(x["prev_lat"], x["prev_lon"], x["latitude"], x["longitude"])
    if pd.notnull(x["prev_lat"]) else None,
    axis=1
)

route_df["time_diff_sec"] = (
    route_df["event_time"] - route_df["prev_time"]
).dt.total_seconds()

route_df["speed_mps"] = route_df["distance_m"] / route_df["time_diff_sec"]

# -----------------------------
# REMOVE INVALID SPEEDS
# -----------------------------
route_df.loc[route_df["time_diff_sec"] <= 0, "speed_mps"] = None
route_df.loc[route_df["speed_mps"] < 0, "speed_mps"] = None
route_df.loc[route_df["speed_mps"] > 25, "speed_mps"] = None  # ~90 km/h cap

# -----------------------------
# SAVE CLEAN DATA
# -----------------------------
output_path = os.path.join(PROCESSED_DIR, "route_with_speed.csv")
route_df.to_csv(output_path, index=False)

# -----------------------------
# PLOT SPEED DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8,5))
route_df["speed_mps"].dropna().hist(bins=60)
plt.title(f"Speed Distribution (Route {selected_route})")
plt.xlabel("Speed (m/s)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "route_speed_distribution.png"))
plt.close()

# -----------------------------
# SUMMARY
# -----------------------------
summary = {
    "route": selected_route,
    "rows": len(route_df),
    "vehicles": route_df["vehicle_id"].nunique(),
    "median_speed": route_df["speed_mps"].median(),
    "p95_speed": route_df["speed_mps"].quantile(0.95)
}

pd.DataFrame([summary]).to_csv(os.path.join(TABLES_DIR, "route_summary.csv"), index=False)

print("\nSummary:")
print(summary)
print("\nSaved:", output_path)