import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

df = pd.read_csv(os.path.join(PROCESSED_DIR, "route_with_speed.csv"))

df["event_time"] = pd.to_datetime(df["event_time"])

# -----------------------------
# SORT
# -----------------------------
df = df.sort_values(["vehicle_id", "event_time"])

# -----------------------------
# SMOOTH SPEED
# -----------------------------
df["speed_smooth"] = (
    df.groupby("vehicle_id")["speed_mps"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# -----------------------------
# REMOVE BAD SPEEDS
# -----------------------------
df = df[(df["speed_smooth"] > 0.5)]  # remove near-zero speeds

# -----------------------------
# NEXT STEP TARGET
# -----------------------------
df["next_distance"] = df.groupby("vehicle_id")["distance_m"].shift(-1)
df["next_time"] = df.groupby("vehicle_id")["time_diff_sec"].shift(-1)

df = df.dropna(subset=["next_distance", "next_time", "speed_smooth"])

# -----------------------------
# PREDICTION
# -----------------------------
df["predicted_time"] = df["next_distance"] / df["speed_smooth"]

# -----------------------------
# ERROR
# -----------------------------
df["error"] = abs(df["predicted_time"] - df["next_time"])

# -----------------------------
# METRICS
# -----------------------------
metrics = {
    "rows": len(df),
    "MAE": df["error"].mean(),
    "Median Error": df["error"].median(),
    "RMSE": ((df["predicted_time"] - df["next_time"])**2).mean()**0.5
}

print("\nMetrics:")
print(metrics)

pd.DataFrame([metrics]).to_csv(os.path.join(TABLES_DIR, "eta_metrics.csv"), index=False)

# -----------------------------
# PLOTS
# -----------------------------
plt.figure(figsize=(7,6))
plt.scatter(df["next_time"], df["predicted_time"], alpha=0.2)
plt.xlabel("Actual Time (sec)")
plt.ylabel("Predicted Time (sec)")
plt.title("ETA Prediction Scatter")
plt.savefig(os.path.join(PLOTS_DIR, "eta_scatter.png"))
plt.close()

plt.figure(figsize=(8,5))
df["error"].clip(0,300).hist(bins=60)
plt.title("ETA Error Distribution")
plt.xlabel("Error (sec)")
plt.savefig(os.path.join(PLOTS_DIR, "eta_error.png"))
plt.close()