import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# PATH SETUP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

INPUT_FILE = os.path.join(PROCESSED_DIR, "route_with_speed.csv")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Missing processed file: {INPUT_FILE}")


# LOAD DATA

df = pd.read_csv(INPUT_FILE)

required_cols = [
    "vehicle_id", "event_time", "distance_m", "time_diff_sec",
    "speed_mps", "current_status"
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
df = df.dropna(subset=["vehicle_id", "event_time", "distance_m", "time_diff_sec", "speed_mps"])

# Keep only plausible rows for evaluation
df = df[(df["time_diff_sec"] > 0) & (df["distance_m"] >= 0) & (df["speed_mps"] > 0)]

# Sort for rolling and next-step targets
df = df.sort_values(["vehicle_id", "event_time"]).copy()

# FEATURE ENGINEERING

# Rolling mean / median speed
df["speed_mean_5"] = (
    df.groupby("vehicle_id")["speed_mps"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

df["speed_median_5"] = (
    df.groupby("vehicle_id")["speed_mps"]
    .rolling(window=5, min_periods=1)
    .median()
    .reset_index(level=0, drop=True)
)

# Next-step target
df["next_distance_m"] = df.groupby("vehicle_id")["distance_m"].shift(-1)
df["next_time_sec"] = df.groupby("vehicle_id")["time_diff_sec"].shift(-1)

# Remove missing next-step rows
df = df.dropna(subset=["next_distance_m", "next_time_sec", "speed_mean_5", "speed_median_5"])

# Additional guardrails to reduce pathologies in benchmarking
df = df[(df["next_time_sec"] > 0) & (df["next_time_sec"] < 300)]  # focus on ordinary next-step regime
df = df[(df["speed_mean_5"] > 0.5) & (df["speed_median_5"] > 0.5)]  # avoid near-zero explosion

# Time-of-day bucket
df["hour"] = df["event_time"].dt.hour
df["hour_bucket"] = pd.cut(
    df["hour"],
    bins=[-1, 5, 11, 16, 20, 23],
    labels=["late_night", "morning", "afternoon", "evening", "night"]
)

# Speed regime
def speed_regime(v):
    if pd.isna(v):
        return "unknown"
    if v < 1:
        return "stopped"
    elif v < 5:
        return "slow"
    elif v < 12:
        return "normal"
    else:
        return "fast"

df["speed_regime"] = df["speed_mps"].apply(speed_regime)

# Clean status text
df["current_status"] = df["current_status"].astype(str).fillna("UNKNOWN")

# MODEL 1: Rolling mean speed predictor

df["pred_time_mean"] = df["next_distance_m"] / df["speed_mean_5"]

# MODEL 2: Rolling median speed predictor

df["pred_time_median"] = df["next_distance_m"] / df["speed_median_5"]

# MODEL 3: Historical baseline

# Historical median next_time by hour bucket and status
hist_key = ["hour_bucket", "current_status"]
hist_lookup = (
    df.groupby(hist_key)["next_time_sec"]
    .median()
    .rename("hist_pred_time")
    .reset_index()
)

df = df.merge(hist_lookup, on=hist_key, how="left")

# Fallback in case of sparse buckets
global_hist_median = df["next_time_sec"].median()
df["hist_pred_time"] = df["hist_pred_time"].fillna(global_hist_median)


# ERRORS

for pred_col, name in [
    ("pred_time_mean", "mean"),
    ("pred_time_median", "median"),
    ("hist_pred_time", "historical"),
]:
    df[f"abs_error_{name}"] = (df[pred_col] - df["next_time_sec"]).abs()
    df[f"sq_error_{name}"] = (df[pred_col] - df["next_time_sec"]) ** 2


# METRIC FUNCTION

def metric_block(sub_df, label):
    out = {
        "model": label,
        "rows": len(sub_df),
        "mae_sec": sub_df[f"abs_error_{label}"].mean(),
        "median_abs_error_sec": sub_df[f"abs_error_{label}"].median(),
        "rmse_sec": np.sqrt(sub_df[f"sq_error_{label}"].mean()),
        "p95_abs_error_sec": sub_df[f"abs_error_{label}"].quantile(0.95),
        "p99_abs_error_sec": sub_df[f"abs_error_{label}"].quantile(0.99),
    }
    return out

metrics = pd.DataFrame([
    metric_block(df, "mean"),
    metric_block(df, "median"),
    metric_block(df, "historical"),
])

metrics.to_csv(os.path.join(TABLES_DIR, "eta_model_comparison.csv"), index=False)


# STATUS-WISE METRICS

status_rows = []
for status, group in df.groupby("current_status"):
    if len(group) < 50:
        continue
    for label in ["mean", "median", "historical"]:
        status_rows.append({
            "current_status": status,
            "model": label,
            "rows": len(group),
            "mae_sec": group[f"abs_error_{label}"].mean(),
            "median_abs_error_sec": group[f"abs_error_{label}"].median(),
            "rmse_sec": np.sqrt(group[f"sq_error_{label}"].mean()),
        })

status_metrics = pd.DataFrame(status_rows)
status_metrics.to_csv(os.path.join(TABLES_DIR, "eta_error_by_status.csv"), index=False)


# SPEED-REGIME-WISE METRICS

regime_rows = []
for regime, group in df.groupby("speed_regime"):
    if len(group) < 50:
        continue
    for label in ["mean", "median", "historical"]:
        regime_rows.append({
            "speed_regime": regime,
            "model": label,
            "rows": len(group),
            "mae_sec": group[f"abs_error_{label}"].mean(),
            "median_abs_error_sec": group[f"abs_error_{label}"].median(),
            "rmse_sec": np.sqrt(group[f"sq_error_{label}"].mean()),
        })

regime_metrics = pd.DataFrame(regime_rows)
regime_metrics.to_csv(os.path.join(TABLES_DIR, "eta_error_by_speed_regime.csv"), index=False)


# WORST CASES

worst_cases = df.sort_values("abs_error_mean", ascending=False).head(100)[[
    "vehicle_id", "event_time", "current_status", "speed_mps", "speed_regime",
    "next_distance_m", "next_time_sec", "pred_time_mean", "pred_time_median",
    "hist_pred_time", "abs_error_mean", "abs_error_median", "abs_error_historical"
]]
worst_cases.to_csv(os.path.join(TABLES_DIR, "eta_worst_cases_top100.csv"), index=False)


# PLOTS

# 1. Model comparison bar chart

plt.figure(figsize=(9, 5))
x = np.arange(len(metrics))
width = 0.25

plt.bar(x - width, metrics["median_abs_error_sec"], width=width, label="Median Abs Error")
plt.bar(x, metrics["mae_sec"], width=width, label="MAE")
plt.bar(x + width, metrics["p95_abs_error_sec"], width=width, label="P95 Abs Error")

plt.xticks(x, metrics["model"])
plt.ylabel("Seconds")
plt.title("ETA Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eta_model_comparison.png"))
plt.close()

# 2. Error distribution comparison
plt.figure(figsize=(9, 5))
df["abs_error_mean"].clip(0, 300).hist(bins=60, alpha=0.5, label="Rolling Mean")
df["abs_error_median"].clip(0, 300).hist(bins=60, alpha=0.5, label="Rolling Median")
df["abs_error_historical"].clip(0, 300).hist(bins=60, alpha=0.5, label="Historical")
plt.xlabel("Absolute Error (sec, clipped at 300)")
plt.ylabel("Count")
plt.title("Error Distribution Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eta_error_distribution_comparison.png"))
plt.close()

# 3. Error by status (best model only: choose lowest MAE)
best_model = metrics.sort_values("mae_sec").iloc[0]["model"]
status_best = status_metrics[status_metrics["model"] == best_model].sort_values("mae_sec", ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(status_best["current_status"].astype(str), status_best["mae_sec"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("MAE (sec)")
plt.title(f"Error by Current Status ({best_model} model)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eta_error_by_status.png"))
plt.close()

# 4. Error by speed regime (best model)
regime_best = regime_metrics[regime_metrics["model"] == best_model].sort_values("mae_sec", ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(regime_best["speed_regime"].astype(str), regime_best["mae_sec"])
plt.ylabel("MAE (sec)")
plt.title(f"Error by Speed Regime ({best_model} model)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "eta_error_by_speed_regime.png"))
plt.close()

print("\nDone.")
print("Overall model comparison:")
print(metrics)
print(f"\nBest model by MAE: {best_model}")
print(f"\nSaved plots to: {PLOTS_DIR}")
print(f"Saved tables to: {TABLES_DIR}")