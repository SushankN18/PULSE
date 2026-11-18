import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# PATH SETUP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# LOAD FILES

files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

print(f"Found {len(files)} raw files.")

df_list = []
for f in files:
    try:
        temp = pd.read_csv(f)
        temp["source_file"] = os.path.basename(f)
        df_list.append(temp)
    except Exception as e:
        print(f"Failed to read {f}: {e}")

if not df_list:
    raise ValueError("No valid CSV files could be loaded.")

df = pd.concat(df_list, ignore_index=True)

# REQUIRED COLUMNS CHECK

required_cols = ["receive_time_utc", "event_time_utc", "vehicle_id"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# TIME CONVERSION

df["event_time"] = pd.to_datetime(df["event_time_utc"], errors="coerce")
df["receive_time"] = pd.to_datetime(df["receive_time_utc"], errors="coerce")

df = df.dropna(subset=["event_time", "receive_time", "vehicle_id"])

# BASIC SUMMARY

summary_lines = []
summary_lines.append(f"Total rows: {len(df)}")
summary_lines.append(f"Unique vehicles: {df['vehicle_id'].nunique()}")
summary_lines.append(f"Date range: {df['event_time'].min()} -> {df['event_time'].max()}")

print("\n".join(summary_lines))

# LATENCY ANALYSIS

df["latency_sec"] = (df["receive_time"] - df["event_time"]).dt.total_seconds()

latency_stats = df["latency_sec"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
print("\nLatency stats:")
print(latency_stats)

latency_stats.to_csv(os.path.join(TABLES_DIR, "latency_stats.csv"))

# GAP ANALYSIS

df = df.sort_values(["vehicle_id", "event_time"])
df["gap_sec"] = df.groupby("vehicle_id")["event_time"].diff().dt.total_seconds()

gap_stats = df["gap_sec"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
print("\nGap stats:")
print(gap_stats)

gap_stats.to_csv(os.path.join(TABLES_DIR, "gap_stats.csv"))

# LONGEST GAP PER VEHICLE

longest_gap = (
    df.groupby("vehicle_id")["gap_sec"]
    .max()
    .sort_values(ascending=False)
    .head(20)
)

print("\nTop 20 longest gaps by vehicle:")
print(longest_gap)

longest_gap.to_csv(os.path.join(TABLES_DIR, "top20_longest_gaps.csv"))

# NULL RATES

null_rates = df.isnull().mean().sort_values(ascending=False)
null_rates.to_csv(os.path.join(TABLES_DIR, "null_rates.csv"))

# PLOTS

plt.figure(figsize=(8, 5))
df["latency_sec"].dropna().clip(-60, 300).hist(bins=60)
plt.title("Latency Distribution")
plt.xlabel("Latency (seconds)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "latency_distribution.png"))
plt.close()

plt.figure(figsize=(8, 5))
df["gap_sec"].dropna().clip(0, 600).hist(bins=60)
plt.title("Ping Gap Distribution")
plt.xlabel("Gap (seconds)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ping_gap_distribution.png"))
plt.close()

# DAILY ROW COUNTS

df["event_date"] = df["event_time"].dt.date
daily_counts = df.groupby("event_date").size()
daily_counts.to_csv(os.path.join(TABLES_DIR, "daily_row_counts.csv"))

plt.figure(figsize=(10, 5))
daily_counts.plot(kind="bar")
plt.title("Rows per Day")
plt.xlabel("Date")
plt.ylabel("Row Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "daily_row_counts.png"))
plt.close()

print("\nDone.")
print(f"Plots saved to: {PLOTS_DIR}")
print(f"Tables saved to: {TABLES_DIR}")