# 🚍 PULSE — Real-Time Transport Monitoring

Live vehicle tracking, ETA prediction, and data diagnostics for Portland's TriMet transit system using GTFS-Realtime feeds.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Data](https://img.shields.io/badge/Pings-7.7M+-green)

---

## Highlights

- **Real-time pipeline** — polls TriMet GTFS-RT Protobuf feeds every 10–15 s and streams 140+ vehicles to a live Streamlit dashboard
- **Large-scale dataset** — 7.7 M+ GPS pings collected over 40 days across 84 routes (96% fleet coverage)
- **ETA prediction** — benchmarked 3 models on 331 K+ samples; best model (historical-median lookup) cut arrival-time error **38%** (MAE 20.8 → 12.9 s) vs. a rolling-mean-speed baseline
- **Data quality** — Haversine-based speed derivation, outlier removal, latency & gap analysis

## Architecture

```
ingest/                 → GTFS-RT Protobuf ingestion (polling loop)
analysis/
  01_data_report.py     → Raw data profiling & row counts
  02_clean_report.py    → Cleaning, speed computation (Haversine)
  03_eta_baseline.py    → Naïve ETA baseline (distance ÷ smoothed speed)
  04_eta_diagnostics.py → 3-model comparison (mean / median / historical)
  05_static_gtfs_prepare.py → Static GTFS stop & route prep
  06_stop_to_stop_history.py → Historical stop-to-stop travel times
  07_next_stop_eta.py   → Live next-stop ETA with fallback logic
realtime/               → Live monitoring & in-memory state store
frontend/app.py         → Streamlit dashboard (5 tabs)
outputs/
  plots/                → Generated charts (error distributions, model comparison)
  tables/               → Metric CSVs (model comparison, worst cases, etc.)
```

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/SushankN18/PULSE.git
cd PULSE
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your TriMet API key
# Get one free at https://developer.trimet.org/appid/registration/
```

### 3. Ingest live data

```bash
python ingest/gtfs_rt_ingest.py
```

This polls TriMet's GTFS-RT feed every 15 s and writes daily CSVs to `data/raw/`.

### 4. Run the analysis pipeline

Run the scripts in order to process data and build ETA models:

```bash
python analysis/01_data_report.py
python analysis/02_clean_report.py
python analysis/03_eta_baseline.py
python analysis/04_eta_diagnostics.py
python analysis/05_static_gtfs_prepare.py
python analysis/06_stop_to_stop_history.py
python analysis/07_next_stop_eta.py
```

### 5. Launch the dashboard

```bash
streamlit run frontend/app.py
```

The dashboard has 5 tabs:

| Tab | What it shows |
|-----|---------------|
| 📡 Live Monitor | Map + vehicle table with real-time positions |
| ⏱ ETA | Next-stop arrival estimates with quality scores |
| 📊 Diagnostics | Latency, ping gaps, daily volumes, speed distributions |
| 📈 Models | Side-by-side model comparison (MAE, RMSE, P95) |
| ⚠ Failures | Top 100 worst prediction errors for debugging |

## Model Comparison

Three ETA models evaluated on **331,624 samples**:

| Model | MAE (s) | Median Error (s) | RMSE (s) | P95 Error (s) |
|-------|---------|-------------------|----------|----------------|
| **Historical median** | **12.9** | **1.0** | **26.7** | **58.0** |
| Rolling mean speed | 20.8 | 12.6 | 41.1 | 68.3 |
| Rolling median speed | 31.8 | 12.8 | 99.5 | 109.7 |

The historical-median model groups past travel times by **(hour-of-day bucket × vehicle status)** and uses the group median as the ETA. It outperforms the rolling-speed baselines because transit vehicles spend significant time stopped at lights and stations, making instantaneous speed a poor predictor.

## Data

> Raw data files are excluded from the repo via `.gitignore` (too large).

- **Source**: [TriMet GTFS-RT API](https://developer.trimet.org/)
- **Period**: Feb 23 – Apr 4, 2026 (40 days)
- **Volume**: 7.7 M+ vehicle position pings
- **Coverage**: 84 routes, 140+ concurrent vehicles, 96% fleet coverage

## Tech Stack

- **Ingestion**: Python, `requests`, `protobuf`, `gtfs-realtime-bindings`
- **Analysis**: `pandas`, `numpy`, `matplotlib`
- **Dashboard**: `streamlit`
- **Data format**: GTFS-Realtime (Protobuf) → CSV

## License

This project is for educational and portfolio purposes.
