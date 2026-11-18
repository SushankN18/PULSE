from __future__ import annotations

import os
import sys
import time
import pandas as pd
import streamlit as st

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from realtime.state_store import live_state
from realtime.realtime_utils import vehicles_to_dataframe

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="PULSE Dashboard",
    layout="wide"
)

st.title("🚍 PULSE — Real-Time Transport Monitoring")
st.caption("Live vehicle tracking • ETA estimation • Diagnostics • Model insights")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Controls")

auto_refresh = st.sidebar.checkbox("Auto Refresh", True)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 5, 30, 10)

# -----------------------------
# LOAD LIVE STATE
# -----------------------------
vehicles = live_state.get_all_vehicles()
meta = live_state.get_meta()

live_df = vehicles_to_dataframe(vehicles)

# -----------------------------
# ROUTE FILTER
# -----------------------------
route_filter = None

if not live_df.empty and "route_id" in live_df.columns:
    routes = sorted(live_df["route_id"].dropna().astype(str).unique())
    route_filter = st.sidebar.selectbox("Route Filter", ["ALL"] + routes)

    if route_filter != "ALL":
        live_df = live_df[live_df["route_id"].astype(str) == route_filter]

# -----------------------------
# TABS
# -----------------------------
tab_live, tab_eta, tab_diag, tab_models, tab_fail = st.tabs(
    ["📡 Live Monitor", "⏱ ETA", "📊 Diagnostics", "📈 Models", "⚠ Failures"]
)

# =========================================================
# LIVE MONITOR
# =========================================================
with tab_live:

    st.subheader("Live Vehicle State")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Vehicles", len(live_df))
    c2.metric("Feed Entities", meta.get("last_feed_entity_count", 0))
    c3.metric("Last Success", str(meta.get("last_success", False)))
    c4.metric("Route", route_filter if route_filter else "ALL")

    st.divider()

    if not live_df.empty:

        # MAP
        map_df = live_df.dropna(subset=["latitude", "longitude"])
        if not map_df.empty:
            map_df = map_df.rename(columns={"latitude": "lat", "longitude": "lon"})
            st.map(map_df[["lat", "lon"]])

        st.write("### Vehicle Snapshot")

        display_cols = [
            "vehicle_id",
            "route_id",
            "current_status",
            "current_stop_sequence",
            "stop_id",
            "eta_min",
            "eta_quality",
            "latency_sec",
        ]

        display_cols = [c for c in display_cols if c in live_df.columns]

        st.dataframe(
            live_df[display_cols].sort_values("vehicle_id"),
            use_container_width=True
        )

    else:
        st.warning("No live data available. Start realtime_monitor.py.")

# =========================================================
# ETA TAB
# =========================================================
with tab_eta:

    st.subheader("Next-Stop ETA (Live)")

    if live_df.empty:
        st.warning("No live data available.")
    else:

        eta_available = live_df["eta_min"].notna()

        c1, c2, c3 = st.columns(3)

        c1.metric("Vehicles with ETA", int(eta_available.sum()))
        c2.metric("Strong ETA", int((live_df["eta_quality"] == "strong").sum()))
        c3.metric("Coverage", f"{eta_available.mean()*100:.1f}%")

        st.divider()

        # SORTED TABLE
        st.write("### ETA Table")

        eta_df = live_df[live_df["eta_min"].notna()].copy()

        display_cols = [
            "vehicle_id",
            "route_id",
            "current_stop_sequence",
            "stop_id",
            "eta_min",
            "eta_quality",
        ]

        display_cols = [c for c in display_cols if c in eta_df.columns]

        st.dataframe(
            eta_df[display_cols].sort_values("eta_min"),
            use_container_width=True
        )

        # ETA DISTRIBUTION
        st.write("### ETA Distribution")

        if not eta_df.empty:
            st.bar_chart(eta_df["eta_min"])

        # QUALITY DISTRIBUTION
        st.write("### ETA Quality Distribution")

        st.bar_chart(live_df["eta_quality"].value_counts())

        st.markdown("""
        **Interpretation**
        - Strong → high-confidence ETA  
        - Moderate → reasonable estimate  
        - Weak → limited data  
        - Missing → no historical match  
        """)

# =========================================================
# DIAGNOSTICS
# =========================================================
with tab_diag:

    st.subheader("Data Diagnostics")

    OUTPUTS_PLOTS = os.path.join(BASE_DIR, "outputs", "plots")

    plots = [
        ("Latency Distribution", "latency_distribution.png"),
        ("Gap Distribution", "ping_gap_distribution.png"),
        ("Rows per Day", "daily_row_counts.png"),
        ("Speed Distribution", "route_speed_distribution.png"),
    ]

    for title, file in plots:
        st.write(f"### {title}")
        path = os.path.join(OUTPUTS_PLOTS, file)
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info(f"Missing: {file}")

# =========================================================
# MODELS
# =========================================================
with tab_models:

    st.subheader("Model Insights")

    OUTPUTS_PLOTS = os.path.join(BASE_DIR, "outputs", "plots")
    OUTPUTS_TABLES = os.path.join(BASE_DIR, "outputs", "tables")

    plots = [
        ("Model Comparison", "eta_model_comparison.png"),
        ("Error Distribution", "eta_error_distribution_comparison.png"),
        ("Error by Speed Regime", "eta_error_by_speed_regime.png"),
        ("Error by Status", "eta_error_by_status.png"),
    ]

    for title, file in plots:
        st.write(f"### {title}")
        path = os.path.join(OUTPUTS_PLOTS, file)
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info(f"Missing: {file}")

    comp_file = os.path.join(OUTPUTS_TABLES, "eta_model_comparison.csv")

    if os.path.exists(comp_file):
        comp_df = pd.read_csv(comp_file)
        st.dataframe(comp_df, use_container_width=True)

        best_model = comp_df.sort_values("mae_sec").iloc[0]["model"]
        st.success(f"Best Model: {best_model}")

# =========================================================
# FAILURES
# =========================================================
with tab_fail:

    st.subheader("Failure Analysis")

    OUTPUTS_TABLES = os.path.join(BASE_DIR, "outputs", "tables")
    file = os.path.join(OUTPUTS_TABLES, "eta_worst_cases_top100.csv")

    if os.path.exists(file):
        df = pd.read_csv(file)
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Failure table not found.")

# =========================================================
# AUTO REFRESH
# =========================================================
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()