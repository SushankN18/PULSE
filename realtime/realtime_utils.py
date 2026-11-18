from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_get_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def fetch_gtfs_realtime_feed(url: str, timeout: int = 10) -> gtfs_realtime_pb2.FeedMessage:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(response.content)
    return feed


def parse_vehicle_positions(feed: gtfs_realtime_pb2.FeedMessage) -> dict[str, dict[str, Any]]:
    """
    Parse GTFS-RT VehiclePositions feed into a vehicle_id-keyed dictionary.
    """
    receive_time_utc = utc_now_iso()
    vehicles: dict[str, dict[str, Any]] = {}

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        vehicle = entity.vehicle
        position = vehicle.position

        vehicle_id = None
        if vehicle.HasField("vehicle") and vehicle.vehicle.id:
            vehicle_id = safe_get_str(vehicle.vehicle.id)

        if not vehicle_id:
            # Skip entities without a stable vehicle id
            continue

        event_time_utc = None
        if vehicle.timestamp:
            event_time_utc = datetime.fromtimestamp(
                vehicle.timestamp, tz=timezone.utc
            ).isoformat()

        route_id = safe_get_str(vehicle.trip.route_id) if vehicle.HasField("trip") else None
        trip_id = safe_get_str(vehicle.trip.trip_id) if vehicle.HasField("trip") else None
        stop_id = safe_get_str(vehicle.stop_id)

        current_status = str(vehicle.current_status) if hasattr(vehicle, "current_status") else None
        current_stop_sequence = (
            int(vehicle.current_stop_sequence)
            if hasattr(vehicle, "current_stop_sequence") and vehicle.current_stop_sequence is not None
            else None
        )

        lat = float(position.latitude) if position.HasField("latitude") else None
        lon = float(position.longitude) if position.HasField("longitude") else None
        bearing = float(position.bearing) if position.HasField("bearing") else None
        speed = float(position.speed) if position.HasField("speed") else None

        vehicles[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "route_id": route_id,
            "trip_id": trip_id,
            "latitude": lat,
            "longitude": lon,
            "bearing": bearing,
            "speed": speed,
            "current_status": current_status,
            "current_stop_sequence": current_stop_sequence,
            "stop_id": stop_id,
            "event_time_utc": event_time_utc,
            "receive_time_utc": receive_time_utc,
            "latency_sec": _compute_latency_seconds(receive_time_utc, event_time_utc),
        }

    return vehicles


def _compute_latency_seconds(receive_time_utc: str | None, event_time_utc: str | None) -> float | None:
    if not receive_time_utc or not event_time_utc:
        return None
    try:
        receive_time = pd.to_datetime(receive_time_utc, utc=True)
        event_time = pd.to_datetime(event_time_utc, utc=True)
        return float((receive_time - event_time).total_seconds())
    except Exception:
        return None


def vehicles_to_dataframe(vehicles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not vehicles:
        return pd.DataFrame(
            columns=[
                "vehicle_id", "route_id", "trip_id", "latitude", "longitude",
                "bearing", "speed", "current_status", "current_stop_sequence",
                "stop_id", "event_time_utc", "receive_time_utc", "latency_sec"
            ]
        )

    df = pd.DataFrame(list(vehicles.values()))
    return df


def load_static_gtfs(gtfs_dir: str) -> dict[str, pd.DataFrame]:
    """
    Loads a minimal set of GTFS static files if present.
    """
    files = {
        "stops": "stops.txt",
        "routes": "routes.txt",
        "trips": "trips.txt",
        "stop_times": "stop_times.txt",
    }

    loaded: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = os.path.join(gtfs_dir, filename)
        if os.path.exists(path):
            loaded[key] = pd.read_csv(path)
    return loaded


def enrich_with_next_stop_info(
    live_df: pd.DataFrame,
    static_gtfs: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Lightweight enrichment using trip_id + stop_times.
    This is NOT full stop-aware ETA yet; it gives next-stop context when available.
    """
    if live_df.empty:
        return live_df

    if "stop_times" not in static_gtfs:
        live_df["next_stop_id"] = None
        return live_df

    stop_times = static_gtfs["stop_times"].copy()

    required_cols = {"trip_id", "stop_sequence", "stop_id"}
    if not required_cols.issubset(set(stop_times.columns)):
        live_df["next_stop_id"] = None
        return live_df

    stop_times["trip_id"] = stop_times["trip_id"].astype(str)
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")

    live_df = live_df.copy()
    live_df["trip_id"] = live_df["trip_id"].astype(str)
    live_df["current_stop_sequence"] = pd.to_numeric(live_df["current_stop_sequence"], errors="coerce")
    live_df["next_stop_id"] = None

    next_stop_map: dict[tuple[str, int], str] = {}

    for trip_id, group in stop_times.groupby("trip_id"):
        group = group.sort_values("stop_sequence")
        stop_sequences = group["stop_sequence"].tolist()
        stop_ids = group["stop_id"].astype(str).tolist()

        for i in range(len(stop_sequences) - 1):
            cur_seq = stop_sequences[i]
            nxt_stop_id = stop_ids[i + 1]
            if pd.notna(cur_seq):
                next_stop_map[(str(trip_id), int(cur_seq))] = nxt_stop_id

    def lookup_next_stop(row: pd.Series) -> str | None:
        trip_id = row.get("trip_id")
        seq = row.get("current_stop_sequence")
        if pd.isna(seq) or trip_id in [None, "None", "nan"]:
            return None
        return next_stop_map.get((str(trip_id), int(seq)))

    live_df["next_stop_id"] = live_df.apply(lookup_next_stop, axis=1)
    return live_df

import pandas as pd

def load_stop_history(history_path):
    if not os.path.exists(history_path):
        return pd.DataFrame()
    df = pd.read_csv(history_path)
    df["route_id"] = df["route_id"].astype(str)
    # stop_id is stored as float (e.g. 13450.0) — convert to clean int string
    df["stop_id"] = pd.to_numeric(df["stop_id"], errors="coerce").astype("Int64").astype(str)
    return df


def compute_live_eta(vehicles: dict, history_df: pd.DataFrame):
    """
    Add ETA to each vehicle using historical stop-to-stop table.
    """
    if history_df.empty:
        return vehicles

    for vid, v in vehicles.items():

        route_id = str(v.get("route_id"))
        stop_id = str(v.get("stop_id"))
        seq = v.get("current_stop_sequence")

        if route_id is None or stop_id is None or seq is None:
            v["eta_min"] = None
            v["eta_quality"] = "missing"
            continue

        match = history_df[
            (history_df["route_id"] == route_id) &
            (history_df["stop_id"] == stop_id) &
            (history_df["current_stop_sequence"] == seq)
        ]

        if match.empty:
            v["eta_min"] = None
            v["eta_quality"] = "missing"
            continue

        row = match.iloc[0]

        eta_sec = row.get("median_travel_time_sec", None)
        obs = row.get("observations", 0)

        if pd.isna(eta_sec):
            v["eta_min"] = None
            v["eta_quality"] = "missing"
            continue

        v["eta_min"] = eta_sec / 60.0

        if obs >= 30:
            v["eta_quality"] = "strong"
        elif obs >= 10:
            v["eta_quality"] = "moderate"
        else:
            v["eta_quality"] = "weak"

    return vehicles