import os
import time
import csv
import requests
from datetime import datetime, timezone
from google.transit import gtfs_realtime_pb2

# =========================
# CONFIG
# =========================
APP_ID = os.environ.get("TRIMET_APP_ID")
BASE_URL = "https://developer.trimet.org/ws/V1/VehiclePositions"
POLL_INTERVAL_SECONDS = 15

if not APP_ID:
    raise ValueError("Please set environment variable TRIMET_APP_ID")

GTFS_RT_URL = f"{BASE_URL}?appID={APP_ID}"

# =========================
# RAW DATA DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def get_output_path():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(RAW_DIR, f"{today}_vehicle_positions.csv")

def initialize_file_if_needed(path):
    file_exists = os.path.isfile(path)
    if not file_exists:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "receive_time_utc",
                "event_time_utc",
                "vehicle_id",
                "route_id",
                "trip_id",
                "latitude",
                "longitude",
                "bearing",
                "speed",
                "current_status",
                "current_stop_sequence",
                "stop_id"
            ])

# =========================
# INGEST LOOP
# =========================
print("Starting TriMet GTFS-RT ingestion...")

while True:
    try:
        receive_time = datetime.now(timezone.utc).isoformat()
        output_path = get_output_path()
        initialize_file_if_needed(output_path)

        response = requests.get(GTFS_RT_URL, timeout=10)
        response.raise_for_status()

        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)

        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)

            for entity in feed.entity:
                if not entity.HasField("vehicle"):
                    continue

                vehicle = entity.vehicle
                position = vehicle.position

                event_time = None
                if vehicle.timestamp:
                    event_time = datetime.fromtimestamp(
                        vehicle.timestamp,
                        tz=timezone.utc
                    ).isoformat()

                writer.writerow([
                    receive_time,
                    event_time,
                    vehicle.vehicle.id if vehicle.vehicle.id else None,
                    vehicle.trip.route_id if vehicle.trip.route_id else None,
                    vehicle.trip.trip_id if vehicle.trip.trip_id else None,
                    position.latitude if position.HasField("latitude") else None,
                    position.longitude if position.HasField("longitude") else None,
                    position.bearing if position.HasField("bearing") else None,
                    position.speed if position.HasField("speed") else None,
                    vehicle.current_status,
                    vehicle.current_stop_sequence,
                    vehicle.stop_id if vehicle.stop_id else None
                ])

        print(f"[{receive_time}] Logged {len(feed.entity)} entities")

        time.sleep(POLL_INTERVAL_SECONDS)

    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(POLL_INTERVAL_SECONDS)