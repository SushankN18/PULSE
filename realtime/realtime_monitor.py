from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from realtime.state_store import live_state
from realtime.realtime_utils import (
    fetch_gtfs_realtime_feed,
    parse_vehicle_positions,
    load_stop_history,
    compute_live_eta
)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

APP_ID = os.environ.get("TRIMET_APP_ID")
BASE_URL = "https://developer.trimet.org/ws/V1/VehiclePositions"
POLL_INTERVAL = 10

GTFS_RT_URL = f"{BASE_URL}?appID={APP_ID}"

# Load stop history once
HISTORY_FILE = os.path.join(BASE_DIR, "data", "processed", "stop_to_stop_history.csv")
history_df = load_stop_history(HISTORY_FILE)

def run_monitor():
    print("Starting realtime monitor with ETA...")

    while True:
        now = datetime.now(timezone.utc).isoformat()

        try:
            feed = fetch_gtfs_realtime_feed(GTFS_RT_URL)
            vehicles = parse_vehicle_positions(feed)

            # 🔥 Compute ETA here
            vehicles = compute_live_eta(vehicles, history_df)

            live_state.bulk_replace(vehicles)

            live_state.update_meta(
                last_poll_time_utc=now,
                last_feed_entity_count=len(feed.entity),
                last_vehicle_count=len(vehicles),
                last_success=True,
            )

            print(f"[{now}] vehicles={len(vehicles)} (ETA computed)")

        except Exception as e:
            print(f"[{now}] ERROR: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_monitor()