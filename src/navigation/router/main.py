# main.py
# Entry point — simulates a GPS loop feeding positions into NavigationSystem.
# In production, replace the test_locations loop with your real GPS source.
#
# For usage there are two mode nav.navigate_to_nearest(current_location, "keyword") or nav.start_navigation(current_location, target_location)
# For using GPS loop use nav.update(current_location) and it will give a feedback

import logging
import time

from models import Coord, RouteStatus
from nav_config import NavConfig
from navigator import NavigationSystem

# ------------------------------------------------------------------
# Logging setup — configure once here, all modules inherit
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ------------------------------------------------------------------
# Config — tweak thresholds or paths here, not inside the modules
# ------------------------------------------------------------------
config = NavConfig(
    waypoint_threshold_m=15.0,
    off_route_threshold_m=40.0,
    log_dir="logs",
    route_filename="active_route.json",
)

# ------------------------------------------------------------------
# Simulation coordinates (Sıhhiye → Kurtuluş, Ankara)
# ------------------------------------------------------------------
test_locations = [
    Coord(39.92409,  32.845382),   # Start
    Coord(39.9240467, 32.8451522), # Step 1: straight
    Coord(39.9232599, 32.8441792), # Step 2: straight
    Coord(39.9240102, 32.8452347), # Step 3: turn begins
    Coord(39.9249406, 32.8462865), # Step 4: straight
    Coord(39.9254588, 32.8477125), # Step 5: turn right
    Coord(39.9208164, 32.8533392), # Step 6: long walk (sharp left)
    Coord(39.920927,  32.8533893), # Step 7: sharp left
    Coord(39.9210086, 32.8529793), # Step 8: arrival
]

ORIGIN      = test_locations[0]
DESTINATION = test_locations[-1]


def main() -> None:
    # 1. Boot system (loads OSM map once)
    nav = NavigationSystem("map.osm", config=config)

    # 2. Request a route
    success, msg, poi = nav.navigate_to_nearest(ORIGIN, "atm")
    #success, msg = nav.start_navigation(ORIGIN, DESTINATION)
    if not success:
        print(f"[Main] Could not start navigation: {msg}")
        return

    print("\n--- GPS Loop Active ---")

    # 3. GPS loop — replace with real GPS feed in production
    for position in test_locations:
        result = nav.update(position)

        print(f"  GPS {position} → [{result.status.name}] {result.message}")

        # React to status
        if result.status == RouteStatus.OFF_ROUTE:
            print("  ⚠  Off-route detected — you could trigger rerouting here.")

        elif result.status == RouteStatus.FINISHED:
            print("  ✓  Destination reached. Navigation ended.")
            break

        # Simulate GPS poll interval (remove in real use)
        time.sleep(0.05)

    print("\n--- Session complete ---")
    print(f"    Log files written to: {config.log_dir}/")
    #print(nav.list_poi_categories())

if __name__ == "__main__":
    main()
