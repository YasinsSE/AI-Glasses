# main.py
# Entry point — simulates a GPS loop feeding positions into NavigationSystem.
# In production, replace the test_locations loop with your real GPS source.
#
# For usage there are two mode nav.navigate_to_nearest(current_location, "keyword") or nav.start_navigation(current_location, target_location)
# For using GPS loop use nav.update(current_location) and it will give a feedback

import logging
import time

from src.navigation.router.models import Coord, RouteStatus
from src.navigation.router.nav_config import NavConfig
from src.navigation.router.navigator import NavigationSystem

# from .models import Coord, RouteStatus
# from .nav_config import NavConfig
# from .navigator import NavigationSystem

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
    # log_dir defaults to src/navigation/router/ via NavConfig._ROUTER_DIR
)

# ------------------------------------------------------------------
# Simulation coordinates (Sıhhiye → Kurtuluş, Ankara)
# ------------------------------------------------------------------
test_locations = [
    Coord(39.924134, 32.845449),
    Coord(39.9919, 32.8649),   # Start
    Coord(39.9887, 32.8635), # Step 8: arrival
    Coord(39.926414, 32.844915),
]

ORIGIN      = test_locations[0]
DESTINATION = test_locations[-1]


def main() -> None:
    # 1. Boot system (loads OSM map once)
    start_boot = time.perf_counter()
    nav = NavigationSystem("/home/alas/ALAS_PROJECT/AI-Glasses/src/navigation/router/map.osm", config=config)
    end_boot = time.perf_counter()
    print(f"[*] Harita yüklenme süresi: {end_boot - start_boot:.4f}saniye")
    # 2. Request a route
    start_boot = time.perf_counter()
    #success, msg, poi = nav.navigate_to_nearest(ORIGIN, "okul")
    success, msg = nav.start_navigation(ORIGIN, DESTINATION)
    end_boot = time.perf_counter()
    if not success:
        print(f"[Main] Could not start navigation: {msg}")
        return
    print(f"[*] Rota hesaplama süresi: {end_boot - start_boot:.4f}saniye")
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
    print(f"    Route saved to: {config.route_filepath}")
    #print(nav.list_poi_categories())

if __name__ == "__main__":
    main()
