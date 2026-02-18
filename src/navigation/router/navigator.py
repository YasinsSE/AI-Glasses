# navigator.py
# Public entry point for the navigation system.
# Owns no business logic — delegates everything to specialist modules.

import logging
from typing import Optional, Tuple, List

from models import Coord, ProgressResult, RouteStatus
from nav_config import NavConfig
from osm_parser import RoutingDB, load_map
from route_calculator import RouteCalculator
from route_tracker import RouteTracker
from nav_logger import NavLogger
from poi_finder import POIFinder, POIResult

logger = logging.getLogger(__name__)


class NavigationSystem:
    """
    High-level navigation facade.

    Typical lifecycle:
        nav = NavigationSystem("map.osm")
        nav.start_navigation(Coord(39.924, 32.845), Coord(39.921, 32.852))

        # GPS loop:
        result = nav.update(Coord(lat, lon))

    POI usage:
        nav.navigate_to_nearest(my_position, "eczane")

    Args:
        osm_map_path: Path to the .osm map file.
        config:       Optional NavConfig; defaults to NavConfig().
    """

    def __init__(self, osm_map_path: str, config: Optional[NavConfig] = None) -> None:
        self.config = config or NavConfig()
        self._osm_path = osm_map_path

        # Load map once at startup
        self._db: RoutingDB = load_map(osm_map_path, self.config)

        # Specialist modules
        self._calculator = RouteCalculator(self._db, self.config)
        self._tracker    = RouteTracker(self.config)
        self._logger     = NavLogger(self.config)
        self._poi_finder = POIFinder(osm_map_path)

    # ------------------------------------------------------------------
    # Navigation control
    # ------------------------------------------------------------------

    def start_navigation(self, origin: Coord, destination: Coord) -> Tuple[bool, str]:
        """
        Calculate a route and begin tracking.

        Args:
            origin:      Starting coordinate.
            destination: Target coordinate.

        Returns:
            (success, message)
        """
        logger.info(f"Calculating route: {origin} → {destination}")
        steps, msg = self._calculator.calculate(origin, destination)

        if not steps:
            logger.warning(f"Route calculation failed: {msg}")
            return False, msg

        self._tracker.load_route(steps)
        self._logger.save_route(steps)

        first_instruction = steps[0].text
        logger.info(f"Route ready — {len(steps)} steps. First: {first_instruction}")
        print(f"[Nav] Route ready — {len(steps)} steps.")
        print(f"[Nav] {first_instruction}")
        return True, f"Route ready. {len(steps)} steps."

    def stop_navigation(self) -> None:
        """Forcibly end the current navigation session."""
        self._tracker.stop()
        logger.info("Navigation stopped by user.")

    # ------------------------------------------------------------------
    # POI navigation
    # ------------------------------------------------------------------
    
    def navigate_to_nearest(
        self,
        position: Coord,
        category: str,
        radius_m: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[POIResult]]:
        """
        En yakın POI'yi bul ve oraya navigasyonu başlat.

        Args:
            position:  Şu anki konumun.
            category:  
            'atm', 'bakkal', 'banka', 'benzin', 'cafe', 'eczane',
            'fuel', 'hastane', 'hospital', 'itfaiye', 'kafe', 
            'klinik', 'market', 'okul', 'otopark', 'park', 
            'pharmacy', 'polis', 'restaurant', 'restoran', 
            'supermarket'
            radius_m:  Sadece bu mesafe içinde ara (None = sınırsız).

        Returns:
            (success, message, poi_result)
            poi_result → bulunan yer bilgisi, başarısızsa None.
        """
        poi = self._poi_finder.find_nearest(position, category, radius_m=radius_m)

        if not poi:
            msg = f"'{category}' kategorisinde yakında yer bulunamadı."
            logger.warning(f"[Nav] {msg}")
            return False, msg, None

        logger.info(f"[Nav] Hedef: {poi}")
        print(f"[Nav] Hedef bulundu: {poi}")

        success, msg = self.start_navigation(position, poi.coord)
        return success, msg, poi

    def find_nearby(
        self,
        position: Coord,
        category: str,
        radius_m: Optional[float] = None,
    ) -> List[POIResult]:
        """
        Kategoriye uyan tüm yakın POI'leri listele (navigasyon başlatmaz).

        Args:
            position:  Şu anki konumun.
            category:  Kategori adı.
            radius_m:  Mesafe filtresi.

        Returns:
            POIResult listesi, en yakından uzağa sıralı.
        """
        return self._poi_finder.find_all(position, category, radius_m)

    def list_poi_categories(self) -> List[str]:
        """Desteklenen POI kategori isimlerini döndürür."""
        return self._poi_finder.list_categories()

    # ------------------------------------------------------------------
    # GPS update — call this on every position fix
    # ------------------------------------------------------------------

    def update(self, position: Coord) -> ProgressResult:
        """
        Process a new GPS position and return the current navigation status.

        Args:
            position: Current geographic coordinate.

        Returns:
            ProgressResult containing RouteStatus, message, and step info.
        """
        result = self._tracker.check_progress(position)
        self._logger.log_event(result, position.lat, position.lon)
        return result

    # ------------------------------------------------------------------
    # Convenience read-only properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._tracker.is_active

    @property
    def remaining_steps(self) -> int:
        return self._tracker.remaining_steps