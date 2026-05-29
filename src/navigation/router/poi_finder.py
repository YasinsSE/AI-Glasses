"""POI finder — locates places of a given category in an .osm file.

Finds places of a given category (pharmacy, hospital, market, …) in an .osm
file, returns the nearest POI, and can be used directly with NavigationSystem.

Example:
    finder = POIFinder("map.osm")
    result = finder.find_nearest(Coord(39.924, 32.845), category="pharmacy")
    if result:
        nav.start_navigation(my_position, result.coord)
"""

import xml.sax as sax
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import Coord
from .geo_utils import haversine_distance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POI categories -> OSM tag equivalents
# ---------------------------------------------------------------------------

# Spoken/typed keyword -> OSM amenity/shop/healthcare tag value. Turkish keys
# are intentional: they are the words the user actually speaks.
CATEGORY_MAP: Dict[str, List[str]] = {
    # Health
    "eczane":        ["pharmacy"],
    "pharmacy":      ["pharmacy"],
    "hastane":       ["hospital", "clinic"],
    "hospital":      ["hospital", "clinic"],
    "klinik":        ["clinic"],

    # Market / shopping
    "market":        ["supermarket", "convenience", "grocery"],
    "supermarket":   ["supermarket"],
    "bakkal":        ["convenience", "grocery"],

    # Other
    "restoran":      ["restaurant", "fast_food", "food_court"],
    "restaurant":    ["restaurant", "fast_food"],
    "kafe":          ["cafe"],
    "cafe":          ["cafe"],
    "benzin":        ["fuel"],
    "fuel":          ["fuel"],
    "atm":           ["atm", "bank"],
    "banka":         ["bank"],
    "park":          ["park"],
    "okul":          ["school", "kindergarten"],
    "polis":         ["police"],
    "itfaiye":       ["fire_station"],
    "otopark":       ["parking"],
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class POIResult:
    """A single discovered POI."""
    name: str
    category: str          # Category keyword the user entered.
    osm_type: str          # amenity/shop value from OSM.
    coord: Coord
    distance_m: float      # Distance from the user (metres).

    def __str__(self) -> str:
        return f"{self.name} ({self.osm_type}) — {int(self.distance_m)} m away"


# ---------------------------------------------------------------------------
# SAX parser — reads only nodes and their relevant tags
# ---------------------------------------------------------------------------

class _POIHandler(sax.ContentHandler):
    """Scans the OSM file and collects POI nodes."""

    def __init__(self, osm_types: List[str]) -> None:
        self.osm_types = set(osm_types)
        self.results: List[dict] = []          # Raw data.

        self._curr_node: Optional[dict] = None
        self._curr_tags: Dict[str, str] = {}

    def startElement(self, name: str, attrs) -> None:
        if name == "node":
            self._curr_node = {
                "lat": float(attrs["lat"]),
                "lon": float(attrs["lon"]),
            }
            self._curr_tags = {}
        elif name == "tag" and self._curr_node is not None:
            self._curr_tags[attrs["k"]] = attrs["v"]

    def endElement(self, name: str) -> None:
        if name == "node" and self._curr_node is not None:
            self._check_and_save()
            self._curr_node = None
            self._curr_tags = {}

    def _check_and_save(self) -> None:
        """If the current node is a relevant POI, add it to the list."""
        amenity = self._curr_tags.get("amenity", "")
        shop    = self._curr_tags.get("shop", "")
        osm_val = amenity or shop

        if osm_val in self.osm_types:
            self.results.append({
                "lat":      self._curr_node["lat"],
                "lon":      self._curr_node["lon"],
                "name":     self._curr_tags.get("name", "Unnamed"),
                "osm_type": osm_val,
            })


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class POIFinder:
    """Finds nearby POIs in an .osm file.

    Re-reads the OSM file on every search — this can be slow on large maps.
    A cache (e.g. a load_map() method) could be added if needed.

    Args:
        osm_file: Path to the .osm file to use.
    """

    def __init__(self, osm_file: str) -> None:
        self.osm_file = osm_file

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_nearest(
        self,
        position: Coord,
        category: str,
        max_results: int = 1,
        radius_m: Optional[float] = None,
    ) -> Optional[POIResult]:
        """Return the POI nearest to the given position.

        Args:
            position:    Current position.
            category:    Turkish or English category name
                         ("eczane", "pharmacy", "hastane", "market", …).
            max_results: How many results to return (default 1 = nearest only).
            radius_m:    Only include POIs within this distance (None = unbounded).

        Returns:
            The nearest POIResult, or None if none found.
        """
        all_results = self.find_all(position, category, radius_m)
        if not all_results:
            return None
        return all_results[0]

    def find_all(
        self,
        position: Coord,
        category: str,
        radius_m: Optional[float] = None,
    ) -> List[POIResult]:
        """Return all matching POIs, sorted by distance.

        Args:
            position:  Current position.
            category:  Category name.
            radius_m:  Distance filter (metres). None = unbounded.

        Returns:
            A list of POIResult, ordered nearest to farthest.
        """
        osm_types = self._resolve_category(category)
        if not osm_types:
            logger.warning(f"[POIFinder] Unknown category: '{category}'")
            logger.info(f"[POIFinder] Valid categories: {sorted(CATEGORY_MAP.keys())}")
            return []

        logger.info(f"[POIFinder] Scanning {self.osm_file} for '{category}'...")
        raw_pois = self._parse(osm_types)

        results: List[POIResult] = []
        for poi in raw_pois:
            dist = haversine_distance(
                position.lat, position.lon,
                poi["lat"], poi["lon"],
            )
            if radius_m is not None and dist > radius_m:
                continue
            results.append(POIResult(
                name=poi["name"],
                category=category,
                osm_type=poi["osm_type"],
                coord=Coord(poi["lat"], poi["lon"]),
                distance_m=dist,
            ))

        results.sort(key=lambda r: r.distance_m)
        logger.info(f"[POIFinder] {len(results)} result(s) found.")
        return results

    def list_categories(self) -> List[str]:
        """Return the supported category names."""
        return sorted(CATEGORY_MAP.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_category(self, category: str) -> List[str]:
        """Translate the user keyword into a list of OSM tags."""
        return CATEGORY_MAP.get(category.lower().strip(), [])

    def _parse(self, osm_types: List[str]) -> List[dict]:
        """Scan the OSM file with SAX and collect the relevant nodes."""
        handler = _POIHandler(osm_types)
        parser = sax.make_parser()
        parser.setContentHandler(handler)
        try:
            parser.parse(self.osm_file)
        except Exception as e:
            logger.error(f"[POIFinder] OSM parse error: {e}")
            return []
        return handler.results
