# poi_finder.py
# .osm dosyasından belirli kategorideki yerleri (eczane, hastane, market vb.) bulur.
# En yakın POI'yi döndürür ve NavigationSystem ile doğrudan kullanılabilir.
#
# Kullanım:
#   finder = POIFinder("map.osm")
#   result = finder.find_nearest(Coord(39.924, 32.845), category="pharmacy")
#   if result:
#       nav.start_navigation(my_position, result.coord)

import xml.sax as sax
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models import Coord
from geo_utils import haversine_distance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POI kategorileri → OSM tag karşılıkları
# ---------------------------------------------------------------------------

# Komuttan gelen kelime → OSM amenity/shop/healthcare tag değeri
CATEGORY_MAP: Dict[str, List[str]] = {
    # Sağlık
    "eczane":        ["pharmacy"],
    "pharmacy":      ["pharmacy"],
    "hastane":       ["hospital", "clinic"],
    "hospital":      ["hospital", "clinic"],
    "klinik":        ["clinic"],

    # Market / Alışveriş
    "market":        ["supermarket", "convenience", "grocery"],
    "supermarket":   ["supermarket"],
    "bakkal":        ["convenience", "grocery"],

    # Diğer
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
# Veri modeli
# ---------------------------------------------------------------------------

@dataclass
class POIResult:
    """Bulunan bir POI noktası."""
    name: str
    category: str          # kullanıcının girdiği kategori kelimesi
    osm_type: str          # OSM'deki amenity/shop değeri
    coord: Coord
    distance_m: float      # kullanıcıdan uzaklık (metre)

    def __str__(self) -> str:
        return f"{self.name} ({self.osm_type}) — {int(self.distance_m)} m uzakta"


# ---------------------------------------------------------------------------
# SAX parser — sadece node'ları ve ilgili tag'leri okur
# ---------------------------------------------------------------------------

class _POIHandler(sax.ContentHandler):
    """OSM dosyasını tarayıp POI node'larını toplar."""

    def __init__(self, osm_types: List[str]) -> None:
        self.osm_types = set(osm_types)
        self.results: List[dict] = []          # ham veri

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
        """Node ilgili bir POI mi? Öyleyse listeye ekle."""
        amenity = self._curr_tags.get("amenity", "")
        shop    = self._curr_tags.get("shop", "")
        osm_val = amenity or shop

        if osm_val in self.osm_types:
            self.results.append({
                "lat":      self._curr_node["lat"],
                "lon":      self._curr_node["lon"],
                "name":     self._curr_tags.get("name", "İsimsiz"),
                "osm_type": osm_val,
            })


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class POIFinder:
    """
    .osm dosyasında yakın çevredeki POI'leri bulan modül.

    OSM dosyasını her aramada baştan okur — büyük haritalarda
    yavaş olabilir. İstersen load_map() gibi bir önbellek eklenebilir.

    Args:
        osm_file: Kullandığın .osm dosyasının yolu.
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
        """
        Konuma en yakın POI'yi döndürür.

        Args:
            position:    Şu anki konumun.
            category:    Türkçe veya İngilizce kategori adı
                         ("eczane", "pharmacy", "hastane", "market" …)
            max_results: Kaç tane sonuç döndürülsün (default 1 = sadece en yakın).
            radius_m:    Sadece bu mesafe içindeki POI'leri dahil et (None = sınırsız).

        Returns:
            En yakın POIResult, bulunamazsa None.
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
        """
        Kategoriye uyan tüm POI'leri mesafeye göre sıralı döndürür.

        Args:
            position:  Şu anki konumun.
            category:  Kategori adı.
            radius_m:  Mesafe filtresi (metre). None = sınırsız.

        Returns:
            POIResult listesi, en yakından en uzağa sıralı.
        """
        osm_types = self._resolve_category(category)
        if not osm_types:
            logger.warning(f"[POIFinder] Bilinmeyen kategori: '{category}'")
            logger.info(f"[POIFinder] Geçerli kategoriler: {sorted(CATEGORY_MAP.keys())}")
            return []

        logger.info(f"[POIFinder] '{category}' için {self.osm_file} taranıyor...")
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
        logger.info(f"[POIFinder] {len(results)} sonuç bulundu.")
        return results

    def list_categories(self) -> List[str]:
        """Desteklenen kategori isimlerini döndürür."""
        return sorted(CATEGORY_MAP.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_category(self, category: str) -> List[str]:
        """Kullanıcı kelimesini OSM tag listesine çevirir."""
        return CATEGORY_MAP.get(category.lower().strip(), [])

    def _parse(self, osm_types: List[str]) -> List[dict]:
        """OSM dosyasını SAX ile tara, ilgili node'ları topla."""
        handler = _POIHandler(osm_types)
        parser = sax.make_parser()
        parser.setContentHandler(handler)
        try:
            parser.parse(self.osm_file)
        except Exception as e:
            logger.error(f"[POIFinder] OSM parse hatası: {e}")
            return []
        return handler.results
