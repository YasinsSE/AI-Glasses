# osm_parser.py
# Reads an .osm file and builds an in-memory routing graph.
# Depends only on: geo_utils, nav_config — nothing else from this project.

import xml.sax as sax
from typing import Dict, List, Optional

from .geo_utils import haversine_distance
from .nav_config import NavConfig, WALKABLE_TYPES, FORBIDDEN_TYPES


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

class Edge:
    """Directed weighted edge between two graph nodes."""

    __slots__ = ["target", "distance", "time", "name", "road_type"]

    def __init__(
        self,
        target: "Node",
        distance: float,
        road_type: str,
        name: str,
        walking_speed_kmh: float,
        steps_penalty: float,
    ) -> None:
        self.target = target
        self.distance = distance
        self.name = name
        self.road_type = road_type

        speed_ms = walking_speed_kmh * 1000 / 3600
        factor = steps_penalty if road_type == "steps" else 1.0
        self.time = (distance / speed_ms) * factor


class Node:
    """A graph node representing an OSM node (intersection or shape point)."""

    __slots__ = ["id", "lat", "lon", "edges"]

    def __init__(self, nid: str, lat: float, lon: float) -> None:
        self.id = nid
        self.lat = float(lat)
        self.lon = float(lon)
        self.edges: List[Edge] = []


# ---------------------------------------------------------------------------
# Routing graph
# ---------------------------------------------------------------------------

class RoutingDB:
    """In-memory graph of walkable nodes and edges."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(
        self,
        u: str,
        v: str,
        road_type: str,
        name: str,
        config: NavConfig,
    ) -> None:
        n1 = self.nodes.get(u)
        n2 = self.nodes.get(v)
        if not n1 or not n2:
            return
        d = haversine_distance(n1.lat, n1.lon, n2.lat, n2.lon)
        edge_kwargs = dict(
            distance=d,
            road_type=road_type,
            name=name,
            walking_speed_kmh=config.walking_speed_kmh,
            steps_penalty=config.steps_time_penalty,
        )
        n1.edges.append(Edge(target=n2, **edge_kwargs))
        n2.edges.append(Edge(target=n1, **edge_kwargs))

    def cleanup(self) -> None:
        """Remove isolated nodes (no edges) to save memory."""
        self.nodes = {k: v for k, v in self.nodes.items() if v.edges}


# ---------------------------------------------------------------------------
# SAX content handler
# ---------------------------------------------------------------------------

class OSMHandler(sax.ContentHandler):
    """Stream-parse an OSM XML file and populate a RoutingDB."""

    def __init__(self, db: RoutingDB, config: NavConfig) -> None:
        self.db = db
        self.config = config
        self._curr_nodes: List[str] = []
        self._tags: Dict[str, str] = {}
        self._in_way = False

    def startElement(self, name: str, attrs) -> None:  # type: ignore[override]
        if name == "node":
            self.db.add_node(Node(attrs["id"], attrs["lat"], attrs["lon"]))
        elif name == "way":
            self._in_way = True
            self._curr_nodes = []
            self._tags = {}
        elif name == "nd" and self._in_way:
            self._curr_nodes.append(attrs["ref"])
        elif name == "tag" and self._in_way:
            self._tags[attrs["k"]] = attrs["v"]

    def endElement(self, name: str) -> None:  # type: ignore[override]
        if name == "way":
            self._process_way()
            self._in_way = False

    def _process_way(self) -> None:
        if "highway" not in self._tags:
            return
        road_type = self._tags["highway"]
        if road_type in FORBIDDEN_TYPES or road_type not in WALKABLE_TYPES:
            return
        name = self._tags.get("name", "Unnamed road")
        for i in range(len(self._curr_nodes) - 1):
            self.db.add_edge(
                self._curr_nodes[i],
                self._curr_nodes[i + 1],
                road_type,
                name,
                self.config,
            )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_map(osm_file: str, config: Optional[NavConfig] = None) -> RoutingDB:
    """
    Parse an OSM file and return a populated RoutingDB.

    Args:
        osm_file: Path to the .osm file.
        config:   NavConfig instance (defaults to NavConfig() if omitted).

    Returns:
        RoutingDB ready for route calculation.

    Raises:
        FileNotFoundError: If osm_file does not exist.
        xml.sax.SAXParseException: If the file is not valid XML.
    """
    if config is None:
        config = NavConfig()

    print(f"[OSMParser] Loading map: {osm_file}")
    db = RoutingDB()
    parser = sax.make_parser()
    parser.setContentHandler(OSMHandler(db, config))
    with open(osm_file, 'r', encoding='utf-8') as f:
        parser.parse(f)
    db.cleanup()
    print(f"[OSMParser] Ready — {len(db.nodes)} routable nodes.")
    return db
