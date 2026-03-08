# route_calculator.py
# A* pathfinding on a RoutingDB graph.
# Returns a list of RouteStep objects.

import heapq
from typing import List, Optional, Tuple

from .models import Coord, RouteStep
from .geo_utils import haversine_distance, calculate_bearing, get_turn_instruction
from .nav_config import NavConfig
from .osm_parser import RoutingDB, Node


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_nearest_node(db: RoutingDB, coord: Coord) -> Tuple[Optional[Node], float]:
    """Return the closest graph node to coord and its distance in metres."""
    best_node = None
    min_dist = float("inf")
    for node in db.nodes.values():
        d = haversine_distance(coord.lat, coord.lon, node.lat, node.lon)
        if d < min_dist:
            min_dist = d
            best_node = node
    return best_node, min_dist


def _reconstruct_path(came_from, start, end):
    path = []
    curr = end
    visited_nodes: set = set()                       # ← YENİ
    while curr != start:
        if curr in visited_nodes:                    # ← YENİ
            raise RuntimeError("Cycle detected...") # ← YENİ
        visited_nodes.add(curr)                      # ← YENİ
        parent, edge = came_from[curr]
        path.append((parent, edge))
        curr = parent
    path.reverse()
    return path


def _build_steps(path, start_node: Node) -> List[RouteStep]:
    """Convert a raw A* path into human-readable RouteStep list."""
    steps: List[RouteStep] = []

    # Step 0: departure
    first_edge = path[0][1]
    steps.append(RouteStep(
        step_id=0,
        text=f"Navigation starting. You are on: {first_edge.name}",
        location=Coord(start_node.lat, start_node.lon),
        action="start",
        distance_meters=0,
        road_name=first_edge.name,
    ))

    curr_name = first_edge.name
    dist_accum = 0.0
    step_id = 1

    for i, (node, edge) in enumerate(path):
        dist_accum += edge.distance
        next_edge = path[i + 1][1] if i + 1 < len(path) else None

        # Emit a step when road name changes or we reach the end
        if not next_edge or next_edge.name != curr_name or next_edge.road_type != edge.road_type:
            action = "continue"
            turn_text = "Go straight"

            if next_edge:
                b1 = calculate_bearing(node.lat, node.lon, edge.target.lat, edge.target.lon)
                b2 = calculate_bearing(
                    edge.target.lat, edge.target.lon,
                    next_edge.target.lat, next_edge.target.lon,
                )
                turn_text = get_turn_instruction(b2 - b1)
                if "right" in turn_text.lower():
                    action = "turn_right"
                elif "left" in turn_text.lower():
                    action = "turn_left"
            else:
                turn_text = "You have reached your destination"
                action = "finish"

            steps.append(RouteStep(
                step_id=step_id,
                text=f"After {int(dist_accum)} m — {turn_text}.",
                location=Coord(edge.target.lat, edge.target.lon),
                action=action,
                distance_meters=int(dist_accum),
                road_name=next_edge.name if next_edge else None,
            ))

            dist_accum = 0.0
            step_id += 1
            if next_edge:
                curr_name = next_edge.name

    return steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RouteCalculator:
    """
    Calculates a walking route between two coordinates using A*.

    Args:
        db:     Populated RoutingDB from osm_parser.load_map().
        config: NavConfig instance.
    """

    def __init__(self, db: RoutingDB, config: Optional[NavConfig] = None) -> None:
        self.db = db
        self.config = config or NavConfig()

    def calculate(
        self, origin: Coord, destination: Coord
    ) -> Tuple[Optional[List[RouteStep]], str]:
        """
        Run A* from origin to destination.

        Returns:
            (steps, message) — steps is None on failure.
        """
        start_node, _ = _find_nearest_node(self.db, origin)
        end_node, _ = _find_nearest_node(self.db, destination)

        if not start_node or not end_node:
            return None, "Could not find nearby nodes for given coordinates."

        if start_node == end_node:
            return None, "Origin and destination map to the same node."

        # A* search
        counter = 0                                          # 
        open_set: list = []
        heapq.heappush(open_set, (0.0, counter, start_node)) # 
        came_from: dict = {start_node: (None, None)}
        cost_so_far: dict = {start_node: 0.0}
        visited: set = set()                                 # 
        speed_ms = self.config.walking_speed_kmh * 1000 / 3600
        
        while open_set:
            _, _, current = heapq.heappop(open_set)          # 
            if current in visited:                           # 
                continue                                     # 
            visited.add(current)                             # 
            if current == end_node:
                break
            for edge in current.edges:
                new_cost = cost_so_far[current] + edge.time
                neighbor = edge.target
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heuristic = haversine_distance(neighbor.lat, neighbor.lon, end_node.lat, end_node.lon) / speed_ms
                    counter += 1                             # ← YENİ
                    heapq.heappush(open_set, (new_cost + heuristic, counter, neighbor)) # ← counter eklendi
                    came_from[neighbor] = (current, edge)

        if end_node not in came_from:
            return None, "No walkable route found between these points."

        path = _reconstruct_path(came_from, start_node, end_node)
        steps = _build_steps(path, start_node)
        return steps, "OK"
