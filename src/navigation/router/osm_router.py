# osm_router.py
import sys
import xml.sax as sax
import heapq
import math
import json

# --- AYARLAR ---
WALKING_SPEED = 5.0
WALKABLE_TYPES = {
    'footway', 'pedestrian', 'path', 'steps', 'cycleway', 'living_street', 'track', 'crossing',
    'residential', 'service', 'unclassified', 'primary', 'primary_link',
    'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link'
}
FORBIDDEN_TYPES = {'motorway', 'motorway_link', 'construction'}

# --- MATEMATİK FONKSİYONLARI ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  # Metre cinsinden hassasiyet için 6371 km * 1000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_bearing(n1, n2):
    lat1, lon1 = math.radians(n1.lat), math.radians(n1.lon)
    lat2, lon2 = math.radians(n2.lat), math.radians(n2.lon)
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def get_turn_instruction(diff):
    diff = (diff + 180) % 360 - 180
    if diff > 45: return "Sağa Keskin Dönün"
    elif diff > 10: return "Sağa Dönün"
    elif diff < -45: return "Sola Keskin Dönün"
    elif diff < -10: return "Sola Dönün"
    return "Düz Devam Edin"

# --- SINIFLAR ---
class Edge:
    __slots__ = ['target', 'distance', 'time', 'name', 'type']
    def __init__(self, target, dist, r_type, name):
        self.target = target
        self.distance = dist
        self.name = name
        self.type = r_type
        factor = 2.0 if r_type == 'steps' else 1.0
        self.time = (dist / (WALKING_SPEED * 1000 / 3600)) * factor # m/s cinsinden

class Node:
    __slots__ = ['id', 'lat', 'lon', 'edges']
    def __init__(self, nid, lat, lon):
        self.id = nid
        self.lat = float(lat)
        self.lon = float(lon)
        self.edges = []

class RoutingDB:
    def __init__(self):
        self.nodes = {}
    def add_node(self, node):
        self.nodes[node.id] = node
    def add_edge(self, u, v, r_type, name):
        n1, n2 = self.nodes.get(u), self.nodes.get(v)
        if not n1 or not n2: return
        d = haversine_distance(n1.lat, n1.lon, n2.lat, n2.lon)
        n1.edges.append(Edge(n2, d, r_type, name))
        n2.edges.append(Edge(n1, d, r_type, name))
    def cleanup(self):
        self.nodes = {k: v for k, v in self.nodes.items() if len(v.edges) > 0}

class OSMHandler(sax.ContentHandler):
    def __init__(self, db):
        self.db = db
        self.curr_nodes = []
        self.tags = {}
        self.in_way = False
    def startElement(self, name, attrs):
        if name == "node":
            self.db.add_node(Node(attrs["id"], attrs["lat"], attrs["lon"]))
        elif name == "way":
            self.in_way = True
            self.curr_nodes = []
            self.tags = {}
        elif name == "nd" and self.in_way:
            self.curr_nodes.append(attrs["ref"])
        elif name == "tag" and self.in_way:
            self.tags[attrs["k"]] = attrs["v"]
    def endElement(self, name):
        if name == "way":
            self.process_way()
            self.in_way = False
    def process_way(self):
        if 'highway' not in self.tags: return
        rtype = self.tags['highway']
        if rtype in FORBIDDEN_TYPES or rtype not in WALKABLE_TYPES: return
        name = self.tags.get('name', 'Yaya Yolu')
        for i in range(len(self.curr_nodes) - 1):
            self.db.add_edge(self.curr_nodes[i], self.curr_nodes[i+1], rtype, name)

# --- ANA FONKSİYONLAR ---
def load_map(osm_file):
    """Haritayı bir kez yükler ve veritabanını döndürür."""
    print(f"[Router] Harita yükleniyor: {osm_file}")
    db = RoutingDB()
    parser = sax.make_parser()
    parser.setContentHandler(OSMHandler(db))
    parser.parse(osm_file)
    db.cleanup()
    print(f"[Router] Harita hazır: {len(db.nodes)} nokta.")
    return db

def find_nearest_node(db, lat, lon):
    best_node = None
    min_dist = float('inf')
    for node in db.nodes.values():
        d = haversine_distance(lat, lon, node.lat, node.lon)
        if d < min_dist:
            min_dist = d
            best_node = node
    return best_node, min_dist

def calculate_route(db, start_lat, start_lon, end_lat, end_lon):
    """
    Verilen koordinatlar için rota hesaplar ve JSON formatında (liste) döndürür.
    Dosyaya yazmaz, veriyi return eder.
    """
    start_node, d1 = find_nearest_node(db, start_lat, start_lon)
    end_node, d2 = find_nearest_node(db, end_lat, end_lon)

    if not start_node or not end_node:
        return None, "Harita kapsamı dışında"

    # A* Algoritması
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {start_node: (None, None)}
    cost_so_far = {start_node: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end_node: break

        for edge in current.edges:
            new_cost = cost_so_far[current] + edge.time
            neighbor = edge.target
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + (haversine_distance(neighbor.lat, neighbor.lon, end_node.lat, end_node.lon) / (WALKING_SPEED * 1000/3600))
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = (current, edge)
    
    if end_node not in came_from: return None, "Rota bulunamadı"

    # Rotayı oluştur
    path = []
    curr = end_node
    while curr != start_node:
        parent, edge = came_from[curr]
        path.append((parent, edge))
        curr = parent
    path.reverse()

    # Talimatları (JSON yapısı) oluştur
    steps_data = []
    curr_name = path[0][1].name
    dist_accum = 0.0
    step_counter = 1
    
    # Başlangıç adımı
    steps_data.append({
        "step_id": 0,
        "text": f"Navigasyon başlıyor. Konum: {curr_name}",
        "location": {"lat": start_node.lat, "lon": start_node.lon},
        "action": "start",
        "distance_meters": 0
    })

    for i, (node, edge) in enumerate(path):
        dist_accum += edge.distance
        next_edge = path[i+1][1] if i+1 < len(path) else None
        
        if not next_edge or next_edge.name != curr_name or next_edge.type != edge.type:
            action_code = "continue"
            turn_msg = "Düz devam edin"
            
            if next_edge:
                b1 = calculate_bearing(node, edge.target)
                b2 = calculate_bearing(edge.target, next_edge.target)
                maneuver = get_turn_instruction(b2-b1)
                turn_msg = f"{maneuver}"
                if "Sağa" in maneuver: action_code = "turn_right"
                elif "Sola" in maneuver: action_code = "turn_left"
            else:
                turn_msg = "Hedefe ulaştınız"
                action_code = "finish"

            full_text = f"{int(dist_accum)}m sonra {turn_msg}."
            
            steps_data.append({
                "step_id": step_counter,
                "text": full_text,
                "location": {"lat": edge.target.lat, "lon": edge.target.lon},
                "distance_meters": int(dist_accum),
                "action": action_code,
                "road_name": next_edge.name if next_edge else "Varış"
            })
            dist_accum = 0
            step_counter += 1
            if next_edge: curr_name = next_edge.name

    return steps_data, "Başarılı"