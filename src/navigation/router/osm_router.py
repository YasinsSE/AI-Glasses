#!/usr/bin/env python3

import sys
import xml.sax as sax
import heapq
import math
import json

# -------------------------------
# AYARLAR
# -------------------------------
SPEED_LIMITS = {
    'motorway': 120, 'motorway_link': 100,
    'trunk': 110, 'trunk_link': 90,
    'primary': 90, 'primary_link': 70,
    'secondary': 70, 'secondary_link': 50,
    'tertiary': 50, 'tertiary_link': 40,
    'residential': 30, 'living_street': 20,
    'service': 20, 'unclassified': 30
}
MAX_SPEED = 120.0

# -------------------------------
# Temel Sınıflar ve Fonksiyonlar
# -------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
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

class Edge:
    __slots__ = ['target', 'distance', 'time', 'name', 'type']
    def __init__(self, target, dist, r_type, name):
        self.target = target
        self.distance = dist
        self.name = name
        self.type = r_type
        self.time = dist / SPEED_LIMITS.get(r_type, 30)

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
    def add_edge(self, u, v, r_type, name, oneway):
        n1, n2 = self.nodes.get(u), self.nodes.get(v)
        if not n1 or not n2: return
        d = haversine_distance(n1.lat, n1.lon, n2.lat, n2.lon)
        n1.edges.append(Edge(n2, d, r_type, name))
        if not oneway:
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
        if rtype not in SPEED_LIMITS: return
        name = self.tags.get('name', 'Bilinmeyen Yol')
        oneway = (self.tags.get('oneway') == 'yes') or (rtype == 'motorway')
        for i in range(len(self.curr_nodes) - 1):
            self.db.add_edge(self.curr_nodes[i], self.curr_nodes[i+1], rtype, name, oneway)

def find_nearest_node(db, target_lat, target_lon):
    best_node = None
    min_dist = float('inf')
    for node in db.nodes.values():
        d = haversine_distance(target_lat, target_lon, node.lat, node.lon)
        if d < min_dist:
            min_dist = d
            best_node = node
    return best_node, min_dist

# -------------------------------
# Teşhis ve Rota Algoritmaları
# -------------------------------

def check_reachability(start_node):
    """Başlangıç noktasından kaç farklı noktaya gidilebildiğini sayar (BFS)."""
    visited = set()
    queue = [start_node]
    visited.add(start_node)
    count = 0
    max_dist = 0
    
    while queue:
        node = queue.pop(0)
        count += 1
        if count > 2000: # Çok büyükse kes
            return ">2000 (Ağ bağlantısı iyi)"
            
        for edge in node.edges:
            if edge.target not in visited:
                visited.add(edge.target)
                queue.append(edge.target)
                
    return f"{count} nokta (Muhtemelen izole bir yol parçası)"

def astar(db, start_node, end_node):
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
                priority = new_cost + (haversine_distance(neighbor.lat, neighbor.lon, end_node.lat, end_node.lon) / MAX_SPEED)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = (current, edge)
    
    if end_node not in came_from: return None
    path = []
    curr = end_node
    while curr != start_node:
        parent, edge = came_from[curr]
        path.append((parent, edge))
        curr = parent
    path.reverse()
    return path

def generate_instructions(path):
    instructions = []
    if not path: return instructions
    
    curr_name = path[0][1].name
    dist_accum = 0.0
    instructions.append(f"Başlangıç: {curr_name}")
    
    for i, (node, edge) in enumerate(path):
        dist_accum += edge.distance
        next_edge = path[i+1][1] if i+1 < len(path) else None
        
        if not next_edge or next_edge.name != curr_name:
            if next_edge:
                b1 = calculate_bearing(node, edge.target)
                b2 = calculate_bearing(edge.target, next_edge.target)
                maneuver = get_turn_instruction(b2-b1)
                turn_msg = f"{next_edge.name} yönüne {maneuver}"
            else:
                turn_msg = "Hedefe ulaştınız"
            
            d_str = f"{int(dist_accum*1000)}m" if dist_accum < 1 else f"{dist_accum:.2f}km"
            instructions.append(f"{d_str} ilerleyin, {turn_msg}.")
            dist_accum = 0
            if next_edge: curr_name = next_edge.name
            
    return instructions

def export_geojson(path, filename="rota.json"):
    coords = [[n.lon, n.lat] for n, _ in path]
    coords.append([path[-1][1].target.lon, path[-1][1].target.lat])
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"stroke": "#ff0000", "stroke-width": 4}, "geometry": {"type": "LineString", "coordinates": coords}}]}
    with open(filename, "w") as f: json.dump(geojson, f)

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python3 nav_v3.py <harita.osm>")
        sys.exit(1)

    print(f"--- Harita Yükleniyor: {sys.argv[1]} ---")
    db = RoutingDB()
    parser = sax.make_parser()
    parser.setContentHandler(OSMHandler(db))
    parser.parse(sys.argv[1])
    db.cleanup()
    print(f"Harita Hazır: {len(db.nodes)} aktif nokta.")

    # -------------------------------------------------------------
    # KOORDİNATLARI BURAYA GİRİN (Google Maps'ten sağ tıkla alın)
    # -------------------------------------------------------------
    # Örnek: İstanbul
    S_LAT, S_LON = 39.92409, 32.845382  # Başlangıç
    E_LAT, E_LON = 39.921117, 32.852903  # Bitiş

    print("\n1. En yakın yol noktaları aranıyor...")
    start_node, d1 = find_nearest_node(db, S_LAT, S_LON)
    end_node, d2 = find_nearest_node(db, E_LAT, E_LON)

    if not start_node or not end_node:
        print("KRİTİK HATA: Harita dosyanız bu koordinatları kapsamıyor!")
        sys.exit()

    print(f"   -> Başlangıç noktası {int(d1*1000)}m mesafede bulundu. (ID: {start_node.id})")
    print(f"   -> Varış noktası {int(d2*1000)}m mesafede bulundu. (ID: {end_node.id})")

    # Eğer en yakın nokta çok uzaksa uyar
    if d1 > 2.0 or d2 > 2.0:
        print("\nUYARI: Seçtiğiniz koordinatlar haritadaki en yakın yola çok uzak (>2km).")
        print("Harita dosyanız bu bölgeyi tam kapsamıyor olabilir.")

    print("\n2. Rota hesaplanıyor...")
    path = astar(db, start_node, end_node)

    if path:
        print("\nBAŞARILI: Rota bulundu!")
        export_geojson(path, "rota.json")
        steps = generate_instructions(path)
        for i, s in enumerate(steps, 1): print(f"{i}. {s}")
        print("\nSonuç: rota.json dosyası oluşturuldu.")
    else:
        print("\n--- HATA ANALİZİ ---")
        print("İki nokta arasında rota bulunamadı.")
        print("Olası Sebepler:")
        
        # Analiz yapalım
        reach = check_reachability(start_node)
        print(f"1. Başlangıç noktasından erişilebilen ağ büyüklüğü: {reach}")
        
        if "nokta" in reach and int(reach.split()[0]) < 50:
            print("   -> TESPİT: Başlangıç noktanız ana yoldan kopuk (izole) bir parça üzerinde.")
            print("      Koordinatlarınız kapalı bir site, otopark veya harita hatası olan bir yere denk gelmiş.")
        else:
            print("   -> TESPİT: Başlangıç noktası geniş bir ağa bağlı. Sorun muhtemelen 'Tek Yön' veya 'Harita Kopukluğu'.")
            print("      Bitiş noktasına giden yol harita sınırları dışında kalıyor olabilir.")

        print(f"\nKontrol Linkleri:")
        print(f"Başlangıç: https://www.openstreetmap.org/node/{start_node.id}")
        print(f"Bitiş:     https://www.openstreetmap.org/node/{end_node.id}")