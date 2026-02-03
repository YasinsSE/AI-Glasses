#!/usr/bin/env python3

import sys
import xml.sax as sax
import heapq
import math
import json

# -------------------------------
# AYARLAR
# -------------------------------
WALKING_SPEED = 5.0

WALKABLE_TYPES = {
    # Saf Yaya Yolları
    'footway', 'pedestrian', 'path', 'steps', 'cycleway', 'living_street', 'track', 'crossing',
    
    # Araç Yolları Kaldırım var gibi davranıyoruz
    'residential', 'service', 'unclassified',
    'primary', 'primary_link',
    'secondary', 'secondary_link',
    'tertiary', 'tertiary_link',
    'trunk', 'trunk_link'  
}
# Yayaların girmesi tehlikeli olan yollar (Muhtemelen kaldırılacak.)
FORBIDDEN_TYPES = {'motorway', 'motorway_link'}

# -------------------------------
# Helper Fonksiyonlar
# -------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    İki coğrafi koordinat (enlem, boylam) arasindaki kuş uçuşu mesafeyi hesaplar.
    Dünyanin küresel şeklini (Haversine formülü) baz alir.
    
    Döndürdüğü Değer: Kilometre (km) cinsinden mesafe.
    """
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_bearing(n1, n2):
    """
    İki nokta arasindaki pusula açisini (0-360 derece) hesaplar.
    0: Kuzey, 90: Doğu, 180: Güney, 270: Bati.
    Bu açi, navigasyonda "sağa dön", "sola dön" demek için kullanilir.
    """
    lat1, lon1 = math.radians(n1.lat), math.radians(n1.lon)
    lat2, lon2 = math.radians(n2.lat), math.radians(n2.lon)
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def get_clock_direction(current_heading, target_bearing):
    # Kullanıcının yüzü (current_heading) ile hedef (target_bearing) arasındaki fark
    diff = (target_bearing - current_heading + 360) % 360
    
    # 360 dereceyi 12 saate böl (Her saat dilimi 30 derece)
    clock_hour = int((diff + 15) // 30)
    if clock_hour == 0: clock_hour = 12
    
    return f"Saat {clock_hour} yönünde"

def get_turn_instruction(diff):
    """
    Giriş açisi ile çikiş açisi arasindaki farka bakarak 
    sözel bir yön tarifi (Sağa dön, Düz git vb.) üretir.
    """
    diff = (diff + 180) % 360 - 180
    if diff > 45: return "Sağa Keskin Dönün"
    elif diff > 10: return "Sağa Dönün"
    elif diff < -45: return "Sola Keskin Dönün"
    elif diff < -10: return "Sola Dönün"
    return "Düz Devam Edin"
# -------------------------------
# Veri Yapıları (Graph)
# -------------------------------
class Edge:
    """
    İki düğüm (Node) arasindaki bağlantiyi (yolu) temsil eder.
    Yolun uzunluğunu, tipini ve geçiş süresini tutar.
    """
    __slots__ = ['target', 'distance', 'time', 'name', 'type']
    def __init__(self, target, dist, r_type, name):
        self.target = target
        self.distance = dist
        self.name = name
        self.type = r_type
        
        factor = 2.0 if r_type == 'steps' else 1.0
        self.time = (dist / WALKING_SPEED) * factor

class Node:
    """
    Haritadaki tek bir noktayi (GPS koordinatini) temsil eder.
    Komşu noktalara giden bağlantilari (edges) listesinde tutar.
    """
    __slots__ = ['id', 'lat', 'lon', 'edges']
    def __init__(self, nid, lat, lon):
        self.id = nid
        self.lat = float(lat)
        self.lon = float(lon)
        self.edges = []

class RoutingDB:
    """
    Tüm harita verisini (Düğümler ve Yollar) hafizada tutan ana veritabani sinifidir.
    XML'den okunan veriler buraya kaydedilir.
    """
    def __init__(self):
        self.nodes = {}
    def add_node(self, node):
        """Sisteme yeni bir nokta ekler."""
        self.nodes[node.id] = node
    def add_edge(self, u, v, r_type, name, oneway):
        """
        İki nokta (u ve v) arasına yol ekler.
        Yayalar için 'oneway' (tek yön) genellikle yok sayılır (False gönderilir).
        """
        n1, n2 = self.nodes.get(u), self.nodes.get(v)
        if not n1 or not n2: return
        d = haversine_distance(n1.lat, n1.lon, n2.lat, n2.lon)
        n1.edges.append(Edge(n2, d, r_type, name))
        if not oneway:
            n2.edges.append(Edge(n1, d, r_type, name))
    def cleanup(self):
        """Herhangi bir yola bağlı olmayan (izole) noktaları hafızadan siler."""
        self.nodes = {k: v for k, v in self.nodes.items() if len(v.edges) > 0}
# -------------------------------
# OSM Dosya Okuyucu
# -------------------------------
class OSMHandler(sax.ContentHandler):
    """
    OpenStreetMap (.osm) XML dosyasını satır satır okuyan (parse eden) sınıftır.
    SAX kütüphanesini kullanır, bu sayede büyük dosyaları belleği şişirmeden okur.
    """
    def __init__(self, db):
        self.db = db
        self.curr_nodes = []
        self.tags = {}
        self.in_way = False
    def startElement(self, name, attrs):
        """XML etiketi açıldığında (<node> veya <way>) çağrılır."""
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
        """XML etiketi kapandığında çağrılır. Way bittiyse işlemeye gönderir."""
        if name == "way":
            self.process_way()
            self.in_way = False
    def process_way(self):
        """
        Okunan bir yolun özelliklerini inceler.
        Eğer yol yürünebilir türdeyse veritabanına ekler.
        """
        if 'highway' not in self.tags: return
        rtype = self.tags['highway']
        if rtype in FORBIDDEN_TYPES or rtype not in WALKABLE_TYPES: return
        name = self.tags.get('name', 'Yaya Yolu')
        # YAYALAR İÇİN ÖNEMLİ: Tek yön kuralını iptal ediyoruz.
        # oneway=False diyerek yolun iki yöne de gidilebilir olduğunu belirtiyoruz.
        for i in range(len(self.curr_nodes) - 1):
            self.db.add_edge(self.curr_nodes[i], self.curr_nodes[i+1], rtype, name, oneway=False)

def find_nearest_node(db, target_lat, target_lon):
    """
    Verilen koordinata (GPS) haritadaki en yakın düğümü (Node) bulur.
    Navigasyonun başlayabilmesi için kullanıcının haritaya 'snap' edilmesi gerekir.
    """
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
# -------------------------------
# Rota Algoritmaları (A*)
# -------------------------------
def astar(db, start_node, end_node):
    """
    A* (A-Star) Algoritması: Başlangıçtan bitişe en maliyetsiz (en hızlı/kısa) yolu bulur.
    Maliyet fonksiyonu olarak zamanı (mesafe / hız) kullanır.
    Heuristic (tahmin) olarak kuş uçuşu mesafeyi kullanır.
    """
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
                # Heuristic: Kalan mesafeyi yürüme hızına bölerek tahmini süre ekliyoruz
                priority = new_cost + (haversine_distance(neighbor.lat, neighbor.lon, end_node.lat, end_node.lon) / WALKING_SPEED)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = (current, edge)
    
    if end_node not in came_from: return None

    # Rotayı geriye doğru takip ederek oluştur
    path = []
    curr = end_node
    while curr != start_node:
        parent, edge = came_from[curr]
        path.append((parent, edge))
        curr = parent
    path.reverse()
    return path

def generate_instructions(path):
    """
    Rotayı analiz eder ve her adımı detaylı bir sözlük (dictionary) olarak döndürür.
    Döndürülen yapı:
    [
      {
        "text": "100m ilerleyin, Sağa dönün",
        "location": {"lat": 39.92..., "lon": 32.85...},
        "distance_meters": 100,
        "action": "turn_right"
      },
      ...
    ]
    """
    steps_data = []
    if not path: return steps_data
    
    curr_name = path[0][1].name
    dist_accum = 0.0
    
    # Başlangıç bilgisini ekle
    start_node = path[0][0]
    steps_data.append({
        "step_id": 0,
        "text": f"Başlangıç: {curr_name}",
        "location": {"lat": start_node.lat, "lon": start_node.lon},
        "action": "start",
        "distance_meters": 0
    })
    
    step_counter = 1
    
    for i, (node, edge) in enumerate(path):
        dist_accum += edge.distance
        next_edge = path[i+1][1] if i+1 < len(path) else None
        
        # Karar noktası mı? (İsim değişti, tip değişti veya yol bitti)
        if not next_edge or next_edge.name != curr_name or next_edge.type != edge.type:
            
            # 1. Ekstra Bilgileri Topla
            extra_info = ""
            if next_edge and next_edge.type == 'steps': extra_info = " (Dikkat: Merdiven!)"
            elif edge.type == 'steps': extra_info = " (Merdiven bitimi)"
            
            # 2. Manevra Tipini Belirle
            action_code = "continue"
            if next_edge:
                b1 = calculate_bearing(node, edge.target)
                b2 = calculate_bearing(edge.target, next_edge.target)
                maneuver = get_turn_instruction(b2-b1)
                turn_msg = f"{next_edge.name} yönüne {maneuver}{extra_info}"
                
                # Basit action kodları (Gözlük arayüzü için ikon seçmede işe yarar)
                if "Sağa" in maneuver: action_code = "turn_right"
                elif "Sola" in maneuver: action_code = "turn_left"
            else:
                turn_msg = "Hedefe ulaştınız"
                action_code = "finish"
            
            # 3. Metni Oluştur
            d_str = f"{int(dist_accum*1000)}m" if dist_accum < 1 else f"{dist_accum:.2f}km"
            full_text = f"{d_str} ilerleyin, {turn_msg}."
            
            # 4. SÖZLÜĞÜ OLUŞTUR VE LİSTEYE EKLE
            step_dict = {
                "step_id": step_counter,
                "text": full_text,
                "location": {"lat": edge.target.lat, "lon": edge.target.lon}, # Karar noktasının koordinatı
                "distance_meters": int(dist_accum * 1000),
                "action": action_code,
                "road_name": next_edge.name if next_edge else "Varış"
            }
            steps_data.append(step_dict)
            
            # Değişkenleri sıfırla
            dist_accum = 0
            step_counter += 1
            if next_edge: curr_name = next_edge.name
            
    return steps_data

def export_geojson(path, filename="rota.json"):
    """
    Rotayı harita uygulamalarında (geojson.io vb.) görüntülemek için
    GeoJSON formatında dosyaya kaydeder.
    """
    coords = [[n.lon, n.lat] for n, _ in path]
    if path:
        last_edge = path[-1][1]
        coords.append([last_edge.target.lon, last_edge.target.lat])
    
    geojson = {
        "type": "FeatureCollection", "features": [{"type": "Feature", 
        "properties": {"stroke": "#0000FF", "stroke-width": 4}, 
        "geometry": {"type": "LineString", "coordinates": coords}}]
    }
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        print(f"-> Dosya kaydedildi: {filename}")
    except PermissionError:
        print(f"HATA: {filename} dosyası açık! Kapatıp tekrar dene.")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # Parametre kontrolü
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

# ------------------------------------------------------------------
# TEST KOORDİNATLARI (Senin Map.osm dosyan için uygun test verisi)
# ------------------------------------------------------------------
    
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
            print("\nBAŞARILI: Yaya rotası oluşturuldu!")
            
            # 1. GeoJSON Kaydı (Görsel Harita için)
            export_geojson(path, "rota.json")
            
            # 2. Akıllı Talimatları Oluştur (Liste içinde Sözlükler)
            steps_data = generate_instructions(path)
            
            # Ekrana Yazdır (Okunaklı özet)
            print("-" * 50)
            for step in steps_data: 
                print(f"{step['step_id']}. {step['text']}")
            print("-" * 50)
                
            # 3. Talimatları DETAYLI TEXT Dosyasına Kaydet
            # Burası tam istediğin gibi "Dictionary Type" formatında kaydedecek.
            txt_filename = "talimatlar.txt"
            try:
                with open(txt_filename, "w", encoding="utf-8") as f:
                    # İstersen direkt JSON olarak da dökebilirsin ama 
                    # okunabilir text istediğin için satır satır yazıyoruz.
                    f.write(json.dumps(steps_data, ensure_ascii=False, indent=4))
                    
                print(f"-> Detaylı Veri Dosyası Kaydedildi: {txt_filename}")
                print(f"   (İçeriğinde koordinatlar, aksiyon kodları ve mesafeler var)")
                
            except PermissionError:
                print(f"HATA: {txt_filename} dosyasına yazılamadı (dosya açık olabilir).")

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