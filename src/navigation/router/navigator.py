# navigator.py
# version 1.2
# Navigasyon sistemine ait ana modül 
# Haritayı yükler, rota hesaplar ve navigasyon durumunu yönetir.
# main loop'ta sürekli çağrılır.
# flag tabanlı kontrol sistemi içerir.
import osm_router
import json
import math

class NavigationSystem:
    def __init__(self, osm_map_path):
        # 1. Haritayı belleğe yükle (Sadece 1 kere yapılır)
        self.db = osm_router.load_map(osm_map_path)
        self.route_data = []  # Hesaplanan rota burada tutulacak
        self.current_step_index = 0
        self.is_navigating = False # C'deki Flag bu (0: Duruyor, 1: Aktif)
        
        # Ayarlar
        self.waypoint_threshold = 15.0  # 15 metre kalınca "Vardın" say
        self.off_route_threshold = 40.0 # 40 metre saparsa uyar

    def start_navigation(self, start_lat, start_lon, end_lat, end_lon):
        """
        Rotayı hesaplar ve navigasyon modunu başlatır (Flag = 1 yapar).
        """
        print("[Nav] Rota hesaplanıyor...")
        route, msg = osm_router.calculate_route(self.db, start_lat, start_lon, end_lat, end_lon)
        
        if not route:
            print(f"[Nav] Hata: {msg}")
            self.is_navigating = False
            return False
        
        self.route_data = route
        self.current_step_index = 0
        self.is_navigating = True
        
        # Rotayı incelemek için JSON kaydet (Opsiyonel)
        with open("aktif_rota.json", "w", encoding="utf-8") as f:
            json.dump(self.route_data, f, ensure_ascii=False, indent=2)
            
        print(f"[Nav] Rota oluşturuldu! {len(route)} adım.")
        print(f"[Nav] İlk Talimat: {route[0]['text']}")
        return True

    def check_progress(self, current_lat, current_lon):
        """
        Bu fonksiyon sürekli döngüde (Loop) çağrılacak.
        Kullanıcının anlık konumuna göre ne yapması gerektiğini söyler.
        """
        if not self.is_navigating:
            return "Navigasyon aktif değil."

        # Hedefe ulaştık mı?
        if self.current_step_index >= len(self.route_data):
            self.is_navigating = False
            return "ROTA_BITTI"

        # Hedeflediğimiz sıradaki nokta (Waypoint)
        target_step = self.route_data[self.current_step_index]
        t_lat = target_step['location']['lat']
        t_lon = target_step['location']['lon']
        
        # O noktaya ne kadar kaldı?
        dist_to_target = osm_router.haversine_distance(current_lat, current_lon, t_lat, t_lon)
        
        # --- DURUM KONTROLÜ ---
        
        # 1. Waypoint'e vardık mı?
        if dist_to_target < self.waypoint_threshold:
            self.current_step_index += 1
            
            # Rota bitti mi tekrar kontrol et
            if self.current_step_index >= len(self.route_data):
                self.is_navigating = False
                return "HEDEFE ULAŞTINIZ!"
            
            # Yeni talimatı ver
            new_step = self.route_data[self.current_step_index]
            return f"YENİ TALİMAT: {new_step['text']}"

        # 2. Rotadan saptık mı? (Basit kontrol)
        # Eğer hedef noktaya olan mesafe, olması gerekenden çok fazlaysa sapmışızdır.
        # (Daha gelişmiş versiyonda çizgiye olan uzaklık hesaplanır, şimdilik nokta uzaklığı yeterli)
        # Bu basit bir off-route kontrolü, geliştirilebilir.
        if dist_to_target > target_step['distance_meters'] + self.off_route_threshold:
             # Burada flag'i kapatıp tekrar start_navigation çağırabilirsin (Rerouting)
             return "DİKKAT: ROTADAN SAPTINIZ!"

        # 3. Henüz varmadık, yola devam
        return f"DEVAM: Hedefe {int(dist_to_target)}m kaldı. ({target_step['action']})"

# --- ÖRNEK KULLANIM (MAIN LOOP SİMÜLASYONU) ---
if __name__ == "__main__":
    # 1. Sistem Başlatılıyor
    nav = NavigationSystem("map.osm") # Harita burada 1 kez yüklenir
    
    # 2. Kullanıcı Rota İstiyor (Flag 0 -> 1 oluyor)
    # Örnek koordinatlar (Sıhhiye -> Kurtuluş)
    basari = nav.start_navigation(39.92409, 32.845382, 39.921117, 32.852903)
    
    if basari:
        # 3. OTO CHECK SİSTEMİ (Simülasyon Döngüsü)
        # Gerçek hayatta burası GPS'ten veri geldikçe çalışacak
        
        # Simülasyon: Kullanıcı yolda yürüyor gibi koordinat verelim
        test_locations = [
            (39.92409, 32.845382), # Başlangıçta
            (39.9240467, 32.8451522), # 1. Adım: Düz Devam
            (39.9232599, 32.8441792), # 2. Adım: Düz Devam
            (39.9240102, 32.8452347), # 3. Adım: Dönüşe girdi
            (39.9249406, 32.8462865), # 4. Adım: Düz Devam
            (39.9254588, 32.8477125), # 5. Adım: Sağa Dönüş
            (39.9208164, 32.8533392), # 6. Adım: Uzun yürüyüş sonu (Sola Keskin)
            (39.920927,  32.8533893), # 7. Adım: Sola Keskin
            (39.9210086, 32.8529793)  # 8. Adım: Varış
            # ...
        ]
        
        print("\n--- ÇEK SİSTEMİ DEVREDE ---")
        for lat, lon in test_locations:
            gps_input = (lat, lon)
            status_msg = nav.check_progress(lat, lon)
            print(f"GPS: {gps_input} -> Sistem: {status_msg}")