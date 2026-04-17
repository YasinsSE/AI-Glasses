"""
Test path guidance — iki mod:

══════════════════════════════════════════════════════════════════
 FOTOĞRAF MODU
 • Model gerekmez.
 • Daha önce kaydedilmiş renkli overlay görselleri ile çalışır.
   (result_test1_overlay.jpg, result_test2_overlay.jpg vb.)
 • Renkleri piksel bazında class ID'ye dönüştürür (rgb_to_mask).

 Nasıl çalıştırılır:
   cd /Users/mehmet/Desktop/ai-glasses-1/AI-Glasses
   venv/bin/python test_guidance_from_overlay.py result_test1_overlay.jpg
   venv/bin/python test_guidance_from_overlay.py result_test1_overlay.jpg --nav
     └─ --nav : navigasyonu aktif simüle eder (yaya geçidi sesli bildirilir)

══════════════════════════════════════════════════════════════════
 KAMERA MODU (PerceptionPipeline)
 • Gerçek segmentasyon modeli (.onnx veya .trt/.engine) gerekir.
 • Kameradan canlı frame alır, modeli çalıştırır, overlay gösterir.
 • Yön kılavuzu hem ekrana hem terminale yazılır.
 • Çıkmak için 'q' tuşuna basın.

 Nasıl çalıştırılır:
   cd /Users/mehmet/Desktop/ai-glasses-1/AI-Glasses
   venv/bin/python test_guidance_from_overlay.py --camera --model models/segmentation/alas_engine.onnx
   venv/bin/python test_guidance_from_overlay.py --camera --model models/segmentation/alas_engine.onnx --camera-index 1
     └─ --camera-index : birden fazla kamera varsa indeks belirt (varsayılan: 0)

══════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import time

import numpy as np

# ── Class tanımları (perception.py ile aynı) ────────────────────────────────

WALKABLE_SURFACE   = 0
CROSSWALK          = 1
VEHICLE_ROAD       = 2
COLLISION_OBSTACLE = 3
FALL_HAZARD        = 4
DYNAMIC_HAZARD     = 5
VEHICLE            = 6

CLASS_NAMES = {
    0: "walkable_surface",
    1: "crosswalk",
    2: "vehicle_road",
    3: "collision_obstacle",
    4: "fall_hazard",
    5: "dynamic_hazard",
    6: "vehicle",
}

CLASS_ALERT_CONFIG = {
    WALKABLE_SURFACE:   {"priority": 0, "alert": None,                                              "cooldown": 0},
    CROSSWALK:          {"priority": 1, "alert": "Yaya geçidi algılandı, geçiş güvenli",           "cooldown": 8.0},
    VEHICLE_ROAD:       {"priority": 4, "alert": "Dikkat, araç yolu, girmeyin",                     "cooldown": 5.0},
    COLLISION_OBSTACLE: {"priority": 3, "alert": "Önünüzde engel var, durun veya yön değiştirin",  "cooldown": 3.0},
    FALL_HAZARD:        {"priority": 3, "alert": "Zemin tehlikesi, yavaşlayın",                     "cooldown": 3.0},
    DYNAMIC_HAZARD:     {"priority": 4, "alert": "Hareketli nesne yakınınızda",                                 "cooldown": 2.0},
    VEHICLE:            {"priority": 5, "alert": "Durun, önünüzde araç var",                      "cooldown": 1.5},
}

CLASS_COLORS_BGR = {
    WALKABLE_SURFACE:   (0, 200, 0),
    CROSSWALK:          (200, 255, 0),
    VEHICLE_ROAD:       (180, 30, 30),
    COLLISION_OBSTACLE: (255, 140, 0),
    FALL_HAZARD:        (255, 80, 0),
    DYNAMIC_HAZARD:     (255, 100, 100),
    VEHICLE:            (255, 0, 0),
}

MIN_ALERT_RATIO      = 0.02
VERY_CLOSE_RATIO     = 0.15
NEARBY_RATIO         = 0.05
PATH_BOTTOM_FRACTION      = 0.5   # maskenin alt yarısına bak
CORRIDOR_MARGIN           = 0.15  # sol/sağ kenarı yoksay — kullanıcı oraya yürümez
MIN_WALKABLE_FOR_GUIDANCE = 0.08


# ════════════════════════════════════════════════════════════════════════════
#  FOTOĞRAF MODU — yardımcı fonksiyonlar
#  (Kamera modu bu fonksiyonları kullanmaz; PerceptionPipeline'a güvenir.)
# ════════════════════════════════════════════════════════════════════════════

def load_image_as_rgb(path: str) -> np.ndarray:
    """Overlay görselini RGB numpy dizisine yükler."""
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


def rgb_to_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Overlay renklerini class ID'lerine dönüştürür.
    Renkler perception.py'deki CLASS_COLORS_BGR ile eşleşir.

    Sıra önemli: daha özgün kurallar önce uygulanır.
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    mx  = np.maximum(np.maximum(r, g), b)
    sat = (mx - np.minimum(np.minimum(r, g), b)) / (mx + 1e-6)
    colourful = sat > 0.15

    # Parlak yeşil: G baskın, R ve B düşük
    is_green     = colourful & (g > r * 1.4) & (g > b * 1.4) & (g > 100)
    # Cyan: G ve B yüksek, R düşük
    is_cyan      = colourful & (g > r * 1.3) & (b > r * 1.3) & (g > 80) & (b > 80)
    # Koyu lacivert: B baskın
    is_navy      = colourful & (b > r * 1.5) & (b > g * 1.5)
    # Turuncu: R yüksek, G orta, B düşük
    is_orange    = colourful & (r > 150) & (g > 60) & (g < r * 0.75) & (b < 60)
    # Koyu turuncu / kahverengi-turuncu (fall hazard)
    is_dk_orange = colourful & (r > 100) & (r < 200) & (g > 40) & (g < r * 0.65) & (b < 40)
    # Pembe: R baskın, G ve B birbirine yakın ve düşük değil
    is_pink      = colourful & (r > g * 1.2) & (r > b * 1.2) & (g > 80) & (b > 80) & (np.abs(g - b) < 40)
    # Kırmızı: R baskın, G ve B düşük
    is_red       = colourful & (r > g * 1.5) & (r > b * 1.5) & (r > 120) & (g < 80)

    # En özgün önce (çakışma engellemek için)
    mask[is_navy]      = VEHICLE_ROAD
    mask[is_cyan]      = CROSSWALK
    mask[is_green]     = WALKABLE_SURFACE
    mask[is_orange]    = COLLISION_OBSTACLE
    mask[is_dk_orange] = FALL_HAZARD
    mask[is_red]       = VEHICLE
    mask[is_pink]      = DYNAMIC_HAZARD

    return mask


def analyse_and_alert(mask: np.ndarray, nav_active: bool = False):
    """Maske üzerinden alert listesi üretir (fotoğraf modu için)."""
    h, w   = mask.shape
    third  = w // 3
    total  = float(h * w)
    alerts = []

    print("── Piksel dağılımı ──────────────────────────────────")
    for cid in range(7):
        px = float(np.sum(mask == cid))
        if px == 0:
            continue
        ratio = px / total
        print(f"  {CLASS_NAMES[cid]:<22} {ratio:5.1%}")

        if cid == WALKABLE_SURFACE or ratio < MIN_ALERT_RATIO:
            continue

        cfg = CLASS_ALERT_CONFIG[cid]
        left_px   = float(np.sum((mask == cid)[:, :third]))
        center_px = float(np.sum((mask == cid)[:, third:2 * third]))
        right_px  = float(np.sum((mask == cid)[:, 2 * third:]))
        zone_dict = {"left": left_px, "center": center_px, "right": right_px}
        zone = max(zone_dict, key=zone_dict.get)

        parts = [cfg["alert"]]
        if ratio > VERY_CLOSE_RATIO:
            parts.append("çok yakın")
        elif ratio > NEARBY_RATIO:
            parts.append("yakın")
        if zone == "left":
            parts.append("solunuzda")
        elif zone == "right":
            parts.append("sağınızda")
        alerts.append((cfg["priority"], ", ".join(parts)))

    alerts.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in alerts]


def generate_path_guidance(mask: np.ndarray):
    """
    Alt yarıdaki merkezi koridoru analiz ederek yürüme yönü üretir.
    Sol/sağ kenar pikselleri (CORRIDOR_MARGIN) yoksayılır — kullanıcı
    ekranın tam kenarına yürümeyeceği için o pikseller yönü yanıltır.
    """
    h, w = mask.shape
    bottom_start = int(h * (1.0 - PATH_BOTTOM_FRACTION))
    bottom = mask[bottom_start:, :]

    # Merkezi koridor: kenar piksellerini çıkar
    c_left  = int(w * CORRIDOR_MARGIN)
    c_right = int(w * (1.0 - CORRIDOR_MARGIN))
    corridor   = bottom[:, c_left:c_right]
    corridor_w = c_right - c_left

    walkable    = (corridor == WALKABLE_SURFACE)
    walkable_px = float(np.sum(walkable))

    if walkable_px == 0:
        return None

    walkable_ratio = walkable_px / float(corridor.size)
    if walkable_ratio < MIN_WALKABLE_FOR_GUIDANCE:
        return "Yürünebilir alan çok azalıyor, dikkatli ilerleyin"

    # Koridoru 3 dilime böl: sol / orta / sağ
    third = corridor_w // 3
    left_ratio   = float(np.sum(walkable[:, :third]))          / float(walkable[:, :third].size)
    center_ratio = float(np.sum(walkable[:, third: 2 * third])) / float(walkable[:, third: 2 * third].size)
    right_ratio  = float(np.sum(walkable[:, 2 * third:]))      / float(walkable[:, 2 * third:].size)

    narrow_suffix = ", yol daralıyor" if walkable_ratio < 0.18 else ""

    # Orta dilim yeterliyse düz git
    if center_ratio >= 0.40:
        return "Düz yürüyün" + narrow_suffix

    # Orta yetersiz — daha dolu yana yönlendir
    if left_ratio > right_ratio:
        return "Hafif sola yönelin" + narrow_suffix
    if right_ratio > left_ratio:
        return "Hafif sağa yönelin" + narrow_suffix

    return "Düz yürüyün" + narrow_suffix


# ════════════════════════════════════════════════════════════════════════════
#  FOTOĞRAF MODU — ana akış
# ════════════════════════════════════════════════════════════════════════════

def run_photo_mode(image_path: str, nav_active: bool) -> None:
    if not os.path.exists(image_path):
        print(f"HATA: Dosya bulunamadı: {image_path}")
        sys.exit(1)

    print(f"[Yükleniyor] {image_path}")
    rgb  = load_image_as_rgb(image_path)
    print(f"[Boyut] {rgb.shape[1]}×{rgb.shape[0]}\n")

    mask   = rgb_to_mask(rgb)
    alerts = analyse_and_alert(mask, nav_active=nav_active)

    # Yaya geçidini navigasyon aktif değilse filtrele
    filtered = [a for a in alerts if not ("geçidi" in a and not nav_active)]
    top_hazard = filtered[0] if filtered else None
    guidance   = generate_path_guidance(mask)

    print()
    print("── Sistem Çıktısı ───────────────────────────────────")
    if guidance:
        print(f"  🗣  {guidance}" + (f" — {top_hazard}" if top_hazard else ""))
    elif top_hazard:
        print(f"  🗣  {top_hazard}")
    else:
        print("  (yürünebilir alan algılanamadı)")

    if not nav_active and any("geçidi" in a for a in alerts):
        print("  ℹ  (yaya geçidi algılandı ama navigasyon aktif değil, sessiz geçildi)")
    print()


# ════════════════════════════════════════════════════════════════════════════
#  KAMERA MODU — gerçek PerceptionPipeline ile canlı kamera
# ════════════════════════════════════════════════════════════════════════════

def run_camera_mode(model_path: str, camera_index: int) -> None:
    """
    Ham kamera frame'lerini PerceptionPipeline'a verir.
    Maske overlay'i OpenCV penceresinde gösterir, yön kılavuzunu
    hem ekrana hem terminale yazar. 'q' tuşu ile çıkış.
    """
    import cv2

    # src/ klasörünü Python path'ine ekle (perception modüllerini bulmak için)
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from ai.perception import PerceptionPipeline, render_overlay

    print(f"[Kamera] Model yükleniyor: {model_path}")
    pipeline = PerceptionPipeline(model_path=model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"HATA: Kamera {camera_index} açılamadı.")
        sys.exit(1)

    print(f"[Kamera] Kamera {camera_index} açıldı. Çıkmak için 'q' basın.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("UYARI: Frame okunamadı, yeniden deneniyor...")
            time.sleep(0.1)
            continue

        result = pipeline.process(frame)

        # ── Terminal çıktısı ────────────────────────────────────
        guidance = result.path_guidance
        alerts   = result.alerts
        top_hazard = alerts[0] if alerts else None

        if guidance:
            msg = f"{guidance} — {top_hazard}" if top_hazard else guidance
        elif top_hazard:
            msg = top_hazard
        else:
            msg = "(yürünebilir alan algılanamadı)"

        print(f"[{time.strftime('%H:%M:%S')}] {msg}  "
              f"(inf: {result.inference_ms:.0f}ms | walkable: {result.scene.walkable_ratio:.0%})")

        # ── Overlay görselleştirme ──────────────────────────────
        overlay = render_overlay(frame, result.mask)

        # Yön metnini frame üzerine yaz
        cv2.putText(
            overlay, msg,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, msg,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA,
        )

        cv2.imshow("ALAS — Path Guidance", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Kamera] Kapatıldı.")


# ════════════════════════════════════════════════════════════════════════════
#  ARGÜMAN ÇÖZÜMLEYICI & GİRİŞ NOKTASI
# ════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ALAS path guidance testi — fotoğraf veya canlı kamera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  Fotoğraf : venv/bin/python test_guidance_from_overlay.py overlay.png\n"
            "  Fotoğraf+nav: venv/bin/python test_guidance_from_overlay.py overlay.png --nav\n"
            "  Kamera   : venv/bin/python test_guidance_from_overlay.py --camera --model model.onnx\n"
        ),
    )

    # Fotoğraf modu
    p.add_argument(
        "image", nargs="?", default=None,
        help="Overlay görsel yolu (fotoğraf modu)",
    )
    p.add_argument(
        "--nav", action="store_true",
        help="Navigasyon aktif simülasyonu (yaya geçidi sesli bildirim açılır)",
    )

    # Kamera modu
    p.add_argument(
        "--camera", action="store_true",
        help="Canlı kamera modunu etkinleştir",
    )
    p.add_argument(
        "--model", default="models/segmentation/alas_engine.trt",
        help="Model dosya yolu (.onnx veya .trt/.engine) — sadece kamera modunda kullanılır",
    )
    p.add_argument(
        "--camera-index", type=int, default=0,
        help="Kamera cihaz indeksi (varsayılan: 0)",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.camera:
        # ── KAMERA MODU ──────────────────────────────────────────
        run_camera_mode(
            model_path=args.model,
            camera_index=args.camera_index,
        )
    elif args.image:
        # ── FOTOĞRAF MODU ────────────────────────────────────────
        run_photo_mode(
            image_path=args.image,
            nav_active=args.nav,
        )
    else:
        build_parser().print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
