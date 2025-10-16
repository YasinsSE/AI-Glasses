# -*- coding: utf-8 -*-
import osmnx as ox
import pandas as pd

# ===== PARAMS =====
OSM_PATH = r"C:/Users/SAS/Desktop/EnesUNLU/GPS/planet_32.789,39.899_32.905,39.952.osm.gz"
OUT_ALL_CSV = "highways_all.csv"
OUT_ALL_GEOJSON = "highways_all.geojson"
OUT_WALK_CSV = "highways_walkable.csv"
OUT_WALK_GEOJSON = "highways_walkable.geojson"
# Yaya için kabul edilen yol türleri
WALK_HWY = {
    "footway","path","pedestrian","steps","living_street",
    "residential","service","track","tertiary","tertiary_link",
    "secondary","secondary_link","primary","primary_link"
}
MOTOR_ONLY = {"motorway","motorway_link","trunk","trunk_link"}
# ===================

print("Graf yükleniyor...")
G = ox.graph_from_xml(OSM_PATH, simplify=True, bidirectional=True)

print("GDF'ler çıkarılıyor...")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

# ---- Tüm highway çizgileri ----
all_edges = edges_gdf.copy()
# CSV için temel kolonlar + WKT
all_csv = all_edges.reset_index()
keep = [c for c in [
    "u","v","key","name","highway","oneway","access","foot","sidewalk","surface",
    "width","maxspeed","length","geometry"
] if c in all_csv.columns]
all_csv = all_csv[keep]
all_csv["wkt"] = all_csv["geometry"].apply(lambda g: g.wkt)
all_csv.drop(columns=["geometry"], inplace=True)
all_csv.to_csv(OUT_ALL_CSV, index=False)
all_edges.to_file(OUT_ALL_GEOJSON, driver="GeoJSON")
print(f"Yazıldı: {OUT_ALL_CSV}, {OUT_ALL_GEOJSON} (satır: {len(all_edges)})")

# ---- Yaya-uygun alt küme (rota için kullanıma hazır çizgiler) ----
def walk_ok(row):
    h = row.get("highway")
    foot = str(row.get("foot","")).lower()
    sidewalk = str(row.get("sidewalk","")).lower()
    hs = set(h if isinstance(h,(list,tuple,set)) else [h]) if h is not None else set()

    if hs & {"footway","path","pedestrian","steps","living_street"}:
        return True
    if hs & MOTOR_ONLY:
        return foot in {"yes","designated","permissive"}
    if hs & WALK_HWY:
        if sidewalk and sidewalk not in {"no","none",""}:
            return True
        if foot in {"yes","designated","permissive"}:
            return True
        # kaldırım/foot etiketi yoksa yine de kabul etmek istersen aç:
        # return True
    return False

walk_edges = edges_gdf[edges_gdf.apply(walk_ok, axis=1)].copy()

walk_csv = walk_edges.reset_index()
keep2 = [c for c in [
    "u","v","key","name","highway","oneway","access","foot","sidewalk","surface",
    "width","maxspeed","length","geometry"
] if c in walk_csv.columns]
walk_csv = walk_csv[keep2]
walk_csv["wkt"] = walk_csv["geometry"].apply(lambda g: g.wkt)
walk_csv.drop(columns=["geometry"], inplace=True)
walk_csv.to_csv(OUT_WALK_CSV, index=False)
walk_edges.to_file(OUT_WALK_GEOJSON, driver="GeoJSON")
print(f"Yazıldı: {OUT_WALK_CSV}, {OUT_WALK_GEOJSON} (satır: {len(walk_edges)})")
