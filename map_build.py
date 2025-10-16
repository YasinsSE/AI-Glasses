#"C:/Users/SAS/Desktop/EnesUNLU/GPS/planet_32.789,39.899_32.905,39.952.osm.pbf"
# "C:/Users/SAS/Desktop/EnesUNLU/GPS/planet_32.789,39.899_32.905,39.952.osm.gz"
# import osmnx as ox

#OSM_PATH = r"C:/Users/SAS/Desktop/EnesUNLU/GPS/planet_32.789,39.899_32.905,39.952.osm.gz"  # senin dosya yolu
#CENTER   = (39.9208, 32.8541)   # (lat, lon)
#RADIUS_M = 8000

#G = ox.graph_from_xml(OSM_PATH, simplify=True)

# 1) Koordinatı node'a çevir
#center_node = ox.distance.nearest_nodes(G, X=CENTER[1], Y=CENTER[0])

# 2) O node'dan RADIUS_M metreye kadar kırp
#G = ox.truncate.truncate_graph_dist(G, center_node, RADIUS_M, weight="length")

#print(len(G.nodes), len(G.edges))
#ox.save_graphml(G, "ankara_walk_8km.graphml")


# build_poi_only.py
# Gerekenler: pip install osmnx pandas
# map_build.py
# OSM dosyasından belirli POI'leri çıkarır (market, eczane, metro)
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

OSM_PATH = r"C:/Users/SAS/Desktop/EnesUNLU/GPS/planet_32.789,39.899_32.905,39.952.osm.gz"
CENTER   = (39.9208, 32.8541)
RADIUS_M = 8000

TAGS = {
    "shop": ["supermarket"],
    "amenity": ["pharmacy"],
    "railway": ["station"],   # metrolar genelde railway=station + station=subway
    "station": ["subway"]
}

import osmnx as ox
import pandas as pd
from shapely.geometry import Point

def extract_poi_csv_multi(osm_path, tags_dict, center_latlon, radius_m, out_csv):
    gdf = ox.features_from_xml(osm_path, tags=tags_dict)
    if gdf.empty:
        print(f"{out_csv}: veri yok"); return

    # Nokta geometriye indir
    gdf = gdf[gdf.geometry.type.isin(["Point","MultiPoint"])].copy()
    if gdf.empty:
        print(f"{out_csv}: nokta tipi veri yok"); return
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.type=="Point"].copy()
    gdf.set_crs(4326, inplace=True)  # güvence

    # Tür etiketle
    def kind(r):
        if r.get("shop") == "supermarket": return "supermarket"
        if r.get("amenity") == "pharmacy":  return "pharmacy"
        if r.get("railway") == "station" and r.get("station") == "subway": return "metro"
        if r.get("station") == "subway": return "metro"
        return "other"
    gdf["kind"] = [kind(r) for _, r in gdf.iterrows()]

    # Projeksiyon + mesafe
    gdf_proj = ox.projection.project_gdf(gdf)
    center_ll = Point(center_latlon[1], center_latlon[0])  # lon,lat
    center_proj, _ = ox.projection.project_geometry(center_ll, crs="EPSG:4326", to_crs=gdf_proj.crs)
    dists = gdf_proj.geometry.distance(center_proj)
    gdf_sel = gdf.loc[dists <= radius_m].to_crs(4326)

    if gdf_sel.empty:
        print(f"{out_csv}: kayıt yok"); return

    df = pd.DataFrame({
        "kind": gdf_sel["kind"].values,
        "name": gdf_sel.get("name", pd.Series([""]*len(gdf_sel))).fillna(""),
        "lat":  gdf_sel.geometry.y.values,
        "lon":  gdf_sel.geometry.x.values,
    })
    df.to_csv(out_csv, index=False)
    print(f"{out_csv}: {len(df)} kayıt kaydedildi")

# ---- ana ----
if __name__ == "__main__":
    TAGS = {
        "shop": ["supermarket"],
        "amenity": ["pharmacy"],
        "railway": ["station"],
        "station": ["subway"],
    }
    extract_poi_csv_multi(OSM_PATH, TAGS, CENTER, RADIUS_M, "pois.csv")
