import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx

ROADS_FPATH = "/Users/nefelikousta/Desktop/work/PhotoLab/projects/UrBREATH/data/Tallinn/optimal_snow_deposit_latitudo/tallinn_roads.geojson"
DEP_MUN_FPATH = "/Users/nefelikousta/Desktop/work/PhotoLab/projects/UrBREATH/data/Tallinn/optimal_snow_deposit_latitudo/snow_capacity_municipality.geojson"
DEP_GOV_FPATH = "/Users/nefelikousta/Desktop/work/PhotoLab/projects/UrBREATH/data/Tallinn/optimal_snow_deposit_latitudo/snow_capacity_government.geojson"

OUTPUT_DIR = "data_prepared/"
print("Loading data...")

roads = gpd.read_file(ROADS_FPATH)
deposits1 = gpd.read_file(DEP_MUN_FPATH)
deposits2 = gpd.read_file(DEP_GOV_FPATH)
deposits = pd.concat([deposits1, deposits2], ignore_index=True)

# Compute per-street area fractions (V-independent); multiply by V at runtime
total_area = roads["area_sqm"].sum()
road_area_fractions = (roads["area_sqm"] / total_area).to_numpy()

capacities = deposits["snow_capacity_m3"].to_numpy()

print("Projecting for centroid extraction...") # EPSG:3301 for Estonia
roads3301 = roads.to_crs(3301) 
deposits3301 = deposits.to_crs(3301)

road_lons = roads3301.geometry.centroid.x.values
road_lats = roads3301.geometry.centroid.y.values

depot_lons = deposits3301.geometry.centroid.x.values
depot_lats = deposits3301.geometry.centroid.y.values

print("Loading OSM graph...")
G = ox.graph_from_place("Tallinn, Estonia", network_type="drive")
ox.distance.add_edge_lengths(G)

print("Finding nearest nodes...")
road_nodes = ox.nearest_nodes(G, road_lons, road_lats)
depot_nodes = ox.nearest_nodes(G, depot_lons, depot_lats)

print("Computing distance matrix...")
n_streets = len(roads)
n_depots = len(deposits)

dist_matrix = np.zeros((n_streets, n_depots))

for j, depot_node in enumerate(depot_nodes):
    dists = nx.shortest_path_length(G, depot_node, weight="length")
    for i, road_node in enumerate(road_nodes):
        dist_matrix[i, j] = dists.get(road_node, 1e9)

print("Saving outputs...")
roads.to_parquet(OUTPUT_DIR + "roads.parquet")
deposits.to_parquet(OUTPUT_DIR + "deposits.parquet")

np.save(OUTPUT_DIR + "road_area_fractions.npy", road_area_fractions)
np.save(OUTPUT_DIR + "capacities.npy", capacities)
np.save(OUTPUT_DIR + "dist_matrix.npy", dist_matrix)
np.save(OUTPUT_DIR + "road_nodes.npy", road_nodes)
np.save(OUTPUT_DIR + "depot_nodes.npy", depot_nodes)

print("Done!")