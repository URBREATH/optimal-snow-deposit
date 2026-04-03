from src.clustering import (
    cluster_streets,
    build_cluster_level_data,
    build_allowed_assignments_cluster,
    map_clusters_to_streets,
)
from src.solvers import solve_capacitated_p_median
import numpy as np
import geopandas as gpd
import os

# User inputs — these will be set by the API layer per job
total_snow_volume = 1_000_000
p = 5
solver_method = "cpsat"

def run_pipeline(job_id, total_snow_volume, p, solver_method):

    path = os.path.join("outputs", str(job_id))
    # print(path)
    os.makedirs(path, exist_ok=True)

    # Load preprocessed data
    roads = gpd.read_parquet("data_prepared/roads.parquet")
    deposits = gpd.read_parquet("data_prepared/deposits.parquet")
    dist_matrix = np.load("data_prepared/dist_matrix.npy")
    road_area_fractions = np.load("data_prepared/road_area_fractions.npy")
    capacities = np.load("data_prepared/capacities.npy")

    snow_weights = road_area_fractions * total_snow_volume

    roads_with_clusters, cluster_ids = cluster_streets(roads, n_clusters=300)
    cluster_dist_matrix, cluster_demands, cluster_to_streets = build_cluster_level_data(
        dist_matrix, snow_weights, cluster_ids
    )
    allowed_indices = build_allowed_assignments_cluster(cluster_dist_matrix, K=100) #(K=100 for cp-sat, K=20 for milp-cbc)

    selected_depots, cluster_assignment, best_obj = solve_capacitated_p_median(
        cluster_dist_matrix,
        cluster_demands,
        capacities,
        p=p,
        allowed_indices=allowed_indices,
        method=solver_method,
        cost_scale=100,
        time_limit_sec=None,
        num_workers=8,
        log_search_progress=False,
    )

    print("Selected depots:", selected_depots)
    print("Objective:", best_obj)

    n_streets = dist_matrix.shape[0]
    street_assignment = map_clusters_to_streets(
        cluster_assignment, cluster_to_streets, n_streets
    )

    # selected deposits 
    selected_deposits_gdf = deposits.iloc[selected_depots].copy()
    selected_deposits_gdf["deposit_index"] = selected_depots
    selected_deposits_gdf.to_file(os.path.join(path,"results_deposits.geojson"), driver="GeoJSON")

    # street assignment (attach the assigned deposit index to each road)
    roads_with_clusters["assigned_deposit_index"] = street_assignment
    roads_with_clusters.to_file(os.path.join(path,"results_street_assignment.geojson"), driver="GeoJSON")

    print("Results saved to results_deposits.geojson and results_street_assignment.geojson")

    return os.path.join(path,"results_deposits.geojson"), os.path.join(path,"results_street_assignment.geojson")