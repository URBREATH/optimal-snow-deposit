import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans

def cluster_streets(roads_gdf, n_clusters=300, metric_crs=3301):
    """
    Cluster streets into spatial demand zones using KMeans on centroids.

    roads_gdf : GeoDataFrame with geometry and 'snow_m3'
    n_clusters : int, number of clusters
    metric_crs : projected CRS for distance-based clustering (e.g., EPSG:3301 for Estonia)

    Returns
    -------
    roads_with_clusters : GeoDataFrame
        Original roads with 'cluster_id' column.
    cluster_ids : np.ndarray shape (n_streets,)
        Cluster index per street.
    """
    roads = roads_gdf.copy()

    # project to metric CRS and compute centroids
    roads_metric = roads.to_crs(metric_crs)
    centroids = roads_metric.geometry.centroid

    coords = np.vstack([centroids.x.values, centroids.y.values]).T

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    cluster_ids = kmeans.fit_predict(coords)

    roads["cluster_id"] = cluster_ids
    return roads, cluster_ids

def build_cluster_level_data(dist_matrix, snow_weights, cluster_ids):
    """
    Aggregate street-level data to cluster-level.

    dist_matrix : (n_streets, n_depots) distances
    snow_weights : (n_streets,) demand per street
    cluster_ids : (n_streets,) cluster index for each street

    Returns
    -------
    cluster_dist_matrix : (n_clusters, n_depots) weighted distances
    cluster_demands : (n_clusters,)
    cluster_to_streets : list of arrays with street indices per cluster
    """
    n_streets, n_depots = dist_matrix.shape
    cluster_ids = np.asarray(cluster_ids)
    snow_weights = np.asarray(snow_weights)

    n_clusters = cluster_ids.max() + 1

    cluster_demands = np.zeros(n_clusters, dtype=float)
    cluster_dist_matrix = np.zeros((n_clusters, n_depots), dtype=float)
    cluster_to_streets = []

    for c in range(n_clusters):
        idx = np.where(cluster_ids == c)[0]
        cluster_to_streets.append(idx)

        if len(idx) == 0:
            continue

        demands_c = snow_weights[idx]
        total_demand_c = demands_c.sum()
        cluster_demands[c] = total_demand_c

        dist_c = dist_matrix[idx, :]  # (n_cluster_streets, n_depots)

        if total_demand_c > 0:
            cluster_dist_matrix[c, :] = (dist_c * demands_c[:, None]).sum(axis=0) / total_demand_c
        else:
            cluster_dist_matrix[c, :] = dist_c.mean(axis=0)

    return cluster_dist_matrix, cluster_demands, cluster_to_streets


def map_clusters_to_streets(cluster_assignment, cluster_to_streets, n_streets):
    """
    Map cluster-level depot assignments back to individual streets.

    cluster_assignment : (n_clusters,) depot index per cluster
    cluster_to_streets : list of arrays with street indices per cluster
    n_streets : total number of streets

    Returns
    -------
    street_assignment : np.ndarray shape (n_streets,)
        Assigned depot index per street.
    """
    street_assignment = np.zeros(n_streets, dtype=int)
    for c, street_indices in enumerate(cluster_to_streets):
        street_assignment[street_indices] = cluster_assignment[c]
    return street_assignment



def build_allowed_assignments_cluster(cluster_dist_matrix, K=100):
    """
    For each cluster, keep only K nearest depots.

    cluster_dist_matrix : (n_clusters, n_depots)
    K : max depots per cluster

    Returns
    -------
    allowed_indices : list of list[int]
        allowed_indices[c] = list of depot indices for cluster c.
    """
    n_clusters, n_depots = cluster_dist_matrix.shape
    allowed_indices = []

    for c in range(n_clusters):
        idx_sorted = np.argsort(cluster_dist_matrix[c, :])
        allowed_indices.append(idx_sorted[:min(K, n_depots)].tolist())

    return allowed_indices
