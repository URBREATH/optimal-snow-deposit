from ortools.sat.python import cp_model
import numpy as np

def solve_capacitated_p_median_cpsat(
    cluster_dist_matrix,
    cluster_demands,
    capacities,
    p,
    allowed_indices,
    cost_scale=100,          # scales distances to integer costs
    time_limit_sec=None,     # optional time limit
    num_workers=8,           # parallel workers
    log_search_progress=False
):
    """
    Solve capacitated p-median on clusters using OR-Tools CP-SAT.

    Parameters
    ----------
    cluster_dist_matrix : (n_clusters, n_depots) float distances
    cluster_demands : (n_clusters,) float demands
    capacities : (n_depots,) float capacities
    p : int, number of depots to open
    allowed_indices : list of list[int], allowed depots by cluster
    cost_scale : float, factor to scale distances to integer costs
    time_limit_sec : float or None, CP-SAT time limit
    num_workers : int, threads for CP-SAT
    log_search_progress : bool, CP-SAT logging

    Returns
    -------
    selected_depots : list[int]
        Indices of depots opened.
    cluster_assignment : np.ndarray shape (n_clusters,)
        Assigned depot index per cluster.
    best_obj : float
        Objective value in original distance units (approx).
    """
    n_clusters, n_depots = cluster_dist_matrix.shape

    # cp-sat requires integers for coefficients
    demands = np.round(cluster_demands).astype(int)
    caps = np.round(capacities).astype(int)

    model = cp_model.CpModel()

    # x[j] = 1 if depot j is opened
    x = [model.NewBoolVar(f"x_{j}") for j in range(n_depots)]

    # y[c,j] = 1 if cluster c assigned to depot j (only for allowed pairs)
    y = {}
    for c in range(n_clusters):
        for j in allowed_indices[c]:
            y[c, j] = model.NewBoolVar(f"y_{c}_{j}")

    # each cluster must be assigned to exactly one depot
    for c in range(n_clusters):
        model.Add(sum(y[c, j] for j in allowed_indices[c]) == 1)

    # we only assign to open depots
    for c in range(n_clusters):
        for j in allowed_indices[c]:
            model.Add(y[c, j] <= x[j])

    # compute for exactly p depots
    model.Add(sum(x[j] for j in range(n_depots)) == p)

    # capacity constraints
    for j in range(n_depots):
        terms = []
        for c in range(n_clusters):
            if (c, j) in y:
                terms.append(demands[c] * y[c, j])
        if terms:
            model.Add(sum(terms) <= caps[j] * x[j])

    # OBJECTIVE: minimize sum(demand_c * dist_cj * y_cj)
    objective_terms = []
    for c in range(n_clusters):
        for j in allowed_indices[c]:
            # scaling distance to integer
            cost_int = int(round(cluster_dist_matrix[c, j] * cost_scale))
            objective_terms.append(cost_int * y[c, j])

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = num_workers

    if time_limit_sec is not None:
        solver.parameters.max_time_in_seconds = time_limit_sec

    solver.parameters.log_search_progress = log_search_progress

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"No solution found. CP-SAT status = {status}")

    selected_depots = [j for j in range(n_depots) if solver.Value(x[j]) == 1]

    cluster_assignment = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        for j in allowed_indices[c]:
            if solver.Value(y[c, j]) == 1:
                cluster_assignment[c] = j
                break

    best_obj_scaled = solver.ObjectiveValue()
    best_obj = best_obj_scaled / cost_scale  # rescale back to original distance units (approx)

    return selected_depots, cluster_assignment, best_obj