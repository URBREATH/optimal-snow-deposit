from ortools.linear_solver import pywraplp
import numpy as np

def solve_capacitated_p_median_cbc(cluster_dist_matrix,
                                               cluster_demands,
                                               capacities,
                                               p,
                                               allowed_indices):
    """
    Capacitated p-median on clusters with sparse y[i,j].

    Parameters
    ----------
    cluster_dist_matrix : np.ndarray, shape (n_clusters, n_depots)
    cluster_demands : np.ndarray, shape (n_clusters,)
    capacities : np.ndarray, shape (n_depots,)
    p : int
        Number of depots to open.
    allowed_indices : list of list[int]
        allowed_indices[c] = depots that cluster c can connect to.

    Returns
    -------
    selected_depots : list[int]
        Depot indices chosen.
    cluster_assignment : np.ndarray, shape (n_clusters,)
        Assigned depot index per cluster.
    """
    n_clusters, n_depots = cluster_dist_matrix.shape
    demands = np.asarray(cluster_demands)
    caps = np.asarray(capacities)

    solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        raise RuntimeError("CBC solver not available")

    # variables
    x = [solver.IntVar(0, 1, f"x_{j}") for j in range(n_depots)]  # depot open
    y = {}  # assignment (cluster, depot)
    for c in range(n_clusters):
        for j in allowed_indices[c]:
            y[c, j] = solver.IntVar(0, 1, f"y_{c}_{j}")

    # each cluster is assigned to exactly one depot (over allowed depots)
    for c in range(n_clusters):
        solver.Add(sum(y[c, j] for j in allowed_indices[c]) == 1)   
    
    # we only assign to open depots
    for c in range(n_clusters):
        for j in allowed_indices[c]:
            solver.Add(y[c, j] <= x[j])   

    # exactly p depots open
    solver.Add(sum(x[j] for j in range(n_depots)) == p)   

    # capacity constraints
    for j in range(n_depots):
        solver.Add(
            sum(demands[c] * y[c, j] for c in range(n_clusters) if (c, j) in y)   
            <= caps[j] * x[j]
        )

    # OBJECTIVE: minimize sum(demand * distance * assignment)
    objective_terms = []
    for c in range(n_clusters):
        for j in allowed_indices[c]:
            cost = demands[c] * cluster_dist_matrix[c, j]
            objective_terms.append(cost * y[c, j])

    solver.Minimize(solver.Sum(objective_terms)) 

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"No optimal solution found (status = {status})")

    selected_depots = [j for j in range(n_depots) if x[j].solution_value() > 0.5]

    cluster_assignment = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        for j in allowed_indices[c]:
            if y[c, j].solution_value() > 0.5:
                cluster_assignment[c] = j
                break

    obj_value = 0.0
    for c in range(n_clusters):
        j = cluster_assignment[c]
        obj_value += cluster_demands[c] * cluster_dist_matrix[c, j]

    return selected_depots, cluster_assignment, obj_value


# CBC uses a sparse y[c,j] set (because we restrict allowed depots), so:
# it never assigns a cluster to a depot outside its allowed set.
# Therefore cluster_dist_matrix[c, j] is safe — since j will always be valid.