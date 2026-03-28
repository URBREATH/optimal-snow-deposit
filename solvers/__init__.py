from .cp_sat import solve_capacitated_p_median_cpsat
from .milp_cbc import solve_capacitated_p_median_cbc

def solve_capacitated_p_median(
    cluster_dist_matrix,
    cluster_demands,
    capacities,
    p,
    allowed_indices,
    method="cpsat",
    **kwargs,
):
    """
    Unified interface for capacitated p-median.

    method : "cpsat" or "cbc"
    kwargs : extra arguments passed to CP-SAT only
             (cost_scale, time_limit_sec, num_workers, log_search_progress)
             CBC does not accept additional arguments and ignores kwargs.
    """
    if method.lower() == "cpsat":
        return solve_capacitated_p_median_cpsat(
            cluster_dist_matrix,
            cluster_demands,
            capacities,
            p,
            allowed_indices,
            **kwargs,
        )
    elif method.lower() == "cbc":
        return solve_capacitated_p_median_cbc(
            cluster_dist_matrix,
            cluster_demands,
            capacities,
            p,
            allowed_indices,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'cpsat' or 'cbc'.")
