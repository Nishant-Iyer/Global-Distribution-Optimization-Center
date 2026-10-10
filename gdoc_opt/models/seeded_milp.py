"""
Facility location MILP seeded by K-Medoids.

The plain MILP is exact only over its candidate list, and a list of the most populous cities
misses the well-placed smaller cities that make the best sites. Here K-Medoids (fast, considers
every city) proposes the candidates instead: its sites for k-1 .. k+2 centers, each site's nearest
neighbours, and the largest demand centers. The MILP, solved with HiGHS through SciPy, then
confirms or improves that answer over the whole candidate list.

The same model handles capacity limits and per-site running costs:

  minimise   sum_ij w_i * D'_ij * x_ij  +  sum_j f_j * y_j
  subject to sum_j x_ij = 1                 every city fully served
             x_ij <= y_j                    only by open centers
             sum_i w_i x_ij <= C * y_j      capacity (optional)
             sum_j y_j = k                  number of centers (optional)

D' is the distance with the optional logistics (LPI) penalty, w the demand, f the running cost.
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from gdoc_opt.distance import compute_distance_matrix
from gdoc_opt.models.base import BaseOptimizer
from gdoc_opt.models.kmedoids import KMedoidsOptimizer


def candidate_sites(D: np.ndarray, seeds: List[int], weights: np.ndarray, neighbours: int, top_demand: int,
                    forced: List[int]) -> np.ndarray:
    cand = set(seeds) | set(forced)
    for s in seeds:
        cand.update(np.argsort(D[:, s])[:neighbours].tolist())
    cand.update(np.argsort(-weights)[:top_demand].tolist())
    return np.array(sorted(cand))


def solve_facility_milp(D: np.ndarray, weights: np.ndarray, cand: np.ndarray, k: Optional[int],
                        site_costs: Optional[np.ndarray] = None, capacity: Optional[float] = None,
                        forced: Optional[List[int]] = None, time_limit: float = 600.0):
    n, m = D.shape[0], len(cand)
    Dc = D[:, cand]
    f = np.zeros(m) if site_costs is None else np.asarray(site_costs, dtype=float)[cand]
    nx = n * m                                     # x_ij at column i*m + j, then y_j
    c = np.concatenate([(weights[:, None] * Dc).ravel(), f])

    rows = []
    # sum_j x_ij = 1
    A1 = coo_matrix((np.ones(nx), (np.repeat(np.arange(n), m), np.arange(nx))), shape=(n, nx + m))
    rows.append(LinearConstraint(A1, 1, 1))
    # x_ij - y_j <= 0
    r = np.arange(nx)
    A2 = coo_matrix((np.concatenate([np.ones(nx), -np.ones(nx)]),
                     (np.concatenate([r, r]), np.concatenate([r, nx + np.tile(np.arange(m), n)]))),
                    shape=(nx, nx + m))
    rows.append(LinearConstraint(A2, -np.inf, 0))
    if capacity is not None:
        cols = np.arange(nx)
        vals = np.repeat(weights, m)
        A3 = coo_matrix((np.concatenate([vals, -np.full(m, capacity)]),
                         (np.concatenate([np.tile(np.arange(m), n), np.arange(m)]),
                          np.concatenate([cols, nx + np.arange(m)]))), shape=(m, nx + m))
        rows.append(LinearConstraint(A3, -np.inf, 0))
    if k is not None:
        A4 = coo_matrix((np.ones(m), (np.zeros(m, dtype=int), nx + np.arange(m))), shape=(1, nx + m))
        rows.append(LinearConstraint(A4, k, k))

    lb = np.zeros(nx + m)
    ub = np.ones(nx + m)
    pos = {int(s): j for j, s in enumerate(cand)}
    for s in forced or []:
        lb[nx + pos[int(s)]] = 1
    integrality = np.concatenate([np.zeros(nx), np.ones(m)])
    res = milp(c, constraints=rows, bounds=Bounds(lb, ub), integrality=integrality,
               options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 1e-6})
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    y = res.x[nx:] > 0.5
    x = res.x[:nx].reshape(n, m)
    return {"open": cand[y], "x": x[:, y], "objective": float(res.fun), "status": res.message,
            "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0)}


class SeededMILPOptimizer(BaseOptimizer):
    def fit(self, df: pd.DataFrame, n_clusters: Optional[int], **kwargs) -> Dict[str, Any]:
        """
        Args:
            df: cities with latitude, longitude, the weight column, lpi, price_level.
            n_clusters: number of centers to open, or None to let running costs decide.
            **kwargs:
                metric: 'haversine', 'geodesic' or 'transport'. Default 'haversine'.
                distance_matrix: precomputed n x n distances (overrides metric).
                weight_column: demand column. Default 'population'.
                lpi_factor: logistics penalty, as in K-Medoids. Default 0.
                capacity_share: max share of total demand one center may serve, e.g. 0.25. Default None.
                site_cost_km: running cost of a center at US prices, expressed as the extra average
                    distance (km) it would be worth to avoid one. Scaled by each country's price level.
                    Default 0.
                neighbours: nearest cities added around each K-Medoids site. Default 40.
                top_demand: largest-demand cities always added. Default 25.
                forced_dcs: city names that must be open.
                time_limit: seconds for HiGHS. Default 600.
        """
        start = time.time()
        weight_col = kwargs.get("weight_column", "population")
        w = df[weight_col].values.astype(float)
        D = kwargs.get("distance_matrix")
        if D is None:
            D = compute_distance_matrix(df, metric=kwargs.get("metric", "haversine"))
        lpi = df["lpi"].values.astype(float)
        Dp = D * (1.0 + kwargs.get("lpi_factor", 0.0) * (5.0 - lpi[None, :]))

        total = w.sum()
        site_km = kwargs.get("site_cost_km", 0.0)
        price = df["price_level"].values.astype(float) if "price_level" in df else np.ones(len(df))
        f = site_km * price * total if site_km else None
        cap_share = kwargs.get("capacity_share")
        cap = cap_share * total if cap_share else None

        names = df["city"].str.lower().values
        forced = [int(np.where(names == c.lower())[0][0]) for c in kwargs.get("forced_dcs", []) if (names == c.lower()).any()]

        # 1. K-Medoids proposals
        km = KMedoidsOptimizer()
        k0 = n_clusters or 5
        seeds, km_result = [], None
        ks = range(2, 13) if n_clusters is None else sorted({max(1, k0 - 1), k0, k0 + 1, k0 + 2})
        for kk in ks:
            r = km.fit(df, kk, distance_matrix=Dp, weight_column=weight_col, site_costs=f)
            seeds += r["medoid_indices"]
            if kk == k0:
                km_result = r
        km_cost = float(np.sum(np.min(Dp[:, km_result["medoid_indices"]], axis=1) * w))
        if f is not None:
            km_cost += float(f[km_result["medoid_indices"]].sum())
        cand = candidate_sites(Dp, seeds, w, kwargs.get("neighbours", 40), kwargs.get("top_demand", 25), forced)

        # 2. Exact solve over the candidate list
        sol = solve_facility_milp(Dp, w, cand, n_clusters, f, cap, forced, kwargs.get("time_limit", 600.0))
        open_ = sol["open"]
        assign = np.argmax(sol["x"], axis=1)
        served = (sol["x"] * w[:, None]).sum(axis=0)
        phys = float((sol["x"] * D[:, open_] * w[:, None]).sum())

        dc_locations = [{
            "city": df["city"].iloc[j], "country": df["country"].iloc[j],
            "latitude": float(df["latitude"].iloc[j]), "longitude": float(df["longitude"].iloc[j]),
            "population": int(df["population"].iloc[j]), "lpi": float(lpi[j]),
            "share_of_demand": float(served[q] / total), "cluster_id": q,
        } for q, j in enumerate(open_)]
        return {
            "dc_locations": dc_locations,
            "medoid_indices": [int(j) for j in open_],
            "assignments": assign.tolist(),
            "total_distance": phys,
            "metrics": {
                "fit_time_seconds": time.time() - start,
                "candidates": int(len(cand)),
                "objective": sol["objective"],
                "kmedoids_objective": km_cost,
                "improvement_over_kmedoids": (km_cost - sol["objective"]) / km_cost if cap is None else None,
                "running_cost": float(f[open_].sum()) if f is not None else 0.0,
                "solver_status": sol["status"],
                "mip_gap": sol["mip_gap"],
                "algorithm": "K-Medoids-seeded MILP (HiGHS)",
            },
        }
