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


def capacitated_assignment(Dp: np.ndarray, w: np.ndarray, sites, cap: float, f=None):
    """Greedy capacity-aware assignment: cities that lose most by missing their nearest site
    choose first, each taking the nearest site with room. Returns (cost, assignment)."""
    sites = list(sites)
    d = Dp[:, sites]
    srt = np.sort(d, axis=1)
    regret = (srt[:, 1] - srt[:, 0]) * w if len(sites) > 1 else w
    order = np.argsort(-regret)
    pref = np.argsort(d, axis=1)
    load = np.zeros(len(sites))
    asg = np.empty(len(w), dtype=int)
    cost = 0.0
    for i in order:
        for q in pref[i]:
            if load[q] + w[i] <= cap * (1 + 1e-9):
                break
        else:
            return np.inf, None
        asg[i] = q
        load[q] += w[i]
        cost += w[i] * d[i, q]
    if f is not None:
        cost += float(np.sum(f[sites]))
    return cost, asg


def capacitated_local_search(Dp: np.ndarray, w: np.ndarray, start, cap: float, f=None, locked=(),
                             neighbours: int = 20, top_demand: int = 30, max_passes: int = 30):
    """Best-swap search with capacity-aware assignment. Any site may move next to any other
    site or to one of the biggest demand centers."""
    sites = list(start)
    best, _ = capacitated_assignment(Dp, w, sites, cap, f)
    near = np.argsort(Dp, axis=0)[1:neighbours + 1, :]          # column s: nearest cities to site s
    big = list(np.argsort(-w)[:top_demand])
    for _ in range(max_passes):
        pool = set(big)
        for s in sites:
            pool.update(near[:, s].tolist())
        move = None
        for m in range(len(sites)):
            if sites[m] in locked:
                continue
            for c in pool:
                if c in sites:
                    continue
                trial = sites.copy()
                trial[m] = c
                cost, _ = capacitated_assignment(Dp, w, trial, cap, f)
                if cost < best - 1e-6 and (move is None or cost < move[1]):
                    move = (trial, cost)
        if move is None:
            break
        sites, best = move
    return sites, best


def solve_facility_milp(D: np.ndarray, weights: np.ndarray, cand: np.ndarray, k: Optional[int],
                        site_costs: Optional[np.ndarray] = None, capacity: Optional[float] = None,
                        forced: Optional[List[int]] = None, time_limit: float = 600.0,
                        strong_links: int = 8):
    n, m = D.shape[0], len(cand)
    # work in demand shares and thousands of km: keeps HiGHS numerically comfortable
    W = float(weights.sum())
    weights = weights / W
    Dc = D[:, cand] / 1000.0
    f = np.zeros(m) if site_costs is None else np.asarray(site_costs, dtype=float)[cand] / (W * 1000.0)
    capacity = None if capacity is None else capacity / W
    nx = n * m                                     # x_ij at column i*m + j, then y_j
    c = np.concatenate([(weights[:, None] * Dc).ravel(), f])

    rows = []
    # sum_j x_ij = 1
    A1 = coo_matrix((np.ones(nx), (np.repeat(np.arange(n), m), np.arange(nx))), shape=(n, nx + m))
    rows.append(LinearConstraint(A1, 1, 1))
    # x_ij - y_j <= 0. With a capacity limit, sum_i w_i x_ij <= C y_j already keeps closed sites
    # empty, so these rows only tighten the relaxation; keeping them for each city's nearest
    # candidates gives most of the benefit at a fraction of the size.
    if capacity is None:
        r = np.arange(nx)
    else:
        near = np.argsort(Dc, axis=1)[:, :min(strong_links, m)]
        r = (np.arange(n)[:, None] * m + near).ravel()
    A2 = coo_matrix((np.concatenate([np.ones(len(r)), -np.ones(len(r))]),
                     (np.concatenate([np.arange(len(r)), np.arange(len(r))]), np.concatenate([r, nx + r % m]))),
                    shape=(len(r), nx + m))
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
    return {"open": cand[y], "x": x[:, y], "objective": float(res.fun) * W * 1000.0, "status": res.message,
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
        if cap is not None:
            # With a capacity limit the uncapacitated sites are a poor guide (one center can't
            # serve all of China), so the seeds come from a capacity-aware swap search instead,
            # and the candidate list stays tight enough for the solver to prove its answer.
            k_cap = max(k0, int(np.ceil(1.0 / cap_share - 1e-9)))
            heur, heur_cost = capacitated_local_search(Dp, w, km_result["medoid_indices"][:k_cap] if k_cap <= k0 else
                                                       km.fit(df, k_cap, distance_matrix=Dp, weight_column=weight_col)["medoid_indices"],
                                                       cap, f, locked=forced)
            km_cost = heur_cost
            seeds = list(heur)
            cand = candidate_sites(Dp, seeds, w, kwargs.get("neighbours", 15), kwargs.get("top_demand", 10), forced)
            strong = len(cand)
        else:
            nb = kwargs.get("neighbours", 40 if n_clusters else 15)
            cand = candidate_sites(Dp, seeds, w, nb, kwargs.get("top_demand", 25), forced)
            strong = 8   # unused without a capacity limit: every x_ij <= y_j row is kept

        # 2. Exact solve over the candidate list
        sol = solve_facility_milp(Dp, w, cand, n_clusters, f, cap, forced, kwargs.get("time_limit", 600.0), strong)
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
                "kmedoids_objective": km_cost,   # capacity-aware heuristic's cost when capacity is set
                "improvement_over_kmedoids": (km_cost - sol["objective"]) / km_cost,
                "running_cost": float(f[open_].sum()) if f is not None else 0.0,
                "solver_status": sol["status"],
                "mip_gap": sol["mip_gap"],
                "algorithm": "K-Medoids-seeded MILP (HiGHS)",
            },
        }
