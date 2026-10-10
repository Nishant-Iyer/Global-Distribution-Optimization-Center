import time
import pandas as pd
import numpy as np
from gdoc_opt.models.base import BaseOptimizer
from gdoc_opt.distance import compute_distance_matrix
from typing import Dict, Any, List

class KMedoidsOptimizer(BaseOptimizer):
    def fit(self, df: pd.DataFrame, n_clusters: int, **kwargs) -> Dict[str, Any]:
        """
        Fits a geodesic K-Medoids model. Custom NumPy implementation of PAM.
        
        Args:
            df (pd.DataFrame): Dataframe containing 'longitude', 'latitude', and population.
            n_clusters (int): Number of DCs (medoids).
            **kwargs:
                metric (str): 'haversine' or 'geodesic'. Default is 'haversine'.
                weight_column (str): Column to use for demand weights. Default is 'population'.
                max_iter (int): Maximum improvement passes. Default is 200.
                random_state (int): Seed for the k-means++ style start. Default is 0.
                distance_matrix (np.ndarray): Precomputed n x n distances; skips computing them.
                site_costs (np.ndarray): Fixed cost of opening a center at each city, in the same
                    units as weight x distance. Default is zero.
        """
        start_time = time.time()
        metric = kwargs.get("metric", "haversine")
        weight_col = kwargs.get("weight_column", "population")
        max_iter = kwargs.get("max_iter", 200)
        rng = np.random.default_rng(kwargs.get("random_state", 0))
        lpi_factor = kwargs.get("lpi_factor", 0.0)
        
        n = len(df)
        if n_clusters > n:
            raise ValueError("n_clusters cannot exceed number of data points.")
            
        weights = df[weight_col].values.astype(float)
        lpi_values = df["lpi"].values.astype(float)
        
        # 1. Compute geodesic distance matrix
        D = kwargs.get("distance_matrix")
        if D is None:
            D = compute_distance_matrix(df, metric=metric)
        site_costs = kwargs.get("site_costs")
        site_costs = np.zeros(n) if site_costs is None else np.asarray(site_costs, dtype=float)
        
        # Compute LPI penalized distance matrix for medoid selection
        # D_penalized[i, j] = D[i, j] * (1.0 + lpi_factor * (5.0 - lpi_j))
        D_penalized = D * (1.0 + lpi_factor * (5.0 - lpi_values[np.newaxis, :]))
        
        # 2. Initialize Medoids (geodesic K-Means++) using penalized distance
        # First medoid is the 1-medoid (minimizer of single DC cost)
        total_costs = np.sum(D_penalized * weights[:, np.newaxis], axis=0) + site_costs
        medoids = [int(np.argmin(total_costs))]
        
        # Add remaining medoids using distance-weighted probability
        for _ in range(1, n_clusters):
            # Compute distance to closest medoid for each city
            min_dist_to_medoid = np.min(D_penalized[:, medoids], axis=1)
            # Distance weighted by demand
            prob = min_dist_to_medoid * weights
            prob = prob / np.sum(prob)
            
            # Select next medoid
            next_medoid = rng.choice(n, p=prob)
            medoids.append(int(next_medoid))
            
        medoids = np.array(medoids)
        
        # 3. PAM swap refinement: each pass evaluates every (medoid, candidate) swap
        #    and applies the best one, until no swap lowers the cost (a true local optimum).
        best_cost = self._calculate_cost(medoids, D_penalized, weights) + site_costs[medoids].sum()
        iterations = 0
        while iterations < max_iter:
            iterations += 1
            best_swap, best_swap_cost = None, best_cost
            for i in range(n_clusters):
                others = np.delete(medoids, i)
                dist_others = np.min(D_penalized[:, others], axis=1) if len(others) else np.full(n, np.inf)
                # cost of replacing medoid i with each candidate j, for all j at once
                swap_costs = (np.minimum(dist_others[:, np.newaxis], D_penalized) * weights[:, np.newaxis]).sum(axis=0)
                swap_costs = swap_costs + site_costs + site_costs[others].sum()
                swap_costs[medoids] = np.inf
                j = int(np.argmin(swap_costs))
                if swap_costs[j] < best_swap_cost - 1e-9:
                    best_swap, best_swap_cost = (i, j), swap_costs[j]
            if best_swap is None:
                break
            medoids[best_swap[0]] = best_swap[1]
            best_cost = best_swap_cost

        # 4. Final assignments (based on penalized distances)
        closest_medoid_indices = np.argmin(D_penalized[:, medoids], axis=1)
        
        # Calculate actual physical distance (unpenalized) for final reporting
        actual_total_distance = self._calculate_cost(medoids, D, weights)
        
        dc_locations = []
        for cluster_id, medoid_idx in enumerate(medoids):
            closest_city = df.iloc[medoid_idx]
            dc_locations.append({
                "city": closest_city["city"],
                "country": closest_city["country"],
                "latitude": float(closest_city["latitude"]),
                "longitude": float(closest_city["longitude"]),
                "population": int(closest_city["population"]),
                "lpi": float(closest_city["lpi"]),
                "gdp_per_capita": float(closest_city["gdp_per_capita"]),
                "cluster_id": cluster_id
            })
            
        fit_time = time.time() - start_time
        
        return {
            "dc_locations": dc_locations,
            "medoid_indices": [int(m) for m in medoids],
            "assignments": closest_medoid_indices.tolist(),
            "total_distance": float(actual_total_distance),
            "metrics": {
                "fit_time_seconds": fit_time,
                "iterations": iterations,
                "algorithm": f"K-Medoids ({metric.capitalize()} distance)"
            }
        }
        
    def _calculate_cost(self, medoids: np.ndarray, D: np.ndarray, weights: np.ndarray) -> float:
        """Helper to calculate total weighted distance cost of configuration."""
        min_dist = np.min(D[:, medoids], axis=1)
        return float(np.sum(min_dist * weights))
