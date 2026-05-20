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
                max_iter (int): Maximum swap iterations. Default is 50.
        """
        start_time = time.time()
        metric = kwargs.get("metric", "haversine")
        weight_col = kwargs.get("weight_column", "population")
        max_iter = kwargs.get("max_iter", 50)
        lpi_factor = kwargs.get("lpi_factor", 0.0)
        
        n = len(df)
        if n_clusters > n:
            raise ValueError("n_clusters cannot exceed number of data points.")
            
        weights = df[weight_col].values.astype(float)
        lpi_values = df["lpi"].values.astype(float)
        
        # 1. Compute geodesic distance matrix
        D = compute_distance_matrix(df, metric=metric)
        
        # Compute LPI penalized distance matrix for medoid selection
        # D_penalized[i, j] = D[i, j] * (1.0 + lpi_factor * (5.0 - lpi_j))
        D_penalized = D * (1.0 + lpi_factor * (5.0 - lpi_values[np.newaxis, :]))
        
        # 2. Initialize Medoids (geodesic K-Means++) using penalized distance
        # First medoid is the 1-medoid (minimizer of single DC cost)
        total_costs = np.sum(D_penalized * weights[:, np.newaxis], axis=0)
        medoids = [int(np.argmin(total_costs))]
        
        # Add remaining medoids using distance-weighted probability
        for _ in range(1, n_clusters):
            # Compute distance to closest medoid for each city
            min_dist_to_medoid = np.min(D_penalized[:, medoids], axis=1)
            # Distance weighted by demand
            prob = min_dist_to_medoid * weights
            prob = prob / np.sum(prob)
            
            # Select next medoid
            next_medoid = np.random.choice(n, p=prob)
            medoids.append(int(next_medoid))
            
        medoids = np.array(medoids)
        
        # 3. PAM Swap Refinement Loop (Vectorized first-improvement)
        best_cost = self._calculate_cost(medoids, D_penalized, weights)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            for i in range(n_clusters):
                # Try swapping medoid i with some non-medoid city
                for j in range(n):
                    if j in medoids:
                        continue
                    
                    test_medoids = medoids.copy()
                    test_medoids[i] = j
                    
                    cost = self._calculate_cost(test_medoids, D_penalized, weights)
                    if cost < best_cost:
                        best_cost = cost
                        medoids = test_medoids
                        improved = True
                        break  # First-improvement swap
                if improved:
                    break
                    
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
