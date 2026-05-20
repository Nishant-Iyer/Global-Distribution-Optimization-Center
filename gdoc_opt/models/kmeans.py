import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from gdoc_opt.models.base import BaseOptimizer
from gdoc_opt.distance import haversine_vectorized
from typing import Dict, Any

class KMeansOptimizer(BaseOptimizer):
    def fit(self, df: pd.DataFrame, n_clusters: int, **kwargs) -> Dict[str, Any]:
        """
        Fits a population-weighted KMeans model on 3D Cartesian coordinates.
        Projects centroids back to the sphere and maps them to the closest cities.
        
        Args:
            df (pd.DataFrame): Dataframe with X, Y, Z, latitude, longitude, and population.
            n_clusters (int): Number of DCs.
            **kwargs:
                weight_column (str): Column to use for demand weights. Default is 'population'.
        """
        start_time = time.time()
        weight_col = kwargs.get("weight_column", "population")
        
        lpi_factor = kwargs.get("lpi_factor", 0.0)
        
        # Prepare coordinates and weights
        coords_3d = df[["X", "Y", "Z"]].values
        weights = df[weight_col].values
        
        # Run KMeans with population weights
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(coords_3d, sample_weight=weights)
        
        centroids = kmeans.cluster_centers_
        
        dc_locations = []
        assignments = np.zeros(len(df), dtype=int)
        
        # Map centroids back to the closest actual cities
        for cluster_id in range(n_clusters):
            # Centroid of the cluster
            c_x, c_y, c_z = centroids[cluster_id]
            
            # Project back to spherical coordinates
            lon_c = np.degrees(np.arctan2(c_y, c_x))
            lat_c = np.degrees(np.arcsin(c_z))
            
            # Find the closest city in this cluster to the centroid
            cluster_mask = df["cluster"] == cluster_id
            cluster_df = df[cluster_mask].copy()
            
            # Calculate distance from centroid to all cities in this cluster
            distances = haversine_vectorized(
                cluster_df["longitude"].values,
                cluster_df["latitude"].values,
                lon_c,
                lat_c
            )
            
            # Penalize locations with low LPI
            effective_distances = distances * (1.0 + lpi_factor * (5.0 - cluster_df["lpi"].values))
            
            # Find closest city index in cluster_df
            closest_idx = cluster_df.index[np.argmin(effective_distances)]
            closest_city = df.loc[closest_idx]
            
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
            
        # Re-assign all cities in df to their nearest open DC
        dc_lons = np.array([dc["longitude"] for dc in dc_locations])
        dc_lats = np.array([dc["latitude"] for dc in dc_locations])
        dc_lpis = np.array([dc["lpi"] for dc in dc_locations])
        
        total_weighted_distance = 0.0
        
        # For each city, compute distance to all DCs and assign to the closest
        for i, row in enumerate(df.itertuples()):
            distances = haversine_vectorized(
                dc_lons,
                dc_lats,
                row.longitude,
                row.latitude
            )
            # Apply LPI penalty during final assignment
            effective_dists = distances * (1.0 + lpi_factor * (5.0 - dc_lpis))
            nearest_dc_id = np.argmin(effective_dists)
            assignments[i] = int(nearest_dc_id)
            total_weighted_distance += getattr(row, weight_col) * distances[nearest_dc_id]
            
        fit_time = time.time() - start_time
        
        return {
            "dc_locations": dc_locations,
            "assignments": assignments.tolist(),
            "total_distance": float(total_weighted_distance),
            "metrics": {
                "fit_time_seconds": fit_time,
                "algorithm": "K-Means (Euclidean 3D)"
            }
        }
