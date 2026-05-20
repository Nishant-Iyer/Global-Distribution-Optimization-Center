import time
import pandas as pd
import numpy as np
import torch
from gdoc_opt.models.base import BaseOptimizer
from gdoc_opt.distance import haversine_vectorized
from typing import Dict, Any

class PyTorchOptimizer(BaseOptimizer):
    def fit(self, df: pd.DataFrame, n_clusters: int, **kwargs) -> Dict[str, Any]:
        """
        Fits a spherical facility location model using PyTorch and projected gradient descent.
        Optimizes coordinates directly on the 2-sphere S^2.
        
        Args:
            df (pd.DataFrame): Dataframe of cities.
            n_clusters (int): Number of DCs.
            **kwargs:
                weight_column (str): Column to use for demand weights. Default is 'population'.
                epochs (int): Number of optimization epochs. Default is 300.
                learning_rate (float): Adam learning rate. Default is 0.05.
                lpi_factor (float): Penalty factor for low LPI (0 to 1). Default is 0.0.
        """
        start_time = time.time()
        weight_col = kwargs.get("weight_column", "population")
        epochs = kwargs.get("epochs", 300)
        lr = kwargs.get("learning_rate", 0.05)
        lpi_factor = kwargs.get("lpi_factor", 0.0)
        
        # Prepare inputs as PyTorch Tensors
        # N x 3 Cartesian coordinates of cities
        coords_np = df[["X", "Y", "Z"]].values
        weights_np = df[weight_col].values.astype(np.float32)
        
        N = len(df)
        
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        X_cities = torch.tensor(coords_np, dtype=torch.float32, device=device) # Shape: (N, 3)
        W_cities = torch.tensor(weights_np, dtype=torch.float32, device=device) # Shape: (N,)
        
        # Initialize DC centers on the unit sphere (randomly select from cities to start)
        init_indices = np.random.choice(N, n_clusters, replace=False)
        centers = torch.tensor(coords_np[init_indices], dtype=torch.float32, device=device, requires_grad=True) # Shape: (K, 3)
        
        optimizer = torch.optim.Adam([centers], lr=lr)
        
        # Geodesic radius of Earth in km
        R = 6371.0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Normalize centers to ensure they stay on the unit sphere
            normalized_centers = centers / torch.norm(centers, p=2, dim=1, keepdim=True)
            
            # Compute spherical distance via dot product: d = R * arccos(clip(X_cities @ normalized_centers.T, -1, 1))
            # Shape: (N, K)
            dot_prod = torch.matmul(X_cities, normalized_centers.T)
            dot_prod = torch.clamp(dot_prod, -0.99999, 0.99999)
            
            # Distance matrix (in km)
            dists = R * torch.acos(dot_prod)
            
            # Find closest DC index for each city
            min_dists, _ = torch.min(dists, dim=1) # Shape: (N,)
            
            # Loss is population-weighted distance sum
            loss = torch.sum(W_cities * min_dists)
            
            loss.backward()
            optimizer.step()
            
        # Final normalized centers on CPU
        with torch.no_grad():
            final_centers = (centers / torch.norm(centers, p=2, dim=1, keepdim=True)).cpu().numpy()
            
        # Map optimized centers to the closest actual cities
        dc_locations = []
        assignments = np.zeros(N, dtype=int)
        
        for cluster_id in range(n_clusters):
            c_x, c_y, c_z = final_centers[cluster_id]
            
            lon_c = np.degrees(np.arctan2(c_y, c_x))
            lat_c = np.degrees(np.arcsin(c_z))
            
            # Find the closest city in df to this optimized center
            distances = haversine_vectorized(
                df["longitude"].values,
                df["latitude"].values,
                lon_c,
                lat_c
            )
            
            closest_idx = df.index[np.argmin(distances)]
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
            
        # Final assignment and distance calculation (taking LPI penalty into account if set)
        dc_lons = np.array([dc["longitude"] for dc in dc_locations])
        dc_lats = np.array([dc["latitude"] for dc in dc_locations])
        dc_lpis = np.array([dc["lpi"] for dc in dc_locations])
        
        total_weighted_distance = 0.0
        
        for i, row in enumerate(df.itertuples()):
            dists = haversine_vectorized(
                dc_lons,
                dc_lats,
                row.longitude,
                row.latitude
            )
            
            # Incorporate LPI Penalty: effective_distance = distance * (1 + lpi_factor * (5.0 - lpi_dc))
            effective_dists = dists * (1.0 + lpi_factor * (5.0 - dc_lpis))
            
            best_dc_id = np.argmin(effective_dists)
            assignments[i] = int(best_dc_id)
            total_weighted_distance += getattr(row, weight_col) * dists[best_dc_id]
            
        fit_time = time.time() - start_time
        
        return {
            "dc_locations": dc_locations,
            "assignments": assignments.tolist(),
            "total_distance": float(total_weighted_distance),
            "metrics": {
                "fit_time_seconds": fit_time,
                "epochs": epochs,
                "device": device,
                "algorithm": "PyTorch Gradient Descent on Sphere"
            }
        }
