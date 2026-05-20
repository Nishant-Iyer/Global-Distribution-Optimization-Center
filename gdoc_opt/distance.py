import numpy as np
import pandas as pd
from geopy.distance import geodesic
from typing import Union

EARTH_RADIUS_KM = 6371.0

def haversine_vectorized(
    lons1: np.ndarray, lats1: np.ndarray, lons2: Union[np.ndarray, float], lats2: Union[np.ndarray, float]
) -> np.ndarray:
    """
    Computes Haversine distance between arrays of coordinates (in degrees).
    Supports broadcasting for single vs multiple coordinate sets.
    """
    lon1, lat1 = np.radians(lons1), np.radians(lats1)
    lon2, lat2 = np.radians(lons2), np.radians(lats2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return c * EARTH_RADIUS_KM

def geodesic_exact(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Computes exact geodesic distance on the WGS-84 ellipsoid using geopy.
    Returned value is in kilometers.
    """
    return geodesic((lat1, lon1), (lat2, lon2)).km

def compute_distance_matrix(
    df: pd.DataFrame, metric: str = "haversine"
) -> np.ndarray:
    """
    Computes a symmetric distance matrix for all cities in the dataframe.
    metric can be 'haversine' or 'geodesic'.
    """
    n = len(df)
    coords = df[["longitude", "latitude"]].values
    
    if metric == "haversine":
        # Vectorized computation using broadcasting
        lons = coords[:, 0]
        lats = coords[:, 1]
        
        # Reshape for broadcasting: (n, 1) and (1, n)
        lons_col = lons[:, np.newaxis]
        lats_col = lats[:, np.newaxis]
        
        lons_row = lons[np.newaxis, :]
        lats_row = lats[np.newaxis, :]
        
        return haversine_vectorized(lons_col, lats_col, lons_row, lats_row)
    
    elif metric == "geodesic":
        # Calculate exactly via geopy. Slower but high accuracy.
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = geodesic_exact(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        return dist_matrix
    else:
        raise ValueError(f"Unknown metric: {metric}")
