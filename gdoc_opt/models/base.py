from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseOptimizer(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, n_clusters: int, **kwargs) -> Dict[str, Any]:
        """
        Fits the model on the provided dataframe containing city coordinates and demand weights.
        
        Args:
            df (pd.DataFrame): Dataframe containing 'longitude', 'latitude', 'population', 'X', 'Y', 'Z' columns.
            n_clusters (int): Number of distribution centers to optimize.
            **kwargs: Extra parameters like capacity, metric type, etc.
            
        Returns:
            Dict[str, Any]: Optimization result dictionary containing:
                - 'dc_locations': List of dicts with DC city metadata.
                - 'assignments': Array of cluster indices mapping each city to its assigned DC.
                - 'total_distance': Float representing total weighted distance (km).
                - 'metrics': Dict of other statistics (e.g. run times, capacities).
        """
        pass
