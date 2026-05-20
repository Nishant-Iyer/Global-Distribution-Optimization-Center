import pandas as pd
import numpy as np
from typing import Tuple, Dict

# Country GDP per Capita (nominal USD, rough 2023 estimates)
GDP_PER_CAPITA: Dict[str, float] = {
    "United States": 80000.0,
    "Germany": 52000.0,
    "Japan": 34000.0,
    "United Kingdom": 46000.0,
    "France": 44000.0,
    "China": 12500.0,
    "India": 2500.0,
    "Brazil": 10000.0,
    "Russia": 12000.0,
    "Bangladesh": 2700.0,
    "Iran": 4000.0,
    "Nigeria": 2000.0,
    "Cameroon": 1600.0,
    "South Africa": 6000.0,
    "Egypt": 4000.0,
    "Indonesia": 5000.0,
    "Mexico": 11000.0,
    "Canada": 53000.0,
    "Australia": 64000.0,
    "Singapore": 82000.0,
    "Finland": 54000.0,
    "Netherlands": 57000.0,
    "Switzerland": 92000.0,
    "Pakistan": 1500.0,
    "Argentina": 13000.0,
    "Turkey": 11000.0,
    "Saudi Arabia": 30000.0,
    "South Korea": 33000.0,
    "Italy": 35000.0,
    "Spain": 30000.0,
    "Colombia": 6500.0,
    "Kenya": 2000.0,
    "Vietnam": 4300.0,
    "Thailand": 7500.0,
    "Philippines": 3600.0,
    "Malaysia": 12000.0,
}

# World Bank Logistics Performance Index (LPI) scores (1 to 5 scale, 2023)
LPI_SCORES: Dict[str, float] = {
    "Singapore": 4.3,
    "Finland": 4.2,
    "Germany": 4.1,
    "Netherlands": 4.1,
    "Switzerland": 4.1,
    "Japan": 3.9,
    "United States": 3.8,
    "United Kingdom": 3.7,
    "Canada": 3.7,
    "Australia": 3.7,
    "France": 3.7,
    "South Korea": 3.8,
    "China": 3.7,
    "Italy": 3.6,
    "Spain": 3.6,
    "Saudi Arabia": 3.4,
    "India": 3.4,
    "Brazil": 3.2,
    "South Africa": 3.1,
    "Indonesia": 3.0,
    "Mexico": 2.9,
    "Turkey": 3.1,
    "Vietnam": 3.3,
    "Thailand": 3.5,
    "Malaysia": 3.6,
    "Philippines": 3.3,
    "Egypt": 2.6,
    "Russia": 2.6,
    "Nigeria": 2.6,
    "Bangladesh": 2.6,
    "Cameroon": 2.5,
    "Pakistan": 2.4,
    "Iran": 2.4,
    "Colombia": 2.9,
    "Argentina": 2.8,
    "Kenya": 2.7,
}

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df = pd.read_csv(filepath)
        self.df = self._clean_data(self.raw_df)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows with missing lat, lon, or population
        clean_df = df.dropna(subset=["population", "latitude", "longitude"]).copy()
        
        # Clean population column (handles string with commas or already numeric)
        if clean_df["population"].dtype == object:
            clean_df["population"] = pd.to_numeric(
                clean_df["population"].astype(str).str.replace(",", ""), errors="coerce"
            )
        
        clean_df = clean_df.dropna(subset=["population"])
        clean_df["population"] = clean_df["population"].astype(float)
        
        # Geodetic coordinates to radians
        clean_df["lat_rad"] = np.radians(clean_df["latitude"])
        clean_df["lon_rad"] = np.radians(clean_df["longitude"])
        
        # 3D Cartesian coordinates for standard Euclidean algorithms (like KMeans)
        clean_df["X"] = np.cos(clean_df["lat_rad"]) * np.cos(clean_df["lon_rad"])
        clean_df["Y"] = np.cos(clean_df["lat_rad"]) * np.sin(clean_df["lon_rad"])
        clean_df["Z"] = np.sin(clean_df["lat_rad"])
        
        # Add GDP/Capita and LPI columns with fallbacks
        clean_df["gdp_per_capita"] = clean_df["country"].map(GDP_PER_CAPITA).fillna(6000.0)
        clean_df["lpi"] = clean_df["country"].map(LPI_SCORES).fillna(2.8)
        
        # Add GDP adjusted population column (Purchasing Power adjusted demand)
        clean_df["gdp_adjusted"] = clean_df["population"] * clean_df["gdp_per_capita"]
        
        return clean_df

    def get_top_cities(self, n: int = 1000) -> pd.DataFrame:
        """Returns the top N cities by population, sorted in descending order."""
        return self.df.nlargest(n, "population").copy()
