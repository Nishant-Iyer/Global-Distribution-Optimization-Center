import os

import numpy as np
import pandas as pd

INDICATORS_PATH = os.path.join(os.path.dirname(__file__), "country_indicators.csv")


def load_country_indicators(path: str = INDICATORS_PATH) -> pd.DataFrame:
    """World Bank GDP per capita, logistics (LPI) score and price level per country.

    Built by scripts/fetch_country_data.py; see the *_source columns for imputed values.
    """
    return pd.read_csv(path).set_index("country")


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df = pd.read_csv(filepath)
        self.df = self._clean_data(self.raw_df)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows with missing lat, lon, or population
        clean_df = df.dropna(subset=["population", "latitude", "longitude"]).copy()
        
        # Unconditionally clean and convert population to numeric (handles spaces, commas, category/string dtypes)
        pop_str = clean_df["population"].astype(str).str.strip().str.replace(",", "")
        clean_df["population"] = pd.to_numeric(pop_str, errors="coerce")
        
        clean_df = clean_df.dropna(subset=["population"])
        clean_df["population"] = clean_df["population"].astype(float)
        
        # Geodetic coordinates to radians
        clean_df["lat_rad"] = np.radians(clean_df["latitude"])
        clean_df["lon_rad"] = np.radians(clean_df["longitude"])
        
        # 3D Cartesian coordinates for standard Euclidean algorithms (like KMeans)
        clean_df["X"] = np.cos(clean_df["lat_rad"]) * np.cos(clean_df["lon_rad"])
        clean_df["Y"] = np.cos(clean_df["lat_rad"]) * np.sin(clean_df["lon_rad"])
        clean_df["Z"] = np.sin(clean_df["lat_rad"])
        
        # Country indicators. The few small territories without an entry get the median country.
        ind = load_country_indicators()
        for col in ("gdp_per_capita", "lpi", "price_level"):
            clean_df[col] = clean_df["country"].map(ind[col]).fillna(ind[col].median())

        # Add GDP adjusted population column (Purchasing Power adjusted demand)
        clean_df["gdp_adjusted"] = clean_df["population"] * clean_df["gdp_per_capita"]
        
        return clean_df

    def get_top_cities(self, n: int = 1000) -> pd.DataFrame:
        """Returns the top N cities by population, sorted in descending order."""
        return self.df.nlargest(n, "population").copy()
