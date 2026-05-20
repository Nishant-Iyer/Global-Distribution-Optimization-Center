import os
import unittest
import pandas as pd
import numpy as np

from gdoc_opt.data import DataLoader
from gdoc_opt.models.kmeans import KMeansOptimizer
from gdoc_opt.models.kmedoids import KMedoidsOptimizer
from gdoc_opt.models.milp import MILPOptimizer
from gdoc_opt.models.pytorch_opt import PyTorchOptimizer

class TestGDOCPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Path to final_dataset.csv
        cls.data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "final_dataset.csv"
        )
        if not os.path.exists(cls.data_path):
            # Create a mock dataset for testing if final_dataset.csv is missing or for isolated environment
            mock_data = {
                "city": ["Tokyo", "Delhi", "Shanghai", "Sao Paulo", "Mumbai", "Mexico City", "Beijing", "Osaka"],
                "country": ["Japan", "India", "China", "Brazil", "India", "Mexico", "China", "Japan"],
                "latitude": [35.6895, 28.6139, 31.2304, -23.5505, 19.0760, 19.4326, 39.9042, 34.6937],
                "longitude": [139.6917, 77.2090, 121.4737, -46.6333, 72.8777, -99.1332, 116.4074, 135.5023],
                "population": [37400088, 32941000, 29210808, 22620000, 21290000, 22200000, 21890000, 19000000]
            }
            cls.df = pd.DataFrame(mock_data)
            cls.df.to_csv("test_dataset.csv", index=False)
            cls.loader = DataLoader("test_dataset.csv")
        else:
            cls.loader = DataLoader(cls.data_path)
            
        cls.df_test = cls.loader.get_top_cities(n=8)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test_dataset.csv"):
            os.remove("test_dataset.csv")

    def test_data_loader(self):
        # Verify columns exist
        df = self.loader.df
        self.assertIn("X", df.columns)
        self.assertIn("Y", df.columns)
        self.assertIn("Z", df.columns)
        self.assertIn("gdp_per_capita", df.columns)
        self.assertIn("lpi", df.columns)
        self.assertIn("gdp_adjusted", df.columns)
        self.assertTrue((df["gdp_adjusted"] == df["population"] * df["gdp_per_capita"]).all())

    def test_kmeans_optimizer(self):
        opt = KMeansOptimizer()
        res = opt.fit(self.df_test, n_clusters=2, weight_column="population")
        self.assertEqual(len(res["dc_locations"]), 2)
        self.assertEqual(len(res["assignments"]), len(self.df_test))
        self.assertGreater(res["total_distance"], 0.0)

    def test_kmedoids_optimizer(self):
        opt = KMedoidsOptimizer()
        res = opt.fit(self.df_test, n_clusters=2, weight_column="gdp_adjusted", lpi_factor=0.5)
        self.assertEqual(len(res["dc_locations"]), 2)
        self.assertEqual(len(res["assignments"]), len(self.df_test))
        self.assertGreater(res["total_distance"], 0.0)

    def test_milp_optimizer(self):
        opt = MILPOptimizer()
        # Test basic facility location
        res = opt.fit(self.df_test, n_clusters=2, weight_column="population", candidate_limit=4)
        self.assertEqual(len(res["dc_locations"]), 2)
        self.assertEqual(len(res["assignments"]), len(self.df_test))
        
        # Test forced DCs
        res_forced = opt.fit(
            self.df_test,
            n_clusters=2,
            weight_column="population",
            candidate_limit=4,
            forced_dcs=["Tokyo"]
        )
        opened_cities = [loc["city"] for loc in res_forced["dc_locations"]]
        self.assertIn("Tokyo", opened_cities)

    def test_pytorch_optimizer(self):
        opt = PyTorchOptimizer()
        res = opt.fit(self.df_test, n_clusters=2, weight_column="population", epochs=50)
        self.assertEqual(len(res["dc_locations"]), 2)
        self.assertEqual(len(res["assignments"]), len(self.df_test))
        self.assertGreater(res["total_distance"], 0.0)

if __name__ == "__main__":
    unittest.main()
