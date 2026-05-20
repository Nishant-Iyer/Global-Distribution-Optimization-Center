import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd

from gdoc_opt.data import DataLoader
from gdoc_opt.models.kmeans import KMeansOptimizer
from gdoc_opt.models.kmedoids import KMedoidsOptimizer
from gdoc_opt.models.milp import MILPOptimizer
from gdoc_opt.models.pytorch_opt import PyTorchOptimizer

app = FastAPI(
    title="Global Distribution Center Optimization API",
    description="Enterprise API serving spatial optimization engines for global supply chain distribution.",
    version="1.0.0"
)

# Initialize DataLoader with final_dataset.csv (local to workspace)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final_dataset.csv")
if not os.path.exists(DATA_PATH):
    # Fallback to local execution directory if module path is unresolved
    DATA_PATH = "final_dataset.csv"

try:
    data_loader = DataLoader(DATA_PATH)
except Exception as e:
    data_loader = None
    print(f"Error loading final_dataset.csv: {e}")

class OptimizeRequest(BaseModel):
    algorithm: str = Field(..., description="Optimization algorithm: 'kmeans', 'kmedoids', 'milp', or 'pytorch'")
    n_clusters: int = Field(5, ge=1, le=20, description="Number of DCs to open")
    metric: str = Field("haversine", description="Distance metric: 'haversine' or 'geodesic'")
    weight_column: str = Field("population", description="Demand weighting column: 'population' or 'gdp_adjusted'")
    lpi_factor: float = Field(0.0, ge=0.0, le=2.0, description="Penalty multiplier for low infrastructure LPI score")
    
    # MILP specific params
    capacity_limit: Optional[float] = Field(None, description="Max demand capacity per DC (only for milp)")
    candidate_limit: Optional[int] = Field(50, ge=10, le=200, description="Candidate DC search space size (only for milp)")
    
    # PyTorch specific params
    epochs: Optional[int] = Field(300, ge=50, le=1000, description="Number of optimization epochs (only for pytorch)")
    learning_rate: Optional[float] = Field(0.05, ge=0.001, le=0.5, description="Adam optimizer learning rate (only for pytorch)")

class DCLocation(BaseModel):
    city: str
    country: str
    latitude: float
    longitude: float
    population: int
    lpi: float
    gdp_per_capita: float
    cluster_id: int

class OptimizeResponse(BaseModel):
    dc_locations: List[DCLocation]
    assignments: List[int]
    total_distance: float
    metrics: Dict[str, Any]

@app.get("/")
def read_root():
    return {
        "status": "online",
        "dataset_loaded": data_loader is not None,
        "total_cities_in_db": len(data_loader.df) if data_loader else 0
    }

@app.post("/optimize", response_model=OptimizeResponse)
def run_optimization(req: OptimizeRequest):
    if data_loader is None:
        raise HTTPException(status_code=500, detail="Database file final_dataset.csv not found or failed to load.")
        
    # Get top 1000 cities
    df = data_loader.get_top_cities(n=1000)
    
    # Select Optimizer
    algo = req.algorithm.lower()
    if algo == "kmeans":
        optimizer = KMeansOptimizer()
    elif algo == "kmedoids":
        optimizer = KMedoidsOptimizer()
    elif algo == "milp":
        optimizer = MILPOptimizer()
    elif algo == "pytorch":
        optimizer = PyTorchOptimizer()
    else:
        raise HTTPException(status_code=400, detail=f"Invalid algorithm: {req.algorithm}. Choose from 'kmeans', 'kmedoids', 'milp', 'pytorch'.")
        
    try:
        # Build kwargs dynamically based on request schema
        opt_kwargs = {
            "weight_column": req.weight_column,
            "metric": req.metric,
            "lpi_factor": req.lpi_factor
        }
        if algo == "milp":
            opt_kwargs["capacity_limit"] = req.capacity_limit
            opt_kwargs["candidate_limit"] = req.candidate_limit
        elif algo == "pytorch":
            opt_kwargs["epochs"] = req.epochs
            opt_kwargs["learning_rate"] = req.learning_rate
            
        res = optimizer.fit(df, req.n_clusters, **opt_kwargs)
        return OptimizeResponse(
            dc_locations=res["dc_locations"],
            assignments=res["assignments"],
            total_distance=res["total_distance"],
            metrics=res["metrics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization engine error: {str(e)}")

@app.get("/compare")
def compare_models(n_clusters: int = 5, lpi_factor: float = 0.0, weight_column: str = "population", metric: str = "haversine"):
    if data_loader is None:
        raise HTTPException(status_code=500, detail="Database file final_dataset.csv not found.")
        
    df = data_loader.get_top_cities(n=1000)
    
    algorithms = {
        "K-Means (Euclidean 3D)": KMeansOptimizer(),
        "K-Medoids (PAM)": KMedoidsOptimizer(),
        "PyTorch Spherical GD": PyTorchOptimizer()
    }
    
    comparison_results = []
    
    for name, optimizer in algorithms.items():
        try:
            res = optimizer.fit(
                df,
                n_clusters,
                weight_column=weight_column,
                metric=metric,
                lpi_factor=lpi_factor
            )
            comparison_results.append({
                "algorithm": name,
                "total_distance_km": res["total_distance"],
                "fit_time_seconds": res["metrics"]["fit_time_seconds"],
                "dc_cities": [loc["city"] for loc in res["dc_locations"]]
            })
        except Exception as e:
            comparison_results.append({
                "algorithm": name,
                "error": str(e)
            })
            
    return comparison_results
