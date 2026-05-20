import time
import pandas as pd
import numpy as np
import pulp
from gdoc_opt.models.base import BaseOptimizer
from gdoc_opt.distance import compute_distance_matrix
from typing import Dict, Any

class MILPOptimizer(BaseOptimizer):
    def fit(self, df: pd.DataFrame, n_clusters: int, **kwargs) -> Dict[str, Any]:
        """
        Fits a Mixed Integer Linear Programming (MILP) model for the Capacitated 
        or Uncapacitated Facility Location Problem (CFLP / UFLP) using PuLP.
        
        To prevent combinatorial explosion (1000x1000 = 1,000,000 binary/continuous variables),
        we select the top M cities as candidate facility locations, while all N cities 
        have customer demands.
        
        Args:
            df (pd.DataFrame): Dataframe of cities.
            n_clusters (int): The number of DCs to open (if fixed_k is True).
            **kwargs:
                metric (str): 'haversine' or 'geodesic'. Default is 'haversine'.
                weight_column (str): Column to use for demand weights. Default is 'population'.
                candidate_limit (int): Number of top cities to consider as potential DC locations. Default is 50.
                capacity_limit (float): Max demand a single DC can serve. Default is None (uncapacitated).
                fixed_k (bool): Enforce opening exactly n_clusters DCs. Default is True.
                setup_cost (float): Fixed cost to open any DC (used if fixed_k is False). Default is 1e8.
        """
        start_time = time.time()
        metric = kwargs.get("metric", "haversine")
        weight_col = kwargs.get("weight_column", "population")
        candidate_limit = min(kwargs.get("candidate_limit", 50), len(df))
        capacity_limit = kwargs.get("capacity_limit", None)
        fixed_k = kwargs.get("fixed_k", True)
        setup_cost = kwargs.get("setup_cost", 1e8)
        lpi_factor = kwargs.get("lpi_factor", 0.0)
        
        # Demands (Customers)
        n_customers = len(df)
        demands = df[weight_col].values.astype(float)
        
        # Facilities (Candidates) - Select top M cities as potential locations
        # Scenario planning: support forced DCs
        forced_dcs = kwargs.get("forced_dcs", [])
        forced_indices = []
        for city in forced_dcs:
            matched_rows = df[df["city"].str.lower() == city.lower()]
            if not matched_rows.empty:
                idx = matched_rows.index[0] # Pick the largest population city match
                if idx not in forced_indices:
                    forced_indices.append(idx)
        
        if len(forced_indices) > n_clusters:
            raise ValueError(f"Number of unique forced DCs ({len(forced_indices)}) cannot exceed n_clusters ({n_clusters}).")
        
        candidates_df = df.nlargest(candidate_limit, "population").copy()
        # Ensure forced DCs are in the candidates dataframe
        for f_idx in forced_indices:
            if f_idx not in candidates_df.index:
                candidates_df = pd.concat([candidates_df, df.loc[[f_idx]]])
                
        candidate_limit = len(candidates_df)
        candidate_indices = [df.index.get_loc(idx) for idx in candidates_df.index]
        candidate_lpis = candidates_df["lpi"].values.astype(float)
        
        # 1. Compute distance matrix between all N customers and M candidates
        # To reuse distance matrix computation, compute full matrix first, then slice
        D_full = compute_distance_matrix(df, metric=metric)
        # Sliced distance matrix: rows = customers, cols = candidates
        D = D_full[:, candidate_indices]
        
        # Sliced LPI penalized distance matrix for objective formulation
        D_penalized = D * (1.0 + lpi_factor * (5.0 - candidate_lpis[np.newaxis, :]))
        
        # 2. Formulate MILP in PuLP
        prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
        
        # Decision Variables:
        # y[j] = 1 if candidate j is open as a DC, 0 otherwise
        y = [pulp.LpVariable(f"y_{j}", cat=pulp.LpBinary) for j in range(candidate_limit)]
        
        # Add constraints for forced DCs
        for j, idx in enumerate(candidates_df.index):
            if idx in forced_indices:
                prob += y[j] == 1.0
                
        # x[i][j] = fraction of customer i's demand served by candidate j
        x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
              for j in range(candidate_limit)] for i in range(n_customers)]
        
        # Objective Function:
        # Minimize total weighted distance + (optionally) setup costs
        transport_cost = pulp.lpSum(demands[i] * D_penalized[i][j] * x[i][j] 
                                    for i in range(n_customers) 
                                    for j in range(candidate_limit))
        
        if fixed_k:
            prob += transport_cost
        else:
            fixed_costs = pulp.lpSum(setup_cost * y[j] for j in range(candidate_limit))
            prob += transport_cost + fixed_costs
            
        # Constraints:
        # 1. Each customer's demand must be fully satisfied
        for i in range(n_customers):
            prob += pulp.lpSum(x[i][j] for j in range(candidate_limit)) == 1.0
            
        # 2. Customers can only be served by open DCs
        for i in range(n_customers):
            for j in range(candidate_limit):
                prob += x[i][j] <= y[j]
                
        # 3. Capacity constraints (if set)
        if capacity_limit is not None:
            # Capacity_limit is in terms of the weighted demand (e.g. population)
            for j in range(candidate_limit):
                prob += pulp.lpSum(demands[i] * x[i][j] for i in range(n_customers)) <= capacity_limit * y[j]
                
        # 4. Enforce exactly K facilities if fixed_k is True
        if fixed_k:
            prob += pulp.lpSum(y[j] for j in range(candidate_limit)) == n_clusters
            
        # Solve using CBC (default bundled solver)
        # Suppress solver output for clean logs unless debugging
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        
        # 3. Process Results
        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError(f"MILP solver failed to find an optimal solution. Status: {pulp.LpStatus[prob.status]}")
            
        # Retrieve open facilities
        open_dc_indices = []
        dc_locations = []
        cluster_id = 0
        for j in range(candidate_limit):
            if pulp.value(y[j]) > 0.5:
                # Retrieve original row from candidates_df
                orig_city_idx = candidates_df.index[j]
                closest_city = df.loc[orig_city_idx]
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
                open_dc_indices.append(j)
                cluster_id += 1
                
        # Compute assignments: map each city to its highest fraction DC
        assignments = []
        total_dist_val = 0.0
        
        for i in range(n_customers):
            # Find the open DC index j with maximum fraction of service
            fractions = [pulp.value(x[i][j]) for j in open_dc_indices]
            best_dc_idx = int(np.argmax(fractions))
            assignments.append(best_dc_idx)
            
            # Distance computation based on final assignments
            chosen_dc = dc_locations[best_dc_idx]
            d = D_full[i, df.index.get_loc(candidates_df.index[open_dc_indices[best_dc_idx]])]
            total_dist_val += demands[i] * d
            
        fit_time = time.time() - start_time
        
        # Calculate capacity utilization
        dc_utilizations = []
        for idx_in_open, j in enumerate(open_dc_indices):
            served_demand = sum(demands[i] * pulp.value(x[i][j]) for i in range(n_customers))
            dc_utilizations.append({
                "city": dc_locations[idx_in_open]["city"],
                "demand_served": float(served_demand),
                "utilization_percent": (served_demand / capacity_limit * 100.0) if capacity_limit else 100.0
            })
            
        return {
            "dc_locations": dc_locations,
            "assignments": assignments,
            "total_distance": float(total_dist_val),
            "metrics": {
                "fit_time_seconds": fit_time,
                "solver_status": pulp.LpStatus[prob.status],
                "objective_value": pulp.value(prob.objective),
                "dc_utilizations": dc_utilizations,
                "algorithm": f"MILP CFLP (Capacity: {capacity_limit if capacity_limit else 'Infinite'})"
            }
        }
