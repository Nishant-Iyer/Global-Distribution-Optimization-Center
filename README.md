# Global Distribution Center Placement

Where should a company put its distribution centers to serve customers worldwide? This project treats it as the classic facility location problem used in supply chain network design: choose sites that minimize demand-weighted freight distance across the 1,000 largest cities (129 countries), optionally with capacity limits and site running costs. It compares fast heuristics against an exact mixed-integer solver and reports where each one falls short.

**[Live demo](https://www.konnectingnots.com/demos/distribution.html)** · **[Case study](https://www.konnectingnots.com/projects/distribution-centers.html)**

## Approach

**Solvers**
- **K-Means** on 3D Cartesian coordinates, projected back to the sphere: a fast baseline.
- **K-Medoids (PAM)** on a precomputed distance matrix, so every center is a real city. This is the p-median problem, the standard formulation for uncapacitated facility location.
- **Spherical gradient descent** in PyTorch, keeping centers on the unit sphere at each step.
- **MILP facility location** (uncapacitated and capacitated) with PuLP/CBC.
- **K-Medoids-seeded MILP:** K-Medoids proposes candidate sites, then HiGHS (via SciPy) proves the answer optimal on that list or improves it, with optional capacity limits and per-site running costs.

**Scenario inputs**
- **Demand weighting:** population, or population × GDP per capita as a proxy for purchasing power.
- **Logistics quality:** a penalty based on the World Bank Logistics Performance Index (LPI), so countries with weaker infrastructure are less attractive:
  $$D'_{ij} = D_{ij} \times \left(1 + \beta \times \left(5.0 - \text{LPI}_j\right)\right)$$
- **Fixed sites:** lock in existing hubs (for example Singapore or Rotterdam) and solve for the rest.
- **Road and sea distances:** road within a landmass, and road, sea and road again between landmasses.
- **Capacity and running costs:** cap the share of demand one center may serve, and charge each site a running cost scaled by its country's price level.

**Delivery:** a FastAPI service for optimization and benchmark requests, and a Streamlit dashboard with a 3D globe of the network.

## Results

Average distance from a person to their nearest distribution center (km), over the 1,000 largest cities in the dataset (129 countries). Lower is better.

| Distance | Weighting | Centers | K-Means | K-Medoids | Seeded MILP |
|---|---|---|---|---|---|
| Straight line | Population | 3 | 2,141 | 2,096 | 2,096 |
| Straight line | Population | 5 | 1,581 | 1,560 | 1,560 |
| Straight line | Population | 8 | 1,131 | 1,103 | 1,103 |
| Straight line | Purchasing power | 3 | 1,789 | 1,744 | 1,744 |
| Straight line | Purchasing power | 5 | 1,449 | 1,379 | 1,379 |
| Straight line | Purchasing power | 8 | 1,015 | 983 | **982** |
| Road and sea | Population | 3 | 2,942 | 2,747 | 2,747 |
| Road and sea | Population | 5 | 2,153 | 1,968 | 1,968 |
| Road and sea | Population | 8 | 1,475 | 1,413 | 1,413 |
| Road and sea | Purchasing power | 3 | 2,296 | 2,200 | 2,200 |
| Road and sea | Purchasing power | 5 | 1,877 | 1,770 | **1,739** |
| Road and sea | Purchasing power | 8 | 1,311 | 1,234 | 1,234 |

- **Road and sea distances are about 25–30% longer** than straight lines, and they move the sites: with 5 centers the network shifts from São Paulo, Alexandria, Gwalior, Macheng and Birmingham to Lu'an, Raipur, Konya, New Orleans and Libreville.
- **The seeded MILP** proves the K-Medoids answer optimal on its candidate list in 10 of 12 cases and improves it in 2 (by 1.8% for purchasing power, road and sea, 5 centers). Candidate lists hold 230–480 cities; each solve takes 25–90 s, against under a second for K-Medoids. By contrast, a MILP limited to the 50 most populous cities lost to K-Medoids at every size with population weighting (1,609 vs 1,560 km with 5 centers, straight line).

**Capacity** (5 centers, road and sea, population; the MILP may split a city between centers):

| Max share of demand per center | Average distance | Sites |
|---|---|---|
| No limit (largest center serves 51%) | 1,968 km | Lu'an, Konya, Raipur, New Orleans, Libreville |
| 30% | 2,091 km | Linyi, Miami, Izmir, Zhaoqing, Indore |
| 25% | 2,170 km | Linyi, Miami, Izmir, Vishakhapatnam, Yiyang |

With a capacity limit, the MILP is seeded by a capacity-aware swap search (2,195 km at 25%) and proves its answer optimal on an 85-city candidate list.

**Running costs** decide how many centers to open (road and sea, population). A center's running cost is set at US prices, as the extra average distance it would be worth to avoid one, and scaled by each country's price level:

| Cost of a US-priced center | Centers | Average distance |
|---|---|---|
| 100 km | 16 | 1,078 km |
| 200 km | 11 | 1,271 km |
| 400 km | 6 | 1,929 km |

Price level alone favours the cheapest economies, including some no company would build in; combine it with the logistics penalty for more realistic sites.

Reproduce with `python benchmarks/compare_solvers.py` (about 40 minutes; results also go to `benchmarks/results.json`).

### How distances work

`metric="transport"` (see `gdoc_opt/transport.py`) estimates freight distance for every pair of cities:

- **Same landmass:** straight line × a road circuity factor per country, from Ballou, Rahardja & Sakai (2002), *Selected country circuity factors for road travel distance estimation*. Unlisted countries use 1.46 in Europe and 1.34 elsewhere.
- **Otherwise:** road to the nearest coast + the shortest sea route + road again. Sea routes are shortest paths over a 0.5° ocean grid built from the Natural Earth outlines, with the Suez and Panama canals and narrow straits added and the frozen Arctic closed. Checked against quoted port distances: Rotterdam–Shanghai ≈ 19,500 km via Suez, New York–Rotterdam ≈ 6,250 km.
- **Landmasses** follow real road links: the Channel Tunnel and the Danish and Japanese bridges join land; the Darien Gap and the closed Korean border split it.

### Country data

`gdoc_opt/country_indicators.csv` holds GDP per capita, the Logistics Performance Index and the price level (GDP per capita at market prices ÷ at PPP) for every country, from the World Bank API. Of the 129 countries in the top 1,000 cities, 128 have World Bank GDP, and 126 have LPI and price levels; the rest use the median of their income group, flagged in the `*_source` columns. Refresh it with `python scripts/fetch_country_data.py` or the **Refresh country indicators** GitHub Action.

**[Try it in your browser](https://www.konnectingnots.com/demos/distribution.html):** pick the number of centers, road-and-sea or straight-line distance, weighting, logistics penalty, running cost and capacity, or lock in a city, and it re-solves live.

## Setup

### Local Setup
1. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Package (Editable Mode)**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Run Unit Tests**:
   ```bash
   python -m unittest tests/test_optimizers.py
   ```


## Running with Docker

### Docker Compose
Run both the API and Streamlit Dashboard concurrently:
```bash
docker-compose up --build
```
- [FastAPI service](http://localhost:8000) on port 8000, with [interactive API docs](http://localhost:8000/docs)
- [Streamlit dashboard](http://localhost:8501) on port 8501

### Kubernetes Manifests
Deploy to a cluster (deployment, service and replica settings):
```bash
kubectl apply -f k8s/deployment.yml
```


## API

### 1. Run Optimization: `POST /optimize`
**Payload**:
```json
{
  "algorithm": "milp",
  "n_clusters": 5,
  "metric": "geodesic",
  "weight_column": "gdp_adjusted",
  "lpi_factor": 0.5,
  "candidate_limit": 50
}
```

### 2. Run Comparative Models: `GET /compare`
Allows comparing runtime, total distance, and selected sites across all solvers in one call.


## Dashboard

### Running Locally
Run the dashboard locally using Streamlit:
```bash
streamlit run app.py
```
Use the sidebar to change algorithms, toggle demand weighting modes, scale infrastructure readiness parameters, and add scenario constraints.

### Hosting
The dashboard can be deployed on [Streamlit Community Cloud](https://share.streamlit.io/) with `app.py` as the main file.

## License

MIT. See [LICENSE](LICENSE).
