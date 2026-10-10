# Global Distribution Optimization Center (GDOC)

🌐 **Enterprise-Grade Global Supply Chain Facility Location & Spatial Optimization Platform.**

This system transitions a simple clustering notebook into an absolute overkill supply chain planning system. It equips analysts and planners with mathematical solvers, REST API endpoints, and a rich, interactive 3D Globe dashboard to perform multi-scenario distribution network optimization.

---

## 🚀 Key Features

### 1. Advanced Math & Optimization Solvers
- **K-Means (Euclidean 3D)**: Fast baseline clustering operating on Cartesian coordinates projected back to the WGS-84 sphere.
- **K-Medoids (Exact Geodesic)**: Vectorized PAM (Partitioning Around Medoids) algorithm operating on pre-computed geodesic distance matrices.
- **PyTorch Spherical Gradient Descent**: Adam optimizer performing gradient descent on the unit sphere $S^2$, projecting centers via $C \leftarrow \frac{C}{\|C\|_2}$ at each epoch.
- **MILP Facility Location**: Solves the Uncapacitated and Capacitated Facility Location Problem (UFLP/CFLP) using the `CBC` solver via `PuLP`. Includes candidate site selection bounds to scale safely.
- **K-Medoids-seeded MILP** (`SeededMILPOptimizer`): K-Medoids proposes the candidate sites, then HiGHS (via SciPy) confirms or improves the answer, with optional capacity limits and per-site running costs.

### 2. Multi-Scenario & Variable Constraints
- **Dynamic Demand Weighting**: Choose between **Raw Population** demand and **Purchasing Power Adjusted** ($Population \times GDP_{per capita}$) demand weighting.
- **Infrastructure Readiness (LPI Index)**: Penalize placing distribution centers in countries with low logistics capabilities using a customizable weight multiplier $\beta$ on the distance function:
  $$D'_{ij} = D_{ij} \times \left(1 + \beta \times \left(5.0 - \text{LPI}_j\right)\right)$$
- **Scenario Planning (Forced DCs)**: Force placement in key supply chain hubs (e.g., Singapore, Rotterdam) and solve for the remaining slots.
- **Road and Sea Distances**: Freight distance by road within a landmass and by road–sea–road between them.
- **Capacity and Running Costs**: Cap the share of demand one center may serve, and charge each site a running cost scaled by its country's price level.

### 3. Application Stack
- **FastAPI Backend Service**: Exposes high-throughput JSON optimization and benchmark endpoints.
- **Streamlit Interactive 3D Globe**: Render networks in the browser, showing geodesic routing arcs, customer demand sizes, and DC profiles.

---

## 📊 Results

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

**[Try it in your browser](https://www.konnectingnots.com/demos/distribution.html)**: pick the number of centers, road-and-sea or straight-line distance, weighting, logistics penalty, running cost and capacity, or lock in a city, and it re-solves live.

## 🛠️ Installation & Setup

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

---

## 🐳 Running with Containerization

### Docker Compose
Run both the API and Streamlit Dashboard concurrently:
```bash
docker-compose up --build
```
- **FastAPI API**: http://localhost:8000
- **FastAPI Swagger Docs**: http://localhost:8000/docs
- **Streamlit Dashboard**: http://localhost:8501

### Kubernetes Manifests
Deploy the platform onto a cluster (with services and replica controls):
```bash
kubectl apply -f k8s/deployment.yml
```

---

## 📈 API Endpoints

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

---

## 📊 Dashboard Usage

### Running Locally
Run the dashboard locally using Streamlit:
```bash
streamlit run app.py
```
Use the sidebar to change algorithms, toggle demand weighting modes, scale infrastructure readiness parameters, and add scenario constraints.

### Deploying to Streamlit Community Cloud (GitHub Hosting)
You can deploy this dashboard directly from GitHub for free using Streamlit Community Cloud:
1. **Push this repository** to your GitHub account.
2. Visit [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**, then select your repository, branch, and set the Main file path to `app.py`.
4. Click **Deploy!** Your interactive dashboard will be live and shared publicly.

