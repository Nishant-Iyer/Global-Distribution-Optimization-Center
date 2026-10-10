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

### 2. Multi-Scenario & Variable Constraints
- **Dynamic Demand Weighting**: Choose between **Raw Population** demand and **Purchasing Power Adjusted** ($Population \times GDP_{per capita}$) demand weighting.
- **Infrastructure Readiness (LPI Index)**: Penalize placing distribution centers in countries with low logistics capabilities using a customizable weight multiplier $\beta$ on the distance function:
  $$D'_{ij} = D_{ij} \times \left(1 + \beta \times \left(5.0 - \text{LPI}_j\right)\right)$$
- **Scenario Planning (Forced DCs)**: Force placement in key supply chain hubs (e.g., Singapore, Rotterdam) and solve for the remaining slots.

### 3. Application Stack
- **FastAPI Backend Service**: Exposes high-throughput JSON optimization and benchmark endpoints.
- **Streamlit Interactive 3D Globe**: Render networks in the browser, showing geodesic routing arcs, customer demand sizes, and DC profiles.

---

## 📊 Results

Average distance from a person to their nearest distribution center (km), over the 1,000 largest cities in the dataset (129 countries, about 56% of the dataset's population). Lower is better; best per row in bold.

| Demand weighting | Centers | K-Means | K-Medoids | MILP (top-50 candidates) |
|---|---|---|---|---|
| Population | 3 | 2,141 | **2,096** | 2,145 |
| Population | 5 | 1,581 | **1,560** | 1,609 |
| Population | 8 | 1,131 | **1,103** | 1,152 |
| Purchasing power | 3 | 1,863 | **1,807** | 1,919 |
| Purchasing power | 5 | 1,499 | **1,429** | 1,470 |
| Purchasing power | 8 | 1,072 | **1,019** | 1,069 |

**Why the "exact" MILP doesn't win:** it is exact only over its candidate list, and by default only the 50 most populous cities may host a center. The best sites are often smaller, well-placed cities. Widening the candidate list closes the gap (5 centers, population weighting: 1,609 km with 50 candidates, 1,569 with 100, 1,565 with 400 in 74 s), while K-Medoids considers all 1,000 cities and reaches 1,560 km in under a second.

**Fixed in this version:**
- K-Means projected cluster centers to latitude/longitude without normalizing them back onto the sphere, which pulled every center toward the equator (5 centers: 1,634 → 1,581 km after the fix).
- K-Medoids stopped after 50 swaps rather than at a local optimum, and used an unseeded random start, so results changed run to run. It now applies the best swap each pass until none helps, with a fixed seed.

**Known limitation:** GDP per capita and logistics (LPI) scores are filled in for 36 countries; every other country uses placeholder values (6,000 USD, LPI 2.8), which matters for the purchasing-power and logistics-penalty options.

Reproduce with `python benchmarks/compare_solvers.py`.

**[Try it in your browser](https://www.konnectingnots.com/demos/distribution.html)**: pick the number of centers, weighting and logistics penalty, or lock in a city, and it re-solves live.

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

