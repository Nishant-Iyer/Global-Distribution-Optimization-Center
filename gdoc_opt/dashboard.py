import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

from gdoc_opt.data import DataLoader
from gdoc_opt.distance import haversine_vectorized
from gdoc_opt.models.kmeans import KMeansOptimizer
from gdoc_opt.models.kmedoids import KMedoidsOptimizer
from gdoc_opt.models.milp import MILPOptimizer
from gdoc_opt.models.pytorch_opt import PyTorchOptimizer

# Cached loader for country outlines in 3D Cartesian coordinates
@st.cache_data
def load_country_borders(radius=0.995):
    import json
    import os
    
    path = os.path.join(os.path.dirname(__file__), "countries.geojson")
    if not os.path.exists(path):
        return [], [], []
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    xs, ys, zs = [], [], []
    for feature in data['features']:
        geom = feature['geometry']
        g_type = geom['type']
        coords = geom['coordinates']
        
        polygons = []
        if g_type == 'Polygon':
            polygons = [coords]
        elif g_type == 'MultiPolygon':
            polygons = coords
            
        for poly in polygons:
            for ring in poly:
                lons = [pt[0] for pt in ring]
                lats = [pt[1] for pt in ring]
                
                lons_rad = np.radians(lons)
                lats_rad = np.radians(lats)
                
                x = radius * np.cos(lats_rad) * np.cos(lons_rad)
                y = radius * np.cos(lats_rad) * np.sin(lons_rad)
                z = radius * np.sin(lats_rad)
                
                xs.extend(x.tolist() + [None])
                ys.extend(y.tolist() + [None])
                zs.extend(z.tolist() + [None])
                
    return xs, ys, zs

# Page Config
st.set_page_config(
    page_title="Global Distribution Optimization Center",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern dark theme and glassmorphism cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Sora:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap');
    
    html, body, [class*="css"], .stApp {
        background: radial-gradient(circle at 50% 50%, #110926 0%, #050505 100%) !important;
        background-attachment: fixed !important;
        color: #ffffff;
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    
    .main {
        background-color: transparent !important;
        color: #ffffff;
    }
    
    /* Typography Hierarchy */
    h1 {
        font-size: 2.0rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.4rem !important;
        color: #ffffff !important;
    }
    
    h2 {
        font-size: 1.35rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        margin-top: 1.0rem !important;
        margin-bottom: 0.5rem !important;
        color: #ffffff !important;
    }
    
    h3 {
        font-size: 1.15rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.3rem !important;
        color: #ffffff !important;
    }
    
    p, li, label, span {
        font-family: 'DM Sans', sans-serif !important;
    }
    
    .stMarkdown p {
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
        color: #dddddd !important;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(8, 8, 8, 0.95) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.1) !important;
    }
    
    .metric-card {
        background: rgba(10, 10, 10, 0.8) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 12px;
        padding: 14px 10px;
        margin-bottom: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 90px;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(168, 85, 247, 0.5) !important;
        box-shadow: 0 12px 30px rgba(168, 85, 247, 0.25);
    }
    
    .metric-title {
        font-size: 0.82rem;
        color: #b0b0b0;
        margin-bottom: 4px;
        font-family: 'Sora', sans-serif;
        font-weight: 500;
        line-height: 1.2;
    }
    
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'Sora', sans-serif;
        line-height: 1.2;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .metric-value.savings {
        color: #00d4ff;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    .metric-value.runtime {
        color: #a855f7;
        text-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%) !important;
        color: #050505 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 12px 28px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4) !important;
        color: #050505 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #888888;
        font-size: 1.05rem;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
    }
    
    /* Streamlit slider customization */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #00d4ff !important;
        border: 2px solid #a855f7 !important;
        width: 18px !important;
        height: 18px !important;
    }
    .stSlider [data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #a855f7) !important;
    }
    
    /* Inputs, selectors, and dropdowns */
    div[data-baseweb="input"] {
        background-color: rgba(10, 10, 10, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #a855f7 !important;
    }
    div[data-baseweb="select"] {
        background-color: rgba(10, 10, 10, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loader
@st.cache_resource
def get_data_loader(filepath: str) -> DataLoader:
    return DataLoader(filepath)

# Path to final_dataset.csv in workspace
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final_dataset.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "final_dataset.csv"

try:
    loader = get_data_loader(DATA_PATH)
    df_clean = loader.df
except Exception as e:
    st.error(f"Error loading final_dataset.csv: {e}")
    st.stop()

# Title banner
st.markdown('<h1 class="gradient-text" style="margin-bottom: 0.2rem;">🌐 Global Distribution Optimization Center</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.02rem; color: #aaaaaa; font-family: \'Sora\', sans-serif; margin-bottom: 1.5rem;">Production-grade multi-scenario facility location & logistics network planner</p>', unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px; padding-top: 10px;">
    <img src="https://img.icons8.com/nolan/256/globe.png" width="90" style="filter: drop-shadow(0 4px 10px rgba(0, 212, 255, 0.3)); margin-bottom: 10px;">
    <h3 style="font-family: 'Sora', sans-serif; margin-top: 10px; color: #ffffff; font-size: 1.30rem;">GDOC Controller</h3>
    <p style="font-size: 0.82rem; color: #888888; font-family: 'DM Sans', sans-serif; line-height: 1.4; padding: 0 10px;">Configure model hyperparameters and constraints dynamically.</p>
</div>
""", unsafe_allow_html=True)

algorithm_name = st.sidebar.selectbox(
    "Optimization Model",
    ["K-Medoids (Exact Geodesic)", "K-Means (Euclidean 3D)", "PyTorch Spherical GD", "MILP Facility Location"]
)

n_clusters = st.sidebar.slider("Number of DCs (K)", 1, 15, 5)

metric = st.sidebar.selectbox(
    "Distance Metric",
    ["haversine", "geodesic"]
)

weight_type = st.sidebar.selectbox(
    "Demand Weighting Type",
    ["population", "gdp_adjusted"],
    format_func=lambda x: "Population" if x == "population" else "Purchasing Power (Pop x GDP/Cap)"
)

lpi_factor = st.sidebar.slider(
    "Infrastructure Penalty (LPI Factor)",
    0.0, 1.5, 0.0, 0.1,
    help="Higher values penalize placing DCs in countries with low Logistics Performance Index (LPI)."
)

# Extra params based on selection
extra_kwargs = {}
if algorithm_name == "MILP Facility Location":
    st.sidebar.subheader("MILP Parameters")
    cap_limit_input = st.sidebar.number_input(
        "Capacity Limit per DC (None = Infinite)",
        value=0.0, step=1e7, format="%e"
    )
    extra_kwargs["capacity_limit"] = cap_limit_input if cap_limit_input > 0 else None
    
    extra_kwargs["candidate_limit"] = st.sidebar.number_input(
        "Candidate Sites Search Size",
        min_value=10, max_value=200, value=50
    )
    
    # Forced DCs Scenario
    st.sidebar.subheader("Scenario Planning")
    forced_dc_input = st.sidebar.text_input(
        "Force Open DCs (Comma-separated city names)",
        ""
    )
    if forced_dc_input:
        # Pre-process forced list
        st.sidebar.info("Scenario active: forcing DC placement in inputs.")
        extra_kwargs["forced_dcs"] = [c.strip() for c in forced_dc_input.split(",") if c.strip()]
        
elif algorithm_name == "PyTorch Spherical GD":
    st.sidebar.subheader("PyTorch Parameters")
    extra_kwargs["epochs"] = st.sidebar.slider("Optimization Epochs", 50, 500, 300, 50)
    extra_kwargs["learning_rate"] = st.sidebar.slider("Learning Rate", 0.01, 0.2, 0.05, 0.01)

# Fit current model
@st.cache_data
def fit_model(algo: str, k: int, metric_choice: str, weight_col: str, lpi: float, **kwargs) -> dict:
    df_fit = loader.get_top_cities(n=1000)
    
    if algo == "K-Means (Euclidean 3D)":
        opt = KMeansOptimizer()
    elif algo == "K-Medoids (Exact Geodesic)":
        opt = KMedoidsOptimizer()
    elif algo == "PyTorch Spherical GD":
        opt = PyTorchOptimizer()
    else:
        opt = MILPOptimizer()
        
    res = opt.fit(
        df_fit,
        k,
        metric=metric_choice,
        weight_column=weight_col,
        lpi_factor=lpi,
        **kwargs
    )
    return res

# Run button or auto-run
with st.spinner("Running optimization algorithms..."):
    t_start = time.time()
    results = fit_model(
        algorithm_name,
        n_clusters,
        metric,
        weight_type,
        lpi_factor,
        **extra_kwargs
    )
    t_elapsed = time.time() - t_start

# Retrieve outputs
dc_locs = results["dc_locations"]
assignments = results["assignments"]
total_dist = results["total_distance"]
fit_time = results["metrics"]["fit_time_seconds"]

# Map assignments back to dataframe
df_1000 = loader.get_top_cities(n=1000)
df_1000["assigned_dc"] = assignments

# Helper to format massive numbers cleanly
def format_metric_value(val: float, is_weighted: bool = False, w_type: str = "population") -> str:
    if not is_weighted:
        return f"{val:,.1f} km"
    
    suffix = "km-pop" if w_type == "population" else "km-GDP"
    if val >= 1e18:
        return f"{val/1e18:.2f} E {suffix}"
    elif val >= 1e15:
        return f"{val/1e15:.2f} Q {suffix}"
    elif val >= 1e12:
        return f"{val/1e12:.2f} T {suffix}"
    elif val >= 1e9:
        return f"{val/1e9:.2f} B {suffix}"
    elif val >= 1e6:
        return f"{val/1e6:.2f} M {suffix}"
    else:
        return f"{val:,.1f} {suffix}"

# Metrics Header
st.markdown("### 📊 Network Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    formatted_total_dist = format_metric_value(total_dist, is_weighted=True, w_type=weight_type)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Weighted Distance</div>
        <div class="metric-value">{formatted_total_dist}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    # Compute average distance
    avg_d = total_dist / df_1000[weight_type].sum()
    formatted_avg_d = format_metric_value(avg_d, is_weighted=False)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Service Distance</div>
        <div class="metric-value savings">{formatted_avg_d}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    avg_lpi = np.mean([dc["lpi"] for dc in dc_locs])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg DC Infrastructure (LPI)</div>
        <div class="metric-value">{avg_lpi:.2f} / 5.0</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Computation Runtime</div>
        <div class="metric-value runtime">{fit_time:.4f} s</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab_map, tab_comparison, tab_dc_details = st.tabs(["🌐 Dynamic 3D Globe", "📈 Sensitivity & Comparisons", "🏢 DC Locations Details"])

# Spherical Arc Interpolation Helper
def get_spherical_arc(x1, y1, z1, x2, y2, z2, steps=15):
    """Interpolates points along the sphere chord, then normalizes back to sphere surface."""
    t = np.linspace(0, 1, steps)
    arc_x = (1 - t) * x1 + t * x2
    arc_y = (1 - t) * y1 + t * y2
    arc_z = (1 - t) * z1 + t * z2
    norms = np.sqrt(arc_x**2 + arc_y**2 + arc_z**2)
    return arc_x / norms, arc_y / norms, arc_z / norms

with tab_map:
    st.markdown("### Interactive 3D Geospatial Network Map")
    st.markdown("This model plots cities (colored by cluster allocation) and links them back to their optimal distribution centers using geodesic arcs.")
    
    fig = go.Figure()
    
    # 1. Draw a wireframe sphere representing the Earth
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    xs = 0.99 * np.outer(np.cos(u), np.sin(v))
    ys = 0.99 * np.outer(np.sin(u), np.sin(v))
    zs = 0.99 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0, "#121721"], [1, "#121721"]],
        showscale=False,
        opacity=0.3,
        hoverinfo="skip"
    ))
    
    # 1.5. Draw country outlines
    bx, by, bz = load_country_borders(radius=0.995)
    if bx:
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode="lines",
            name="Country Outlines",
            line=dict(color="rgba(255, 255, 255, 0.2)", width=1.0),
            hoverinfo="skip",
            showlegend=False
        ))
    
    # 2. Add customer cities, sized by population and colored by cluster assignment
    colors = [
        "#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3", 
        "#33FFF6", "#AE33FF", "#FFAE33", "#33FFAE", "#AEFFAE",
        "#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1"
    ]
    
    for cluster_id in range(n_clusters):
        c_df = df_1000[df_1000["assigned_dc"] == cluster_id]
        if len(c_df) == 0:
            continue
            
        color = colors[cluster_id % len(colors)]
        
        # Add city points
        fig.add_trace(go.Scatter3d(
            x=c_df["X"], y=c_df["Y"], z=c_df["Z"],
            mode="markers",
            name=f"Cluster {cluster_id + 1}",
            marker=dict(
                size=c_df["population"] / 2.5e6 + 2.0,
                color=color,
                opacity=0.7
            ),
            text=c_df["city"] + ", " + c_df["country"] + "<br>Population: " + c_df["population"].map('{:,.0f}'.format),
            hoverinfo="text"
        ))
        
        # Get DC coordinates
        dc = dc_locs[cluster_id]
        # Retrieve DC 3D coordinates from the index
        dc_city_df = df_1000[(df_1000["city"] == dc["city"]) & (df_1000["country"] == dc["country"])].iloc[0]
        dc_x, dc_y, dc_z = dc_city_df["X"], dc_city_df["Y"], dc_city_df["Z"]
        
        # Add connection lines from each city to its assigned DC
        line_x, line_y, line_z = [], [], []
        for row in c_df.itertuples():
            ax, ay, az = get_spherical_arc(row.X, row.Y, row.Z, dc_x, dc_y, dc_z)
            line_x.extend(ax.tolist() + [None])
            line_y.extend(ay.tolist() + [None])
            line_z.extend(az.tolist() + [None])
            
        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode="lines",
            line=dict(color=color, width=1.0),
            opacity=0.25,
            showlegend=False,
            hoverinfo="skip"
        ))
        
    # 3. Add open DCs as distinctive gold diamonds
    dc_x_vals = []
    dc_y_vals = []
    dc_z_vals = []
    dc_names = []
    for dc in dc_locs:
        dc_city_df = df_1000[(df_1000["city"] == dc["city"]) & (df_1000["country"] == dc["country"])].iloc[0]
        dc_x_vals.append(dc_city_df["X"])
        dc_y_vals.append(dc_city_df["Y"])
        dc_z_vals.append(dc_city_df["Z"])
        dc_names.append(f"<b>DC {dc['cluster_id'] + 1}: {dc['city']}, {dc['country']}</b><br>LPI: {dc['lpi']}<br>GDP/Capita: ${dc['gdp_per_capita']:,.0f}")
        
    fig.add_trace(go.Scatter3d(
        x=dc_x_vals, y=dc_y_vals, z=dc_z_vals,
        mode="markers",
        name="Distribution Centers",
        marker=dict(
            size=10,
            color="#FFDF00",
            symbol="diamond",
            line=dict(color="#ffffff", width=1.5)
        ),
        text=dc_names,
        hoverinfo="text"
    ))
    
    # Styling and View
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=""),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1)
        ),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05),
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0a0a",
        font={"color": "white", "family": "DM Sans, sans-serif"}
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab_comparison:
    st.subheader("📈 Algorithm Sensitivity Analysis")
    
    # 1. Show dynamic Elbow curve for K=1 to 8
    st.markdown("#### Geodesic Elbow Method (K-Medoids)")
    if st.checkbox("Generate Elbow Curve (Takes ~5 seconds)", value=False):
        elbow_k = range(1, 9)
        elbow_costs = []
        progress_bar = st.progress(0.0)
        
        for idx, k_val in enumerate(elbow_k):
            res_elbow = fit_model("K-Medoids (Exact Geodesic)", k_val, metric, weight_type, lpi_factor)
            elbow_costs.append(res_elbow["total_distance"])
            progress_bar.progress((idx + 1) / len(elbow_k))
            
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(elbow_k), y=elbow_costs,
            mode="lines+markers",
            line=dict(color="#00d2ff", width=3),
            marker=dict(size=8, color="#ffffff")
        ))
        fig_elbow.update_layout(
            template="plotly_dark",
            title="Elbow Curve: Weighted Geodesic Distance vs. K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Total Weighted Geodesic Distance (km)",
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a0a0a",
            font={"color": "white", "family": "DM Sans, sans-serif"}
        )
        fig_elbow.update_xaxes(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont={"size": 10},
            title_font={"size": 11, "family": "Sora, sans-serif"}
        )
        fig_elbow.update_yaxes(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont={"size": 10},
            title_font={"size": 11, "family": "Sora, sans-serif"}
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        
    # 2. Performance Comparison Table
    st.markdown("#### Algorithm Performance Benchmark")
    if st.button("Run Benchmark Across Models"):
        with st.spinner("Benchmarking models..."):
            benchmark_list = []
            algos_to_test = {
                "K-Means (Euclidean 3D)": KMeansOptimizer(),
                "K-Medoids (Geodesic)": KMedoidsOptimizer(),
                "PyTorch Spherical GD": PyTorchOptimizer()
            }
            
            for name, optimizer in algos_to_test.items():
                t0 = time.time()
                res = optimizer.fit(
                    df_1000,
                    n_clusters,
                    weight_column=weight_type,
                    metric=metric,
                    lpi_factor=lpi_factor
                )
                t_run = time.time() - t0
                formatted_dist = format_metric_value(res['total_distance'], is_weighted=True, w_type=weight_type)
                benchmark_list.append({
                    "Algorithm": name,
                    "Total Weighted Distance": formatted_dist,
                    "Avg DC LPI Score": f"{np.mean([dc['lpi'] for dc in res['dc_locations']]):.2f}",
                    "Fit Time (seconds)": f"{t_run:.4f}",
                    "Optimal DCs": ", ".join([loc["city"] for loc in res["dc_locations"]])
                })
                
            st.table(pd.DataFrame(benchmark_list))

with tab_dc_details:
    st.subheader("🏢 Optimized Distribution Center Profiles")
    st.markdown("Detailed profiles of selected optimal distribution locations.")
    
    dc_data = []
    for dc in dc_locs:
        # Calculate cluster specific statistics
        c_cities = df_1000[df_1000["assigned_dc"] == dc["cluster_id"]]
        total_cluster_pop = c_cities["population"].sum()
        
        # Calculate average distance in this cluster
        distances = haversine_vectorized(
            c_cities["longitude"].values,
            c_cities["latitude"].values,
            dc["longitude"],
            dc["latitude"]
        )
        avg_cluster_dist = np.sum(distances * c_cities[weight_type].values) / c_cities[weight_type].sum()
        
        dc_data.append({
            "Cluster ID": dc["cluster_id"] + 1,
            "DC City": dc["city"],
            "Country": dc["country"],
            "Latitude": f"{dc['latitude']:.4f}",
            "Longitude": f"{dc['longitude']:.4f}",
            "Country LPI Index": f"{dc['lpi']:.2f}",
            "Country GDP per Capita": f"${dc['gdp_per_capita']:,.0f}",
            "Served Population (Total)": f"{total_cluster_pop:,.0f}",
            "Avg Service Distance (km)": f"{avg_cluster_dist:,.2f}"
        })
        
    st.dataframe(pd.DataFrame(dc_data), use_container_width=True)
