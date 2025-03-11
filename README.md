# Global Distribution Center Optimization

## Project Overview
This project optimizes the placement of distribution centers (DCs) worldwide to minimize the total population-weighted distance, using 2023 population data for the top 1000 cities. The analysis addresses two objectives:

1. **Single DC Optimization**: Identify the city closest to the global population.
2. **Multiple DC Optimization**: Determine the optimal number and locations of DCs using K-means clustering.

The results provide insights for strategically placing DCs to improve global distribution efficiency.

---

## Table of Contents
- [Data Source](#data-source)
- [Methodology](#methodology)
  - [Single DC Analysis](#single-dc-analysis)
  - [Multiple DC Analysis](#multiple-dc-analysis)
- [Results](#results)
  - [Single DC Result](#single-dc-result)
  - [Multiple DC Results (K=5)](#multiple-dc-results-k5)
- [Assumptions](#assumptions)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Installation and Usage](#installation-and-usage)
- [Future Work](#future-work)
- [License](#license)

---

## Data Source
- **Dataset**: [Simple Maps World Cities Database (2023)](https://simplemaps.com/data/world-cities)
  - Includes city names, countries, populations, latitudes, and longitudes for the top 1000 cities by population.
  - Meets project requirements for current (2023) and sufficient (1000 cities) data.

---

## Methodology

### Single DC Analysis
- **Objective**: Minimize the total population-weighted distance from a single DC to all cities.
- **Approach**:
  - Calculate great-circle distances using the **Haversine formula**.
  - For each city \(i\), compute:
    \[
    \text{Total Weighted Distance}_i = \sum_{j=1}^{N} (\text{Population}_j \times \text{Distance}_{i,j})
    \]
  - Select the city with the smallest total weighted distance.

### Multiple DC Analysis
- **Objective**: Identify the optimal number and locations of multiple DCs.
- **Approach**:
  - Convert coordinates to **3D Cartesian coordinates** for clustering.
  - Apply **K-means clustering** (K=1 to 10), weighted by population.
  - Assign the nearest city to each cluster centroid as the DC.
  - Use the **elbow method** to find the optimal K.

---

## Results

### Single DC Result
- **Optimal Location**: **Narayanganj, Bangladesh**
- **Reason**: Proximity to dense population centers in South Asia.

### Multiple DC Results (K=5)
- **Optimal Number of DCs**: 5
- **DC Locations**:
  - Yichun, China
  - Esfahan, Iran
  - Campo Grande, Brazil
  - New Orleans, United States
  - Yaounde, Cameroon
- **Total Weighted Distance**: 4,822,347,640,365.60 km

---

## Assumptions
- **Distance Metric**: Haversine formula for geodesic distances.
- **Earth Model**: Perfect sphere (radius 6371 km).
- **Population Data**: Static and concentrated at city coordinates.
- **DC Placement**: Located at city centers with no capacity limits.
- **Objective**: Minimize total weighted distance (no cost or logistics constraints).

---

## Sensitivity Analysis
- **Varying K**:
  - K=3: Higher distance, inadequate coverage.
  - K=5: Optimal balance.
  - K=7: Marginal improvement, added complexity.
- **Number of Cities**: Minor impact on DC locations.
- **Population Noise**: Negligible effect on primary DC selections.

---

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nishant-iyer/global-distribution-optimization-center.git
   cd global-dc-optimization
