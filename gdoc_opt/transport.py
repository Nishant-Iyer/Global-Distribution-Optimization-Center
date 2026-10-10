"""
Road and sea distances between cities.

Straight lines over the globe understate real freight distances: roads wind, and goods can't
drive across oceans. This module estimates a freight distance for every pair of cities:

  * Same landmass:  road distance = great-circle distance x a road circuity factor.
  * Otherwise:      road to the nearest coast + sea route between the two coasts + road again.

For a pair on the same landmass the shorter of the two options is used.

Road circuity factors come from Ballou, Rahardja & Sakai (2002), "Selected country circuity
factors for road travel distance estimation", Transportation Research Part A 36(9). Countries
they didn't measure use the median of their values (1.34). A pair of cities in two countries
uses the average of the two factors.

Sea routes are shortest paths over a 0.5-degree ocean grid built from the Natural Earth country
outlines in countries.geojson, with the ice-bound Arctic closed and explicit links for canals and narrow straits (Suez, Panama,
Gibraltar, the Turkish and Danish straits, and others) that the coarse grid would otherwise close.

Landmasses are groups of country outlines that touch, adjusted for real road links and gaps:
the Channel Tunnel and the Danish and Japanese bridges join land, while the Darien Gap
(Panama-Colombia) and the closed Korean border split it.
"""
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.path import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

R = 6371.0
HERE = os.path.dirname(__file__)
GRID = 0.5

# Ballou et al. (2002), road distance / straight-line distance
CIRCUITY: Dict[str, float] = {
    "Argentina": 1.22, "Australia": 1.28, "Belarus": 1.12, "Brazil": 1.23, "Canada": 1.30, "China": 1.33,
    "Egypt": 2.10, "United Kingdom": 1.40, "France": 1.65, "Germany": 1.32, "Italy": 1.18, "Spain": 1.58,
    "Hungary": 1.35, "India": 1.31, "Indonesia": 1.43, "Japan": 1.41, "Mexico": 1.46, "New Zealand": 2.05,
    "Poland": 1.21, "Russia": 1.37, "Saudi Arabia": 1.34, "South Africa": 1.23, "Thailand": 1.42,
    "Turkey": 1.36, "Ukraine": 1.29, "United States": 1.20,
}
EUROPE_DEFAULT = 1.46     # Ballou's "Europe" figure, for European countries not listed above
DEFAULT_CIRCUITY = 1.34   # median of the values above

# Sea links the 0.5-degree grid can't resolve: (name, (lat, lon) on one side, (lat, lon) on the other)
SEA_GATEWAYS: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = [
    ("Suez Canal", (31.4, 32.4), (29.8, 32.6)),
    ("Panama Canal", (9.5, -79.9), (8.8, -79.5)),
    ("Strait of Gibraltar", (36.0, -6.2), (36.1, -4.9)),
    ("Turkish Straits", (41.4, 29.2), (40.0, 26.0)),
    ("Danish Straits", (57.5, 11.0), (55.0, 13.0)),
    ("Strait of Dover", (51.2, 1.0), (50.7, 1.8)),
    ("Bab-el-Mandeb", (12.8, 43.0), (12.4, 43.6)),
    ("Strait of Hormuz", (26.4, 56.0), (25.9, 57.0)),
    ("Singapore Strait", (1.2, 103.5), (1.3, 104.3)),
    ("Strait of Malacca", (2.5, 101.0), (1.3, 103.5)),
    ("Bosphorus to Black Sea", (41.4, 29.2), (42.0, 30.0)),
    ("Kattegat to Baltic", (55.0, 13.0), (55.5, 15.0)),
    ("Taiwan Strait", (24.5, 119.5), (24.0, 119.0)),
    ("Korea Strait", (34.6, 129.0), (34.0, 129.8)),
    ("Strait of Messina", (38.3, 15.6), (38.0, 15.6)),
]

# Road links that join landmasses: (lat, lon) inside each
LAND_LINKS = [
    ("Channel Tunnel", (51.5, -0.1), (48.9, 2.35)),
    ("Oresund Bridge", (55.68, 12.57), (55.6, 13.0)),
    ("Great Belt Bridge", (55.68, 12.57), (56.2, 9.5)),
    ("Seikan Tunnel", (43.06, 141.35), (35.69, 139.69)),
    ("Kanmon Bridge", (33.59, 130.40), (35.69, 139.69)),
    ("Seto Bridges", (33.84, 132.77), (35.69, 139.69)),
    ("Strait of Messina ferries", (38.1, 13.36), (41.9, 12.5)),
]
# Borders with no road freight between them: country pairs whose outlines touch
LAND_CUTS = [("Panama", "Colombia"), ("Korea", "Dem. Rep. Korea")]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def circuity_for(country: str, continent: str = "") -> float:
    if country in CIRCUITY:
        return CIRCUITY[country]
    if continent == "Europe":
        return EUROPE_DEFAULT
    return DEFAULT_CIRCUITY


# ---------------------------------------------------------------------------- geometry
def load_polygons():
    """Returns a list of (country_name, continent, Path) for every polygon piece."""
    geo = json.load(open(os.path.join(HERE, "countries.geojson")))
    out = []
    for f in geo["features"]:
        p, g = f["properties"], f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0], dtype=float)
            out.append((p["name"], p.get("continent", ""), p.get("name_long", p["name"]), Path(ring[:, :2])))
    return out


def land_mask(polys):
    lats = np.arange(-90 + GRID / 2, 90, GRID)
    lons = np.arange(-180 + GRID / 2, 180, GRID)
    LON, LAT = np.meshgrid(lons, lats)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    land = np.zeros(len(pts), dtype=bool)
    for _, _, _, path in polys:
        x0, y0 = path.vertices.min(axis=0)
        x1, y1 = path.vertices.max(axis=0)
        box = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        if box.any():
            idx = np.where(box)[0]
            land[idx[path.contains_points(pts[idx])]] = True
    return lats, lons, land.reshape(len(lats), len(lons))


class OceanGraph:
    """Shortest sea routes on a 0.5-degree grid, 16-neighbour moves, with canal and strait links."""

    def __init__(self, polys=None):
        polys = polys or load_polygons()
        self.lats, self.lons, self.land = land_mask(polys)
        ny, nx = self.land.shape
        # Arctic sea ice: no year-round commercial route north of 70N, except the ice-free
        # Norwegian and Barents seas (30W-60E, below 78N).
        LAT = self.lats[:, None] + 0 * self.lons[None, :]
        LON = self.lons[None, :] + 0 * self.lats[:, None]
        ice = (LAT >= 70) & ~((LON >= -30) & (LON <= 60) & (LAT < 78))
        ice |= LAT <= -66   # Antarctic coast
        water = ~self.land & ~ice
        idx = -np.ones((ny, nx), dtype=np.int64)
        idx[water] = np.arange(water.sum())
        self.cell_lat = np.repeat(self.lats[:, None], nx, axis=1)[water]
        self.cell_lon = np.repeat(self.lons[None, :], ny, axis=0)[water]
        self.idx = idx
        rows, cols, wts = [], [], []
        moves = [(dy, dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)
                 if (dy, dx) != (0, 0) and np.gcd(abs(dy), abs(dx)) == 1]
        Y, X = np.nonzero(water)
        for dy, dx in moves:
            y2, x2 = Y + dy, (X + dx) % nx
            ok = (y2 >= 0) & (y2 < ny)
            y2c = np.clip(y2, 0, ny - 1)
            ok &= water[y2c, x2]
            if abs(dy) == 2:      # knight move: both cells it brushes past must be water
                yb = np.clip(Y + np.sign(dy), 0, ny - 1)
                ok &= water[yb, X] & water[yb, (X + dx) % nx]
            elif abs(dx) == 2:
                xb = (X + np.sign(dx)) % nx
                ok &= water[Y, xb] & water[y2c, xb]
            a = idx[Y[ok], X[ok]]
            b = idx[y2c[ok], x2[ok]]
            rows.append(a); cols.append(b)
            wts.append(haversine(self.cell_lat[a], self.cell_lon[a], self.cell_lat[b], self.cell_lon[b]))
        rows, cols, wts = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        n = len(self.cell_lat)
        g = coo_matrix((wts, (rows, cols)), shape=(n, n)).tocsr()
        # keep the connected ocean; drop lakes and inland seas such as the Caspian
        ncomp, lab = connected_components(g, directed=False)
        self.main = lab == np.bincount(lab).argmax()
        extra_r, extra_c, extra_w = [], [], []
        self.gateways = []
        for name, a, b in SEA_GATEWAYS:
            ia, ib = self.nearest_cell(*a, main_only=False), self.nearest_cell(*b, main_only=False)
            w = float(haversine(a[0], a[1], b[0], b[1])) + 1e-3
            extra_r += [ia, ib]; extra_c += [ib, ia]; extra_w += [w, w]
            self.gateways.append((name, ia, ib))
        g = g + coo_matrix((extra_w, (extra_r, extra_c)), shape=(n, n)).tocsr()
        ncomp, lab = connected_components(g, directed=False)
        self.main = lab == np.bincount(lab).argmax()
        self.graph = g

    def nearest_cell(self, lat, lon, main_only=True):
        d = haversine(lat, lon, self.cell_lat, self.cell_lon)
        if main_only:
            d = np.where(self.main, d, np.inf)
        return int(np.argmin(d))

    def sea_distances(self, cells):
        """Sea km between every pair of the given ocean cells."""
        cells = np.asarray(cells)
        D = dijkstra(self.graph, directed=False, indices=cells)
        return D[:, cells]


# ---------------------------------------------------------------------------- landmasses
def landmasses(polys):
    """Group polygon pieces that share border vertices; then apply LAND_LINKS / LAND_CUTS."""
    key = {}
    parent = list(range(len(polys)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cut = {frozenset(c) for c in LAND_CUTS}
    for i, (name, _, _, path) in enumerate(polys):
        for v in np.round(path.vertices, 3):
            k = (v[0], v[1])
            if k in key:
                j = key[k]
                if frozenset((polys[j][0], name)) not in cut:
                    parent[find(i)] = find(j)
            else:
                key[k] = i
    return find, parent


def city_polygon(polys, lat, lon):
    for i, (_, _, _, path) in enumerate(polys):
        if path.contains_point((lon, lat)):
            return i
    # small islands and city-states (Singapore, Hong Kong) are missing at this resolution: nearest outline
    best, bd = 0, np.inf
    for i, (_, _, _, path) in enumerate(polys):
        v = path.vertices
        d = np.min(haversine(lat, lon, v[:, 1], v[:, 0]))
        if d < bd:
            best, bd = i, d
    return best


class TransportModel:
    """Freight distances for a fixed list of cities, plus the compact pieces the browser demo needs."""

    def __init__(self, cities):
        """cities: DataFrame with city, country, latitude, longitude."""
        polys = load_polygons()
        find, parent = landmasses(polys)
        lat = cities["latitude"].values.astype(float)
        lon = cities["longitude"].values.astype(float)
        poly_of = np.array([city_polygon(polys, a, b) for a, b in zip(lat, lon)])
        # names of the outlines containing each LAND_LINKS point
        for _, a, b in LAND_LINKS:
            pa, pb = city_polygon(polys, *a), city_polygon(polys, *b)
            parent[find(pa)] = find(pb)
        roots = np.array([find(p) for p in poly_of])
        _, self.landmass = np.unique(roots, return_inverse=True)

        geo = json.load(open(os.path.join(HERE, "countries.geojson")))
        continent = {"Czechia": "Europe"}
        for f in geo["features"]:
            p = f["properties"]
            for k in ("name", "name_long", "admin", "formal_en"):
                if p.get(k):
                    continent.setdefault(p[k], p.get("continent", ""))
        self.circuity = np.array([circuity_for(c, continent.get(c, "")) for c in cities["country"]])

        ocean = OceanGraph(polys)
        cell = np.array([ocean.nearest_cell(a, b) for a, b in zip(lat, lon)])
        ports, self.port = np.unique(cell, return_inverse=True)
        self.port_lat, self.port_lon = ocean.cell_lat[ports], ocean.cell_lon[ports]
        self.road_to_port = haversine(lat, lon, self.port_lat[self.port], self.port_lon[self.port]) * self.circuity
        self.sea = ocean.sea_distances(ports)
        self.lat, self.lon = lat, lon
        self.ocean = ocean

    def matrix(self):
        """n x n freight distance matrix (km)."""
        gc = haversine(self.lat[:, None], self.lon[:, None], self.lat[None, :], self.lon[None, :])
        c = (self.circuity[:, None] + self.circuity[None, :]) / 2
        road = gc * c
        inter = self.road_to_port[:, None] + self.sea[self.port[:, None], self.port[None, :]] + self.road_to_port[None, :]
        same = self.landmass[:, None] == self.landmass[None, :]
        D = np.where(same, np.minimum(road, inter), inter)
        np.fill_diagonal(D, 0.0)
        return D
