"""
Export the city data and the road-and-sea distance model for the in-browser demo.

Writes demo/distribution-data.json:
  fields    column names for each city row
  cities    name, country, lat, lon, population, GDP per capita, LPI, price level,
            landmass id, port id, road km to that port, road circuity x 100
  ports     number of port cells
  sea       upper triangle (row-major, i < j) of the port-to-port sea distance matrix, km,
            as base64 little-endian uint16

The browser rebuilds the full freight distance from these pieces exactly as
gdoc_opt.transport.TransportModel.matrix() does.

Run: python scripts/export_demo.py
"""
import base64
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from gdoc_opt.data import DataLoader  # noqa: E402
from gdoc_opt.distance import transport_model  # noqa: E402


def main():
    df = DataLoader(os.path.join(ROOT, "final_dataset.csv")).get_top_cities(1000).reset_index(drop=True)
    tm = transport_model(df)
    P = len(tm.port_lat)
    iu = np.triu_indices(P, 1)
    sea = np.round(tm.sea[iu]).astype("<u2")
    assert tm.sea[iu].max() < 65535
    cities = []
    for i, r in df.iterrows():
        cities.append([r["city"], r["country"], round(float(r["latitude"]), 3), round(float(r["longitude"]), 3),
                       int(r["population"]), round(float(r["gdp_per_capita"])), round(float(r["lpi"]), 2),
                       round(float(r["price_level"]), 3), int(tm.landmass[i]), int(tm.port[i]),
                       int(round(tm.road_to_port[i])), int(round(tm.circuity[i] * 100))])
    out = {
        "fields": ["city", "country", "lat", "lon", "population", "gdp_per_capita", "lpi", "price_level",
                   "landmass", "port", "road_to_port_km", "circuity_x100"],
        "cities": cities,
        "ports": P,
        "sea": base64.b64encode(sea.tobytes()).decode("ascii"),
    }
    os.makedirs(os.path.join(ROOT, "demo"), exist_ok=True)
    path = os.path.join(ROOT, "demo", "distribution-data.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"), ensure_ascii=False)
    # reference distances for checking the browser's reconstruction
    D = tm.matrix()
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(df), size=(200, 2))
    with open(os.path.join(ROOT, "demo", "distribution-check.json"), "w") as fh:
        json.dump([[int(a), int(b), float(D[a, b])] for a, b in pairs], fh)
    print(f"wrote {path}: {len(cities)} cities, {P} ports, {os.path.getsize(path) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
