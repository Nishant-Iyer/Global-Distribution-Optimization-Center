"""Benchmarks on the 1,000 largest cities.

    python benchmarks/compare_solvers.py            # everything (about 40 minutes)
    python benchmarks/compare_solvers.py --quick    # 5 centers only

Prints three tables and writes benchmarks/results.json:
  1. Average distance from a person to their nearest center, by solver, for straight-line and
     road-and-sea distances.
  2. The same with a capacity limit per center (seeded MILP).
  3. Letting running costs decide how many centers to open (seeded MILP).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gdoc_opt.data import DataLoader  # noqa: E402
from gdoc_opt.distance import compute_distance_matrix  # noqa: E402
from gdoc_opt.models.kmeans import KMeansOptimizer  # noqa: E402
from gdoc_opt.models.kmedoids import KMedoidsOptimizer  # noqa: E402
from gdoc_opt.models.seeded_milp import SeededMILPOptimizer  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "final_dataset.csv")


def site_indices(df, result):
    if "medoid_indices" in result:
        return list(result["medoid_indices"])
    # K-Means reports the city nearest each centroid
    keys = list(zip(df["city"], df["country"]))
    return [keys.index((d["city"], d["country"])) for d in result["dc_locations"]]


def avg_km(D, sites, w):
    return float(np.sum(np.min(D[:, sites], axis=1) * w) / w.sum())


def main(quick=False):
    df = DataLoader(DATA).get_top_cities(1000).reset_index(drop=True)
    t = time.time()
    mats = {"straight line": compute_distance_matrix(df, "haversine"),
            "road and sea": compute_distance_matrix(df, "transport")}
    print(f"distance matrices built in {time.time() - t:.0f} s\n")
    ks = (5,) if quick else (3, 5, 8)
    out = {"solvers": [], "capacity": [], "running_cost": []}

    print(f"{'distance':<14}{'weighting':<18}{'K':>3}{'K-Means':>10}{'K-Medoids':>11}{'Seeded MILP':>13}{'candidates':>12}{'secs':>7}")
    for dname, D in mats.items():
        for weight, wlabel in (("population", "population"), ("gdp_adjusted", "purchasing power")):
            w = df[weight].values.astype(float)
            for k in ks:
                km = KMeansOptimizer().fit(df.copy(), k, weight_column=weight)
                kmd = KMedoidsOptimizer().fit(df, k, distance_matrix=D, weight_column=weight)
                ms = SeededMILPOptimizer().fit(df, k, distance_matrix=D, weight_column=weight)
                row = {"distance": dname, "weighting": wlabel, "k": k,
                       "kmeans": avg_km(D, site_indices(df, km), w),
                       "kmedoids": avg_km(D, kmd["medoid_indices"], w),
                       "seeded_milp": avg_km(D, ms["medoid_indices"], w),
                       "candidates": ms["metrics"]["candidates"], "milp_seconds": ms["metrics"]["fit_time_seconds"],
                       "sites": [d["city"] for d in ms["dc_locations"]]}
                out["solvers"].append(row)
                print(f"{dname:<14}{wlabel:<18}{k:>3}{row['kmeans']:>10,.0f}{row['kmedoids']:>11,.0f}"
                      f"{row['seeded_milp']:>13,.0f}{row['candidates']:>12}{row['milp_seconds']:>7.0f}", flush=True)

    D = mats["road and sea"]
    w = df["population"].values.astype(float)
    print("\nCapacity limits (5 centers, road and sea, population)")
    for cap in (None, 0.30, 0.25):
        r = SeededMILPOptimizer().fit(df, 5, distance_matrix=D, capacity_share=cap)
        shares = sorted((d["share_of_demand"] for d in r["dc_locations"]), reverse=True)
        row = {"capacity_share": cap, "avg_km": r["total_distance"] / w.sum(), "largest_share": shares[0],
               "sites": [d["city"] for d in r["dc_locations"]], "seconds": r["metrics"]["fit_time_seconds"]}
        out["capacity"].append(row)
        print(f"  cap {'none' if cap is None else f'{cap:.0%}':>5}: {row['avg_km']:,.0f} km, largest center serves "
              f"{row['largest_share']:.0%}: {', '.join(row['sites'])}", flush=True)

    print("\nRunning costs decide the number of centers (road and sea, population)")
    for site_km in (100, 200, 400):
        r = SeededMILPOptimizer().fit(df, None, distance_matrix=D, site_cost_km=site_km)
        row = {"site_cost_km": site_km, "centers": len(r["dc_locations"]), "avg_km": r["total_distance"] / w.sum(),
               "sites": [f"{d['city']}, {d['country']}" for d in r["dc_locations"]], "seconds": r["metrics"]["fit_time_seconds"]}
        out["running_cost"].append(row)
        print(f"  {site_km} km per US-priced center: {row['centers']} centers, {row['avg_km']:,.0f} km: "
              f"{'; '.join(row['sites'])}", flush=True)

    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
