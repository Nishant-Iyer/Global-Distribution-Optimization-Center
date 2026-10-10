"""Compare the solvers on the 1,000 largest cities.

Prints the average distance (km) from a person to the nearest distribution
center for each solver, demand weighting and number of centers.

    python benchmarks/compare_solvers.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gdoc_opt.data import DataLoader  # noqa: E402
from gdoc_opt.models import KMeansOptimizer, KMedoidsOptimizer, MILPOptimizer  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), '..', 'final_dataset.csv')


def main():
    df = DataLoader(DATA).get_top_cities(1000)
    solvers = {
        'K-Means': (KMeansOptimizer(), {}),
        'K-Medoids': (KMedoidsOptimizer(), {}),
        'MILP (top-50)': (MILPOptimizer(), {'candidate_limit': 50}),
    }
    print(f"{'weighting':<14}{'K':>3}  " + ''.join(f'{name:>16}' for name in solvers))
    for weight in ('population', 'gdp_adjusted'):
        total = df[weight].sum()
        for k in (3, 5, 8):
            cells = []
            for opt, extra in solvers.values():
                res = opt.fit(df.copy(), k, weight_column=weight, **extra)
                cells.append(f"{res['total_distance'] / total:>16,.0f}")
            print(f'{weight:<14}{k:>3}  ' + ''.join(cells))


if __name__ == '__main__':
    main()
