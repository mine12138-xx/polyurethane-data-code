from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN


ROOT = Path(__file__).resolve().parents[1]


def cluster(data):
    coordinates = data[["UMAP_1", "UMAP_2"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("UMAP coordinates must be finite.")
    if len(data) < 400:
        raise ValueError("The configured HDBSCAN setting requires at least 400 rows.")
    result = data.copy()
    result["Cluster"] = HDBSCAN(min_cluster_size=200, min_samples=400, metric="euclidean").fit_predict(coordinates)
    return result


def main():
    parser = argparse.ArgumentParser(description="Cluster the UMAP coordinates with HDBSCAN.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/UMAP.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/Clusters.csv")
    args = parser.parse_args()
    result = cluster(pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = result.groupby(["Cluster", "Data_label"], sort=True).size().unstack(fill_value=0)
    summary.to_csv(args.output.with_name(args.output.stem + "_composition.csv"), encoding="utf-8-sig")
    print(f"Clustered {len(result)} rows: {args.output}")


if __name__ == "__main__":
    main()
