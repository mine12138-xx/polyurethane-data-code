from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "HS_NumHAcceptors_Density", "HS_NumHDonors_Density", "HS_NOCount_Density", "HS_CH3_groups_Density",
    "HS_NumRotatableBonds_Density", "HS_RingCount_Density", "HS_FractionCSP3", "HS_FpDensityMorgan3",
    "HS_Acc", "HS_MolWt", "PU_RingCount_Density", "PU_MPC", "PU_ROG",
]
K2 = 11
M = 11
EPS = 1e-9
LABEL_COL = "Data_label"
LABEL_HIGH = "Literature_double_high"
UMAP_COLUMNS = ["UMAP_1", "UMAP_2"]
LITERATURE_LABELS = ["Literature_single_high", "Literature_double_high", "Literature_regular"]


def score_region(data, cluster_id):
    region = data.loc[data["Cluster"] == cluster_id].copy().reset_index(drop=True)
    if len(region) < 3:
        raise ValueError(f"Region {cluster_id} has fewer than three rows; check the clustering output.")
    if not np.isfinite(region[UMAP_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("UMAP coordinates must be finite.")
    raw_features = region[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    filled_features = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(raw_features.values)
    standardized_features = StandardScaler().fit_transform(filled_features)
    high_mask = (region[LABEL_COL].astype(str).str.strip() == LABEL_HIGH).values
    high_features = standardized_features[high_mask]
    coordinates = region[UMAP_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).values
    center = np.mean(coordinates, axis=0)
    covariance = np.cov(coordinates, rowvar=False)
    if np.linalg.matrix_rank(covariance) < 2:
        raise ValueError("The UMAP regional covariance is singular.")
    inverse_covariance = np.linalg.inv(covariance)
    mahalanobis_distances = np.array([mahalanobis(point, center, inverse_covariance) for point in coordinates])
    s1 = np.exp(-mahalanobis_distances**2 / 2.0)
    region["Mahalanobis_core_distance"] = np.round(mahalanobis_distances, 4)
    if len(high_features) < 1:
        raise ValueError("The region has no double-high literature anchors.")

    high_coordinates = coordinates[high_mask]
    k2_use = min(K2, len(high_coordinates))
    _, neighbor_indices = NearestNeighbors(n_neighbors=k2_use).fit(high_coordinates).kneighbors(coordinates)
    s2 = np.zeros(len(coordinates))
    for row_index in range(len(coordinates)):
        neighbors = high_coordinates[neighbor_indices[row_index]]
        high_center = neighbors.mean(axis=0)
        vector = coordinates[row_index] - high_center
        vector_norm = np.linalg.norm(vector)
        if vector_norm < EPS:
            s2[row_index] = 1.0
            continue
        axis = PCA(n_components=1).fit(neighbors).components_[0] if len(neighbors) >= 2 else vector / vector_norm
        s2[row_index] = np.abs(np.dot(vector, axis)) / (vector_norm + EPS)

    reference_mask = region[LABEL_COL].astype(str).str.strip().isin(LITERATURE_LABELS).values
    references = standardized_features[reference_mask]
    if len(references) < 2:
        raise ValueError("The region needs at least two literature scoring references.")
    m_use = min(M, len(references) - 1)
    distances = NearestNeighbors(n_neighbors=m_use + 1).fit(references).kneighbors(standardized_features)[0][:, 1:].mean(axis=1)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances) + EPS
    probability = np.exp(-0.5 * ((distances - mean_distance) / std_distance) ** 2)
    s3_raw = distances * probability
    s3 = (s3_raw - s3_raw.min()) / (s3_raw.max() - s3_raw.min() + EPS)
    score = (s1 * s2 * s3) ** (1.0 / 3.0)
    region["S1_maha_sim"] = s1
    region["S2_alignment"] = s2
    region["S3_density"] = s3
    region["Score"] = score
    region["Rank"] = region["Score"].rank(ascending=False, method="min").astype(int)
    region["Percentile"] = region["Score"].rank(pct=True) * 100
    for column in ["S1_maha_sim", "S2_alignment", "S3_density"]:
        region[column + "_pct"] = region[column].rank(pct=True) * 100
    return region.sort_values("Score", ascending=False, kind="stable")


def main():
    parser = argparse.ArgumentParser(description="Apply the fixed three-axis regional scoring method.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/Clusters.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--clusters", type=int, nargs="+", default=[40, 47])
    args = parser.parse_args()
    data = pd.read_excel(args.input) if args.input.suffix.lower() == ".xlsx" else pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    results = {cluster_id: score_region(data, cluster_id) for cluster_id in args.clusters}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, result in results.items():
        result.to_csv(args.output_dir / f"Cluster{cluster_id}_Scores.csv", index=False, encoding="utf-8-sig")
        virtual = result.loc[result[LABEL_COL] == "Virtual_library"].copy()
        virtual.insert(0, "Virtual_queue_position", range(1, len(virtual) + 1))
        virtual.head(10).to_csv(args.output_dir / f"Cluster{cluster_id}_Top10.csv", index=False, encoding="utf-8-sig")
        print(f"Scored region {cluster_id}: {len(result)} rows, {len(virtual)} virtual candidates")


if __name__ == "__main__":
    main()
