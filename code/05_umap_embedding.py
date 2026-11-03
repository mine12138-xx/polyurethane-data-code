from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap


ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "HS_FpDensityMorgan3", "HS_NumHAcceptors_Density", "HS_FractionCSP3", "HS_Acc",
    "HS_CH3_groups_Density", "PU_RingCount_Density", "HS_MolWt", "HS_NOCount_Density",
    "HS_NumHDonors_Density", "PU_ROG", "HS_RingCount_Density", "HS_NumRotatableBonds_Density", "PU_MPC",
]


def embed(data):
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing fixed descriptors: {sorted(missing)}")
    if len(data) < 4:
        raise ValueError("UMAP needs at least four rows.")
    values = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0)
    values = np.clip(values, -1e15, 1e15).astype(np.float64)
    scaled = StandardScaler().fit_transform(values)
    coordinates = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.5,
        metric="euclidean",
        random_state=42,
        n_jobs=1,
    ).fit_transform(scaled)
    result = data.copy()
    result["UMAP_1"] = coordinates[:, 0]
    result["UMAP_2"] = coordinates[:, 1]
    return result


def prepare_inputs(literature_path, virtual_path):
    literature = pd.read_csv(literature_path, encoding="utf-8-sig", low_memory=False)
    virtual = pd.read_csv(virtual_path, encoding="utf-8-sig", low_memory=False)
    strength = pd.to_numeric(literature["Tensile_strength"], errors="coerce")
    healing = pd.to_numeric(literature["Healing_efficiency"], errors="coerce")
    literature["Data_label"] = np.select(
        [(strength > 20) & (healing > 0.9), (strength > 20) | (healing > 0.9)],
        ["Literature_double_high", "Literature_single_high"],
        default="Literature_regular",
    )
    virtual["Data_label"] = "Virtual_library"
    return pd.concat([literature, virtual], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Embed fixed formulation descriptors with UMAP.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--literature", type=Path, default=ROOT / "outputs/Literature_Features.csv")
    parser.add_argument("--virtual", type=Path, default=ROOT / "outputs/Virtual_Features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/UMAP.csv")
    args = parser.parse_args()
    data = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False) if args.input else prepare_inputs(args.literature, args.virtual)
    result = embed(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Embedded {len(result)} rows: {args.output}")


if __name__ == "__main__":
    main()
