# Workflow and data fields

This document describes the released implementation. See the [README](../README.md) for execution and the [version notes](VERSION_NOTES.md) for computational settings and output conventions.

## 1. Inputs

`data/raw/SMILES.xlsx` contains 86 building blocks: 22 soft segments, 8 diisocyanates, and 56 chain extenders. Its main fields are `Abbreviation`, `Role`, `SMILES`, and `Molecular_weight`.

`data/raw/OriginData.xlsx` contains 296 literature records. The formulation fields are `Record_ID`, `Formulation`, `Soft_segment_1`, `Soft_segment_2`, `Diisocyanate_1`, `Diisocyanate_2`, `Chain_extender_1` through `Chain_extender_3`, and `Composition`. Metadata fields are `Hard_segment_content`, `Tensile_strength`, `Healing_efficiency`, and `DOI`.

Composition values are colon-separated. The literature parser accepts a compact ratio for the non-empty component positions or a ratio containing all seven component positions. Component names must match `SMILES.xlsx`. Preserve the original measurements and formulation fields when comparing releases. Tensile strength is recorded in MPa and healing efficiency as a fraction (0.90 denotes 90%). Missing measurements remain blank in the source workbook and are excluded from threshold checks.

## 2. Pipeline stages

| Stage | Script | Main input | Default output |
| --- | --- | --- | --- |
| 1 | `01_enumerate_virtual_library.py` | `SMILES.xlsx` | `outputs/Virtual_PU.csv` |
| 2 | `02_repair_virtual_library_ratios.py` | `Virtual_PU.csv` | `outputs/Virtual_PU_Fixed.csv` |
| 3 | `03_extract_virtual_descriptors.py` | `SMILES.xlsx`, repaired virtual table | `outputs/Virtual_Features.csv` |
| 4 | `04_extract_literature_descriptors.py` | `SMILES.xlsx`, `OriginData.xlsx` | `outputs/Literature_Features.csv` |
| 5 | `05_umap_embedding.py` | the two descriptor tables | `outputs/UMAP.csv` |
| 6 | `06_hdbscan_clustering.py` | `UMAP.csv` | `outputs/Clusters.csv`, `Clusters_composition.csv` |
| 7 | `07_final_scoring.py` | `Clusters.csv` | `ClusterID_Scores.csv`, `ClusterID_Top10.csv` |

Stages 3 and 4 are independent descriptor branches. Stage 5 concatenates their rows and adds labels before fitting UMAP. All scripts provide `--help` for custom paths.

## 3. Virtual enumeration and ratio representation

Each virtual formulation contains one soft segment, one diisocyanate, and two chain-extender positions. Chain-extender pairs are sampled with replacement and order is ignored, giving `56 x 57 / 2 = 1,596` pairs. The nominal enumeration is `22 x 8 x 1,596 = 280,896` formulations at ratios 2:4:1:1.

The nominal hard-segment content is the mass fraction of diisocyanate plus chain extenders. The repair stage changes a unit ratio of FBA, OBB, ABB, or PBA to `0.66`, including each repeated position independently. This is the documented trimer representation rule; it is applied by component identity to every matching chain-extender position.

## 4. Component descriptors and segment weighting

RDKit calculates component descriptors from SMILES. The three-dimensional branch adds hydrogens, calls `EmbedMolecule(randomSeed=42)`, calls `MMFFOptimizeMolecule`, and then calculates shape descriptors. This is component conformer calculation, not polymer-chain construction, network simulation, or molecular dynamics.

Each formulation is represented by soft-segment (`SS`), hard-segment (`HS`), and all-component (`PU`) groups. Each group contains the same base descriptor family. Density features use mass weighting after normalization by the SMILES molecular weight; `MPC`, `FractionCSP3`, and related features use molar weighting; shape and other mass-average features use mass weighting. `CED` is retained as zero in the current implementation and is not a calculated cohesive-energy density.

## 5. Fixed 13-dimensional mapping

The UMAP and scoring stages use exactly these 13 columns:

```text
HS_NumHAcceptors_Density
HS_NumHDonors_Density
HS_NOCount_Density
HS_CH3_groups_Density
HS_NumRotatableBonds_Density
HS_RingCount_Density
HS_FractionCSP3
HS_FpDensityMorgan3
HS_Acc
HS_MolWt
PU_RingCount_Density
PU_MPC
PU_ROG
```

Values are converted to numeric, non-finite values are treated as missing, missing values are filled with zero for UMAP, values are clipped to `±1e15`, and all rows are standardized together. UMAP uses two dimensions, `n_neighbors=30`, `min_dist=0.5`, Euclidean distance, `random_state=42`, and `n_jobs=1`.

Measured properties, names, DOI, and data labels do not enter descriptor fitting or UMAP fitting.

## 6. Labels and HDBSCAN

The literature thresholds are tensile strength `> 20 MPa` and healing efficiency `> 0.90`. Both thresholds produce `Literature_double_high`; exactly one produces `Literature_single_high`; remaining literature rows produce `Literature_regular`. Generated rows use `Virtual_library`.

HDBSCAN is applied to `UMAP_1` and `UMAP_2` with `min_cluster_size=200`, `min_samples=400`, and Euclidean distance. Non-negative values in `Cluster` identify fitted regions; `-1` is noise. The numeric labels are algorithm outputs and do not have fixed chemical identities across reruns.

## 7. Three-axis regional score

The scorer uses `K2 = M = 11` and `EPS = 1e-9`. A region needs at least three rows, one double-high literature anchor, two literature references, and a nonsingular two-dimensional covariance matrix.

- `S1_maha_sim` is the exponential similarity from the Mahalanobis distance to the region center in UMAP space.
- `S2_alignment` measures directional alignment with the first principal axis of the nearest double-high literature anchors. The nearest `min(11, number of anchors)` points are used.
- `S3_density` is based on distances in the independently standardized 13-dimensional descriptor space. Median imputation is used within the region. The nearest-reference distance is multiplied by a Gaussian distribution weight and then min-max normalized within the region.

The total score is `(S1 * S2 * S3) ** (1/3)`. It is a regional ordering statistic, not a direct property prediction or a universal ranking across clusters. The scorer outputs full-region `Rank` and `Percentile` values and, separately, the first ten virtual rows in score order with `Virtual_queue_position`.

## 8. Interpretation and execution notes

The structural representation uses component SMILES and weighted descriptors. Soft-segment SMILES may represent a shorter repeat-unit or representative structure, while the workbook supplies its nominal molecular weight for weighting. Shape descriptors describe the component conformer, rather than a full polymer network. The three-dimensional branch uses the returned conformer without an explicit optimization-convergence filter; its values should be interpreted within that implementation.

The descriptor tables contain a larger descriptor family than the 13-column analysis set. `CED` is a zero placeholder, and the released SPS API lookup also yields zero with the recorded dependency environment. Neither column belongs to the final 13 descriptors. Ratio parsing assumes one amount per represented component; check component counts and finite, nonnegative amounts when supplying new inputs.

The 0.66 step changes the descriptor ratio representation. Other enumeration metadata, including nominal hard-segment content and total molecular weight, retain their nominal values. The feature calculations use the represented composition. S3 drops the first nearest-reference distance for every query, including virtual formulations, before averaging the remaining distances. Its weighting is centered on the regional distance distribution; it is not a monotonic measure of novelty or a calibrated uncertainty.

The three-axis score combines regional geometry and literature references. It is a quantitative prioritization method for organizing heterogeneous literature formulations and virtual candidates; it is not a direct tensile-strength or healing-efficiency measurement.
