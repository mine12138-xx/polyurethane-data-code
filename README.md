# Polyurethane formulation data and analysis code

Data and Python scripts for polyurethane chemical-space analysis, including virtual-library enumeration, molecular descriptors, UMAP, HDBSCAN, and three-axis regional scoring.

## Contents

| Location | Contents |
| --- | --- |
| `data/raw/OriginData.xlsx` | 296 literature formulation records and source DOI fields |
| `data/raw/SMILES.xlsx` | 86 molecular building blocks: 22 soft segments, 8 diisocyanates, and 56 chain extenders |
| `code/` | Seven numbered analysis scripts |
| [Workflow](docs/WORKFLOW.md) | Input fields, parameters, descriptor definitions, and outputs |
| [Version notes](docs/VERSION_NOTES.md) | Computational versions, region identifiers, and recorded run information |

## Installation

Use Python 3.11 or later. Python 3.12.13 on Windows was used for the recorded local checks. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
```

`requirements-tested.txt` records the package versions used for those checks; `requirements.txt` lists compatible dependency ranges. On Linux or macOS, use `.venv/bin/python` in the commands below.

## Run the analysis

```powershell
.\.venv\Scripts\python.exe code/01_enumerate_virtual_library.py
.\.venv\Scripts\python.exe code/02_repair_virtual_library_ratios.py
.\.venv\Scripts\python.exe code/03_extract_virtual_descriptors.py
.\.venv\Scripts\python.exe code/04_extract_literature_descriptors.py
.\.venv\Scripts\python.exe code/05_umap_embedding.py
.\.venv\Scripts\python.exe code/06_hdbscan_clustering.py
```

These steps generate 280,896 virtual formulations and combine them with the literature records. Files are written to `outputs/`. Allow several gigabytes of free disk space for the intermediate tables.

For regional scoring, inspect `outputs/Clusters_composition.csv` and select a region with double-high literature anchors, at least two literature references, and nonsingular coordinate covariance:

```powershell
.\.venv\Scripts\python.exe code/07_final_scoring.py --clusters 40
```

Each script supports `--help`. Replace `40` with a selected cluster identifier from `Clusters_composition.csv`. Rankings are relative to the selected region. The release contains inputs and code; computed tables are generated locally. Input fields, analysis parameters, and output definitions are described in the [workflow documentation](docs/WORKFLOW.md).

## Citation and reuse

Repository: https://github.com/mine12138-xx/polyurethane-data-code

When referencing these materials, record the release tag or commit used and retain the source DOI fields in the literature data. No reuse license is specified in this package.
