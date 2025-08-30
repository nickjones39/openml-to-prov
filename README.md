# openml-to-prov

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17007366.svg)](https://doi.org/10.5281/zenodo.17007366)

**Export OpenML study tasks and runs into W3C PROV-JSON with optional PROV graph rendering for reproducible ML provenance.**

## Overview

`openml-to-prov` is a reproducible pipeline for converting machine learning experiments from [OpenML](https://www.openml.org/) study suites (e.g., CC18, or any other suite/task) into [W3C PROV](https://www.w3.org/TR/prov-dm/) compliant JSON records. The script captures datasets, flows, models, predictions, evaluation metrics, cross-validation splits, timing, and environment metadata, and optionally renders the provenance graph using the `prov` Python package.

Designed for research in provenance-aware machine learning, reproducibility, and provenance-based graph compression, the tool supports local execution (no uploads to OpenML servers) and emits both raw “pre-PROV” bundles and full PROV-JSON exports suitable for further analysis, visualisation, or archival.

The script captures:
- Datasets (metadata, targets, versions)
- Flows and parameter settings
- Trained models (artefacts, checksums, features/classes)
- Predictions and CV splits
- Evaluation metrics (per-fold + aggregate)
- Timing and environment metadata
- Agents and activities in PROV semantics

Optional rendering with the [`prov` Python package](https://pypi.org/project/prov/) generates publication-quality provenance graphs (Graphviz).

## Features

- **Generic**: Works with any OpenML suite ID (default: CC18, `suite_id=99`).
- **Provenance fidelity**: Conforms to the W3C PROV data model.
- **Reproducibility**: Environment and parameter capture included.
- **Outputs**:
  - PROV-JSON (`*.prov.json`)
  - Raw “pre-PROV” bundles
  - Rendered provenance graphs (PNG)
  - Prediction/split CSV artefacts
  - Summary CSV (nodes, edges, accuracy, sizes)

## Installation

Clone the repository and install dependencies in a fresh environment:

```bash
git clone https://github.com/<your-username>/openml-to-prov.git
cd openml-to-prov

# Conda example
conda create -n openml-prov python=3.10
conda activate openml-prov
pip install -r requirements.txt
```

**Dependencies**:
- `openml`
- `scikit-learn`
- `numpy`
- `pandas`
- `joblib`
- `prov` (optional, for rendering)
- `graphviz` (optional, for rendering)

## Usage

Run the script directly:

```bash
python openml-to-prov.py --suite-id 99 --n-tasks 3
```

**Arguments**:
- `--suite-id` (int): OpenML suite ID (default: 99 for CC18)
- `--n-tasks` (int): Number of tasks to process (default: 5)

**Outputs** will be written to:
- `prov_out/`
- `preprov_out/`
- `models_out/`
- `predictions_out/`
- `splits_out/`
- `prov_dot_png/` (if rendering enabled)

## Example

Example provenance output (`openml_run_<uuid>.prov.json`):

```json
{
  "prefix": {
    "prov": "http://www.w3.org/ns/prov#",
    "openml": "https://openml.org/def/",
    "e": "urn:entity:"
  },
  "entity": {
    "e:dataset/11": {
      "prov:label": "OpenML dataset 11 (balance-scale)",
      "prov:type": "openml:Dataset",
      "openml:data_id": 11
    }
  },
  "activity": { ... },
  "agent": { ... }
}
```

Rendered graph example: *(Image not shown here; see `prov_dot_png/` for output)*

## Citation

If you use this software, please cite:

```bibtex
@software{jones_openml_to_prov_2025,
  author       = {Nicholas Jones},
  title        = {openml-to-prov: Export OpenML runs to W3C PROV-JSON},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.17007366},
  url          = {https://doi.org/10.5281/zenodo.17007366}
}
```

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Built on [OpenML](https://www.openml.org/) and [scikit-learn](https://scikit-learn.org/).
- Provenance representation via [W3C PROV](https://www.w3.org/TR/prov-dm/).
- Graph rendering through the [`prov` Python library](https://pypi.org/project/prov/).
- This work acknowledges the [OpenML](https://www.openml.org/) platform as described in:  
  Vanschoren, J., van Rijn, J. N., Bischl, B., & Torgo, L. (2017). *OpenML: networked science in machine learning.*  
  [arXiv:1708.03731](https://arxiv.org/abs/1708.03731).
