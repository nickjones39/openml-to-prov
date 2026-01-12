# openml-to-prov

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17007366.svg)](https://doi.org/10.5281/zenodo.17007366)

**Generate W3C PROV-JSON provenance corpora from OpenML benchmark tasks for reproducible ML provenance research.**

## Overview

`openml-to-prov` is a reproducible pipeline for generating [W3C PROV](https://www.w3.org/TR/prov-dm/) compliant JSON provenance graphs from [OpenML](https://www.openml.org/) benchmark tasks. The tool is centred on the OpenML-CC18 curated classification benchmark, with extended task coverage for scalability studies.

Designed for research in provenance-aware machine learning, reproducibility, and **provenance-based graph compression**, the tool supports multiple corpus sizes from ~2 MB to ~2+ GB, enabling validation of compression algorithms at scale.

The pipeline captures:
- Datasets (metadata, targets, versions)
- Flows and parameter settings
- Trained models (artefacts, checksums, features/classes)
- Predictions and CV splits (per-fold granularity)
- Evaluation metrics (per-fold + aggregate)
- Timing and environment metadata
- Agents and activities in PROV semantics

## Features

- **Scalable corpora**: Four modes from ~2 MB to ~2+ GB for compression research
- **OpenML-CC18 core**: Built on the official curated classification benchmark suite
- **Extended coverage**: Additional classification and regression tasks for scale
- **Provenance fidelity**: Conforms to the W3C PROV data model
- **Reproducibility**: Environment and parameter capture included
- **Per-fold granularity**: Separate train/predict/evaluate chains for each CV fold
- **144 model configurations**: 12 classifiers/regressors × 12 hyperparameter settings
- **Outputs**:
  - PROV-JSON files per run (`prov_*.json`)
  - Corpus manifest with statistics
  - Organized by task and model type

## Corpus Modes

| Mode | Tasks | Configs | Runs | Size | Use Case |
|------|-------|---------|------|------|----------|
| `light` | 72 | 1 | 72 | ~2.2 MB | Quick testing, CI/CD |
| `scaled` | 72 | 144 | 10,368 | ~308 MB | Medium-scale experiments |
| `large` | 172 | 144 | 24,768 | ~734 MB | Large-scale validation |
| `full` | 422 | 144 | 60,768 | ~2+ GB | Production benchmarking |

### Task Sources

- **CC18**: OpenML-CC18 benchmark suite — 72 curated classification tasks ([Suite ID 99](https://www.openml.org/search?type=benchmark&study_type=task&id=99))
- **Extended classification**: Additional OpenML classification tasks selected for dataset diversity (~100–175 tasks)
- **Regression**: OpenML regression tasks for supervised learning coverage (~250 tasks)

### Model Configurations

**Classification** (12 classifiers × 12 configs = 144):
- RandomForest, GradientBoosting, AdaBoost, ExtraTrees
- LogisticRegression, SVM, KNN, MLP
- DecisionTree, NaiveBayes, BaggingClassifier, HistGradientBoosting

**Regression** (12 regressors × 12 configs = 144):
- RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
- Ridge, Lasso, ElasticNet, SVR
- KNeighborsRegressor, MLPRegressor, DecisionTreeRegressor, HistGradientBoostingRegressor

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/nickjones39/openml-to-prov.git
cd openml-to-prov

# Conda example
conda create -n openml-prov python=3.10
conda activate openml-prov
pip install -r requirements.txt
```

**Dependencies**:
- `numpy`

**Optional** (for original OpenML integration):
- `openml`
- `scikit-learn`
- `pandas`
- `joblib`
- `prov` (for rendering)
- `graphviz` (for rendering)

## Usage

### Command Line

```bash
# Generate light corpus (~2.2 MB, 72 runs)
python -m openml_to_prov --mode light

# Generate scaled corpus (~308 MB, 10,368 runs)
python -m openml_to_prov --mode scaled

# Generate large corpus (~734 MB, 24,768 runs)
python -m openml_to_prov --mode large

# Generate full corpus (~2+ GB, 60,768 runs)
python -m openml_to_prov --mode full
```

### Options

```bash
python -m openml_to_prov --mode full --output my_corpus   # Custom output directory
python -m openml_to_prov --mode full --compact            # Minified JSON (smaller files)
python -m openml_to_prov --mode full --quiet              # Suppress progress output
python -m openml_to_prov --mode full --max-tasks 10       # Limit tasks (for testing)
```

### Programmatic API

```python
from openml_to_prov import CorpusGenerator, CorpusConfig

# Configure corpus generation
config = CorpusConfig(
    mode="large",           # light, scaled, large, or full
    output_dir="prov_corpus",
    n_folds=5,
    pretty_print=True,      # Human-readable JSON
    verbose=True
)

# Generate corpus
generator = CorpusGenerator(config)
stats = generator.generate()

print(f"Generated {stats['total_runs']:,} runs")
print(f"Total size: {stats['total_bytes'] / 1e6:.1f} MB")
```

## Output Structure

```
prov_corpus/
├── corpus_manifest.json          # Corpus metadata and statistics
├── task_3/
│   ├── RandomForest/
│   │   ├── prov_<uuid>.json      # PROV document for config 0
│   │   ├── prov_<uuid>.json      # PROV document for config 1
│   │   └── ...                   # 12 configs total
│   ├── GradientBoosting/
│   │   └── ...
│   └── ...                       # 12 classifiers total
├── task_6/
│   └── ...
└── ...                           # 72-422 tasks depending on mode
```

## PROV Document Structure

Each PROV-JSON file contains a complete provenance graph for one experiment run:

### Entities
- **Dataset**: OpenML dataset with metadata, checksums
- **Task**: OpenML task definition
- **Flow**: ML model/algorithm specification
- **FlowParameters**: Hyperparameter configuration
- **Environment**: Python version, platform, library versions
- **Split**: CV fold train/test split indices
- **Model**: Trained model artifact per fold
- **Predictions**: Model predictions per fold
- **Metrics**: Evaluation metrics per fold
- **AggregateMetrics**: Cross-fold mean/std metrics

### Activities
- **Experiment**: Top-level experiment coordination
- **Train**: Model training per fold
- **Predict**: Model inference per fold
- **Evaluate**: Metric computation per fold
- **Aggregate**: Cross-fold aggregation

### Relations
- `used`: Activity consumed entity
- `wasGeneratedBy`: Entity produced by activity
- `wasAssociatedWith`: Activity performed by agent
- `wasInformedBy`: Activity communication chain
- `wasAttributedTo`: Entity attributed to agent
- `wasDerivedFrom`: Entity derivation lineage

## Example

Example provenance output (`prov_<uuid>.json`):

```json
{
  "prefix": {
    "prov": "http://www.w3.org/ns/prov#",
    "openml": "https://openml.org/def/",
    "ml": "https://ml-schema.org/",
    "e": "urn:entity:",
    "a": "urn:activity:",
    "ag": "urn:agent:"
  },
  "entity": {
    "e:dataset/11": {
      "prov:label": "Dataset 11",
      "prov:type": "openml:Dataset",
      "openml:data_id": 11,
      "ml:n_samples": 625,
      "ml:n_features": 4
    },
    "e:metrics/abc123_fold1": {
      "prov:label": "Metrics fold 1",
      "prov:type": "openml:Metrics",
      "openml:accuracy": 0.856
    }
  },
  "activity": {
    "a:train/abc123_fold1": {
      "prov:label": "Train fold 1",
      "prov:type": "openml:Train",
      "prov:startTime": "2025-01-09T12:00:00Z",
      "prov:endTime": "2025-01-09T12:00:03Z"
    }
  },
  "agent": {
    "ag:system/OpenML": {
      "prov:type": "prov:SoftwareAgent",
      "prov:label": "OpenML sklearn - RandomForest"
    }
  },
  "used": [...],
  "wasGeneratedBy": [...],
  "wasAssociatedWith": [...],
  "wasInformedBy": [...],
  "wasDerivedFrom": [...]
}
```

## Package Structure

```
openml_to_prov/
├── __init__.py       # Package exports
├── __main__.py       # CLI entry point
├── config.py         # CorpusConfig, task IDs (CC18, extended, regression)
├── generator.py      # CorpusGenerator class
├── models.py         # Classifier/regressor configurations
├── prov_builder.py   # W3C PROV document builder
└── utils.py          # Utility functions (hashing, timestamps, metrics)
```

## Use Cases

### Compression Research

Generate corpora at multiple scales to validate provenance compression algorithms:

```bash
# Generate test corpus
python -m openml_to_prov --mode light --output corpus_light

# Generate validation corpus
python -m openml_to_prov --mode large --output corpus_large

# Measure compression ratio
python your_compressor.py corpus_large/
```

### Reproducibility Studies

Capture complete ML experiment provenance for reproducibility analysis:

```python
from openml_to_prov import CorpusGenerator, CorpusConfig

config = CorpusConfig(mode="scaled")
generator = CorpusGenerator(config)
stats = generator.generate()

# Analyze provenance structure
import json
with open("prov_corpus/task_3/RandomForest/prov_xxx.json") as f:
    prov = json.load(f)
    print(f"Entities: {len(prov['entity'])}")
    print(f"Activities: {len(prov['activity'])}")
```

## Citation

If you use this software, please cite:

```bibtex
@software{jones_openml_to_prov_2025,
  author       = {Nicholas Jones},
  title        = {openml-to-prov: Generate W3C PROV corpora from OpenML benchmarks},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {2.0.0},
  doi          = {10.5281/zenodo.17007366},
  url          = {https://doi.org/10.5281/zenodo.17007366}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Built on [OpenML](https://www.openml.org/) benchmark tasks and [scikit-learn](https://scikit-learn.org/) model configurations.
- Provenance representation via [W3C PROV](https://www.w3.org/TR/prov-dm/).
- This work acknowledges the [OpenML](https://www.openml.org/) platform as described in:  
  Vanschoren, J., van Rijn, J. N., Bischl, B., & Torgo, L. (2017). *OpenML: networked science in machine learning.*  
  [arXiv:1708.03731](https://arxiv.org/abs/1708.03731).
