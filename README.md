# openml-to-prov

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18218060.svg)](https://doi.org/10.5281/zenodo.18218060)

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
| `light` | 71 | 1 | 71 | ~2.1 MB | Quick testing, CI/CD |
| `scaled` | 71 | 144 | 10,224 | ~296 MB | Medium-scale experiments |
| `large` | 170 | 144 | 24,480 | ~710 MB | Large-scale validation |
| `full` | 527 | 144 | 75,888 | ~2.2 GB | Production benchmarking |

### Task Sources

- **CC18**: OpenML-CC18 benchmark suite — 71 validated classification tasks from the hardcoded list, or 72 fetched live from OpenML ([Suite ID 99](https://www.openml.org/search?type=benchmark&study_type=task&id=99)) when `--real` is used
- **Extended classification**: 99 additional OpenML classification tasks for `large` mode; 179 additional tasks for `full` mode
- **Regression**: 277 OpenML regression tasks for `full` mode supervised learning coverage

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
- `openml` (required for `--real` mode)
- `scikit-learn` (required for `--real` mode)
- `pandas` (required for `--real` mode)
- `python-dotenv` (loads OpenML API key from `.env`)
- `joblib`

**Optional** (for provenance graph rendering):
- `prov`
- `graphviz`

### OpenML API Key (for `--real` mode)

Real execution against the OpenML API requires an API key to avoid rate limiting. Get one free at [openml.org](https://www.openml.org/) → profile → API key, then create a `.env` file in the repo root:

```bash
# .env (never commit this file — already in .gitignore)
OPENML_API_KEY=your_openml_api_key_here
```

The key is loaded automatically whenever `--real` is used. At startup you should see:

```
OpenML API key loaded (ends ...XXXX)
```

If no key is found you'll see a warning and requests may be throttled.

## Usage

### Command Line

```bash
# Generate light corpus (~2.1 MB, 71 runs)
python -m openml_to_prov --mode light

# Generate scaled corpus (~296 MB, 10,224 runs)
python -m openml_to_prov --mode scaled

# Generate large corpus (~710 MB, 24,480 runs)
python -m openml_to_prov --mode large

# Generate full corpus (~2.2 GB, 75,888 runs)
python -m openml_to_prov --mode full
```

### Options

```bash
python -m openml_to_prov --mode full --output my_corpus   # Custom output directory
python -m openml_to_prov --mode full --compact            # Minified JSON (smaller files)
python -m openml_to_prov --mode full --quiet              # Suppress progress output
python -m openml_to_prov --mode full --max-tasks 10       # Limit tasks (for testing)
python -m openml_to_prov --mode light --real              # Real OpenML + sklearn execution
```

### Real Execution Mode (`--real`)

By default the corpus is generated with synthetic per-fold metrics and timestamps so it can be produced quickly without any network calls. The `--real` flag switches to **actual OpenML execution**: each task is downloaded from OpenML, scikit-learn is trained with the configured hyperparameters using OpenML's official cross-validation splits, and the PROV graph records real accuracy/R² scores and real wall-clock timestamps.

```bash
# Real execution for the 71-task CC18 light corpus (~45 min)
python -m openml_to_prov --mode light --real --output prov_light_real

# Sample real runs from larger modes (use --max-tasks to limit scope)
python -m openml_to_prov --mode scaled --real --max-tasks 5 --output prov_scaled_sample
python -m openml_to_prov --mode full   --real --max-tasks 3 --output prov_full_sample
```

> **Warning**
> Running `--real` on anything other than `light` mode will take many hours to several days. The CLI prints a runtime estimate and prompts for confirmation before starting. Use `--max-tasks N` to validate a subset before committing to a full run. Pass `--skip-confirm` to bypass the prompt (for scripting / CI).

**Notes on `--real` mode:**

- Requires an OpenML API key in `.env` (see [Installation](#installation)).
- Fetches the live CC18 task list from OpenML suite 99 at startup, so the corpus always matches the official suite.
- Supports both classification (accuracy) and regression (R²) tasks.
- Tasks that fail (unsupported type, server unavailable) are skipped gracefully and that run falls back to synthetic data.
- Transient network/server errors trigger automatic retries with exponential backoff.
- The corpus manifest records `"real_execution": true` so consumers can distinguish real from synthetic runs.

Estimated wall-clock per mode at `--real` (measured baseline: `light` = ~45 min on an M-series MacBook Pro; larger modes also use heavier classifiers on bigger datasets so real time may exceed the linear estimate):

| Mode | Runs | Estimated time |
|------|------|----------------|
| `light` | 71 | ~45 min (measured) |
| `scaled` | 10,224 | ~4–7 days |
| `large` | 24,480 | ~10–17 days |
| `full` | 75,888 | ~30–50 days |

For scales beyond `light`, use `--max-tasks` to validate a representative subset rather than running the entire corpus.

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
├── executor.py       # Real OpenML + sklearn execution engine (--real mode)
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
  doi          = {10.5281/zenodo.18218060},
  url          = {https://doi.org/10.5281/zenodo.18218060}
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
