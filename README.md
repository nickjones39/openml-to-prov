# openml-to-prov

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23383570.svg)](https://doi.org/10.5281/zenodo.23383570)

**Generate W3C PROV-JSON provenance corpora from OpenML benchmark tasks for reproducible ML provenance research.**

## Overview

`openml-to-prov` is a reproducible pipeline for generating [W3C PROV](https://www.w3.org/TR/prov-dm/) compliant JSON provenance graphs from [OpenML](https://www.openml.org/) benchmark tasks. The tool is centred on the OpenML-CC18 curated classification benchmark, with extended task coverage for scalability studies.

Designed for research in provenance-aware machine learning, reproducibility, and **provenance-based graph compression**, the tool supports four corpus sizes spanning 2.5 MB to 2,503 MB (measured on disk, decimal MB), enabling validation of compression algorithms across three orders of magnitude.

The pipeline captures:
- Datasets (metadata, targets, versions)
- Flows and parameter settings
- Trained models (artefacts; content digests in real-execution mode)
- Predictions and CV splits (per-fold granularity)
- Evaluation metrics (per-fold + aggregate)
- Timing and environment metadata
- Agents and activities in PROV semantics

## Features

- **Scalable corpora**: Four modes from 2.5 MB to 2,503 MB for compression research
- **OpenML-CC18 core**: Built on the official curated classification benchmark suite
- **Extended coverage**: Additional classification and regression tasks for scale
- **Provenance fidelity**: Conforms to the W3C PROV data model
- **Reproducibility**: real-execution graphs record what a re-execution needs (installed library
  versions, the preprocessing pipeline, OpenML fold indices, content digests of data, model and
  predictions); `tools/reproduce_from_prov.py` re-executes runs from their graphs alone
- **Per-fold granularity**: Separate train/predict/evaluate chains for each CV fold
- **144 model configurations**: 12 classifiers/regressors × 12 hyperparameter settings
- **Conformance validation**: Optional pre-write checks (PROV-JSON schema, SHACL shapes, PROV-O round-trip) with a machine-readable `conformance_report.json`
- **Semantic validation**: `tools/semantic_check.py` checks that every graph carries exactly the relations of the published mapping, a subset of W3C PROV-CONSTRAINTS, and value consistency
- **Any fold count**: `--n-folds k` (default 5, the first five of OpenML-CC18's ten); the template is `7k+9` nodes / `24k+8` edges
- **Outputs**:
  - PROV-JSON files per run (`prov_*.json`)
  - Corpus manifest with statistics
  - Organized by task and model type

## Corpus Modes

| Mode | Tasks | Configs | Runs | Size | Use Case |
|------|-------|---------|------|------|----------|
| `light` | 72 | 1 | 72 | 2.5 MB | CI fixture; the only mode using real execution |
| `scaled` | 72 | 144 | 10,368 | 341 MB | Mid-scale structural studies |
| `large` | 171 | 144 | 24,624 | 810 MB | Structural scaling validation |
| `full` | 528 | 144 | 76,032 | 2,503 MB | Largest structural scale |

The four generated corpora are archived on Zenodo as **openml-to-prov Corpora v1.1**,
[10.5281/zenodo.23385491](https://doi.org/10.5281/zenodo.23385491) (all versions:
[10.5281/zenodo.22217218](https://doi.org/10.5281/zenodo.22217218)). In v1.1 the real-execution
`light` corpus is generated with v1.4.0 (under scikit-learn 1.8.0); the three synthetic corpora
are those of v1.3.0, and generating them with v1.4.0 gives graphs of identical structure and
attribute set, since the v1.4.0 changes affect real-execution graphs only.

### Task Sources

- **CC18**: 72 classification tasks from the OpenML-CC18 benchmark suite ([Suite ID 99](https://www.openml.org/search?type=benchmark&study_type=task&id=99)). The hardcoded list is verified against the live suite; with `--real`, task IDs are fetched directly from OpenML at runtime
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

Each family contributes 12 configurations to the 144-configuration grid. The hyperparameter
dimensions varied per classifier family are:

| Classifier | Key hyperparameters | Variation range |
|---|---|---|
| RandomForest | `n_estimators`, `max_depth` | 50–300 trees, depth 5–30 |
| GradientBoosting | `n_estimators`, `learning_rate`, `max_depth` | 50–200, 0.01–0.2, 3–6 |
| AdaBoost | `n_estimators`, `learning_rate` | 50–200, 0.3–1.5 |
| ExtraTrees | `n_estimators`, `max_depth` | 50–200, 8–30 |
| LogisticRegression | `C`, `max_iter` | 0.01–100, 1000–2000 |
| SVM | `C`, `kernel` | 0.01–10, rbf/linear/poly |
| KNN | `n_neighbors`, `weights` | 1–25, uniform/distance |
| MLP | `hidden_layer_sizes`, `max_iter` | (50,)–(256,128), 500–1000 |
| DecisionTree | `max_depth`, `criterion` | 3–25, gini/entropy |
| NaiveBayes | `var_smoothing` | 1e-12–1e-5 |
| BaggingClassifier | `n_estimators`, `max_samples` | 10–100, 0.7–1.0 |
| HistGradientBoosting | `max_iter`, `learning_rate`, `max_depth` | 50–200, 0.01–0.2, 4–10 |

Every (flow, parameter) combination is content-addressed by SHA-256. Because the hash is taken
over the hyperparameter dictionary, it identifies a *configuration* rather than a model family:
the classification-only modes therefore contain exactly 144 distinct flow hashes, and `full`
contains 203 (85 configurations are parameterised identically for a classifier and its
regression counterpart, so they hash alike).

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

**Optional** (for conformance validation `--validate`, and the `openml_to_prov.evaluate` RDF/SPARQL layers):
- `rdflib`
- `pyshacl`
- `prov`
- `matplotlib` (for `evaluate` plots only)

> The schema check and the compression layer of `evaluate` run with no extra dependencies. The SHACL and PROV-O round-trip checks, and the RDF/SPARQL layers of `evaluate`, require the packages above.
>
> **By default a check whose dependency is missing is skipped, and a skipped check does not fail the document** — so a corpus generated in an environment without these packages yields a report in which every document "passed" while only the schema check actually ran. Pass `--require-deps` whenever the resulting `conformance_report.json` will be cited as evidence: it turns a missing dependency into a hard failure instead.

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
# Generate light corpus (2.5 MB, 72 runs)
python -m openml_to_prov --mode light

# Generate scaled corpus (341 MB, 10,368 runs)
python -m openml_to_prov --mode scaled

# Generate large corpus (810 MB, 24,624 runs)
python -m openml_to_prov --mode large

# Generate full corpus (2,503 MB, 76,032 runs)
python -m openml_to_prov --mode full
```

### Options

```bash
python -m openml_to_prov --mode full --output my_corpus   # Custom output directory
python -m openml_to_prov --mode full --compact            # Minified JSON (smaller files)
python -m openml_to_prov --mode full --quiet              # Suppress progress output
python -m openml_to_prov --mode full --max-tasks 10       # Limit tasks (for testing)
python -m openml_to_prov --mode light --real              # Real OpenML + sklearn execution
python -m openml_to_prov --mode light --validate          # Validate each graph before writing
python -m openml_to_prov --mode light --real --n-folds 10 # Full ten-fold OpenML-CC18 protocol
```

### Fold Coverage

OpenML-CC18 tasks are defined on the platform with a **ten-fold** stratified cross-validation
estimation procedure. This tool consumes OpenML's own split indices — so every fold is a
genuine platform-defined partition, not one re-derived locally — but takes the **first
`n_folds` of the ten defined folds**, with `n_folds = 5` by default.

Two consequences are worth being explicit about:

- Each fold keeps OpenML's 90/10 train/test ratio, so the individual splits are exactly those
  the platform specifies. For `credit-g` (task 31, n=1000): train 900, test 100.
- The five test partitions together cover **half** of each dataset. An aggregate accuracy from
  a default run is therefore a mean over five of ten folds, and is **not** directly comparable
  with the ten-fold figures OpenML publishes for the same task.

Pass `--n-folds 10` (or set `n_folds=10` in `CorpusConfig`) for full coverage and
OpenML-comparable aggregates. The per-run template follows from `k`: `4k+6` entities,
`3k+2` activities and one agent (`7k+9` nodes) and `24k+8` edges — 44/128 at `k=5`, 79/248
at `k=10`. Since v1.4.0 the conformance check computes the expected cardinality from
`n_folds`, so a ten-fold corpus validates without `--no-cardinality-check`.

Why `k=5` by default: it halves real-execution time and corpus size relative to ten folds.
On the 72 CC18 tasks under the `light` RandomForest configuration, the five-fold and
ten-fold means differ little (see `tools/tenfold_compare.py`), but they are not the same
estimate, and only the ten-fold figure is comparable with OpenML's published evaluations.

### Real Execution Mode (`--real`)

By default the corpus is generated with synthetic per-fold metrics and timestamps so it can be produced quickly without any network calls. The `--real` flag switches to **actual OpenML execution**: each task is downloaded from OpenML, scikit-learn is trained with the configured hyperparameters using OpenML's official cross-validation splits, and the PROV graph records real accuracy/R² scores and real wall-clock timestamps.

```bash
# Real execution for the 72-task CC18 light corpus (~44 min measured)
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
- Since v1.4.0, real-execution graphs record everything needed to re-execute a run from the
  graph alone, and content digests that let a re-execution be checked against it:

  | Element | Attribute (real mode) |
  |---|---|
  | Flow | `ml:estimator_class` (fully qualified), `ml:pipeline` (preprocessing specification the pipeline is built from) |
  | Split | `openml:fold_index`, `openml:repeat` (the OpenML split actually used) |
  | Dataset | `checksum:sha256` over the loaded data, `openml:data_version`, `openml:md5_checksum` |
  | Model | `checksum:sha256` of the pickled fitted pipeline |
  | Predictions | `checksum:sha256` of the predicted labels |
  | Environment | installed versions of scikit-learn, openml, NumPy, pandas, SciPy |
  | Train / Predict / Evaluate | `ml:duration_s` (sub-second; `prov:*Time` is whole-second) |
  | Experiment / Aggregate | times taken from the execution, so `wasInformedBy` ordering holds |

  Synthetic graphs keep the v1.3.0 attribute set, so their size and structure are unchanged.
  Their dataset, model and prediction `checksum:sha256` values are digests of the element
  identifier, not of content (there is no content), and serve only as stable keys. The same
  was true of real-execution graphs up to v1.3.0.

Estimated wall-clock per mode at `--real` (measured baseline: `light` = ~44 min wall-clock on an M-series MacBook Pro, of which 41.4 min is fold-level training; larger modes also use heavier classifiers on bigger datasets so real time may exceed the linear estimate):

| Mode | Runs | Estimated time |
|------|------|----------------|
| `light` | 72 | ~44 min (measured) |
| `scaled` | 10,368 | ~4–7 days |
| `large` | 24,624 | ~10–17 days |
| `full` | 76,032 | ~30–50 days |

For scales beyond `light`, use `--max-tasks` to validate a representative subset rather than running the entire corpus.

### Conformance Validation (`--validate`)

By default, documents are written without validation. The `--validate` flag turns on conformance checking: each PROV-JSON document is validated **before** it is written to the corpus, and a `conformance_report.json` is written alongside `corpus_manifest.json`. This provides machine-checkable evidence of PROV conformance rather than assertions of validity.

Three independent checks are available:

- **`schema`** — Structural PROV-JSON conformance: required top-level blocks and prefixes, every element carries a `prov:type`, every relation references declared elements, and (by default) the expected per-run cardinality for the configured fold count — `7k+9` nodes and `24k+8` edges, i.e. 44/128 at `k=5` and 79/248 at `k=10`. No extra dependencies.
- **`shacl`** — SHACL shapes validated over the PROV-O graph (requires `rdflib` + `pyshacl`).
- **`round_trip`** — PROV-O round-trip: the document is parsed to a PROV model, serialised to RDF, round-tripped through PROV-JSON again, and the two RDF graphs are checked for isomorphism (requires `prov` + `rdflib`).

```bash
# Validate with all three checks (default), strict policy
python -m openml_to_prov --mode light --validate

# Fast CI run: schema check only, no optional dependencies needed
python -m openml_to_prov --mode light --validate --validation-checks schema

# Choose specific checks
python -m openml_to_prov --mode scaled --validate --validation-checks schema,round_trip

# Failure policy: warn (write anyway, flag in report) instead of strict (skip)
python -m openml_to_prov --mode large --validate --validation-policy warn

# Ten-fold corpus: the cardinality assertion follows --n-folds automatically
python -m openml_to_prov --mode light --n-folds 10 --validate

# Disable the cardinality assertion (only if you extend the mapping itself)
python -m openml_to_prov --mode light --validate --no-cardinality-check
```

**Options:**

- `--validation-checks` — Comma-separated subset of `schema`, `shacl`, `round_trip`. Default: `schema,shacl,round_trip`.
- `--validation-policy` — On failure: `strict` skips the document and logs it (default), `warn` writes it anyway but flags it in the report, `abort` stops the run.
- `--no-cardinality-check` — Disable the per-run template assertion in the schema check. Since v1.4.0 the expected counts are computed from `--n-folds`, so this is needed only when the Entity–Activity–Agent mapping itself is extended.
- `--require-deps` — Treat a missing optional dependency (`pyshacl`, `rdflib`, `prov`) as a check **failure** rather than skipping it. Use this whenever the conformance report is intended as evidence that a named check actually executed.

The `conformance_report.json` records the failure policy, per-check pass/fail/skip counts, total documents validated, and a capped list of any failures with their file paths. It distinguishes `checks_requested` from `checks_executed`: a check is listed as executed only if it ran against **every** validated document with no skips, and anything short of that is recorded under `checks_not_fully_executed`. Read `checks_executed`, not `checks_requested`, when interpreting a report. Programmatically, the same configuration is available via `CorpusConfig(validate=True, validation_checks=(...), validation_policy="strict", validation_cardinality=True)`, and the `ProvValidator` class can be used standalone:

```python
import json
from openml_to_prov import ProvValidator

doc = json.load(open("prov_corpus/task_3/RandomForest/prov_xxx.json"))
result = ProvValidator(checks=("schema", "shacl", "round_trip")).validate(doc)
print(result.valid, result.error_summary())
```

#### Verified results

Both released corpora have been validated exhaustively with all three checks executed against
every document (`--require-deps`, so nothing was silently skipped):

| Corpus | Documents | Passed | Failed | Skipped | Wall clock |
|---|---|---|---|---|---|
| `light` (real execution, corpora v1.1) | 72 | 72 | 0 | 0 | 11.1 s |
| `full` (synthetic) | 76,032 | 76,032 | 0 | 0 | 9,050 s |
| `prov_light10_real` (real, `k=10`, v1.4.0) | 72 | 72 | 0 | 0 | — |

The PROV-O round-trip returns **exactly 535 triples for every `full` document and 579 for every
`light` document** (its additional real-execution fields) — minimum and maximum alike — so the
RDF projection is invariant within each corpus in the same way the PROV-JSON graph is.

Reports are written to `conformance_report.json` alongside the corpus manifest.

Beyond conformance, the verification tools (below) established, with outputs in `repro_r3/`:

- **Semantic validation** of all 111,096 documents of corpora v1: every graph matches the published
  mapping exactly. It also found that 58 of the 72 real `light` graphs of v1 timestamp the
  experiment activity after some of its folds (it was stamped when the document was built),
  contradicting PROV-CONSTRAINTS `wasInformedBy` ordering. Fixed in v1.4.0; the v1.1 `light`
  corpus and the ten-fold corpus pass every check.
- **Re-execution from the graphs alone** of the 72 real `light` runs of corpora v1: 340/360 folds
  exact under the scikit-learn version the graphs record, and 360/360 under 1.5.2 or 1.8.0 —
  v1.3.0 wrote the scikit-learn version as a fixed string ("1.7.1"), and that corpus was not
  produced under it. The v1.1 `light` corpus, regenerated with v1.4.0, reproduces 360/360 folds
  and 360/360 prediction digests under its recorded versions, with no assumptions and accuracies
  identical to v1; the ten-fold corpus reproduces 720/720.

### Downstream Evaluation (`openml_to_prov.evaluate`)

The `evaluate` module runs a baseline downstream evaluation over a generated corpus, demonstrating practical usability across three access patterns: generic **compression** (gzip/bz2/lzma ratios), **graph-database loading** (PROV-JSON → PROV-O RDF into an in-memory RDFLib store, reporting triples and load time), and **SPARQL querying** (five representative provenance queries with result counts and timing). It emits a JSON results file, two LaTeX tables, and two plots.

```bash
# Evaluate the light corpus (small — runs all of it)
python -m openml_to_prov.evaluate --corpus prov_light_corpus --label light

# Evaluate the full corpus; sample the RDF/SPARQL layers for tractability
# (compression always runs over the whole corpus)
python -m openml_to_prov.evaluate --corpus prov_full_synthetic \
    --label full --sample 1000
```

**Options:**

- `--corpus` — Path to a generated corpus directory (required).
- `--label` — Label used in output filenames (default: corpus directory name).
- `--sample N` — Sample `N` runs for the RDF loading and SPARQL layers. Compression always runs over the whole corpus. Recommended for large corpora (e.g. `--sample 1000`); omit to use every run.
- `--outdir` — Output directory (default: `eval_results`).
- `--seed` — Sampling seed (default: 42).
- `--quiet` — Suppress progress output.

Outputs, written under `--outdir`:

- `eval_<label>.json` — Machine-readable results (raw/compressed bytes, ratios, triple counts, load and query timings).
- `eval_<label>_compression.tex`, `eval_<label>_sparql.tex` — LaTeX tables.
- `eval_<label>_compression.png`, `eval_<label>_query_timing.png` — Plots.

The RDF and SPARQL layers require `rdflib` and `prov`; plots require `matplotlib`. If a dependency is missing, the corresponding layer is skipped gracefully. The compression layer needs only the standard library.

> **Note on compression figures.** The codecs are applied to the corpus concatenated into a single byte stream, which isolates the cross-run redundancy that the corpus is designed to exhibit. Per-file archiving (e.g. a default `.zip`) reports a lower ratio (~6.5×) because it cannot reference structure shared across files, and `tar.gz` falls between the two owing to per-file header and padding overhead.

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
├── conformance_report.json       # Validation results (only when --validate is used)
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
└── ...                           # 72-528 tasks depending on mode
```

## Verification Tools

`tools/` holds four scripts that check a corpus beyond schema conformance or measure its cost.
None is needed to generate a corpus. Their outputs for the paper's round-3 revision are in
`repro_r3/` (see its README), and the ten-fold real-execution corpus they were run against is
included as `prov_light10_real/` (72 runs, `--n-folds 10`, v1.4.0).

```bash
# Semantic validation: exact agreement with the published mapping, a subset of
# W3C PROV-CONSTRAINTS (typing, disjointness, unique generation, derivation
# acyclicity, timestamp consistency with event ordering), and value consistency
python tools/semantic_check.py --corpus prov_light_corpus --out sem.json

# Independent re-execution from the PROV records alone. Does not import
# openml_to_prov; run it in a separate environment pinned to the versions the
# graphs record (see the script's docstring)
python tools/reproduce_from_prov.py --corpus prov_light_corpus --out repro.json --cache-dir .cache-repro

# Five-of-ten versus ten-of-ten folds, and placement against OpenML's published
# RandomForest evaluations
python tools/tenfold_compare.py --corpus prov_light10 --out ten.json --openml

# Generation time / memory, and RDF load + SPARQL latency at increasing size
python tools/scaling_eval.py gen --mode scaled --package-root . --output /tmp/g --out gen.json
python tools/scaling_eval.py rdflib --corpus prov_full_synthetic --out rdflib.json
python tools/scaling_eval.py oxigraph --corpus prov_full_synthetic --store /tmp/ox --out ox.json
```

## PROV Document Structure

Each PROV-JSON file contains a complete provenance graph for one experiment run:

### Entities
- **Dataset**: OpenML dataset with metadata, checksums
- **Task**: OpenML task definition
- **Flow**: ML model/algorithm specification
- **FlowParameters**: Hyperparameter configuration
- **Environment**: Python version, platform, library versions
- **Split**: CV fold number and train/test sizes (real mode also: OpenML fold index and repeat)
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
- `used`: Activity consumed entity — Experiment uses dataset, task, flow, parameters,
  environment; Train uses dataset, split, flow, parameters, environment; Predict uses model,
  dataset, split; Evaluate uses predictions, split; Aggregate uses the per-fold metrics
- `wasGeneratedBy`: Model ← Train, Predictions ← Predict, Metrics ← Evaluate,
  AggregateMetrics ← Aggregate
- `wasAssociatedWith`: every activity with the software agent
- `wasInformedBy`: Train ← Experiment, Predict ← Train, Evaluate ← Predict
- `wasAttributedTo`: each per-fold Model to the software agent
- `wasDerivedFrom`: Metrics ← Predictions, Split; AggregateMetrics ← per-fold Metrics

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
├── validator.py      # Conformance validation (schema, SHACL, PROV-O round-trip)
├── evaluate.py       # Baseline downstream evaluation (compression, RDF, SPARQL)
└── utils.py          # Utility functions (hashing, timestamps, metrics)

tools/                # Verification and measurement; independent of corpus generation
├── semantic_check.py       # Mapping conformance, PROV-CONSTRAINTS subset, value consistency
├── reproduce_from_prov.py  # Re-execute runs from PROV records alone (imports nothing above)
├── tenfold_compare.py      # Five- vs ten-fold means; placement against OpenML results
└── scaling_eval.py         # Generation, RDF load and SPARQL cost at increasing size

repro_r3/             # Outputs of the tools for the paper's round-3 revision (see its README)
prov_light10_real/    # Ten-fold real-execution corpus (72 runs, v1.4.0)
```

## Use Cases

### Compression Research

Generate corpora at multiple scales, then run the built-in baseline evaluation (compression, RDF loading, SPARQL) to characterise them:

```bash
# Generate corpora at two scales
python -m openml_to_prov --mode light --output corpus_light
python -m openml_to_prov --mode large --output corpus_large

# Baseline compression + RDF + SPARQL over each
python -m openml_to_prov.evaluate --corpus corpus_light --label light
python -m openml_to_prov.evaluate --corpus corpus_large --label large --sample 1000
```

### Reproducibility Studies

Re-execute real-execution runs from their PROV records and check them against what was recorded
(run in a separate environment pinned to the versions the graphs record):

```bash
python tools/reproduce_from_prov.py --corpus prov_light10_real --out repro.json --cache-dir .cache-repro
```

Or capture and inspect provenance programmatically:

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

If you use this software, please cite the version you used. The current release is v1.4.0,
[10.5281/zenodo.23383570](https://doi.org/10.5281/zenodo.23383570); the concept DOI
[10.5281/zenodo.17007365](https://doi.org/10.5281/zenodo.17007365) always resolves to the latest.

```bibtex
@software{jones_openml_to_prov_2025,
  author       = {Nicholas Jones},
  title        = {openml-to-prov: Generate W3C PROV corpora from OpenML benchmarks},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.4.0},
  doi          = {10.5281/zenodo.23383570},
  url          = {https://doi.org/10.5281/zenodo.23383570}
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