"""
openml_to_prov - W3C PROV corpus generator for OpenML benchmarks

Generates W3C PROV-JSON provenance graphs for OpenML benchmark tasks.
Supports multiple corpus sizes for compression research validation.

Usage:
  python -m openml_to_prov --mode light    # ~2.2 MB, 72 runs
  python -m openml_to_prov --mode scaled   # ~308 MB, 10,656 runs
  python -m openml_to_prov --mode large    # ~734 MB, 24,912 runs
  python -m openml_to_prov --mode full     # ~2+ GB, 76320 runs

Modes:
  light:   ~2.2 MB (72 runs, CC18 × single classifier)
  scaled:  ~308 MB (10,656 runs, CC18 × 144 configs)
  large:   ~734 MB (24,912 runs, CC18 + extended classification × 144 configs)
  full:    ~2+ GB (76,320 runs, CC18 + extended + regression × 144 configs)

Task Sources:
  - OpenML-CC18: 72 curated classification tasks (official benchmark suite)
  - Extended classification: Additional OpenML classification tasks for scale
  - Regression: OpenML regression tasks for supervised learning coverage
"""

__version__ = "2.0.0"
__author__ = "Nicholas Jones"

from .config import CorpusConfig
from .generator import CorpusGenerator
from .prov_builder import ProvDocumentBuilder

__all__ = [
    "CorpusConfig",
    "CorpusGenerator", 
    "ProvDocumentBuilder",
]
