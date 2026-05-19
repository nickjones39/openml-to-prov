#!/usr/bin/env python3
"""
CLI entry point for openml_to_prov.

Usage:
  python -m openml_to_prov --mode light    # ~2.2 MB, 72 runs
  python -m openml_to_prov --mode scaled   # ~308 MB, 10,656 runs
  python -m openml_to_prov --mode large    # ~734 MB, 24,912 runs
  python -m openml_to_prov --mode full     # ~2+ GB, 76,320 runs
"""

import argparse
from .config import CorpusConfig
from .generator import CorpusGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate W3C PROV corpus for OpenML benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m openml_to_prov --mode light    # ~2.2 MB (72 runs)
  python -m openml_to_prov --mode scaled   # ~308 MB (10,656 runs, CC18)
  python -m openml_to_prov --mode large    # ~734 MB (24,912 runs, CC18 + extended)
  python -m openml_to_prov --mode full     # ~2+ GB (76,320 runs, CC18 + extended + regression)

Corpus sizes:
  light:   72 tasks × 1 config = 72 runs (~2.2 MB)
  scaled:  72 tasks × 144 configs = 10,656 runs (~308 MB)
  large:   172 tasks × 144 configs = 24,912 runs (~734 MB)
  full:    422 tasks × 144 configs = 76,320 runs (~2+ GB)

Task sources:
  CC18:       72 curated classification tasks (OpenML-CC18 benchmark suite)
  Extended:   100 additional OpenML classification tasks
  Regression: 250 OpenML regression tasks
        """
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["light", "scaled", "large", "full"],
        default="full",
        help="Corpus mode (default: full)"
    )
    parser.add_argument(
        "--output", "-o",
        default="prov_corpus",
        help="Output directory (default: prov_corpus)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum tasks to process (for testing)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact JSON output (smaller files)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help=(
            "Run real OpenML + sklearn execution (downloads datasets, trains classifiers). "
            "Works for all modes; combine with --max-tasks to limit scope for larger modes."
        ),
    )

    args = parser.parse_args()

    if args.real and args.mode != "light":
        from .config import CC18_TASK_IDS, EXTENDED_CLASSIFICATION_TASK_IDS, REGRESSION_TASK_IDS
        mode_runs = {"scaled": 10656, "large": 24912, "full": 76320}
        n = mode_runs.get(args.mode, "?")
        print(f"NOTE: --real with --mode {args.mode} will execute {n:,} real sklearn runs. "
              f"Use --max-tasks N to limit scope.")

    config = CorpusConfig(
        mode=args.mode,
        output_dir=args.output,
        verbose=not args.quiet,
        pretty_print=not args.compact,
        use_real_execution=args.real,
    )

    generator = CorpusGenerator(config)
    return generator.generate(max_tasks=args.max_tasks)


if __name__ == "__main__":
    main()
