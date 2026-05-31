#!/usr/bin/env python3
"""
CLI entry point for openml_to_prov.

Usage:
  python -m openml_to_prov --mode light    # ~2.1 MB, 72 runs
  python -m openml_to_prov --mode scaled   # ~301 MB, 10,368 runs
  python -m openml_to_prov --mode large    # ~714 MB, 24,624 runs
  python -m openml_to_prov --mode full     # ~2.2 GB, 76,032 runs
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
  python -m openml_to_prov --mode light    # ~2.1 MB (72 runs)
  python -m openml_to_prov --mode scaled   # ~301 MB (10,368 runs, CC18)
  python -m openml_to_prov --mode large    # ~714 MB (24,624 runs, CC18 + extended)
  python -m openml_to_prov --mode full     # ~2.2 GB (76,032 runs, CC18 + extended + regression)

Corpus sizes:
  light:   72 tasks × 1 config = 72 runs (~2.1 MB)
  scaled:  72 tasks × 144 configs = 10,368 runs (~301 MB)
  large:   171 tasks × 144 configs = 24,624 runs (~714 MB)
  full:    528 tasks × 144 configs = 76,032 runs (~2.2 GB)

Task sources:
  CC18:       72 curated classification tasks (OpenML-CC18 suite 99)
  Extended:   99 (large) / 179 (full) additional OpenML classification tasks
  Regression: 277 OpenML regression tasks (full mode only)
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
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Skip the runtime confirmation prompt shown for --real with non-light modes.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Validate each PROV-JSON document for conformance before writing it "
            "to the corpus, and emit conformance_report.json. SHACL and PROV-O "
            "round-trip checks need optional deps (pyshacl, rdflib, prov); they "
            "are skipped gracefully if absent."
        ),
    )
    parser.add_argument(
        "--validation-checks",
        default="schema,shacl,round_trip",
        help=(
            "Comma-separated checks to run with --validate. "
            "Options: schema, shacl, round_trip. "
            "Default: schema,shacl,round_trip. For fast CI use 'schema'."
        ),
    )
    parser.add_argument(
        "--validation-policy",
        choices=["strict", "warn", "abort"],
        default="strict",
        help=(
            "On validation failure: 'strict' skips and logs the doc (default), "
            "'warn' writes it anyway but flags it, 'abort' stops the run."
        ),
    )
    parser.add_argument(
        "--no-cardinality-check",
        action="store_true",
        help=(
            "Disable the 44-node/128-edge template assertion in the schema "
            "check (use when extending the EAA mapping, e.g. 10-fold CV)."
        ),
    )

    args = parser.parse_args()

    if args.real and args.mode != "light":
        mode_runs = {"scaled": 10368, "large": 24624, "full": 76032}
        # Measured baseline: light mode (72 runs) ~45 min on M-series MacBook Pro.
        # Linear extrapolation: ~37s per run before classifier-cost growth.
        sec_per_run = 37
        n = mode_runs.get(args.mode, 0)
        est_hours = n * sec_per_run / 3600
        est_days = est_hours / 24
        print("\n" + "=" * 70)
        print(f"  WARNING: --real with --mode {args.mode}")
        print("=" * 70)
        print(f"  This will execute {n:,} real sklearn runs against OpenML.")
        print(f"  Estimated wall-clock: ~{est_hours:.0f} hours (~{est_days:.1f} days)")
        print(f"  Baseline: light mode (72 runs) took ~45 min on M-series MacBook Pro.")
        print(f"  Larger modes also use heavier classifiers (SVM, MLP) on bigger")
        print(f"  datasets, so real time may be significantly longer than the linear")
        print(f"  estimate above.")
        print()
        print(f"  Strongly recommended: use --max-tasks N to validate a sample first.")
        print("=" * 70)
        if not args.skip_confirm:
            response = input("\n  Continue anyway? [y/N]: ").strip().lower()
            if response != "y":
                print("  Aborted.")
                return

    config = CorpusConfig(
        mode=args.mode,
        output_dir=args.output,
        verbose=not args.quiet,
        pretty_print=not args.compact,
        use_real_execution=args.real,
        validate=args.validate,
        validation_checks=tuple(
            c.strip() for c in args.validation_checks.split(",") if c.strip()
        ),
        validation_policy=args.validation_policy,
        validation_cardinality=not args.no_cardinality_check,
    )

    generator = CorpusGenerator(config)
    return generator.generate(max_tasks=args.max_tasks)


if __name__ == "__main__":
    main()
