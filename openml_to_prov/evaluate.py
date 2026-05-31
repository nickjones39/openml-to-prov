"""Baseline downstream evaluation for generated PROV-JSON corpora.

Demonstrates practical value across three usage scenarios named by reviewers:

  1. Compression    - gzip / bz2 / lzma ratios over the raw PROV-JSON corpus
                      (whole corpus; cheap, no optional deps).
  2. Graph-DB load  - PROV-JSON -> PROV-O RDF (via ``prov``) loaded into an
                      in-memory rdflib graph; reports triples and load time
                      (sampled; needs ``prov`` + ``rdflib``).
  3. SPARQL queries - a fixed set of representative PROV-O / provenance
                      queries run over the loaded graph, with result counts
                      and timing (sampled; needs ``rdflib``).

In-memory rdflib is used as the triplestore so the evaluation is reproducible
with no external service: ``pip install rdflib prov`` is the only requirement
for layers 2-3; layer 1 needs nothing beyond the standard library.

Run as a module:
    python -m openml_to_prov.evaluate --corpus prov_light_real --label light
    python -m openml_to_prov.evaluate --corpus prov_full --label full \\
        --sample 500 --outdir eval_out

Outputs (under --outdir, default ``eval_results/``):
    eval_<label>.json        machine-readable results
    eval_<label>_*.tex       LaTeX tables (compression, loading, queries)
    eval_<label>_*.png       plots (compression ratios, query timings)
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Corpus discovery
# ---------------------------------------------------------------------------

def find_prov_files(corpus_dir: Path) -> List[Path]:
    """Return all per-run PROV-JSON files under a corpus directory."""
    return sorted(corpus_dir.rglob("prov_*.json"))


# ---------------------------------------------------------------------------
# Layer 1: compression baselines (whole corpus)
# ---------------------------------------------------------------------------

def compression_baselines(files: List[Path]) -> Dict:
    """Measure raw size and per-codec compressed size over the corpus.

    Concatenates all documents into one byte stream per codec so the result
    reflects cross-file redundancy (the property the corpus is designed to
    exhibit), then reports ratio = raw / compressed.
    """
    raw = bytearray()
    for f in files:
        raw += f.read_bytes()
    raw_bytes = bytes(raw)
    raw_size = len(raw_bytes)

    results = {"raw_bytes": raw_size, "n_files": len(files), "codecs": {}}
    for name, fn in (
        ("gzip", lambda b: gzip.compress(b, compresslevel=9)),
        ("bz2", lambda b: bz2.compress(b, compresslevel=9)),
        ("lzma", lambda b: lzma.compress(b, preset=9)),
    ):
        t0 = time.perf_counter()
        comp = fn(raw_bytes)
        dt = time.perf_counter() - t0
        results["codecs"][name] = {
            "compressed_bytes": len(comp),
            "ratio": round(raw_size / len(comp), 2) if comp else 0.0,
            "space_saving_pct": round(100 * (1 - len(comp) / raw_size), 2) if raw_size else 0.0,
            "seconds": round(dt, 3),
        }
    return results


# ---------------------------------------------------------------------------
# Layer 2/3: RDF loading + SPARQL (sampled)
# ---------------------------------------------------------------------------

def _missing(module: str) -> Optional[str]:
    try:
        __import__(module)
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def _to_rdf_graph(files: List[Path]):
    """Convert and merge a list of PROV-JSON files into one rdflib Graph."""
    import rdflib
    from prov.model import ProvDocument

    g = rdflib.Graph()
    for f in files:
        doc = json.loads(f.read_text())
        d = ProvDocument.deserialize(content=json.dumps(doc), format="json")
        ttl = d.serialize(format="rdf", rdf_format="turtle")
        g.parse(data=ttl, format="turtle")
    return g


# Five representative provenance queries over the PROV-O projection.
# Namespaces match the RDF the ``prov`` library emits for our PROV-JSON:
#   prov:   http://www.w3.org/ns/prov#
#   openml: https://openml.org/def/
_QUERY_PREAMBLE = """
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX openml: <https://openml.org/def/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""

# Note on the type patterns below. When the ``prov`` library serialises our
# PROV-JSON to RDF, the domain-specific ``prov:type`` values (e.g.
# "openml:Experiment") are emitted as rdf:type to a *string literal*, not a
# URI, while the PROV relations (wasInformedBy, wasGeneratedBy, wasDerivedFrom)
# are emitted as proper URIRef edges. We therefore match openml types with
# rdf:type ?t . FILTER(STR(?t) = "openml:..."), which is robust to the literal
# datatype, and traverse relations directly.
QUERIES: List[Tuple[str, str, str]] = [
    (
        "count_runs",
        "Count experiment runs (Experiment activities) in the graph.",
        _QUERY_PREAMBLE + """
        SELECT (COUNT(DISTINCT ?e) AS ?n) WHERE {
            ?e rdf:type ?t . FILTER(STR(?t) = "openml:Experiment")
        }""",
    ),
    (
        "per_fold_accuracies",
        "Retrieve every per-fold accuracy recorded on Metrics entities.",
        _QUERY_PREAMBLE + """
        SELECT ?m ?acc WHERE {
            ?m openml:accuracy ?acc .
        }""",
    ),
    (
        "train_predict_chain",
        "Find Predict activities informed by a Train activity (workflow ordering).",
        _QUERY_PREAMBLE + """
        SELECT ?predict ?train WHERE {
            ?predict rdf:type ?tp . FILTER(STR(?tp) = "openml:Predict")
            ?predict prov:wasInformedBy ?train .
            ?train rdf:type ?tt . FILTER(STR(?tt) = "openml:Train")
        }""",
    ),
    (
        "model_lineage",
        "Trace each trained Model back to the Train activity that generated it.",
        _QUERY_PREAMBLE + """
        SELECT ?model ?act WHERE {
            ?model rdf:type ?tm . FILTER(STR(?tm) = "openml:Model")
            ?model prov:wasGeneratedBy ?act .
        }""",
    ),
    (
        "aggregate_derivation",
        "Find AggregateMetrics and the per-fold Metrics they were derived from.",
        _QUERY_PREAMBLE + """
        SELECT ?agg ?fold WHERE {
            ?agg rdf:type ?ta . FILTER(STR(?ta) = "openml:AggregateMetrics")
            ?agg prov:wasDerivedFrom ?fold .
        }""",
    ),
]


def rdf_and_sparql(files: List[Path], sample: Optional[int], seed: int = 42) -> Dict:
    """Load a sample of the corpus into rdflib and run the query suite."""
    miss = _missing("rdflib") or _missing("prov")
    if miss:
        return {"skipped": True, "reason": f"rdflib/prov not installed ({miss})"}

    sampled = list(files)
    if sample is not None and sample < len(sampled):
        rng = random.Random(seed)
        sampled = rng.sample(sampled, sample)

    t0 = time.perf_counter()
    g = _to_rdf_graph(sampled)
    load_s = time.perf_counter() - t0
    n_triples = len(g)

    loading = {
        "skipped": False,
        "files_loaded": len(sampled),
        "files_total": len(files),
        "triples": n_triples,
        "load_seconds": round(load_s, 3),
        "triples_per_file": round(n_triples / len(sampled), 1) if sampled else 0,
        "load_seconds_per_file": round(load_s / len(sampled), 4) if sampled else 0,
    }

    queries = []
    for qid, desc, q in QUERIES:
        t0 = time.perf_counter()
        rows = list(g.query(q))
        dt = time.perf_counter() - t0
        # Surface a scalar for COUNT queries; otherwise the row count.
        result_value = None
        if len(rows) == 1 and len(rows[0]) == 1:
            try:
                result_value = int(rows[0][0])
            except (TypeError, ValueError):
                result_value = str(rows[0][0])
        queries.append({
            "id": qid,
            "description": desc,
            "result_rows": len(rows),
            "result_value": result_value,
            "seconds": round(dt, 4),
        })

    return {"loading": loading, "queries": queries}


# ---------------------------------------------------------------------------
# LaTeX + plot emitters
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_latex(results: Dict, label: str, outdir: Path) -> List[Path]:
    paths = []

    # Compression table
    comp = results["compression"]
    lines = [
        r"\begin{table}[pos=tbp]", r"  \centering",
        r"  \begin{tabular}{|l|r|r|r|}",
        r"    \hline",
        r"    \textbf{Codec} & \textbf{Size (MB)} & \textbf{Ratio} & \textbf{Saving (\%)} \\",
        r"    \hline",
        f"    Raw (uncompressed) & {comp['raw_bytes']/1e6:.2f} & 1.00 & 0.00 \\\\\\hline",
    ]
    for name, c in comp["codecs"].items():
        lines.append(
            f"    {name} & {c['compressed_bytes']/1e6:.2f} & "
            f"{c['ratio']:.2f} & {c['space_saving_pct']:.2f} \\\\\\hline"
        )
    lines += [
        r"  \end{tabular}",
        f"  \\caption{{Compression baselines over the \\texttt{{{label}}} corpus "
        f"({comp['n_files']:,} runs, {comp['raw_bytes']/1e6:.1f}~MB raw). "
        r"Ratio is raw/compressed; higher is better.}",
        f"  \\label{{tab:eval_compression_{label}}}",
        r"\end{table}",
    ]
    p = outdir / f"eval_{label}_compression.tex"
    p.write_text("\n".join(lines) + "\n")
    paths.append(p)

    # Query table (if not skipped)
    rs = results["rdf_sparql"]
    if not rs.get("skipped"):
        ld = rs["loading"]
        qlines = [
            r"\begin{table}[pos=tbp]", r"  \centering",
            r"  \begin{tabular}{|l|r|r|}",
            r"    \hline",
            r"    \textbf{SPARQL query} & \textbf{Rows} & \textbf{Time (ms)} \\",
            r"    \hline",
        ]
        for q in rs["queries"]:
            val = f" ({q['result_value']})" if q["result_value"] is not None else ""
            qlines.append(
                f"    {_tex_escape(q['id'])}{val} & {q['result_rows']:,} & "
                f"{q['seconds']*1000:.1f} \\\\\\hline"
            )
        qlines += [
            r"  \end{tabular}",
            f"  \\caption{{Representative PROV-O/SPARQL queries over the "
            f"\\texttt{{{label}}} corpus sample ({ld['files_loaded']:,} of "
            f"{ld['files_total']:,} runs; {ld['triples']:,} triples loaded into "
            f"rdflib in {ld['load_seconds']:.2f}~s). Each query exercises a distinct "
            r"provenance access pattern.}",
            f"  \\label{{tab:eval_sparql_{label}}}",
            r"\end{table}",
        ]
        p = outdir / f"eval_{label}_sparql.tex"
        p.write_text("\n".join(qlines) + "\n")
        paths.append(p)

    return paths


def write_plots(results: Dict, label: str, outdir: Path) -> List[Path]:
    if _missing("matplotlib"):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    # Compression ratio bar chart
    comp = results["compression"]["codecs"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    names = list(comp)
    ratios = [comp[n]["ratio"] for n in names]
    ax.bar(names, ratios, color="#1f5fa8")
    ax.set_ylabel("Compression ratio (raw / compressed)")
    ax.set_title(f"Compression baselines — {label} corpus")
    for i, r in enumerate(ratios):
        ax.text(i, r, f"{r:.1f}\u00d7", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = outdir / f"eval_{label}_compression.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # Query timing bar chart
    rs = results["rdf_sparql"]
    if not rs.get("skipped"):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        qids = [q["id"] for q in rs["queries"]]
        times = [q["seconds"] * 1000 for q in rs["queries"]]
        ax.barh(qids, times, color="#b8860b")
        ax.set_xlabel("Query time (ms)")
        ax.set_title(f"PROV-O/SPARQL query latency — {label} corpus sample")
        ax.invert_yaxis()
        fig.tight_layout()
        p = outdir / f"eval_{label}_query_timing.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(
    corpus_dir: Path,
    label: str,
    sample: Optional[int],
    outdir: Path,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    files = find_prov_files(corpus_dir)
    if not files:
        raise SystemExit(f"No prov_*.json files found under {corpus_dir}")
    outdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Corpus '{label}': {len(files):,} PROV-JSON runs under {corpus_dir}")
        print("  [1/3] Compression baselines (whole corpus)...")
    compression = compression_baselines(files)

    if verbose:
        n = min(sample, len(files)) if sample else len(files)
        print(f"  [2/3] RDF loading + [3/3] SPARQL (sample of {n:,})...")
    rdf_sparql = rdf_and_sparql(files, sample, seed=seed)

    results = {
        "label": label,
        "corpus_dir": str(corpus_dir),
        "n_runs": len(files),
        "compression": compression,
        "rdf_sparql": rdf_sparql,
    }

    json_path = outdir / f"eval_{label}.json"
    json_path.write_text(json.dumps(results, indent=2))
    tex_paths = write_latex(results, label, outdir)
    png_paths = write_plots(results, label, outdir)

    if verbose:
        _print_summary(results)
        print(f"\n  Wrote: {json_path}")
        for p in tex_paths + png_paths:
            print(f"         {p}")
    return results


def _print_summary(results: Dict) -> None:
    c = results["compression"]
    print(f"\n  === Compression ({results['label']}, {c['n_files']:,} runs, "
          f"{c['raw_bytes']/1e6:.1f} MB raw) ===")
    for name, cc in c["codecs"].items():
        print(f"    {name:5s}: {cc['compressed_bytes']/1e6:7.2f} MB  "
              f"ratio {cc['ratio']:6.2f}x  ({cc['space_saving_pct']:.1f}% saved)")

    rs = results["rdf_sparql"]
    if rs.get("skipped"):
        print(f"\n  === RDF/SPARQL skipped: {rs['reason']} ===")
        return
    ld = rs["loading"]
    print(f"\n  === RDF loading ({ld['files_loaded']:,}/{ld['files_total']:,} runs) ===")
    print(f"    {ld['triples']:,} triples in {ld['load_seconds']:.2f}s "
          f"({ld['triples_per_file']:.0f} triples/run)")
    print(f"\n  === SPARQL queries ===")
    for q in rs["queries"]:
        val = f" -> {q['result_value']}" if q["result_value"] is not None else f" -> {q['result_rows']} rows"
        print(f"    {q['id']:22s}{val}  ({q['seconds']*1000:.1f} ms)")


def main():
    ap = argparse.ArgumentParser(
        description="Baseline downstream evaluation (compression + RDF loading + SPARQL) for a PROV-JSON corpus."
    )
    ap.add_argument("--corpus", required=True, help="Path to a generated corpus directory")
    ap.add_argument("--label", default=None, help="Label for outputs (default: corpus dir name)")
    ap.add_argument("--sample", type=int, default=None,
                    help="Sample N runs for the RDF/SPARQL layers (default: all). "
                         "Recommended for large corpora, e.g. --sample 500.")
    ap.add_argument("--outdir", default="eval_results", help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    label = args.label or corpus_dir.name
    run_evaluation(
        corpus_dir=corpus_dir,
        label=label,
        sample=args.sample,
        outdir=Path(args.outdir),
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
