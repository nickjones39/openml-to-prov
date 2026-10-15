#!/usr/bin/env python3
"""Scalability measurements beyond corpus size.

Three measurements, each written as JSON:

  gen      Wall time and peak resident memory of synthetic corpus generation
           for one mode, measured around a child ``python -m openml_to_prov``
           process (``--package-root`` selects which checkout is timed).

  rdflib   RDF loading and SPARQL latency in an in-memory RDFLib store at
           increasing corpus sizes. Each size runs in a fresh process so peak
           memory is per size. Samples are nested prefixes of one seeded
           permutation of the corpus, so a larger sample contains every smaller
           one. Uses the same PROV-JSON -> PROV-O conversion and the same five
           queries as ``openml_to_prov.evaluate``.

  oxigraph The whole corpus loaded incrementally into an indexed, on-disk
           Oxigraph store, with the five queries timed at checkpoints along the
           way, up to and including all documents.

Every cost is recorded twice: as CPU time (user + system, of the measured
process or of the generation child) and as wall time. CPU time is the primary
figure because it counts only the time the work actually ran on a core, so it
is largely insensitive to other jobs on the machine; wall time is kept for
transparency. Query latencies are the minimum over ``--repeats`` executions,
since interference can only lengthen a run. The load average before and after
each measurement is recorded so the conditions travel with the figures.

Setting ``SCALING_IDLE_LOAD`` makes each measurement first wait (up to
``SCALING_IDLE_MAX_WAIT`` seconds) for the one-minute load average to fall
below that value; by default no waiting is done.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def wait_for_idle() -> Dict:
    """Record the machine's load; optionally wait for it to fall first."""
    limit = os.environ.get("SCALING_IDLE_LOAD")
    t0 = time.time()
    if limit is not None:
        max_wait = float(os.environ.get("SCALING_IDLE_MAX_WAIT", 3 * 3600))
        while os.getloadavg()[0] > float(limit) and time.time() - t0 < max_wait:
            time.sleep(30)
    return {"idle_load_limit": float(limit) if limit else None,
            "waited_seconds": round(time.time() - t0),
            "logical_cpus": os.cpu_count(),
            "loadavg_before": [round(x, 2) for x in os.getloadavg()]}


def child_cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def peak_rss_bytes(children: bool = False) -> int:
    r = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF)
    # macOS reports bytes, Linux kilobytes.
    return r.ru_maxrss if sys.platform == "darwin" else r.ru_maxrss * 1024


def permutation(corpus: str, seed: int) -> List[str]:
    files = sorted(str(p) for p in Path(corpus).rglob("prov_*.json"))
    random.Random(seed).shuffle(files)
    return files


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# --------------------------------------------------------------------------
def cmd_gen(args):
    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    cmd = [sys.executable, "-m", "openml_to_prov", "--mode", args.mode, "--output", str(out), "-q"]
    conditions = wait_for_idle()
    c0 = child_cpu_seconds()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=args.package_root, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    cpu = child_cpu_seconds() - c0
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-2000:])
    files = list(out.rglob("prov_*.json"))
    manifest = json.loads((out / "corpus_manifest.json").read_text())
    res = {
        "mode": args.mode,
        "package_root": args.package_root,
        "runs": len(files),
        "bytes_prov_json": sum(f.stat().st_size for f in files),
        "cpu_seconds": round(cpu, 2),
        "cpu_ms_per_run": round(1000 * cpu / max(len(files), 1), 3),
        "wall_seconds": round(wall, 2),
        "wall_ms_per_run": round(1000 * wall / max(len(files), 1), 3),
        "peak_rss_bytes": peak_rss_bytes(children=True),
        "manifest_total_bytes": manifest.get("stats", {}).get("total_bytes"),
        "conditions": {**conditions, "loadavg_after": [round(x, 2) for x in os.getloadavg()]},
    }
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    if not args.keep:
        shutil.rmtree(out)


# --------------------------------------------------------------------------
def _timed_queries(run_query, queries, repeats) -> List[Dict]:
    out = []
    for qid, _, q in queries:
        cpus, walls, rows = [], [], None
        for _ in range(repeats):
            c0, t0 = time.process_time(), time.perf_counter()
            rows = run_query(q)
            walls.append(time.perf_counter() - t0)
            cpus.append(time.process_time() - c0)
        out.append({"id": qid, "rows": rows,
                    "cpu_seconds_min": round(min(cpus), 4),
                    "wall_seconds_min": round(min(walls), 4),
                    "cpu_seconds_all": [round(t, 4) for t in cpus],
                    "wall_seconds_all": [round(t, 4) for t in walls]})
    return out


def cmd_rdflib_one(args):
    from openml_to_prov.evaluate import QUERIES, _to_rdf_graph
    files = [Path(f) for f in permutation(args.corpus, args.seed)[: args.n]]
    conditions = wait_for_idle()
    c0, t0 = time.process_time(), time.perf_counter()
    g = _to_rdf_graph(files)
    load = time.perf_counter() - t0
    load_cpu = time.process_time() - c0
    res = {"store": "rdflib-memory", "runs": len(files), "triples": len(g),
           "load_cpu_seconds": round(load_cpu, 2), "load_cpu_ms_per_run": round(1000 * load_cpu / len(files), 2),
           "load_wall_seconds": round(load, 2)}
    res["queries"] = _timed_queries(lambda q: len(list(g.query(q))), QUERIES, args.repeats)
    res["peak_rss_bytes"] = peak_rss_bytes()
    res["conditions"] = {**conditions, "loadavg_after": [round(x, 2) for x in os.getloadavg()]}
    print(json.dumps(res))


def cmd_rdflib(args):
    results = []
    for n in args.sizes:
        cmd = [sys.executable, __file__, "rdflib-one", "--corpus", args.corpus, "--n", str(n),
               "--seed", str(args.seed), "--repeats", str(args.repeats)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            results.append({"runs": n, "error": proc.stderr[-1500:]})
            print(f"rdflib n={n}: ERROR", flush=True)
            break
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        results.append(r)
        print(f"rdflib n={n}: {r['triples']} triples, load cpu {r['load_cpu_seconds']}s, "
              f"rss {r['peak_rss_bytes']/1e9:.2f} GB", flush=True)
        Path(args.out).write_text(json.dumps({"seed": args.seed, "results": results}, indent=1))


# --------------------------------------------------------------------------
def _to_ntriples(path: str):
    """Convert one document; return (N-Triples, CPU seconds spent converting)."""
    from prov.model import ProvDocument
    c0 = time.process_time()
    d = ProvDocument.deserialize(content=Path(path).read_text(), format="json")
    nt = d.serialize(format="rdf", rdf_format="nt")
    return nt, time.process_time() - c0


def cmd_oxigraph(args):
    import pyoxigraph as ox
    from openml_to_prov.evaluate import QUERIES

    files = permutation(args.corpus, args.seed)
    store_dir = Path(args.store)
    if store_dir.exists():
        shutil.rmtree(store_dir)
    store = ox.Store(str(store_dir))
    checkpoints = sorted({c for c in args.checkpoints if c < len(files)} | {len(files)})

    def run_query(q):
        return sum(1 for _ in store.query(q))

    results, loaded, convert_s, load_s = [], 0, 0.0, 0.0
    convert_cpu, load_cpu = 0.0, 0.0
    start_conditions = wait_for_idle()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for cp in checkpoints:
            batch = files[loaded:cp]
            for i in range(0, len(batch), args.batch):
                chunk = batch[i:i + args.batch]
                t0 = time.perf_counter()
                out = list(pool.map(_to_ntriples, chunk, chunksize=32))
                t1 = time.perf_counter()
                c1 = time.process_time()
                store.bulk_load("".join(nt for nt, _ in out).encode(), format=ox.RdfFormat.N_TRIPLES)
                load_cpu += time.process_time() - c1
                t2 = time.perf_counter()
                convert_cpu += sum(c for _, c in out)
                convert_s += t1 - t0
                load_s += t2 - t1
            loaded = cp
            store.optimize() if hasattr(store, "optimize") else None
            r = {"runs": loaded, "triples": len(store),
                 "cumulative_convert_cpu_seconds": round(convert_cpu, 1),
                 "cumulative_load_cpu_seconds": round(load_cpu, 1),
                 "cumulative_convert_wall_seconds": round(convert_s, 1),
                 "cumulative_load_wall_seconds": round(load_s, 1),
                 "store_bytes_on_disk": dir_bytes(store_dir),
                 "peak_rss_bytes": peak_rss_bytes()}
            r["query_conditions"] = wait_for_idle()
            r["queries"] = _timed_queries(run_query, QUERIES, args.repeats)
            r["query_conditions"]["loadavg_after"] = [round(x, 2) for x in os.getloadavg()]
            results.append(r)
            print(f"oxigraph n={loaded}: {r['triples']} triples, load cpu {load_cpu:.0f}s, "
                  + ", ".join(f"{q['id']}={q['cpu_seconds_min']}s" for q in r["queries"]), flush=True)
            Path(args.out).write_text(json.dumps({"seed": args.seed, "workers": args.workers,
                                                  "start_conditions": start_conditions,
                                                  "results": results}, indent=1))
    if not args.keep:
        shutil.rmtree(store_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--mode", required=True)
    g.add_argument("--package-root", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--keep", action="store_true")

    for name in ("rdflib", "rdflib-one"):
        r = sub.add_parser(name)
        r.add_argument("--corpus", required=True)
        r.add_argument("--seed", type=int, default=42)
        r.add_argument("--repeats", type=int, default=3)
        if name == "rdflib":
            r.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000])
            r.add_argument("--out", required=True)
        else:
            r.add_argument("--n", type=int, required=True)

    o = sub.add_parser("oxigraph")
    o.add_argument("--corpus", required=True)
    o.add_argument("--store", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--seed", type=int, default=42)
    o.add_argument("--repeats", type=int, default=3)
    o.add_argument("--workers", type=int, default=8)
    o.add_argument("--batch", type=int, default=2000)
    o.add_argument("--checkpoints", type=int, nargs="+",
                   default=[1000, 2000, 5000, 10000, 20000, 40000])
    o.add_argument("--keep", action="store_true")

    args = ap.parse_args()
    {"gen": cmd_gen, "rdflib": cmd_rdflib, "rdflib-one": cmd_rdflib_one,
     "oxigraph": cmd_oxigraph}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
