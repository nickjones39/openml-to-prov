#!/usr/bin/env python3
"""Semantic validation of openml-to-prov PROV-JSON documents.

The conformance pipeline (schema, SHACL, PROV-O round-trip) establishes that a
document is well-formed, standards-compliant PROV. It does not establish that
the document *means* what the published mapping says it means. This tool checks
three further properties, exhaustively over a corpus:

  A. Mapping conformance. Every relation in the document is classified by the
     classes of its endpoints (e.g. ``used: a:train -> e:split``) and the
     resulting multiset is compared, exactly, against the published mapping
     (``MAPPING`` below) instantiated for the document's own fold count k.
     A missing or an unexpected relation pattern fails the document.

  B. PROV-CONSTRAINTS (W3C Recommendation, 2013), a subset:
       - Constraint 50 (typing) for every relation present;
       - Constraint 55 (entity-activity-disjoint);
       - Constraint 24 (unique-generation), checked in the stronger form that
         every generated entity has exactly one generating activity;
       - acyclicity of wasDerivedFrom, which Constraint 42
         (derivation-generation-generation-ordering, *strict* precedence)
         requires of any valid document;
     and, for recorded times, consistency with the event orderings that
     Constraints 30, 33/34, 35, 37 and 42 impose. PROV-CONSTRAINTS defines
     validity over the ordering of events, not over timestamps, so the time
     checks are reported separately: they test whether the recorded times
     *contradict* an ordering the constraints require. At whole-second
     resolution only a reversed ordering is detectable, so non-strict
     comparisons are used throughout.

  C. Value consistency. Attribute values that the mapping relates must agree:
     the flow hash and parameter checksum must be the SHA-256 of the recorded
     parameters; aggregate mean/std must equal the mean/std of the per-fold
     metrics; per-fold train+test sizes must partition the dataset; metric
     sample counts must equal the split's test size; fold numbers must be 1..k.

Usage:
    python tools/semantic_check.py --corpus prov_light_corpus --out sem_light.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

RELATIONS = {
    # relation: (subject key, object key, subject kind, object kind) -- Constraint 50
    "used": ("prov:activity", "prov:entity", "activity", "entity"),
    "wasGeneratedBy": ("prov:entity", "prov:activity", "entity", "activity"),
    "wasInformedBy": ("prov:informed", "prov:informant", "activity", "activity"),
    "wasDerivedFrom": ("prov:generatedEntity", "prov:usedEntity", "entity", "entity"),
    "wasAssociatedWith": ("prov:activity", "prov:agent", "activity", "agent"),
    "wasAttributedTo": ("prov:entity", "prov:agent", "entity", "agent"),
}

# The published Entity--Activity--Agent mapping, as relation patterns with
# their per-document multiplicity as a function of the fold count k.
MAPPING = {
    ("used", "a:experiment", "e:dataset"): lambda k: 1,
    ("used", "a:experiment", "e:task"): lambda k: 1,
    ("used", "a:experiment", "e:flow"): lambda k: 1,
    ("used", "a:experiment", "e:flowparams"): lambda k: 1,
    ("used", "a:experiment", "e:env"): lambda k: 1,
    ("used", "a:train", "e:dataset"): lambda k: k,
    ("used", "a:train", "e:split"): lambda k: k,
    ("used", "a:train", "e:flow"): lambda k: k,
    ("used", "a:train", "e:flowparams"): lambda k: k,
    ("used", "a:train", "e:env"): lambda k: k,
    ("used", "a:predict", "e:model"): lambda k: k,
    ("used", "a:predict", "e:dataset"): lambda k: k,
    ("used", "a:predict", "e:split"): lambda k: k,
    ("used", "a:evaluate", "e:predictions"): lambda k: k,
    ("used", "a:evaluate", "e:split"): lambda k: k,
    ("used", "a:aggregate", "e:metrics"): lambda k: k,
    ("wasGeneratedBy", "e:model", "a:train"): lambda k: k,
    ("wasGeneratedBy", "e:predictions", "a:predict"): lambda k: k,
    ("wasGeneratedBy", "e:metrics", "a:evaluate"): lambda k: k,
    ("wasGeneratedBy", "e:metrics_agg", "a:aggregate"): lambda k: 1,
    ("wasInformedBy", "a:train", "a:experiment"): lambda k: k,
    ("wasInformedBy", "a:predict", "a:train"): lambda k: k,
    ("wasInformedBy", "a:evaluate", "a:predict"): lambda k: k,
    ("wasDerivedFrom", "e:metrics", "e:predictions"): lambda k: k,
    ("wasDerivedFrom", "e:metrics", "e:split"): lambda k: k,
    ("wasDerivedFrom", "e:metrics_agg", "e:metrics"): lambda k: k,
    ("wasAssociatedWith", "a:experiment", "ag:system"): lambda k: 1,
    ("wasAssociatedWith", "a:train", "ag:system"): lambda k: k,
    ("wasAssociatedWith", "a:predict", "ag:system"): lambda k: k,
    ("wasAssociatedWith", "a:evaluate", "ag:system"): lambda k: k,
    ("wasAssociatedWith", "a:aggregate", "ag:system"): lambda k: 1,
    ("wasAttributedTo", "e:model", "ag:system"): lambda k: k,
}

NODE_CLASSES = {
    "e:dataset": lambda k: 1, "e:task": lambda k: 1, "e:flow": lambda k: 1,
    "e:flowparams": lambda k: 1, "e:env": lambda k: 1, "e:split": lambda k: k,
    "e:model": lambda k: k, "e:predictions": lambda k: k, "e:metrics": lambda k: k,
    "e:metrics_agg": lambda k: 1, "a:experiment": lambda k: 1, "a:train": lambda k: k,
    "a:predict": lambda k: k, "a:evaluate": lambda k: k, "a:aggregate": lambda k: 1,
    "ag:system": lambda k: 1,
}


def cls(node_id: str) -> str:
    return node_id.split("/", 1)[0]


def _v(x):
    return x["$"] if isinstance(x, dict) and "$" in x else x


def _t(s):
    return datetime.strptime(_v(s), "%Y-%m-%dT%H:%M:%SZ") if s else None


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def check_document(path: str) -> Dict:
    doc = json.loads(Path(path).read_text())
    ents, acts, agents = doc.get("entity", {}), doc.get("activity", {}), doc.get("agent", {})
    kind = {}
    for ids, kname in ((ents, "entity"), (acts, "activity"), (agents, "agent")):
        for i in ids:
            kind.setdefault(i, set()).add(kname)

    fails: Dict[str, List[str]] = defaultdict(list)
    na: Counter = Counter()

    # ---- C55 entity-activity-disjoint (and agent overlap) -------------------
    for i, ks in kind.items():
        if "entity" in ks and "activity" in ks:
            fails["C55_entity_activity_disjoint"].append(i)

    edges: List[Tuple[str, str, str]] = []
    for rel, (sk, ok, skind, okind) in RELATIONS.items():
        for _, r in doc.get(rel, {}).items():
            s, o = _v(r.get(sk)), _v(r.get(ok))
            edges.append((rel, s, o))
            # ---- C50 typing -----------------------------------------------
            if skind not in kind.get(s, set()) or okind not in kind.get(o, set()):
                fails["C50_typing"].append(f"{rel}({s},{o})")

    k = sum(1 for e in ents if cls(e) == "e:split")

    # ---- A. mapping conformance --------------------------------------------
    got = Counter((rel, cls(s), cls(o)) for rel, s, o in edges)
    want = {p: f(k) for p, f in MAPPING.items()}
    for p in set(got) | set(want):
        if got.get(p, 0) != want.get(p, 0):
            fails["A_mapping_relations"].append(f"{p}: got {got.get(p, 0)} want {want.get(p, 0)}")
    node_got = Counter(cls(i) for i in kind)
    for c in set(node_got) | set(NODE_CLASSES):
        w = NODE_CLASSES[c](k) if c in NODE_CLASSES else 0
        if node_got.get(c, 0) != w:
            fails["A_mapping_nodes"].append(f"{c}: got {node_got.get(c, 0)} want {w}")

    # ---- C24 unique generation (strong form) + derivation acyclicity ---------
    gen_by = defaultdict(list)
    for rel, s, o in edges:
        if rel == "wasGeneratedBy":
            gen_by[s].append(o)
    for e, a in gen_by.items():
        if len(a) != 1:
            fails["C24_unique_generation"].append(f"{e}: {a}")
    deriv = defaultdict(list)
    for rel, s, o in edges:
        if rel == "wasDerivedFrom":
            deriv[s].append(o)
    state = {}

    def cyclic(n) -> bool:
        state[n] = 1
        for m in deriv.get(n, []):
            if state.get(m) == 1 or (state.get(m) is None and cyclic(m)):
                return True
        state[n] = 2
        return False

    if any(state.get(n) is None and cyclic(n) for n in list(deriv)):
        fails["C42_derivation_acyclic"].append("cycle in wasDerivedFrom")

    # ---- B. recorded times vs. event ordering -------------------------------
    start = {a: _t(v.get("prov:startTime")) for a, v in acts.items()}
    end = {a: _t(v.get("prov:endTime")) for a, v in acts.items()}
    untimed = [a for a in acts if start[a] is None or end[a] is None]
    for a in untimed:
        na[f"untimed_activity:{cls(a)}"] += 1

    def before(x, y):  # None when either time is unrecorded
        return None if x is None or y is None else x <= y

    for a in acts:
        r = before(start[a], end[a])
        if r is False:
            fails["T30_start_precedes_end"].append(a)
    for rel, s, o in edges:
        if rel == "wasInformedBy":           # C35: start(informant) <= end(informed)
            r = before(start.get(o), end.get(s))
            if r is False:
                fails["T35_wasInformedBy_ordering"].append(f"{s} informedBy {o}")
            elif r is None:
                na["T35_untimed"] += 1
        elif rel == "used" and s in acts and o in gen_by:   # C37 via C33/C34
            ga = gen_by[o][0]
            r = before(start.get(ga), end.get(s))
            if r is False:
                fails["T37_generation_precedes_usage"].append(f"{o}: gen by {ga} after use by {s}")
            elif r is None:
                na["T37_untimed"] += 1
        elif rel == "wasDerivedFrom" and s in gen_by and o in gen_by:   # C42 via C34
            g2, g1 = gen_by[s][0], gen_by[o][0]
            r = before(start.get(g1), end.get(g2))
            if r is False:
                fails["T42_derivation_generation_ordering"].append(f"{s} from {o}")
            elif r is None:
                na["T42_untimed"] += 1

    # ---- C. value consistency -----------------------------------------------
    def one(prefix):
        hits = [v for i, v in ents.items() if cls(i) == prefix]
        return hits[0] if len(hits) == 1 else None

    params, flow, ds, task, agg = (one("e:flowparams"), one("e:flow"), one("e:dataset"),
                                   one("e:task"), one("e:metrics_agg"))
    if params and flow:
        p = _v(params.get("openml:parameters"))
        if _v(params.get("checksum:sha256")) != sha(p):
            fails["V_params_checksum"].append("checksum != sha256(parameters)")
        if _v(flow.get("openml:flow_hash")) != sha(p)[:16]:
            fails["V_flow_hash"].append("flow_hash != sha256(parameters)[:16]")
    regression = "Regression" in str(_v(task.get("openml:task_type", ""))) if task else False
    mkey = "openml:r2_score" if regression else "openml:accuracy"
    if ds and task and str(_v(ds.get("ml:task_type"))).lower() not in str(_v(task.get("openml:task_type"))).lower():
        fails["V_task_type_consistent"].append("dataset/task task_type disagree")

    splits = {i.rsplit("_fold", 1)[1]: v for i, v in ents.items() if cls(i) == "e:split"}
    numbers = sorted(int(_v(v["fold:number"])) for v in splits.values())
    if numbers != list(range(1, k + 1)):
        fails["V_fold_numbers"].append(str(numbers))
    n = int(_v(ds["ml:n_samples"])) if ds else None
    scores = []
    for i, v in ents.items():
        if cls(i) == "e:split" and n is not None:
            if int(_v(v["fold:train_size"])) + int(_v(v["fold:test_size"])) != n:
                fails["V_split_partitions_dataset"].append(i)
        if cls(i) == "e:metrics":
            f = i.rsplit("_fold", 1)[1]
            if mkey not in v:
                fails["V_metric_matches_task_type"].append(i)
                continue
            scores.append(float(_v(v[mkey])))
            if not regression and not (0.0 <= scores[-1] <= 1.0):
                fails["V_accuracy_range"].append(i)
            sp = splits.get(f)
            if sp and int(_v(v.get("openml:n_samples", -1))) != int(_v(sp["fold:test_size"])):
                fails["V_metric_n_equals_test_size"].append(i)
        if cls(i) == "e:predictions":
            sp = splits.get(i.rsplit("_fold", 1)[1])
            if sp and int(_v(v.get("ml:n_predictions", -1))) != int(_v(sp["fold:test_size"])):
                fails["V_predictions_n_equals_test_size"].append(i)
    if agg and scores:
        mean = sum(scores) / len(scores)
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
        if abs(float(_v(agg[mkey + "_mean"])) - mean) > 1e-6:
            fails["V_aggregate_mean"].append(f"{_v(agg[mkey + '_mean'])} vs {mean}")
        if abs(float(_v(agg[mkey + "_std"])) - std) > 1e-6:
            fails["V_aggregate_std"].append(f"{_v(agg[mkey + '_std'])} vs {std}")
        if int(_v(agg.get("openml:n_folds", -1))) != k:
            fails["V_aggregate_n_folds"].append(str(_v(agg.get("openml:n_folds"))))

    return {"path": path, "k": k, "fails": dict(fails), "na": dict(na)}


CHECKS = [
    "A_mapping_relations", "A_mapping_nodes", "C50_typing", "C55_entity_activity_disjoint",
    "C24_unique_generation", "C42_derivation_acyclic", "T30_start_precedes_end",
    "T35_wasInformedBy_ordering", "T37_generation_precedes_usage", "T42_derivation_generation_ordering",
    "V_params_checksum", "V_flow_hash", "V_task_type_consistent", "V_fold_numbers",
    "V_split_partitions_dataset", "V_metric_matches_task_type", "V_accuracy_range",
    "V_metric_n_equals_test_size", "V_predictions_n_equals_test_size", "V_aggregate_mean",
    "V_aggregate_std", "V_aggregate_n_folds",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    files = [str(p) for p in sorted(Path(args.corpus).rglob("prov_*.json"))]
    docs_failing = Counter()
    edge_fail_counts = Counter()
    examples = defaultdict(list)
    na = Counter()
    ks = Counter()
    any_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(check_document, files, chunksize=256), 1):
            ks[r["k"]] += 1
            na.update(r["na"])
            if r["fails"]:
                any_fail += 1
            for c, items in r["fails"].items():
                docs_failing[c] += 1
                edge_fail_counts[c] += len(items)
                if len(examples[c]) < args.examples:
                    examples[c].append({"path": r["path"], "items": items[:3]})
            if i % 10000 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    summary = {
        "corpus": args.corpus,
        "documents": len(files),
        "fold_counts": dict(ks),
        "documents_passing_all": len(files) - any_fail,
        "checks": {c: {"documents_failing": docs_failing.get(c, 0),
                       "violations": edge_fail_counts.get(c, 0)} for c in CHECKS},
        "not_checkable": dict(na),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "examples": examples}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
