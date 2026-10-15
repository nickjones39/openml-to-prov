#!/usr/bin/env python3
"""Five-of-ten versus ten-of-ten folds, and placement against OpenML.

Reads a corpus generated with ``--n-folds 10`` and, per run:

  * mean accuracy over folds 1-5 (what a default ``k=5`` corpus records) and
    over folds 1-10 (the full OpenML-CC18 estimation procedure);
  * optionally (``--openml``), the distribution of predictive_accuracy that
    OpenML publishes for the same task from scikit-learn RandomForestClassifier
    flows, and the percentile at which our ten-fold mean falls in it. Those
    published runs use other hyperparameters and library versions, so this is a
    plausibility check, not a reproduction.

Optionally (``--compare-v130``) the first five folds are also matched, fold by
fold, against the per-fold accuracies of a v1.3.0 five-fold corpus.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List


def _v(x):
    return x["$"] if isinstance(x, dict) and "$" in x else x


def fold_scores(path: Path) -> Dict:
    d = json.loads(path.read_text())
    ents = d["entity"]
    task = next(v for k, v in ents.items() if k.startswith("e:task/"))
    scores = {}
    for k, v in ents.items():
        if k.startswith("e:metrics/"):
            scores[int(k.rsplit("_fold", 1)[1])] = float(_v(v["openml:accuracy"]))
    return {"task_id": int(_v(task["openml:task_id"])), "scores": [scores[i] for i in sorted(scores)]}


def openml_rf_accuracies(task_id: int, flow_ids: List[int]) -> List[float]:
    """Published predictive_accuracy for this task from the given flows.

    The server rejects pages above 1000 rows, so pages are fetched manually.
    """
    import openml
    import pandas as pd
    parts, offset = [], 0
    while True:
        for attempt in range(5):
            try:
                df = openml.evaluations.list_evaluations(
                    function="predictive_accuracy", tasks=[task_id], flows=flow_ids,
                    output_format="dataframe", size=1000, offset=offset)
                break
            except Exception as exc:
                if "No results" in str(exc) or "372" in str(exc):
                    df = None
                    break
                if attempt == 4:
                    raise
                time.sleep(5 * 2 ** attempt)
        if df is None or len(df) == 0:
            break
        parts.append(df)
        offset += len(df)
        if len(df) < 1000:
            break
    if not parts:
        return []
    return [float(v) for v in pd.concat(parts)["value"].tolist()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--openml", action="store_true")
    ap.add_argument("--rf-flows", default="repro_r3/openml_rf_flows.json",
                    help="JSON with the OpenML flow ids treated as scikit-learn RandomForest flows")
    ap.add_argument("--compare-v130", default=None)
    args = ap.parse_args()

    v130 = {}
    if args.compare_v130:
        for p in Path(args.compare_v130).rglob("prov_*.json"):
            r = fold_scores(p)
            v130[r["task_id"]] = r["scores"]

    flow_ids = json.loads(Path(args.rf_flows).read_text())["flow_ids"] if args.openml else []
    rows = []
    for p in sorted(Path(args.corpus).rglob("prov_*.json")):
        r = fold_scores(p)
        s = r["scores"]
        row = {"task_id": r["task_id"], "k": len(s),
               "mean_5": statistics.fmean(s[:5]), "mean_10": statistics.fmean(s),
               "sd_10": statistics.pstdev(s)}
        row["diff_5_minus_10"] = row["mean_5"] - row["mean_10"]
        if r["task_id"] in v130:
            row["first5_match_v130"] = sum(round(a, 6) == round(b, 6) for a, b in zip(s[:5], v130[r["task_id"]]))
        if args.openml:
            pub = openml_rf_accuracies(r["task_id"], flow_ids)
            row["openml_rf_runs"] = len(pub)
            if pub:
                row["openml_rf_median"] = statistics.median(pub)
                row["openml_rf_q1"], _, row["openml_rf_q3"] = statistics.quantiles(pub, n=4) if len(pub) > 1 else (pub[0],) * 3
                row["openml_rf_max"] = max(pub)
                row["percentile_in_openml_rf"] = 100.0 * sum(v <= row["mean_10"] for v in pub) / len(pub)
            print(f"task {r['task_id']}: 10-fold {row['mean_10']:.4f}, OpenML RF runs {len(pub)}", flush=True)
        rows.append(row)

    diffs = [r["diff_5_minus_10"] for r in rows]
    absd = [abs(d) for d in diffs]
    summary = {
        "runs": len(rows),
        "mean_acc_10": statistics.fmean(r["mean_10"] for r in rows),
        "mean_acc_5": statistics.fmean(r["mean_5"] for r in rows),
        "mean_diff_5_minus_10": statistics.fmean(diffs),
        "median_abs_diff": statistics.median(absd),
        "max_abs_diff": max(absd),
        "tasks_abs_diff_over_0.01": sum(a > 0.01 for a in absd),
        "tasks_abs_diff_over_0.02": sum(a > 0.02 for a in absd),
    }
    if v130:
        m = [r for r in rows if "first5_match_v130" in r]
        summary["v130_tasks_compared"] = len(m)
        summary["v130_first5_folds_matching"] = sum(r["first5_match_v130"] for r in m)
        summary["v130_first5_folds_total"] = 5 * len(m)
    if args.openml:
        pc = [r["percentile_in_openml_rf"] for r in rows if "percentile_in_openml_rf" in r]
        summary["tasks_with_openml_rf_runs"] = len(pc)
        summary["median_openml_rf_runs_per_task"] = statistics.median(
            [r["openml_rf_runs"] for r in rows if r.get("openml_rf_runs")]) if pc else 0
        summary["median_percentile"] = statistics.median(pc) if pc else None
        summary["tasks_within_openml_rf_iqr"] = sum(
            r["openml_rf_q1"] <= r["mean_10"] <= r["openml_rf_q3"] for r in rows if "openml_rf_q1" in r)
        summary["tasks_above_openml_rf_median"] = sum(
            r["mean_10"] >= r["openml_rf_median"] for r in rows if "openml_rf_median" in r)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
