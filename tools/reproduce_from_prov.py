#!/usr/bin/env python3
"""Re-execute benchmark runs from their PROV-JSON records alone, and compare.

This script is deliberately independent of the ``openml_to_prov`` package: it
does not import it, and it does not share its execution code path. Everything
it needs to re-run an experiment is read out of the PROV-JSON document:

  * OpenML task id               <- e:task/*        openml:task_id
  * estimator                    <- e:flow/*        openml:model (or ml:estimator_class)
  * hyperparameters              <- e:flowparams/*  openml:parameters
  * which OpenML split per fold  <- e:split/*       fold:number (or openml:fold_index/openml:repeat)
  * software environment         <- e:env/*         ml:*_version
  * recorded per-fold metric     <- e:metrics/*     openml:accuracy | openml:r2_score
  * recorded prediction digest   <- e:predictions/* checksum:sha256 (content digest from v1.4.0)

Where a document does not record something a re-execution needs, the script
falls back to a documented assumption and reports *which* assumptions it had
to make. For graphs produced by openml-to-prov v1.3.0 there are three:

  1. the preprocessing pipeline is not recorded; the fallback below
     (``FALLBACK_PREPROCESSING``) was transcribed from the v1.3.0 source,
     ``openml_to_prov/executor.py::_build_pipeline``;
  2. ``openml:model`` is a short name ("RandomForest"); the fallback mapping to
     a scikit-learn class is ``FALLBACK_ESTIMATORS`` below;
  3. ``fold:number`` is 1-based and the OpenML repeat is not recorded; the
     fallback reads fold n as OpenML fold n-1 of repeat 0, and the recorded
     train/test sizes are used to check that reading.

Graphs produced by v1.4.0 or later record all three, and the script then uses
the recorded values instead of the fallbacks.

Usage:
    python tools/reproduce_from_prov.py --corpus prov_light_corpus \
        --out repro_light.json [--workers 4] [--cache-dir .cache-repro]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Fallbacks for information a v1.3.0 graph does not carry
# --------------------------------------------------------------------------

FALLBACK_ESTIMATORS = {
    # classification
    "RandomForest": "sklearn.ensemble.RandomForestClassifier",
    "GradientBoosting": "sklearn.ensemble.GradientBoostingClassifier",
    "AdaBoost": "sklearn.ensemble.AdaBoostClassifier",
    "ExtraTrees": "sklearn.ensemble.ExtraTreesClassifier",
    "LogisticRegression": "sklearn.linear_model.LogisticRegression",
    "SVM": "sklearn.svm.SVC",
    "KNN": "sklearn.neighbors.KNeighborsClassifier",
    "MLP": "sklearn.neural_network.MLPClassifier",
    "DecisionTree": "sklearn.tree.DecisionTreeClassifier",
    "NaiveBayes": "sklearn.naive_bayes.GaussianNB",
    "BaggingClassifier": "sklearn.ensemble.BaggingClassifier",
    "HistGradientBoosting": "sklearn.ensemble.HistGradientBoostingClassifier",
    # regression
    "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
    "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
    "AdaBoostRegressor": "sklearn.ensemble.AdaBoostRegressor",
    "ExtraTreesRegressor": "sklearn.ensemble.ExtraTreesRegressor",
    "Ridge": "sklearn.linear_model.Ridge",
    "Lasso": "sklearn.linear_model.Lasso",
    "ElasticNet": "sklearn.linear_model.ElasticNet",
    "SVR": "sklearn.svm.SVR",
    "KNeighborsRegressor": "sklearn.neighbors.KNeighborsRegressor",
    "MLPRegressor": "sklearn.neural_network.MLPRegressor",
    "DecisionTreeRegressor": "sklearn.tree.DecisionTreeRegressor",
    "HistGradientBoostingRegressor": "sklearn.ensemble.HistGradientBoostingRegressor",
}

FALLBACK_PREPROCESSING = {
    "numeric": [{"step": "SimpleImputer", "strategy": "mean"}],
    "categorical": [
        {"step": "SimpleImputer", "strategy": "most_frequent"},
        {"step": "OrdinalEncoder", "handle_unknown": "use_encoded_value", "unknown_value": -1},
    ],
    "column_split": "pandas select_dtypes(include='number') vs the rest",
    "remainder": "drop",
}

VERSION_KEYS = {
    "ml:python_version": "python",
    "ml:sklearn_version": "sklearn",
    "ml:numpy_version": "numpy",
    "ml:openml_version": "openml",
    "ml:pandas_version": "pandas",
    "ml:scipy_version": "scipy",
}


# --------------------------------------------------------------------------
# Reading the PROV-JSON record
# --------------------------------------------------------------------------

def _one(entities: Dict, prefix: str) -> Tuple[str, Dict]:
    hits = [(k, v) for k, v in entities.items() if k.startswith(prefix)]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one {prefix} entity, found {len(hits)}")
    return hits[0]


def _val(v):
    """PROV-JSON typed literals may appear as {'$': value, 'type': ...}."""
    if isinstance(v, dict) and "$" in v:
        return v["$"]
    return v


def read_record(path: Path) -> Dict:
    doc = json.loads(path.read_text())
    ents = doc["entity"]
    _, task = _one(ents, "e:task/")
    _, flow = _one(ents, "e:flow/")
    _, params = _one(ents, "e:flowparams/")
    _, env = _one(ents, "e:env/")

    task_type = "regression" if "Regression" in str(_val(task.get("openml:task_type", ""))) else "classification"
    metric_key = "openml:r2_score" if task_type == "regression" else "openml:accuracy"

    folds = {}
    for k, v in ents.items():
        if k.startswith("e:split/"):
            n = int(_val(v["fold:number"]))
            folds.setdefault(n, {})["split"] = v
        elif k.startswith("e:metrics/"):
            n = int(k.rsplit("_fold", 1)[1])
            folds.setdefault(n, {})["metrics"] = v
        elif k.startswith("e:predictions/"):
            n = int(k.rsplit("_fold", 1)[1])
            folds.setdefault(n, {})["predictions"] = v

    assumptions = []
    pipeline_spec = _val(flow.get("ml:pipeline"))
    if pipeline_spec is None:
        assumptions.append("preprocessing")
        pipeline_spec = FALLBACK_PREPROCESSING
    elif isinstance(pipeline_spec, str):
        pipeline_spec = json.loads(pipeline_spec)

    est_path = _val(flow.get("ml:estimator_class"))
    if est_path is None:
        assumptions.append("estimator_class")
        est_path = FALLBACK_ESTIMATORS[_val(flow["openml:model"])]

    fold_specs = []
    for n in sorted(folds):
        s = folds[n]["split"]
        if "openml:fold_index" in s:
            fold_idx, repeat = int(_val(s["openml:fold_index"])), int(_val(s.get("openml:repeat", 0)))
        else:
            fold_idx, repeat = n - 1, 0
        pred = folds[n].get("predictions", {})
        fold_specs.append({
            "fold_number": n,
            "fold_index": fold_idx,
            "repeat": repeat,
            "recorded_train_size": int(_val(s["fold:train_size"])),
            "recorded_test_size": int(_val(s["fold:test_size"])),
            "recorded_score": float(_val(folds[n]["metrics"][metric_key])),
            "recorded_pred_digest": _val(pred.get("checksum:sha256")),
            "pred_digest_is_content": _val(pred.get("ml:digest_of")) is not None,
        })
    if fold_specs and "openml:fold_index" not in folds[min(folds)]["split"]:
        assumptions.append("fold_index_mapping")

    return {
        "path": str(path),
        "task_id": int(_val(task["openml:task_id"])),
        "task_type": task_type,
        "estimator_class": est_path,
        "params": json.loads(_val(params["openml:parameters"])),
        "pipeline_spec": pipeline_spec,
        "env": {VERSION_KEYS[k]: str(_val(v)) for k, v in env.items() if k in VERSION_KEYS},
        "folds": fold_specs,
        "assumptions": assumptions,
    }


# --------------------------------------------------------------------------
# Re-execution
# --------------------------------------------------------------------------

def _import(path: str):
    mod, _, name = path.rpartition(".")
    return getattr(importlib.import_module(mod), name)


def _make_step(spec: Dict):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    kind = spec["step"]
    kwargs = {k: v for k, v in spec.items() if k != "step"}
    return {"SimpleImputer": SimpleImputer, "OrdinalEncoder": OrdinalEncoder}[kind](**kwargs)


def build_pipeline(X, spec: Dict, estimator_cls, params: Dict):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    transformers = []
    if num_cols and spec.get("numeric"):
        steps = [_make_step(s) for s in spec["numeric"]]
        transformers.append(("num", steps[0] if len(steps) == 1 else
                             Pipeline([(f"s{i}", s) for i, s in enumerate(steps)]), num_cols))
    if cat_cols and spec.get("categorical"):
        steps = [_make_step(s) for s in spec["categorical"]]
        transformers.append(("cat", steps[0] if len(steps) == 1 else
                             Pipeline([(f"s{i}", s) for i, s in enumerate(steps)]), cat_cols))
    pre = ColumnTransformer(transformers, remainder=spec.get("remainder", "drop"))
    return Pipeline([("pre", pre), ("est", estimator_cls(**params))])


def predictions_digest(y_pred) -> str:
    """Content digest of a prediction vector, independent of dtype quirks."""
    payload = json.dumps([str(v) for v in np.asarray(y_pred).tolist()]).encode()
    return hashlib.sha256(payload).hexdigest()


def reexecute(record: Dict, cache_dir: Optional[str]) -> Dict:
    import openml
    from sklearn.metrics import accuracy_score, r2_score

    if cache_dir:
        openml.config.set_root_cache_directory(cache_dir)
    t0 = time.time()
    task = None
    for attempt in range(5):
        try:
            task = openml.tasks.get_task(record["task_id"], download_splits=True)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * 2 ** attempt)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    est_cls = _import(record["estimator_class"])

    out_folds = []
    for f in record["folds"]:
        tr, te = task.get_train_test_split_indices(fold=f["fold_index"], repeat=f["repeat"])
        pipe = build_pipeline(X, record["pipeline_spec"], est_cls, record["params"])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        y_pred = pipe.predict(X.iloc[te])
        y_true = y.iloc[te]
        score = float(accuracy_score(y_true, y_pred) if record["task_type"] == "classification"
                      else r2_score(y_true, y_pred))
        digest = predictions_digest(y_pred)
        out_folds.append({
            **f,
            "train_size": int(len(tr)),
            "test_size": int(len(te)),
            "sizes_match": int(len(tr)) == f["recorded_train_size"] and int(len(te)) == f["recorded_test_size"],
            "score": score,
            "score_rounded": round(score, 6),
            "abs_diff": abs(round(score, 6) - f["recorded_score"]),
            "score_match": round(score, 6) == round(f["recorded_score"], 6),
            "pred_digest": digest,
            "pred_digest_match": (digest == f["recorded_pred_digest"]) if f["pred_digest_is_content"] else None,
        })
    return {
        **{k: record[k] for k in ("path", "task_id", "task_type", "estimator_class", "params", "assumptions", "env")},
        "dataset_id": int(task.dataset_id),
        "dataset_version": int(getattr(dataset, "version", -1) or -1),
        "dataset_md5": getattr(dataset, "md5_checksum", None),
        "folds": out_folds,
        "seconds": round(time.time() - t0, 2),
    }


def _installed_versions() -> Dict[str, str]:
    import platform
    import numpy, sklearn, openml, pandas, scipy  # noqa: E401
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "sklearn": sklearn.__version__, "openml": openml.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-dir", default=None, help="Fresh OpenML cache, for an independent download")
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    files = sorted(Path(args.corpus).rglob("prov_*.json"))[: args.max_runs]
    records = [read_record(p) for p in files]
    installed = _installed_versions()
    print(f"{len(records)} PROV records; installed: {installed}", flush=True)

    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(reexecute, r, args.cache_dir): r for r in records}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                ok = sum(f["score_match"] for f in res["folds"])
                print(f"[{i}/{len(records)}] task {res['task_id']}: {ok}/{len(res['folds'])} folds match "
                      f"({res['seconds']}s)", flush=True)
            except Exception as exc:  # report, never hide
                errors.append({"path": r["path"], "task_id": r["task_id"], "error": repr(exc)})
                print(f"[{i}/{len(records)}] task {r['task_id']}: ERROR {exc!r}", flush=True)

    folds = [f for r in results for f in r["folds"]]
    env_mismatch = sorted({f"{k}: recorded {v} vs installed {installed.get(k)}"
                           for r in results for k, v in r["env"].items()
                           if installed.get(k) is not None and installed[k] != v})
    summary = {
        "corpus": args.corpus,
        "installed_versions": installed,
        "recorded_env_mismatches": env_mismatch,
        "runs_attempted": len(records),
        "runs_reexecuted": len(results),
        "runs_failed": len(errors),
        "folds_reexecuted": len(folds),
        "folds_score_exact": sum(f["score_match"] for f in folds),
        "folds_sizes_match": sum(f["sizes_match"] for f in folds),
        "runs_all_folds_exact": sum(all(f["score_match"] for f in r["folds"]) for r in results),
        "max_abs_diff": max((f["abs_diff"] for f in folds), default=None),
        "pred_digests_checked": sum(f["pred_digest_match"] is not None for f in folds),
        "pred_digests_match": sum(bool(f["pred_digest_match"]) for f in folds),
        "assumptions_needed": sorted({a for r in results for a in r["assumptions"]}),
        "fallback_preprocessing": FALLBACK_PREPROCESSING,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "errors": errors,
                                          "runs": sorted(results, key=lambda r: r["task_id"])}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
