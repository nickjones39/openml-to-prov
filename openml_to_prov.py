#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# A pure-local pipeline for executing OpenML CC18 tasks and exporting
# W3C PROV-compliant provenance records in PROV-JSON, with optional
# Graphviz-style rendering via the prov Python package.
#
# Author: Nicholas Jones
# Copyright (c) 2025 Nicholas Jones
#
# ----------------------------------------------------------------------
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------

import json, pathlib
import os, hashlib
import uuid
import csv
import sys

from sklearn.base import clone
from joblib import dump as joblib_dump
import openml
import numpy as np
import pandas as pd

from datetime import datetime


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# --- PROV-package rendering (prov.dot) ---
# Enable/disable per-format emission
RENDER_PROV_PNG = False
RENDER_PROV_SVG = False

PROV_DOT_PNG_DIR = pathlib.Path("prov_dot_png"); PROV_DOT_PNG_DIR.mkdir(exist_ok=True)
PROV_DOT_SVG_DIR = pathlib.Path("prov_dot_svg"); PROV_DOT_SVG_DIR.mkdir(exist_ok=True)

try:
    from prov.model import ProvDocument
    from prov.dot import prov_to_dot
    _PROV_AVAILABLE = True
except Exception:
    _PROV_AVAILABLE = False

def render_prov_with_prov_pkg(prov_doc: dict, out_path: pathlib.Path, fmt: str = "png") -> None:
    """
    Render provenance as PNG or SVG using the PROV Python package (prov.model + prov.dot).
    fmt ∈ {"png","svg"} (case-insensitive).
    """
    if not _PROV_AVAILABLE:
        raise RuntimeError("prov package (prov.model/prov.dot) not available")

    # Build ProvDocument from our dict
    doc = ProvDocument()

    # prefixes
    for pfx, uri in prov_doc.get("prefix", {}).items():
        try:
            doc.add_namespace(pfx, uri)
        except Exception:
            pass

    # core records
    for eid, attrs in prov_doc.get("entity", {}).items():
        try:
            doc.entity(eid, other_attributes={str(k): v for k, v in attrs.items()})
        except Exception:
            doc.entity(eid)

    for aid, attrs in prov_doc.get("activity", {}).items():
        try:
            doc.activity(aid, other_attributes={str(k): v for k, v in attrs.items()})
        except Exception:
            doc.activity(aid)

    for gid, attrs in prov_doc.get("agent", {}).items():
        try:
            doc.agent(gid, other_attributes={str(k): v for k, v in attrs.items()})
        except Exception:
            doc.agent(gid)

    # relations
    for rel in prov_doc.get("used", []):
        a, e = rel.get("activity"), rel.get("entity")
        if a and e:
            doc.used(a, e, other_attributes={k: v for k, v in rel.items() if k not in ("activity", "entity")})

    for rel in prov_doc.get("wasGeneratedBy", []):
        e, a = rel.get("entity"), rel.get("activity")
        if e and a:
            doc.wasGeneratedBy(e, a, other_attributes={k: v for k, v in rel.items() if k not in ("entity", "activity")})

    for rel in prov_doc.get("wasAssociatedWith", []):
        a, g = rel.get("activity"), rel.get("agent")
        if a and g:
            doc.wasAssociatedWith(a, g, other_attributes={k: v for k, v in rel.items() if k not in ("activity", "agent")})

    for rel in prov_doc.get("wasDerivedFrom", []):
        ge, ue = rel.get("generatedEntity"), rel.get("usedEntity")
        if ge and ue:
            doc.wasDerivedFrom(ge, ue, other_attributes={k: v for k, v in rel.items() if k not in ("generatedEntity", "usedEntity")})

    for rel in prov_doc.get("wasInformedBy", []):
        informed, informant = rel.get("informed"), rel.get("informant")
        if informed and informant:
            doc.wasInformedBy(informed, informant, other_attributes={k: v for k, v in rel.items() if k not in ("informed", "informant")})

    # Produce DOT with default PROV styling and write image
    dot = prov_to_dot(doc)
    try:
        # BT often reads naturally for lineage-from-output; change to "TB" if preferred.
        dot.set_rankdir('BT')
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or "png").lower()
    try:
        if fmt == "svg":
            dot.write_svg(out_path.as_posix())
        else:
            dot.write_png(out_path.as_posix())
    except Exception:
        # Fallback to Graphviz CLI if pydot backend cannot write directly
        tmp = out_path.with_suffix('.dot')
        dot.write(tmp.as_posix())
        import subprocess
        tool = "svg" if fmt == "svg" else "png"
        subprocess.run(['dot', f'-T{tool}', tmp.as_posix(), '-o', out_path.as_posix()], check=True)

#
# OpenML 0.15.x uses sklearn_to_flow (not flow_from_sklearn)
try:
    from openml.flows import sklearn_to_flow
except Exception:
    sklearn_to_flow = None

# --- Helper: make nested structures JSON-serialisable (convert sklearn/np objects)
def _to_jsonable(obj):
    import numpy as _np
    import pandas as _pd
    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # numpy scalars
    if isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
        return obj.item()
    # numpy arrays, lists, tuples, sets
    if isinstance(obj, (_np.ndarray, list, tuple, set)):
        return [_to_jsonable(x) for x in list(obj)]
    # pandas types
    if isinstance(obj, (_pd.Series, _pd.Index)):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, _pd.DataFrame):
        return obj.to_dict(orient="list")
    # dicts
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # sklearn/other objects -> compact string
    try:
        return f"<{obj.__class__.__name__}>"
    except Exception:
        return str(obj)

def _flatten_params(pipe):
    """
    Return a list of {'name': <qualified_param>, 'value': <string>} for sklearn pipelines.
    Values are JSON-safe strings for stability.
    """
    flat = []
    try:
        params = pipe.get_params(deep=True)
    except Exception:
        return flat
    for k, v in sorted(params.items()):
        # Skip very large objects; prefer simple scalars/strings
        if hasattr(v, "__class__") and v.__class__.__name__ in ("ColumnTransformer", "Pipeline"):
            val = f"<{v.__class__.__name__}>"
        else:
            jv = _to_jsonable(v)
            if isinstance(jv, (dict, list)):
                # Compact string form for nested structures
                try:
                    val = json.dumps(jv, sort_keys=True, separators=(",", ":"))
                except Exception:
                    val = str(jv)
            else:
                val = str(jv)
        flat.append({"name": k, "value": val})
    return flat

SUITE_ID = 99            # OpenML-CC18
N_TASKS  = 5             # increase/remove slice to scale
OUT_DIR  = pathlib.Path("prov_out")
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = pathlib.Path("models_out")
MODELS_DIR.mkdir(exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary.csv"
PRED_DIR = pathlib.Path("predictions_out")
PRED_DIR.mkdir(exist_ok=True)
SPLIT_DIR = pathlib.Path("splits_out")
SPLIT_DIR.mkdir(exist_ok=True)
PREPROV_DIR = pathlib.Path("preprov_out")
PREPROV_DIR.mkdir(exist_ok=True)


def run_to_prov(run_id, dataset, flow_label, flow_params, flowparams_label, task_id, metrics, times, env_info, cv_config, model_info, pred_info, split_info, run_record, eval_records):
    rid = f"Run{run_id}"
    # Canonical hash of flow parameters for deduplication
    _flowparams_canonical = json.dumps(flow_params, sort_keys=True, separators=(",", ":"))
    _flowparams_sha256 = hashlib.sha256(_flowparams_canonical.encode("utf-8")).hexdigest()
    task_eid = f"e:task/{task_id}"
    flowparams_eid = f"e:flowparams/{rid}"
    env_eid = f"e:env/{rid}"
    group_aid = f"a:experiment/d{dataset.dataset_id}"
    prov = {
        "prefix": {
            "prov": "http://www.w3.org/ns/prov#",
            "openml": "https://openml.org/def/",
            "e": "urn:entity:",
            "a": "urn:activity:",
            "ag": "urn:agent:",
            "checksum": "urn:checksum:",
            "pred": "urn:pred:",
            "split": "urn:split:"
        },
        "entity": {
            f"e:dataset/{dataset.dataset_id}": {
                "prov:label": f"OpenML dataset {dataset.dataset_id} ({dataset.name})",
                "prov:type": "openml:Dataset",
                "openml:data_id": dataset.dataset_id,
                "openml:version": dataset.version,
                "openml:default_target": dataset.default_target_attribute
            },
            f"e:flow/{rid}": {
                "prov:label": flow_label,
                "prov:type": "openml:Flow",
                "openml:parameters": flow_params
            },
            f"e:model/{rid}": {
                "prov:label": "Trained model",
                "prov:type": "openml:Model",
                **({"prov:location": model_info.get("prov_location")} if model_info.get("prov_location") else {}),
                **({"model:path": model_info.get("path")} if model_info.get("path") else {}),
                **({"model:size_bytes": model_info.get("size_bytes")} if model_info.get("size_bytes") is not None else {}),
                **({"checksum:sha256": model_info.get("sha256")} if model_info.get("sha256") else {}),
                **({"model:feature_count": model_info.get("feature_count")} if model_info.get("feature_count") is not None else {}),
                **({"model:classes": model_info.get("classes")} if model_info.get("classes") is not None else {})
            },
            f"e:predictions/{rid}": {
                "prov:label": "Predictions",
                "prov:type": "openml:Predictions",
                **({"prov:location": pred_info.get("prov_location")} if pred_info.get("prov_location") else {}),
                **({"pred:size_bytes": pred_info.get("size_bytes")} if pred_info.get("size_bytes") is not None else {}),
                **({"pred:rows": pred_info.get("rows")} if pred_info.get("rows") is not None else {}),
                **({"pred:folds": pred_info.get("folds")} if pred_info.get("folds") is not None else {}),
                **({"checksum:sha256": pred_info.get("sha256")} if pred_info.get("sha256") else {})
            },
            f"e:split/{rid}": {
                "prov:label": "CV split indices",
                "prov:type": "openml:Split",
                **({"prov:location": split_info.get("prov_location")} if split_info.get("prov_location") else {}),
                **({"split:size_bytes": split_info.get("size_bytes")} if split_info.get("size_bytes") is not None else {}),
                **({"checksum:sha256": split_info.get("sha256")} if split_info.get("sha256") else {})
            },
            f"e:metrics/{rid}":     {"prov:label": "Evaluation metrics", "prov:type": "openml:Metrics", **metrics},
            task_eid: {
                "prov:label": f"OpenML task {task_id}",
                "prov:type": "openml:Task",
                "openml:task_id": task_id
            },
            flowparams_eid: {
                "prov:label": flowparams_label,
                "prov:type": "openml:FlowParameters",
                "params": flow_params,
                "flowparams:sha256": _flowparams_sha256
            },
            env_eid: {
                "prov:type": "Environment",
                **env_info
            }
        },
        "activity": {
            group_aid: {
                "prov:label": f"Experiment group for dataset {dataset.dataset_id} ({dataset.name})",
                "prov:type": "openml:Experiment"
            },
            f"a:train/{rid}":    {"prov:label": "Model training", "prov:type": "openml:Train",
                                   "prov:startTime": times["train_start"],
                                   "prov:endTime": times["train_end"],
                                   "openml:task_id": task_id},
            f"a:predict/{rid}":  {"prov:label": "Generate predictions", "prov:type": "openml:Predict",
                                   "prov:startTime": times["predict_start"],
                                   "prov:endTime": times["predict_end"]},
            f"a:evaluate/{rid}": {
                "prov:label": "Evaluate predictions",
                "prov:type": "openml:Evaluate",
                "prov:startTime": times["eval_start"],
                "prov:endTime": times["eval_end"],
                "openml:split_scheme": "StratifiedKFold",
                "openml:cv": cv_config
            }
        },
        "agent": {
            "ag:system/OpenML": {
                "prov:type": "prov:SoftwareAgent",
                "prov:label": "Local sklearn pipeline (no upload)"
            }
        },
        "used": [
            {"activity": f"a:train/{rid}",   "entity": f"e:dataset/{dataset.dataset_id}", "prov:role": "train-dataset"},
            {"activity": f"a:train/{rid}",   "entity": f"e:flow/{rid}",                   "prov:role": "flow"},
            {"activity": f"a:train/{rid}",   "entity": flowparams_eid,                    "prov:role": "flow-params"},
            {"activity": f"a:predict/{rid}", "entity": f"e:model/{rid}",                  "prov:role": "trained-model"},
            {"activity": f"a:train/{rid}",   "entity": task_eid,                          "prov:role": "task"},
            {"activity": f"a:train/{rid}",   "entity": env_eid,                           "prov:role": "environment"},
            {"activity": f"a:predict/{rid}", "entity": f"e:dataset/{dataset.dataset_id}", "prov:role": "predict-dataset"},
            {"activity": f"a:evaluate/{rid}","entity": f"e:predictions/{rid}",            "prov:role": "predictions"},
            {"activity": f"a:evaluate/{rid}","entity": f"e:split/{rid}",                  "prov:role": "cv-splits"},
            {"activity": f"a:evaluate/{rid}","entity": flowparams_eid,                    "prov:role": "flow-params"},
            {"activity": f"a:evaluate/{rid}","entity": f"e:dataset/{dataset.dataset_id}", "prov:role": "eval-dataset"}
        ],
        "wasGeneratedBy": [
            {"entity": f"e:model/{rid}",       "activity": f"a:train/{rid}"},
            {"entity": f"e:predictions/{rid}", "activity": f"a:predict/{rid}"},
            {"entity": f"e:metrics/{rid}",     "activity": f"a:evaluate/{rid}"}
        ],
        "wasAssociatedWith": [
            {"activity": f"a:train/{rid}",    "agent": "ag:system/OpenML"},
            {"activity": f"a:predict/{rid}",  "agent": "ag:system/OpenML"},
            {"activity": f"a:evaluate/{rid}", "agent": "ag:system/OpenML"}
        ],
        "wasInformedBy": [
            {"informed": f"a:train/{rid}",    "informant": group_aid},
            {"informed": f"a:predict/{rid}",  "informant": f"a:train/{rid}"},
            {"informed": f"a:evaluate/{rid}", "informant": f"a:predict/{rid}"}
        ]
        # Concrete local run record (OpenML-style stub) and evaluations
        ,
        "run": run_record,
        "evaluations": eval_records
    }
    if "wasAttributedTo" in prov:
        prov["wasAttributedTo"].append({"entity": f"e:model/{rid}", "agent": "ag:system/OpenML"})
    else:
        prov["wasAttributedTo"] = [{"entity": f"e:model/{rid}", "agent": "ag:system/OpenML"}]
    # Add derivation edges to strengthen causal semantics
    if "wasDerivedFrom" not in prov:
        prov["wasDerivedFrom"] = []
    prov["wasDerivedFrom"].append({"generatedEntity": f"e:metrics/{rid}", "usedEntity": f"e:predictions/{rid}"})
    prov["wasDerivedFrom"].append({"generatedEntity": f"e:metrics/{rid}", "usedEntity": f"e:split/{rid}"})
    return prov

def build_pipeline(X_df):
    # Robust to mixed types
    num_cols = X_df.select_dtypes(include=[np.number]).columns
    cat_cols = X_df.columns.difference(num_cols)

    num_pipe = make_pipeline(SimpleImputer())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))

    pre = ColumnTransformer(
        [("num", num_pipe, list(num_cols)),
         ("cat", cat_pipe, list(cat_cols))],
        remainder="drop"
    )
    clf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    return make_pipeline(pre, clf)


def main():
    suite = openml.study.get_suite(SUITE_ID)
    print("CC18 tasks:", len(suite.tasks))

    for idx, task_id in enumerate(suite.tasks[:N_TASKS], 1):
        task    = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        # Load as DataFrame to detect types
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )

        # Local pseudo run id (early for file naming)
        run_id = uuid.uuid4().hex  # deterministic-format unique ID

        # Some CC18 sets have missing or non-binary targets — drop NA rows quickly
        if y is not None:
            mask = pd.notna(y)
            X, y = X[mask], y[mask]

        pipe = build_pipeline(X)

        # Summarise key hyperparameters for label (best-effort)
        try:
            rf = pipe.named_steps.get("randomforestclassifier", None)
            if rf is not None:
                flowparams_label = f"RF({int(getattr(rf, 'n_estimators', 0))} trees, seed={getattr(rf, 'random_state', None)})"
            else:
                flowparams_label = "Pipeline parameters"
        except Exception:
            flowparams_label = "Pipeline parameters"

        import sys, platform, sklearn, numpy as _np, pandas as _pd

        # Train timing on full data (separate pipeline instance to avoid CV leakage)
        pipe_train = build_pipeline(X)
        from time import time as _t
        t0 = _t(); pipe_train.fit(X, y); t1 = _t()
        # Enrich model metadata with feature count and classes (best-effort)
        try:
            preproc = pipe_train.named_steps.get("columntransformer", None)
            if preproc is not None and hasattr(preproc, "get_feature_names_out"):
                feature_count = int(len(preproc.get_feature_names_out()))
            else:
                feature_count = None
        except Exception:
            feature_count = None
        try:
            classes = sorted(pd.unique(y).tolist()) if y is not None else None
        except Exception:
            classes = None

        model_path = MODELS_DIR / f"model_{run_id}.joblib"
        try:
            joblib_dump(pipe_train, model_path)
            size_bytes = os.path.getsize(model_path)
            sha256 = hashlib.sha256()
            with open(model_path, "rb") as _fh:
                for chunk in iter(lambda: _fh.read(1024 * 1024), b""):
                    sha256.update(chunk)
            model_info = {"path": str(model_path.resolve()), "size_bytes": int(size_bytes), "sha256": sha256.hexdigest()}
        except Exception:
            model_info = {"path": None, "size_bytes": None, "sha256": None}

        # Also record a portable (relative) path for provenance viewers
        if model_path:
            model_info["prov_location"] = str(model_path)
        else:
            model_info["prov_location"] = None
        if model_info is not None:
            model_info["feature_count"] = feature_count
            model_info["classes"] = classes

        # Predict timing on a manageable slice
        n_pred = int(min(len(X), 10000))
        t2 = _t(); _ = pipe_train.predict(X.iloc[:n_pred]); t3 = _t()

        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_path = SPLIT_DIR / f"splits_{run_id}.csv"
        pred_path = PRED_DIR / f"preds_{run_id}.csv"
        per_fold = []
        t4 = _t()
        total_pred_rows = 0
        with open(pred_path, "w", newline="") as _csvfh, open(split_path, "w", newline="") as _splitfh:
            writer = csv.writer(_csvfh)
            split_writer = csv.writer(_splitfh)
            writer.writerow(["row_index", "fold", "y_true", "y_pred"])
            split_writer.writerow(["row_index", "fold"])
            fold_id = 0
            for fold_id, (tr, te) in enumerate(cv.split(X, y), 1):
                model = clone(pipe)
                model.fit(X.iloc[tr], y.iloc[tr])
                y_true = y.iloc[te]
                y_pred = model.predict(X.iloc[te])
                # Write predictions rows
                for idx_row, yt, yp in zip(y_true.index.tolist(), y_true.tolist(), y_pred.tolist()):
                    writer.writerow([idx_row, fold_id, yt, yp])
                # Write split indices rows
                for idx_row in y_true.index.tolist():
                    split_writer.writerow([idx_row, fold_id])
                # Accuracy per fold
                import numpy as _np_local
                acc = float((_np_local.array(y_pred) == _np_local.array(y_true)).mean())
                per_fold.append({"fold": int(fold_id), "n": int(len(te)), "predictive_accuracy": round(acc, 6)})
                total_pred_rows += int(len(y_true))
        t5 = _t()
        # Record predictions row count and folds
        pred_info_extra_rows = total_pred_rows
        pred_info_extra_folds = int(fold_id)
        # Aggregate metrics
        import numpy as _np_agg
        accs = _np_agg.array([pf["predictive_accuracy"] for pf in per_fold], dtype=float)
        metrics = {
            "openml:accuracy": round(float(accs.mean()), 6),
            "openml:std": round(float(accs.std(ddof=0)), 6)
        }
        # Predictions artefact metadata
        try:
            size_bytes_p = os.path.getsize(pred_path)
            sha256_p = hashlib.sha256()
            with open(pred_path, "rb") as _pfh:
                for chunk in iter(lambda: _pfh.read(1024 * 1024), b""):
                    sha256_p.update(chunk)
            pred_info = {"prov_location": str(pred_path), "path": str(pred_path.resolve()), "size_bytes": int(size_bytes_p), "sha256": sha256_p.hexdigest()}
        except Exception:
            pred_info = {"prov_location": None, "path": None, "size_bytes": None, "sha256": None}
        pred_info["rows"] = pred_info_extra_rows
        pred_info["folds"] = pred_info_extra_folds

        # Split indices artefact metadata
        try:
            size_bytes_s = os.path.getsize(split_path)
            sha256_s = hashlib.sha256()
            with open(split_path, "rb") as _sfh:
                for chunk in iter(lambda: _sfh.read(1024 * 1024), b""):
                    sha256_s.update(chunk)
            split_info = {"prov_location": str(split_path), "path": str(split_path.resolve()), "size_bytes": int(size_bytes_s), "sha256": sha256_s.hexdigest()}
        except Exception:
            split_info = {"prov_location": None, "path": None, "size_bytes": None, "sha256": None}

        # “Flow” label/params without contacting the server
        if sklearn_to_flow is not None:
            flow = sklearn_to_flow(pipe)
            flow_label   = flow.name
            flow_params  = _to_jsonable(flow.parameters)
        else:
            # Fallback: avoid OpenML flow object; use pipeline metadata directly
            flow = None
            flow_label  = pipe.__class__.__name__
            try:
                flow_params = _to_jsonable(pipe.get_params(deep=False))
            except Exception:
                flow_params = {}

        # Pseudo flow_id from canonical flow parameters (content hash)
        _flowparams_canonical_main = json.dumps(flow_params, sort_keys=True, separators=(",", ":"))
        flow_id_hash = hashlib.sha256(_flowparams_canonical_main.encode("utf-8")).hexdigest()

        # Build evaluation + timing + environment + run record before pre-PROV
        # Per-fold and aggregate evaluation records
        eval_records = [{"function": "predictive_accuracy", "fold": pf["fold"], "value": pf["predictive_accuracy"], "n": pf["n"]} for pf in per_fold]
        eval_records.append({"function": "predictive_accuracy", "aggregate": "mean", "value": metrics["openml:accuracy"], "stdev": metrics["openml:std"]})

        # CV config summary
        cv_config = {"n_splits": int(cv.get_n_splits()), "shuffle": bool(cv.shuffle), "random_state": int(cv.random_state) if cv.random_state is not None else None}

        # ISO timestamps for phases
        def _iso(ts):
            import datetime as _dt
            return _dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
        times = {
            "train_start": _iso(t0), "train_end": _iso(t1),
            "predict_start": _iso(t2), "predict_end": _iso(t3),
            "eval_start": _iso(t4), "eval_end": _iso(t5)
        }

        # Environment snapshot (versions/platform)
        import sys as _sys_env, platform as _platform_env, sklearn as _sk_env, numpy as _np_env, pandas as _pd_env
        env_info = {
            "python": _sys_env.version.split()[0],
            "platform": _platform_env.platform(),
            "sklearn": _sk_env.__version__,
            "numpy": _np_env.__version__,
            "pandas": _pd_env.__version__,
            "openml": openml.__version__
        }

        # Concrete (local) run record stub
        run_record = {
            "run_id": str(run_id),
            "task_id": int(task_id),
            "flow_id": flow_id_hash,
            "flow_name": flow_label,
            "parameter_settings": _flatten_params(pipe),
            "start_time": times["train_start"],
            "end_time": times["eval_end"],
            "uploader": None,
            "server_start_time": None,
            "server_end_time": None,
            "files": {
                "predictions": pred_info.get("prov_location"),
                "model": model_info.get("prov_location"),
                "log": None
            }
        }

        # --- Build "original-form" (pre-PROV) bundle for inspection/archival
        try:
            X_head = X.head(5).to_dict(orient="list") if hasattr(X, "head") else None
            y_head = y.head(5).tolist() if hasattr(y, "head") else None
        except Exception:
            X_head, y_head = None, None

        preprov_bundle = {
            "dataset": {
                "data_id": int(dataset.dataset_id),
                "name": dataset.name,
                "version": int(getattr(dataset, "version", 0) or 0),
                "default_target": dataset.default_target_attribute,
                "qualities": getattr(dataset, "qualities", None)
            },
            "task": {
                "task_id": int(task_id),
                "task_type": getattr(task, "task_type", None),
                "target_name": dataset.default_target_attribute
            },
            "flow": {
                "flow_id": flow_id_hash,
                "flow_name": flow_label,
                "parameters": flow_params,
                "parameter_settings": _flatten_params(pipe)
            },
            "run": { **run_record },
            "evaluations": eval_records,
            "cv": {
                "config": cv_config,
                "splits_file": split_info.get("prov_location"),
                "predictions_file": pred_info.get("prov_location"),
                "total_rows": pred_info.get("rows"),
                "folds": pred_info.get("folds")
            },
            "environment": env_info,
            "samples": {
                "X_head": X_head,
                "y_head": y_head
            },
            "files": {
                "model": model_info.get("prov_location"),
                "predictions": pred_info.get("prov_location"),
                "splits": split_info.get("prov_location")
            },
            "timing": times
        }
        preprov_path = PREPROV_DIR / f"preprov_run_{run_id}.json"
        preprov_path.write_text(json.dumps(preprov_bundle, indent=2, ensure_ascii=False, default=str))
        print(f"         Pre-PROV bundle: {preprov_path}")

        prov = run_to_prov(run_id, dataset, flow_label, flow_params, flowparams_label, task_id, metrics, times, env_info, cv_config, model_info, pred_info, split_info, run_record, eval_records)
        out = OUT_DIR / f"openml_run_{run_id}.prov.json"
        out.write_text(json.dumps(prov, indent=2, ensure_ascii=False, default=str))
        print(f"[{idx}/{N_TASKS}] {dataset.name} → {out}")

        # --- Optional PROV-package (Graphviz-style) rendering
        if _PROV_AVAILABLE and (RENDER_PROV_PNG or RENDER_PROV_SVG):
            if RENDER_PROV_PNG:
                dot_png_path = PROV_DOT_PNG_DIR / (out.stem + "_provpkg.png")
                try:
                    render_prov_with_prov_pkg(prov, dot_png_path, fmt="png")
                    print(f"         PROV(dot, png): {dot_png_path}")
                except Exception as e:
                    print(f"         PROV(dot) PNG render failed: {e.__class__.__name__}: {e}")
            if RENDER_PROV_SVG:
                dot_svg_path = PROV_DOT_SVG_DIR / (out.stem + "_provpkg.svg")
                try:
                    render_prov_with_prov_pkg(prov, dot_svg_path, fmt="svg")
                    print(f"         PROV(dot) SVG: {dot_svg_path}")
                except Exception as e:
                    print(f"         PROV(dot) SVG render failed: {e.__class__.__name__}: {e}")
        elif (RENDER_PROV_PNG or RENDER_PROV_SVG) and not _PROV_AVAILABLE:
            print("         PROV(dot) render skipped: 'prov' package not available")

        # --- Append summary row (create header if new)
        try:
            prov_bytes = out.stat().st_size
        except Exception:
            prov_bytes = None

        nodes = len(prov.get("entity", {})) + len(prov.get("activity", {})) + len(prov.get("agent", {}))
        edges = sum(len(prov.get(k, [])) for k in ["used", "wasGeneratedBy", "wasAssociatedWith", "wasInformedBy", "wasAttributedTo", "wasDerivedFrom"])

        header = "run_id,dataset_id,dataset_name,task_id,nodes,edges,prov_bytes,accuracy,std,model_bytes,pred_bytes,split_bytes\n"
        row = f"{run_id},{dataset.dataset_id},{dataset.name},{task_id},{nodes},{edges},{prov_bytes},{metrics.get('openml:accuracy')},{metrics.get('openml:std')},{model_info.get('size_bytes')},{pred_info.get('size_bytes')},{split_info.get('size_bytes')}\n"

        if not SUMMARY_CSV.exists():
            SUMMARY_CSV.write_text(header)
        with SUMMARY_CSV.open("a") as _fh:
            _fh.write(row)

if __name__ == "__main__":
    main()