"""Utility functions for PROV corpus generation."""

import hashlib
import json
import platform
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import numpy as np


def generate_run_id() -> str:
    """Generate a unique run identifier."""
    return uuid.uuid4().hex


def compute_sha256(data: Any) -> str:
    """Compute SHA256 hash of data."""
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def get_timestamp(offset_seconds: int = 0) -> str:
    """Get ISO format timestamp with optional offset."""
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_environment_info() -> Dict[str, str]:
    """Get runtime environment information.

    Keys are qualified with the ``ml:`` namespace so the dict can be splatted
    into a PROV entity's attribute dict without producing bare keys (which
    would violate the PROV-JSON spec).
    """
    return {
        "ml:python_version": platform.python_version(),
        "ml:platform": platform.platform(),
        "ml:sklearn_version": "1.7.1",
        "ml:numpy_version": np.__version__,
        "ml:openml_version": "0.15.1",
    }


def generate_synthetic_metric(
    model_name: str, config_idx: int, task_id: int, fold: int, task_type: str
) -> float:
    """Generate realistic synthetic performance metrics."""
    clf_base = {
        "RandomForest": 0.85,
        "GradientBoosting": 0.86,
        "AdaBoost": 0.82,
        "ExtraTrees": 0.84,
        "LogisticRegression": 0.78,
        "SVM": 0.80,
        "KNN": 0.75,
        "MLP": 0.81,
        "DecisionTree": 0.72,
        "NaiveBayes": 0.70,
        "BaggingClassifier": 0.83,
        "HistGradientBoosting": 0.87,
    }
    reg_base = {
        "RandomForestRegressor": 0.82,
        "GradientBoostingRegressor": 0.84,
        "AdaBoostRegressor": 0.78,
        "ExtraTreesRegressor": 0.81,
        "Ridge": 0.72,
        "Lasso": 0.70,
        "ElasticNet": 0.71,
        "SVR": 0.75,
        "KNeighborsRegressor": 0.68,
        "MLPRegressor": 0.76,
        "DecisionTreeRegressor": 0.65,
        "HistGradientBoostingRegressor": 0.85,
    }

    base_map = clf_base if task_type == "classification" else reg_base
    base = base_map.get(model_name, 0.75)
    random.seed(task_id * 1000 + config_idx * 10 + fold)
    return max(0.0, min(1.0, base + random.gauss(0, 0.05)))
