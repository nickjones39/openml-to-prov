"""Real OpenML + sklearn execution engine."""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

_RETRYABLE = (OSError, TimeoutError, ConnectionError)

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSIFIER_MAP = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "AdaBoost": AdaBoostClassifier,
    "ExtraTrees": ExtraTreesClassifier,
    "LogisticRegression": LogisticRegression,
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "MLP": MLPClassifier,
    "DecisionTree": DecisionTreeClassifier,
    "NaiveBayes": GaussianNB,
    "BaggingClassifier": BaggingClassifier,
    "HistGradientBoosting": HistGradientBoostingClassifier,
}

REGRESSOR_MAP = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "AdaBoostRegressor": AdaBoostRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "MLPRegressor": MLPRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_pipeline(X: pd.DataFrame, estimator_class, cfg: Dict) -> Pipeline:
    """Preprocessing + estimator pipeline; handles missing values and categoricals."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="mean"), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            cat_cols,
        ))

    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("est", estimator_class(**cfg))])


class OpenMLExecutor:
    """Fetches OpenML tasks and runs real sklearn cross-validation."""

    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = True):
        import os
        import openml
        from pathlib import Path
        from dotenv import load_dotenv

        # Load .env from repo root (two levels up from this file)
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")

        api_key = os.environ.get("OPENML_API_KEY")
        if api_key and api_key != "your_key_here":
            openml.config.apikey = api_key
            print(f"  OpenML API key loaded (ends ...{api_key[-4:]})")
        else:
            print("  WARNING: No OpenML API key found — requests may be rate-limited")

        if cache_dir:
            openml.config.cache_directory = cache_dir
        self._openml = openml
        self.verbose = verbose

    SUPPORTED_TASK_TYPES = {
        "classification": "Supervised Classification",
        "regression": "Supervised Regression",
    }

    def get_suite_task_ids(self, suite_id: int) -> Optional[List[int]]:
        """Fetch live task IDs for an OpenML benchmark suite (e.g. 99 = CC18). Returns None on failure."""
        try:
            suite = self._openml.study.get_suite(suite_id)
            return [int(t) for t in suite.tasks]
        except Exception as exc:
            if self.verbose:
                print(f"  WARNING: could not fetch suite {suite_id} from OpenML: {exc}")
            return None

    def _get_task_with_retry(self, task_id: int, retries: int = 5, delay: float = 10.0):
        """Fetch an OpenML task, retrying on transient network or server errors."""
        from openml.exceptions import OpenMLServerException
        for attempt in range(retries):
            try:
                return self._openml.tasks.get_task(task_id)
            except NotImplementedError as exc:
                # openml-python raises this for task types it can't parse
                # (e.g. Supervised Data Stream Classification). Treat as skip.
                raise ValueError(f"task {task_id} type not supported by openml-python: {exc}") from exc
            except OpenMLServerException as exc:
                # Code 151 can mean rate-limited as well as genuinely missing.
                # Retry with backoff; only treat as permanent after all retries fail.
                if attempt == retries - 1:
                    raise ValueError(
                        f"task {task_id} unavailable after {retries} attempts: {exc}"
                    ) from exc
                wait = delay * (2 ** attempt)
                if self.verbose:
                    print(f"      server error for task {task_id} "
                          f"(attempt {attempt+1}/{retries}): {exc} — retrying in {wait:.0f}s")
                time.sleep(wait)
            except _RETRYABLE as exc:
                if attempt == retries - 1:
                    raise
                wait = delay * (2 ** attempt)
                if self.verbose:
                    print(f"      network error fetching task {task_id} "
                          f"(attempt {attempt+1}/{retries}): {exc} — retrying in {wait:.0f}s")
                time.sleep(wait)
            except Exception as exc:
                if "requests" in type(exc).__module__ or "urllib" in type(exc).__module__:
                    if attempt == retries - 1:
                        raise
                    wait = delay * (2 ** attempt)
                    if self.verbose:
                        print(f"      network error fetching task {task_id} "
                              f"(attempt {attempt+1}/{retries}): {exc} — retrying in {wait:.0f}s")
                    time.sleep(wait)
                else:
                    raise

    def validate_task_type(self, task_id: int, task_type: str) -> None:
        """Raise ValueError if the OpenML task type is not supported. Cheap — no dataset download."""
        task = self._get_task_with_retry(task_id)
        expected = self.SUPPORTED_TASK_TYPES.get(task_type)
        if task.task_type != expected:
            raise ValueError(
                f"unsupported type '{task.task_type}' (expected '{expected}')"
            )

    def get_task_data(self, task_id: int, task_type: str) -> Tuple:
        """Return (task, X, y, dataset_meta_dict)."""
        task = self._get_task_with_retry(task_id)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")

        is_clf = task_type == "classification"
        meta = {
            "dataset_id": task.dataset_id,
            "dataset_name": dataset.name,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))) if is_clf else None,
            "task_type": task_type,
            "metric": "accuracy" if is_clf else "r2_score",
        }
        return task, X, y, meta

    def run_fold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        estimator_class,
        cfg: Dict,
        task_type: str,
    ) -> Dict:
        """Train and evaluate one fold; return timing + metric score."""
        pipe = _build_pipeline(X, estimator_class, cfg)

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        t0 = time.time()
        pipe.fit(X_train, y_train)
        t1 = time.time()

        y_pred = pipe.predict(X_test)
        t2 = time.time()

        if task_type == "classification":
            score = float(accuracy_score(y_test, y_pred))
        else:
            score = float(r2_score(y_test, y_pred))

        t3 = time.time()

        return {
            "score": score,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "train_start": _iso(t0),
            "train_end": _iso(t1),
            "pred_start": _iso(t1),
            "pred_end": _iso(t2),
            "eval_start": _iso(t2),
            "eval_end": _iso(t3),
            "train_duration_s": round(t1 - t0, 4),
            "pred_duration_s": round(t2 - t1, 4),
        }

    def execute(self, task_id: int, model_name: str, cfg: Dict,
                task_type: str = "classification", n_folds: int = 5) -> Dict:
        """
        Run a full n-fold CV for one (task, model, config) triple using OpenML splits.
        Returns dataset metadata and per-fold results suitable for build_prov.
        Supports both classification and regression task types.
        """
        model_map = CLASSIFIER_MAP if task_type == "classification" else REGRESSOR_MAP
        estimator_class = model_map.get(model_name)
        if estimator_class is None:
            raise ValueError(
                f"Unknown model '{model_name}' for task_type='{task_type}'. "
                f"Available: {list(model_map)}"
            )

        task, X, y, dataset_meta = self.get_task_data(task_id, task_type)
        metric_name = dataset_meta["metric"]

        fold_results: List[Dict] = []
        for fold in range(n_folds):
            train_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=0)
            result = self.run_fold(X, y, train_idx, test_idx, estimator_class, cfg, task_type)
            fold_results.append(result)
            if self.verbose:
                print(f"      fold {fold+1}/{n_folds}: "
                      f"{metric_name}={result['score']:.4f}  "
                      f"train={result['train_duration_s']:.2f}s")

        return {"dataset_meta": dataset_meta, "fold_results": fold_results}
