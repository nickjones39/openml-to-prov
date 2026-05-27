"""PROV corpus generator for OpenML benchmarks."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    CC18_TASK_IDS,
    DATASET_TEMPLATES,
    EXTENDED_CLASSIFICATION_FULL_TASK_IDS,
    EXTENDED_CLASSIFICATION_TASK_IDS,
    REGRESSION_TASK_IDS,
    CorpusConfig,
)
from .executor import OpenMLExecutor
from .models import (
    get_classification_configs,
    get_original_config,
    get_regression_configs,
)
from .prov_builder import ProvDocumentBuilder
from .utils import (
    compute_sha256,
    generate_run_id,
    generate_synthetic_metric,
    get_environment_info,
    get_timestamp,
)


class CorpusGenerator:
    """Generates PROV corpus in various modes."""

    def __init__(self, config: CorpusConfig):
        self.config = config
        self.env_info = get_environment_info()
        self.stats = {
            "total_runs": 0,
            "total_files": 0,
            "total_bytes": 0,
            "tasks_processed": 0,
            "classification_tasks": 0,
            "regression_tasks": 0,
            "real_execution": False,
            "graph_structure": {
                "nodes_per_graph": {"min": float("inf"), "max": 0, "total": 0},
                "edges_per_graph": {"min": float("inf"), "max": 0, "total": 0},
                "entities_per_graph": {"min": float("inf"), "max": 0, "total": 0},
                "activities_per_graph": {"min": float("inf"), "max": 0, "total": 0},
                "agents_per_graph": {"min": float("inf"), "max": 0, "total": 0},
                "edge_breakdown": {
                    "used": 0,
                    "wasGeneratedBy": 0,
                    "wasAssociatedWith": 0,
                    "wasInformedBy": 0,
                    "wasAttributedTo": 0,
                    "wasDerivedFrom": 0,
                },
            },
        }
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor = (
            OpenMLExecutor(verbose=config.verbose)
            if config.use_real_execution
            else None
        )

    def _cc18_ids(self) -> List[int]:
        """Return CC18 task IDs — live from OpenML suite 99 if executor is on, else hardcoded fallback."""
        if self.executor is not None:
            live = self.executor.get_suite_task_ids(99)
            if live:
                print(f"  Loaded {len(live)} CC18 task IDs from OpenML suite 99")
                return live
            print(f"  Falling back to hardcoded CC18 list ({len(CC18_TASK_IDS)} tasks)")
        return CC18_TASK_IDS

    def get_tasks(self) -> List[Tuple[int, str]]:
        """Get task IDs with types based on mode."""
        cc18 = self._cc18_ids()
        if self.config.mode == "light":
            return [(t, "classification") for t in cc18]
        elif self.config.mode == "scaled":
            return [(t, "classification") for t in cc18]
        elif self.config.mode == "large":
            tasks = [(t, "classification") for t in cc18]
            tasks += [(t, "classification") for t in EXTENDED_CLASSIFICATION_TASK_IDS]
            return tasks
        else:  # full
            tasks = [(t, "classification") for t in cc18]
            tasks += [
                (t, "classification") for t in EXTENDED_CLASSIFICATION_FULL_TASK_IDS
            ]
            tasks += [(t, "regression") for t in REGRESSION_TASK_IDS]
            return tasks

    def get_model_configs(self, task_type: str) -> Dict[str, List[Dict]]:
        """Get model configurations based on mode and task type."""
        if self.config.mode == "light":
            return get_original_config()
        if task_type == "classification":
            return get_classification_configs()
        return get_regression_configs()

    def generate_dataset_info(self, task_id: int, task_type: str) -> Dict:
        """Generate synthetic dataset information."""
        random.seed(task_id)
        templates = DATASET_TEMPLATES[task_type]
        return {
            "dataset_id": task_id,
            "dataset_name": random.choice(templates).format(task_id),
            "n_samples": random.randint(500, 50000),
            "n_features": random.randint(10, 500),
            "n_classes": random.randint(2, 10)
            if task_type == "classification"
            else None,
            "task_type": task_type,
            "metric": "accuracy" if task_type == "classification" else "r2_score",
        }

    def build_prov(
        self,
        task_id: int,
        ds: Dict,
        model: str,
        cfg: Dict,
        cfg_idx: int,
        task_type: str,
        real_fold_results: Optional[List[Dict]] = None,
    ) -> ProvDocumentBuilder:
        """Build a PROV document for a single run."""
        b = ProvDocumentBuilder()
        run_id = generate_run_id()
        ts = get_timestamp()
        metric = ds["metric"]

        # Agent
        agent = "ag:system/OpenML"
        b.add_agent(
            agent,
            {
                "prov:type": "prov:SoftwareAgent",
                "prov:label": f"OpenML sklearn - {model}",
            },
        )

        # Dataset entity
        ds_ent = f"e:dataset/{ds['dataset_id']}"
        b.add_entity(
            ds_ent,
            {
                "prov:label": f"Dataset {ds['dataset_id']}",
                "prov:type": "openml:Dataset",
                "openml:data_id": ds["dataset_id"],
                "ml:n_samples": ds["n_samples"],
                "ml:n_features": ds["n_features"],
                "ml:task_type": task_type,
                "checksum:sha256": compute_sha256(f"ds_{ds['dataset_id']}"),
            },
        )

        # Task entity
        task_ent = f"e:task/{task_id}"
        b.add_entity(
            task_ent,
            {
                "prov:label": f"Task {task_id}",
                "prov:type": "openml:Task",
                "openml:task_id": task_id,
                "openml:task_type": f"Supervised {task_type.capitalize()}",
            },
        )

        # Flow entity
        flow_ent = f"e:flow/{run_id}"
        b.add_entity(
            flow_ent,
            {
                "prov:label": f"Flow: {model}",
                "prov:type": "openml:Flow",
                "openml:model": model,
                "openml:flow_hash": compute_sha256(cfg)[:16],
            },
        )

        # Flow parameters entity
        params_ent = f"e:flowparams/{run_id}"
        b.add_entity(
            params_ent,
            {
                "prov:label": f"{model} cfg {cfg_idx}",
                "prov:type": "openml:FlowParameters",
                "openml:parameters": json.dumps(cfg, sort_keys=True),
                "checksum:sha256": compute_sha256(cfg),
            },
        )

        # Environment entity
        env_ent = f"e:env/{run_id}"
        b.add_entity(env_ent, {"prov:type": "openml:Environment", **self.env_info})

        # Experiment activity
        exp_act = f"a:experiment/t{task_id}_{model}_cfg{cfg_idx}"
        b.add_activity(
            exp_act,
            {
                "prov:label": f"Experiment: {model} cfg{cfg_idx} task{task_id}",
                "prov:type": "openml:Experiment",
                "prov:startTime": ts,
                "prov:endTime": get_timestamp(60),
            },
        )
        b.add_association(exp_act, agent)
        for ent, role in [
            (ds_ent, "dataset"),
            (task_ent, "task"),
            (flow_ent, "flow"),
            (params_ent, "params"),
            (env_ent, "env"),
        ]:
            b.add_used(exp_act, ent, role)

        # Per-fold provenance
        fold_metrics, scores = [], []
        default_train_sz = int(ds["n_samples"] * 0.8)
        default_test_sz = ds["n_samples"] - default_train_sz

        for f in range(self.config.n_folds):
            fn = f + 1
            fs = f"{run_id}_fold{fn}"
            fo = f * 10

            if real_fold_results is not None:
                rd = real_fold_results[f]
                score = rd["score"]
                train_sz = rd["train_size"]
                test_sz = rd["test_size"]
                train_start = rd["train_start"]
                train_end = rd["train_end"]
                pred_start = rd["pred_start"]
                pred_end = rd["pred_end"]
                eval_start = rd["eval_start"]
                eval_end = rd["eval_end"]
            else:
                score = generate_synthetic_metric(model, cfg_idx, task_id, f, task_type)
                train_sz = default_train_sz
                test_sz = default_test_sz
                train_start = get_timestamp(fo)
                train_end = get_timestamp(fo + 3)
                pred_start = get_timestamp(fo + 4)
                pred_end = get_timestamp(fo + 5)
                eval_start = get_timestamp(fo + 6)
                eval_end = get_timestamp(fo + 7)

            scores.append(score)

            # Split entity
            split = f"e:split/{fs}"
            b.add_entity(
                split,
                {
                    "prov:label": f"Split fold {fn}",
                    "prov:type": "openml:Split",
                    "fold:number": fn,
                    "fold:train_size": train_sz,
                    "fold:test_size": test_sz,
                },
            )

            # Train activity
            train = f"a:train/{fs}"
            b.add_activity(
                train,
                {
                    "prov:label": f"Train fold {fn}",
                    "prov:type": "openml:Train",
                    "prov:startTime": train_start,
                    "prov:endTime": train_end,
                },
            )
            b.add_association(train, agent)
            for ent, role in [
                (ds_ent, "data"),
                (split, "split"),
                (flow_ent, "flow"),
                (params_ent, "params"),
                (env_ent, "env"),
            ]:
                b.add_used(train, ent, role)
            b.add_communication(train, exp_act)

            # Model entity
            mdl = f"e:model/{fs}"
            b.add_entity(
                mdl,
                {
                    "prov:label": f"Model fold {fn}",
                    "prov:type": "openml:Model",
                    "openml:model_name": model,
                    "checksum:sha256": compute_sha256(f"mdl_{fs}"),
                },
            )
            b.add_generation(mdl, train)
            b.add_attribution(mdl, agent)

            # Predict activity
            pred_act = f"a:predict/{fs}"
            b.add_activity(
                pred_act,
                {
                    "prov:label": f"Predict fold {fn}",
                    "prov:type": "openml:Predict",
                    "prov:startTime": pred_start,
                    "prov:endTime": pred_end,
                },
            )
            b.add_association(pred_act, agent)
            for ent, role in [(mdl, "model"), (ds_ent, "data"), (split, "split")]:
                b.add_used(pred_act, ent, role)
            b.add_communication(pred_act, train)

            # Predictions entity
            preds = f"e:predictions/{fs}"
            b.add_entity(
                preds,
                {
                    "prov:label": f"Predictions fold {fn}",
                    "prov:type": "openml:Predictions",
                    "ml:n_predictions": test_sz,
                    "checksum:sha256": compute_sha256(f"preds_{fs}"),
                },
            )
            b.add_generation(preds, pred_act)

            # Evaluate activity
            eval_act = f"a:evaluate/{fs}"
            b.add_activity(
                eval_act,
                {
                    "prov:label": f"Evaluate fold {fn}",
                    "prov:type": "openml:Evaluate",
                    "prov:startTime": eval_start,
                    "prov:endTime": eval_end,
                },
            )
            b.add_association(eval_act, agent)
            b.add_used(eval_act, preds, "predictions")
            b.add_used(eval_act, split, "split")
            b.add_communication(eval_act, pred_act)

            # Metrics entity
            metrics = f"e:metrics/{fs}"
            b.add_entity(
                metrics,
                {
                    "prov:label": f"Metrics fold {fn}",
                    "prov:type": "openml:Metrics",
                    f"openml:{metric}": round(score, 6),
                    "openml:n_samples": test_sz,
                },
            )
            b.add_generation(metrics, eval_act)
            b.add_derivation(metrics, preds)
            b.add_derivation(metrics, split)
            fold_metrics.append(metrics)

        # Aggregate activity
        agg_act = f"a:aggregate/{run_id}"
        b.add_activity(
            agg_act,
            {"prov:label": "Aggregate", "prov:type": "openml:AggregateEvaluate"},
        )
        b.add_association(agg_act, agent)
        for fm in fold_metrics:
            b.add_used(agg_act, fm, "fold-metrics")

        # Aggregate metrics entity
        agg_met = f"e:metrics_agg/{run_id}"
        b.add_entity(
            agg_met,
            {
                "prov:label": "Aggregate Metrics",
                "prov:type": "openml:AggregateMetrics",
                f"openml:{metric}_mean": round(float(np.mean(scores)), 6),
                f"openml:{metric}_std": round(float(np.std(scores)), 6),
                "openml:n_folds": self.config.n_folds,
            },
        )
        b.add_generation(agg_met, agg_act)
        for fm in fold_metrics:
            b.add_derivation(agg_met, fm)

        return b

    def _update_graph_structure_stats(self, gs: Dict):
        """Accumulate per-run node/edge counts into running min/max/total."""
        gs_out = self.stats["graph_structure"]
        for key, field in [
            ("nodes", "nodes_per_graph"),
            ("edges", "edges_per_graph"),
            ("entities", "entities_per_graph"),
            ("activities", "activities_per_graph"),
            ("agents", "agents_per_graph"),
        ]:
            bucket = gs_out[field]
            v = gs[key]
            if v < bucket["min"]:
                bucket["min"] = v
            if v > bucket["max"]:
                bucket["max"] = v
            bucket["total"] += v
        for rel in [
            "used",
            "wasGeneratedBy",
            "wasAssociatedWith",
            "wasInformedBy",
            "wasAttributedTo",
            "wasDerivedFrom",
        ]:
            gs_out["edge_breakdown"][rel] += gs[rel]

    def process_task(self, task_id: int, task_type: str, idx: int, total: int) -> Dict:
        """Process a single task with all model configurations."""
        ds = self.generate_dataset_info(task_id, task_type)
        cfgs = self.get_model_configs(task_type)
        task_dir = self.output_dir / f"task_{task_id}"
        task_dir.mkdir(exist_ok=True)
        stats = {"task_id": task_id, "runs": 0, "bytes": 0}

        # Pre-check task type once before the config loop so we don't
        # re-fetch OpenML metadata and repeat the skip message per config.
        task_executor = self.executor
        if task_executor is not None:
            try:
                task_executor.validate_task_type(task_id, task_type)
            except ValueError as exc:
                if self.config.verbose:
                    print(f"    SKIP task {task_id}: {exc} — using synthetic data")
                task_executor = None

        for model, model_cfgs in cfgs.items():
            model_dir = task_dir / model
            model_dir.mkdir(exist_ok=True)
            for cfg_idx, cfg in enumerate(model_cfgs):
                real_fold_results = None
                if task_executor is not None:
                    try:
                        result = task_executor.execute(
                            task_id, model, cfg, task_type, self.config.n_folds
                        )
                        ds = result["dataset_meta"]
                        real_fold_results = result["fold_results"]
                    except Exception as exc:
                        print(
                            f"    WARNING: real execution failed unexpectedly ({exc}), falling back to synthetic"
                        )
                if real_fold_results is not None:
                    self.stats["real_execution"] = True
                prov = self.build_prov(
                    task_id, ds, model, cfg, cfg_idx, task_type, real_fold_results
                )
                gs = prov.graph_stats()
                prov_json = prov.to_json(self.config.pretty_print)
                (model_dir / f"prov_{generate_run_id()}.json").write_text(prov_json)
                sz = len(prov_json.encode())
                self.stats["total_runs"] += 1
                self.stats["total_files"] += 1
                self.stats["total_bytes"] += sz
                stats["runs"] += 1
                stats["bytes"] += sz
                self._update_graph_structure_stats(gs)

        self.stats["tasks_processed"] += 1
        if task_type == "classification":
            self.stats["classification_tasks"] += 1
        else:
            self.stats["regression_tasks"] += 1

        if self.config.verbose:
            mb = self.stats["total_bytes"] / (1024 * 1024)
            print(
                f"  [{idx + 1}/{total}] Task {task_id} ({task_type[:3]}): "
                f"{stats['runs']} runs, {stats['bytes'] / 1024:.1f} KB (Total: {mb:.1f} MB)"
            )
        return stats

    def generate(self, max_tasks: Optional[int] = None) -> Dict:
        """Generate the full corpus."""
        tasks = self.get_tasks()
        if max_tasks:
            tasks = tasks[:max_tasks]

        n_clf = sum(1 for _, t in tasks if t == "classification")
        n_reg = sum(1 for _, t in tasks if t == "regression")
        clf_cfgs = sum(len(c) for c in get_classification_configs().values())
        reg_cfgs = sum(len(c) for c in get_regression_configs().values())
        expected = (
            n_clf * (1 if self.config.mode == "light" else clf_cfgs) + n_reg * reg_cfgs
        )

        print("=" * 70)
        print(f"PROV Corpus Generator - {self.config.mode.upper()} MODE")
        print("=" * 70)
        print(f"Classification tasks: {n_clf}")
        print(f"Regression tasks:     {n_reg}")
        print(f"Total tasks:          {len(tasks)}")
        print(f"Expected runs:        {expected:,}")
        print(f"Output:               {self.output_dir}")
        print("=" * 70)

        for i, (tid, ttype) in enumerate(tasks):
            self.process_task(tid, ttype, i, len(tasks))

        # Finalise graph structure averages
        n = self.stats["total_runs"]
        if n > 0:
            gs = self.stats["graph_structure"]
            for field in [
                "nodes_per_graph",
                "edges_per_graph",
                "entities_per_graph",
                "activities_per_graph",
                "agents_per_graph",
            ]:
                bucket = gs[field]
                bucket["avg"] = round(bucket["total"] / n, 2)
                if bucket["min"] == float("inf"):
                    bucket["min"] = 0

        # Save manifest
        manifest = {
            "corpus_name": f"OpenML PROV Corpus ({self.config.mode})",
            "version": "2.0",
            "mode": self.config.mode,
            "created": get_timestamp(),
            "stats": self.stats,
            "task_sources": {
                "CC18": len(CC18_TASK_IDS),
                "extended_classification": len(EXTENDED_CLASSIFICATION_TASK_IDS)
                if self.config.mode == "large"
                else (
                    len(EXTENDED_CLASSIFICATION_FULL_TASK_IDS)
                    if self.config.mode == "full"
                    else 0
                ),
                "regression": len(REGRESSION_TASK_IDS)
                if self.config.mode == "full"
                else 0,
            },
        }
        (self.output_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        gb = self.stats["total_bytes"] / (1024**3)
        gs = self.stats["graph_structure"]
        nodes = gs["nodes_per_graph"]
        edges = gs["edges_per_graph"]
        ents = gs["entities_per_graph"]
        acts = gs["activities_per_graph"]
        agts = gs["agents_per_graph"]
        print(f"\n{'=' * 70}\nCOMPLETE\n{'=' * 70}")
        print(
            f"Classification: {self.stats['classification_tasks']} | "
            f"Regression: {self.stats['regression_tasks']}"
        )
        print(f"Total runs:     {self.stats['total_runs']:,}")
        print(
            f"Total size:     {self.stats['total_bytes'] / (1024**2):.1f} MB ({gb:.2f} GB)"
        )
        print(f"\nGraph structure (per run):")
        print(
            f"  Nodes:      avg={nodes.get('avg', 'n/a')}  "
            f"min={nodes['min']}  max={nodes['max']}"
        )
        print(
            f"  Edges:      avg={edges.get('avg', 'n/a')}  "
            f"min={edges['min']}  max={edges['max']}"
        )
        print(
            f"  Entities:   avg={ents.get('avg', 'n/a')}  "
            f"min={ents['min']}  max={ents['max']}"
        )
        print(
            f"  Activities: avg={acts.get('avg', 'n/a')}  "
            f"min={acts['min']}  max={acts['max']}"
        )
        print(
            f"  Agents:     avg={agts.get('avg', 'n/a')}  "
            f"min={agts['min']}  max={agts['max']}"
        )
        eb = gs["edge_breakdown"]
        print(f"  Edge breakdown (totals across all runs):")
        for rel, count in eb.items():
            print(f"    {rel}: {count:,}")
        return self.stats
