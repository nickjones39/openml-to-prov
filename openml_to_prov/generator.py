"""PROV corpus generator for OpenML benchmarks."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from .config import (
    CorpusConfig, CC18_TASK_IDS, EXTENDED_CLASSIFICATION_TASK_IDS, 
    EXTENDED_CLASSIFICATION_FULL_TASK_IDS, REGRESSION_TASK_IDS, DATASET_TEMPLATES
)
from .models import get_original_config, get_classification_configs, get_regression_configs
from .prov_builder import ProvDocumentBuilder
from .utils import (
    generate_run_id, compute_sha256, get_timestamp,
    get_environment_info, generate_synthetic_metric
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
        }
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_tasks(self) -> List[Tuple[int, str]]:
        """Get task IDs with types based on mode."""
        if self.config.mode == "light":
            return [(t, "classification") for t in CC18_TASK_IDS]
        elif self.config.mode == "scaled":
            return [(t, "classification") for t in CC18_TASK_IDS]
        elif self.config.mode == "large":
            tasks = [(t, "classification") for t in CC18_TASK_IDS]
            tasks += [(t, "classification") for t in EXTENDED_CLASSIFICATION_TASK_IDS]
            return tasks
        else:  # full
            tasks = [(t, "classification") for t in CC18_TASK_IDS]
            tasks += [(t, "classification") for t in EXTENDED_CLASSIFICATION_FULL_TASK_IDS]
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
            "n_classes": random.randint(2, 10) if task_type == "classification" else None,
            "task_type": task_type,
            "metric": "accuracy" if task_type == "classification" else "r2_score",
        }

    def build_prov(
        self, task_id: int, ds: Dict, model: str, 
        cfg: Dict, cfg_idx: int, task_type: str
    ) -> ProvDocumentBuilder:
        """Build a PROV document for a single run."""
        b = ProvDocumentBuilder()
        run_id = generate_run_id()
        ts = get_timestamp()
        metric = ds["metric"]

        # Agent
        agent = "ag:system/OpenML"
        b.add_agent(agent, {
            "prov:type": "prov:SoftwareAgent",
            "prov:label": f"OpenML sklearn - {model}"
        })

        # Dataset entity
        ds_ent = f"e:dataset/{ds['dataset_id']}"
        b.add_entity(ds_ent, {
            "prov:label": f"Dataset {ds['dataset_id']}",
            "prov:type": "openml:Dataset",
            "openml:data_id": ds['dataset_id'],
            "ml:n_samples": ds['n_samples'],
            "ml:n_features": ds['n_features'],
            "ml:task_type": task_type,
            "checksum:sha256": compute_sha256(f"ds_{ds['dataset_id']}")
        })

        # Task entity
        task_ent = f"e:task/{task_id}"
        b.add_entity(task_ent, {
            "prov:label": f"Task {task_id}",
            "prov:type": "openml:Task",
            "openml:task_id": task_id,
            "openml:task_type": f"Supervised {task_type.capitalize()}"
        })

        # Flow entity
        flow_ent = f"e:flow/{run_id}"
        b.add_entity(flow_ent, {
            "prov:label": f"Flow: {model}",
            "prov:type": "openml:Flow",
            "openml:model": model,
            "openml:flow_hash": compute_sha256(cfg)[:16]
        })

        # Flow parameters entity
        params_ent = f"e:flowparams/{run_id}"
        b.add_entity(params_ent, {
            "prov:label": f"{model} cfg {cfg_idx}",
            "prov:type": "openml:FlowParameters",
            "params": cfg,
            "flowparams:sha256": compute_sha256(cfg)
        })

        # Environment entity
        env_ent = f"e:env/{run_id}"
        b.add_entity(env_ent, {"prov:type": "openml:Environment", **self.env_info})

        # Experiment activity
        exp_act = f"a:experiment/t{task_id}_{model}_cfg{cfg_idx}"
        b.add_activity(exp_act, {
            "prov:label": f"Experiment: {model} cfg{cfg_idx} task{task_id}",
            "prov:type": "openml:Experiment",
            "prov:startTime": ts,
            "prov:endTime": get_timestamp(60)
        })
        b.add_association(exp_act, agent)
        for ent, role in [(ds_ent, "dataset"), (task_ent, "task"), 
                          (flow_ent, "flow"), (params_ent, "params"), (env_ent, "env")]:
            b.add_used(exp_act, ent, role)

        # Per-fold provenance
        fold_metrics, scores = [], []
        train_sz = int(ds['n_samples'] * 0.8)
        test_sz = ds['n_samples'] - train_sz

        for f in range(self.config.n_folds):
            fn = f + 1
            fs = f"{run_id}_fold{fn}"
            fo = f * 10
            score = generate_synthetic_metric(model, cfg_idx, task_id, f, task_type)
            scores.append(score)

            # Split entity
            split = f"e:split/{fs}"
            b.add_entity(split, {
                "prov:label": f"Split fold {fn}",
                "prov:type": "openml:Split",
                "fold:number": fn,
                "fold:train_size": train_sz,
                "fold:test_size": test_sz
            })

            # Train activity
            train = f"a:train/{fs}"
            b.add_activity(train, {
                "prov:label": f"Train fold {fn}",
                "prov:type": "openml:Train",
                "prov:startTime": get_timestamp(fo),
                "prov:endTime": get_timestamp(fo + 3)
            })
            b.add_association(train, agent)
            for ent, role in [(ds_ent, "data"), (split, "split"), (flow_ent, "flow"),
                              (params_ent, "params"), (env_ent, "env")]:
                b.add_used(train, ent, role)
            b.add_communication(train, exp_act)

            # Model entity
            mdl = f"e:model/{fs}"
            b.add_entity(mdl, {
                "prov:label": f"Model fold {fn}",
                "prov:type": "openml:Model",
                "model:name": model,
                "checksum:sha256": compute_sha256(f"mdl_{fs}")
            })
            b.add_generation(mdl, train)
            b.add_attribution(mdl, agent)

            # Predict activity
            pred_act = f"a:predict/{fs}"
            b.add_activity(pred_act, {
                "prov:label": f"Predict fold {fn}",
                "prov:type": "openml:Predict",
                "prov:startTime": get_timestamp(fo + 4),
                "prov:endTime": get_timestamp(fo + 5)
            })
            b.add_association(pred_act, agent)
            for ent, role in [(mdl, "model"), (ds_ent, "data"), (split, "split")]:
                b.add_used(pred_act, ent, role)
            b.add_communication(pred_act, train)

            # Predictions entity
            preds = f"e:predictions/{fs}"
            b.add_entity(preds, {
                "prov:label": f"Predictions fold {fn}",
                "prov:type": "openml:Predictions",
                "pred:rows": test_sz,
                "checksum:sha256": compute_sha256(f"preds_{fs}")
            })
            b.add_generation(preds, pred_act)

            # Evaluate activity
            eval_act = f"a:evaluate/{fs}"
            b.add_activity(eval_act, {
                "prov:label": f"Evaluate fold {fn}",
                "prov:type": "openml:Evaluate",
                "prov:startTime": get_timestamp(fo + 6),
                "prov:endTime": get_timestamp(fo + 7)
            })
            b.add_association(eval_act, agent)
            b.add_used(eval_act, preds, "predictions")
            b.add_used(eval_act, split, "split")
            b.add_communication(eval_act, pred_act)

            # Metrics entity
            metrics = f"e:metrics/{fs}"
            b.add_entity(metrics, {
                "prov:label": f"Metrics fold {fn}",
                "prov:type": "openml:Metrics",
                f"openml:{metric}": round(score, 6),
                "openml:n_samples": test_sz
            })
            b.add_generation(metrics, eval_act)
            b.add_derivation(metrics, preds)
            b.add_derivation(metrics, split)
            fold_metrics.append(metrics)

        # Aggregate activity
        agg_act = f"a:aggregate/{run_id}"
        b.add_activity(agg_act, {
            "prov:label": "Aggregate",
            "prov:type": "openml:AggregateEvaluate"
        })
        b.add_association(agg_act, agent)
        for fm in fold_metrics:
            b.add_used(agg_act, fm, "fold-metrics")

        # Aggregate metrics entity
        agg_met = f"e:metrics_agg/{run_id}"
        b.add_entity(agg_met, {
            "prov:label": "Aggregate Metrics",
            "prov:type": "openml:AggregateMetrics",
            f"openml:{metric}_mean": round(float(np.mean(scores)), 6),
            f"openml:{metric}_std": round(float(np.std(scores)), 6),
            "openml:n_folds": self.config.n_folds
        })
        b.add_generation(agg_met, agg_act)
        for fm in fold_metrics:
            b.add_derivation(agg_met, fm)

        return b

    def process_task(self, task_id: int, task_type: str, idx: int, total: int) -> Dict:
        """Process a single task with all model configurations."""
        ds = self.generate_dataset_info(task_id, task_type)
        cfgs = self.get_model_configs(task_type)
        task_dir = self.output_dir / f"task_{task_id}"
        task_dir.mkdir(exist_ok=True)
        stats = {"task_id": task_id, "runs": 0, "bytes": 0}

        for model, model_cfgs in cfgs.items():
            model_dir = task_dir / model
            model_dir.mkdir(exist_ok=True)
            for cfg_idx, cfg in enumerate(model_cfgs):
                prov = self.build_prov(task_id, ds, model, cfg, cfg_idx, task_type)
                prov_json = prov.to_json(self.config.pretty_print)
                (model_dir / f"prov_{generate_run_id()}.json").write_text(prov_json)
                sz = len(prov_json.encode())
                self.stats["total_runs"] += 1
                self.stats["total_files"] += 1
                self.stats["total_bytes"] += sz
                stats["runs"] += 1
                stats["bytes"] += sz

        self.stats["tasks_processed"] += 1
        if task_type == "classification":
            self.stats["classification_tasks"] += 1
        else:
            self.stats["regression_tasks"] += 1

        if self.config.verbose:
            mb = self.stats["total_bytes"] / (1024 * 1024)
            print(f"  [{idx+1}/{total}] Task {task_id} ({task_type[:3]}): "
                  f"{stats['runs']} runs, {stats['bytes']/1024:.1f} KB (Total: {mb:.1f} MB)")
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
        expected = n_clf * (1 if self.config.mode == "light" else clf_cfgs) + n_reg * reg_cfgs

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

        # Save manifest
        manifest = {
            "corpus_name": f"OpenML PROV Corpus ({self.config.mode})",
            "version": "2.0",
            "mode": self.config.mode,
            "created": get_timestamp(),
            "stats": self.stats,
            "task_sources": {
                "CC18": len(CC18_TASK_IDS),
                "extended_classification": len(EXTENDED_CLASSIFICATION_TASK_IDS) if self.config.mode == "large" else (
                    len(EXTENDED_CLASSIFICATION_FULL_TASK_IDS) if self.config.mode == "full" else 0
                ),
                "regression": len(REGRESSION_TASK_IDS) if self.config.mode == "full" else 0,
            }
        }
        (self.output_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))

        gb = self.stats["total_bytes"] / (1024**3)
        print(f"\n{'='*70}\nCOMPLETE\n{'='*70}")
        print(f"Classification: {self.stats['classification_tasks']} | "
              f"Regression: {self.stats['regression_tasks']}")
        print(f"Total runs:     {self.stats['total_runs']:,}")
        print(f"Total size:     {self.stats['total_bytes']/(1024**2):.1f} MB ({gb:.2f} GB)")
        return self.stats
