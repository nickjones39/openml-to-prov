# Evidence for the round-3 revision (openml-to-prov v1.4.0)

Raw outputs of the experiments reported in *A Scalable Provenance Framework for Reproducible
Machine Learning Benchmarking*, §6.4, §6.8 and §6.10. Each JSON is written by a script in
`tools/`; the matching `.log` is its console output. The corpora they were run against are the
v1.3.0 corpora (`prov_*_corpus`, `prov_*_synthetic`) and the ten-fold corpus `prov_light10_real`
generated with v1.4.0. Paths inside the files are those of the machine the runs were made on.

| Files | Script | Paper |
|---|---|---|
| `sem_*_v130.json` | `semantic_check.py` over the four v1.3.0 corpora (111,096 documents) | §6.8 |
| `sem_light10_v140.json` | `semantic_check.py` over the v1.4.0 ten-fold corpus | §6.10 |
| `repro_light_v130.json` | `reproduce_from_prov.py`, light corpus, scikit-learn 1.7.1 as recorded (340/360 folds) | §6.10 |
| `sweep_sk*.json` | the same, on the five failing tasks, under 1.5.2 / 1.6.1 / 1.7.0 / 1.7.2 / 1.8.0 | §6.10 |
| `repro_light_v130_sk152.json`, `..._sk180.json` | all 72 runs under 1.5.2 and 1.8.0 (360/360 each) | §6.10 |
| `light10_real.log` | `python -m openml_to_prov --mode light --real --n-folds 10 --validate --require-deps` | §6.10 |
| `repro_light10_v140.json` | re-execution of the ten-fold corpus (720/720 folds and prediction digests) | §6.10 |
| `tenfold_compare.json`, `openml_rf_flows.json` | `tenfold_compare.py --openml --compare-v130`; the 19 OpenML RandomForest flow ids used | §6.10 |
| `light_v2_generation.log`, `conformance_light_v2_full.json` | regeneration of the `light` corpus with v1.4.0 under scikit-learn 1.8.0 (corpora v1.1), and its exhaustive conformance pass | §6.7, §6.8 |
| `repro_light_v2.json`, `sem_light_v2.json`, `light_v2_vs_v1.json` | re-execution of the v1.1 `light` corpus (360/360 folds and prediction digests), its semantic check, and fold-by-fold accuracy agreement with v1 (360/360) | §6.10 |
| `eval_light_v2.json` | `openml_to_prov.evaluate` over the v1.1 `light` corpus | §6.9 |
| `gen_*.json` | `scaling_eval.py gen`, synthetic generation from the archived v1.3.0 code | §6.4, Table 6 |
| `rdflib_series.json`, `oxigraph_series.json` | `scaling_eval.py rdflib` / `oxigraph` over nested samples of `full` | §6.4, Table 6 |

Costs are CPU time measured on an Apple M2 Max while other workloads ran; each measurement
records the load average before and after it.
