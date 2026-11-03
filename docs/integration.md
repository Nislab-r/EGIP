# 🔌 How EGIP connects to MedAgentSim

The adapter reuses MedAgentSim's scenario API and the experiment's `compare_results` scorer. EGIP replaces the interaction loop for EGIP runs. It does not train the backbone or modify an external native workflow.

```text
MedQA JSONL → ScenarioMedQA → EGIP acquisition → e_T → terminal prediction → scorer
                              R + B + O         E     Direct / Vote / Full
```

- **R — role-specific views:** `context_engineering.py` and `engine.py` construct separate doctor/patient inputs. The doctor receives disclosed history, not the raw patient record or reference answer.
- **B — result-blinded action menus:** `protocol.py` derives test identifiers/names; `planner.py` orders available actions without reading their results.
- **O — record lookup:** `actions.py` resolves a requested test to its stored result. Guards reject invalid requests and redirect repeats or premature diagnoses. Unknown values remain unknown.
- **E — acquired-evidence interface:** `engine.py` passes the objective, candidate labels and ordered Patient/Measurement observations to `terminals.py`. Doctor assertions are not independent observations.

`hosts.py` connects these operations to MedAgentSim. `run.py` calls `diagnosis_information()` only after prediction, then invokes the scorer. The scenario object holds the reference internally, but it is never passed to the model during acquisition or terminal prediction.

The bundled `_medagentsim/` contains the original scenario classes, original MeasurementAgent for its ablation, and the experiment's similarity scorer. `query_model.py` is a new small HTTP transport exposing the same `BAgent.query_model` interface. It replaces deployment-specific providers and aliases, preserves system/user messages, and explicitly sends temperature/max tokens and non-thinking configuration. HTTP timeout defaults to 120 seconds. The scorer retains its original three attempts and returns incorrect after invalid replies or exhausted failures. This behavior is not an independent clinical judgment.

For your own compatible checkout, pass `--host-root /path/to/MedAgentSim` to `egip-run` or `egip-replay`. That path must contain `medsim/query_model.py` (`BAgent`), `medsim/core/scenario.py` (scenario classes and `resolve_model_name`), `medsim/core/agent.py` (`compare_results`, `MeasurementAgent`) and `datasets/`. External checkouts use their own dependencies and transport settings. The bundled setup requires no external checkout.

Run software tests with `python -m unittest discover -s tests -v`. The release changes packaging and transport, not EGIP's acquisition prompts or selection/validation rules. Model accuracy must be measured against a live service.
