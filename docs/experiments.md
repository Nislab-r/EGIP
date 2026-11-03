# 📊 Experiment Commands

Install the package and start the model service as described in the [README](../README.md). Commands below use the bundled MedQA dataset and `Qwen/Qwen3-8B`. Model service calls perform real inference and scoring.

## 🧠 Terminal Strategies

Run three repetitions with fresh acquisition:

```bash
egip-reproduce --terminal vote --output results/medqa_vote
egip-reproduce --terminal full --output results/medqa_full
```

For comparisons on identical acquired observations, reuse a saved Direct result instead:

```bash
egip-replay --source results/medqa_direct/repeat_1/case_0000/result.json \
  --model Qwen/Qwen3-8B --terminal vote \
  --first-direct results/medqa_direct/repeat_1/case_0000/result.json \
  --score --dataset MedQA --case-id 0 --output results/vote_case0
```

Vote uses five samples, including the saved Direct answer when `--first-direct` is provided. For Full, change `--terminal` to `full` and omit `--first-direct`. Without `--score`, fixed-evidence prediction returns an answer without automated scoring.

## 🧩 Ablations

Acquisition interventions keep Direct fixed:

```bash
egip-reproduce --acquisition-ablation no_action_menu --output results/no_menu
egip-reproduce --acquisition-ablation native_measurement --output results/native_measurement
```

Remove observations only at the terminal stage for a saved case:

```bash
egip-replay --source results/medqa_direct/repeat_1/case_0000/result.json \
  --model Qwen/Qwen3-8B --arm no_observations \
  --score --dataset MedQA --case-id 0 --output results/no_observations_case0
```

Repeat fixed-evidence commands for the desired saved cases and repetitions. MedQA provides no candidate labels, so `no_candidate_labels` does not apply to this dataset.

## 📂 Outputs and Native Baseline

Use a new output directory for each configuration. `failure.json` records errors. `egip-summarize results/medqa_direct/repeat_1` summarizes a single acquisition run. Pass multiple repetition directories to obtain their mean and sample SD.

The bundled MedAgentSim subset runs EGIP. To execute the original native doctor workflow, use a complete compatible MedAgentSim checkout with its dependencies:

```bash
egip-native --root /path/to/full/MedAgentSim -- <native arguments>
```

Native arguments and transport settings are passed through unchanged. See [Integration](integration.md) for the external checkout interface.
