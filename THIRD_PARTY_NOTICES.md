# Data and source attribution

## MedAgentSim

The bundled MedQA file and MedAgentSim-derived code come from [MAXNORM8650/MedAgentSim](https://github.com/MAXNORM8650/MedAgentSim), by Mohammad Almansoori, Komal Kumar and Hisham Cholakkal. Cite *Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions*, MICCAI 2025.

The reference dataset is the repository's 107-case [`datasets/_medqa.jsonl`](https://github.com/MAXNORM8650/MedAgentSim/blob/be691b458bed9cd9fca15ef197675af7b9c8f21c/datasets/_medqa.jsonl), stored here at `egip_aligned/_medagentsim/datasets/_medqa.jsonl`. It is the adapted interactive benchmark, not the full original MedQA dataset. It is copied unchanged. MedAgentSim credits [AgentClinic](https://github.com/samuelschmidgall/agentclinic) for inherited simulation components; MedQA originates from Jin et al., *What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams*, 2021.

MedAgentSim publishes its repository under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Its license notice is retained at `egip_aligned/_medagentsim/LICENSE.txt`. Attribution, noncommercial use and share-alike conditions apply to the bundled MedAgentSim materials. These notices do not assign a new license to independently governed source datasets.

Bundled scenario classes and MeasurementAgent are unchanged excerpts. The similarity scorer comes from the MedAgentSim version used in the EGIP experiments, rather than the current upstream scorer. The transport is replaced with a minimal OpenAI-compatible HTTP client; unused model backends, training code, credentials and machine paths are excluded.

## EGIP

EGIP-authored code is licensed under the [Apache License 2.0](LICENSE). Copyright 2026 EGIP contributors. This license does not replace the licenses of MedAgentSim-derived components or the bundled MedQA data described above.
