"""
Structured context maps for MedAgentSim experiments.

The goal is to give each simuated role a compact map of boundaries, phase
order, and priorities instead of a long linear instruction list. The map is
rendered into prompts, but remains structured in code so different model sizes
can receive different scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class ModelStrategy:
    """Controls how much scaffolding or boundary enforcement to inject."""

    name: str
    intent: str
    prompt_budget: str
    doctor_mode: str
    patient_mode: str
    measurement_mode: str
    final_mode: str


MODEL_STRATEGIES: Dict[str, ModelStrategy] = {
    "small": ModelStrategy(
        name="small",
        intent="increase agent capability with explicit phase scaffolds and evidence ledger",
        prompt_budget="compact but directive",
        doctor_mode="Use the phase map as a checklist. Prefer one high-value question or one test per turn.",
        patient_mode="Answer directly from patient facts. If asked about missing information, say you do not know.",
        measurement_mode="Match requested tests to available findings before answering. Do not invent unavailable results.",
        final_mode="Build a short evidence ledger first, then choose one diagnosis using the strongest positive findings.",
    ),
    "balanced": ModelStrategy(
        name="balanced",
        intent="balance role fidelity, diagnostic breadth, and concise reasoning",
        prompt_budget="moderate",
        doctor_mode="Use the phase map to choose the next highest-yield action; avoid repeating covered ground.",
        patient_mode="Stay in character and reveal only symptom-level information.",
        measurement_mode="Return only data supported by the exam map; otherwise use UNAVAILABLE / NOT RECORDED.",
        final_mode="Weigh supporting and conflicting evidence before the final diagnosis.",
    ),
    "large": ModelStrategy(
        name="large",
        intent="prevent capability overreach with strict role boundaries and evidence-only decisions",
        prompt_budget="minimal",
        doctor_mode="Do not use hidden labels, ground truth, or unsupported tests. Ask only clinically justified questions.",
        patient_mode="Do not diagnose yourself, explain disease mechanisms, or reveal the hidden diagnosis.",
        measurement_mode="Do not infer beyond recorded test data. Never add new abnormal findings.",
        final_mode="Only diagnose from observed dialogue and returned tests. Prefer a specific etiology over a broad syndrome when returned evidence supports it. State exactly one final diagnosis.",
    ),
}


def infer_model_strategy(model_name: str, override: Optional[str] = None) -> ModelStrategy:
    """Infer the prompt strategy from model name unless explicitly overridden."""

    if override:
        key = override.strip().lower()
        if key not in MODEL_STRATEGIES:
            raise ValueError(f"Unknown context strategy '{override}'. Expected one of {sorted(MODEL_STRATEGIES)}.")
        return MODEL_STRATEGIES[key]

    name = (model_name or "").lower()
    size = _extract_model_size_b(name)
    if size is not None:
        if size <= 14:
            return MODEL_STRATEGIES["small"]
        if size >= 60:
            return MODEL_STRATEGIES["large"]
    if any(token in name for token in ["0.5b", "1.5b", "3b", "7b", "8b", "11b", "qwen2", "small"]):
        return MODEL_STRATEGIES["small"]
    if any(token in name for token in ["70b", "72b", "405b", "large", "gpt4", "gpt-4", "claude"]):
        return MODEL_STRATEGIES["large"]
    return MODEL_STRATEGIES["balanced"]


def _extract_model_size_b(model_name: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", model_name)
    return float(match.group(1)) if match else None


def render_role_map(
    role: str,
    strategy: ModelStrategy,
    scenario_context: Any,
    *,
    max_infs: Optional[int] = None,
    used_infs: Optional[int] = None,
    img_request: bool = False,
    bias_prompt: str = "",
) -> str:
    """Render a compact structured map for a role."""

    role_key = role.lower()
    sections: List[str] = [
        "CONTEXT MAP",
        f"role: {role_key}",
        f"model_strategy: {strategy.name}",
        f"strategy_goal: {strategy.intent}",
        "priority_order:",
        "- obey role boundary",
        "- use case facts and dialogue history",
        "- follow the current phase",
        "- keep output format exact",
    ]

    if role_key == "doctor":
        sections.extend(_doctor_map(strategy, scenario_context, max_infs, used_infs, img_request))
    elif role_key == "patient":
        sections.extend(_patient_map(strategy, scenario_context))
    elif role_key == "measurement":
        sections.extend(_measurement_map(strategy, scenario_context))
    elif role_key == "council":
        sections.extend(_council_map(strategy, scenario_context))
    else:
        raise ValueError(f"Unsupported role map: {role}")

    if bias_prompt:
        sections.extend(["bias_context:", _indent(bias_prompt.strip())])

    return "\n".join(sections).strip()


def render_turn_prompt(role: str, history: str, latest_input: str, *, phase_hint: str = "") -> str:
    """Keep turn prompts short and consistent across roles."""

    return "\n".join(
        part
        for part in [
            f"PHASE_HINT: {phase_hint}" if phase_hint else "",
            "DIALOGUE_HISTORY:",
            history.strip() if history.strip() else "(none yet)",
            "LATEST_INPUT:",
            latest_input.strip() if latest_input.strip() else "(none)",
            f"NEXT_{role.upper()}_OUTPUT:",
        ]
        if part
    )


def evidence_ledger_prompt(patient_statement: str, history: str, doctor_discussion: str) -> str:
    """Ask the model to compress discussion into a sortable evidence ledger."""

    return (
        "Build a compact diagnostic evidence ledger from the case.\n"
        "Return exactly these fields:\n"
        "positive_findings: semicolon-separated facts\n"
        "negative_or_normal_findings: semicolon-separated facts\n"
        "candidate_diagnoses: 3-5 diagnoses ranked by evidence\n"
        "role_boundary_check: confirm no hidden ground-truth or unavailable data was used\n\n"
        f"patient_statement: {patient_statement}\n"
        f"dialogue_history: {history}\n"
        f"doctor_discussion: {doctor_discussion}\n"
    )


def _doctor_map(
    strategy: ModelStrategy,
    presentation: Any,
    max_infs: Optional[int],
    used_infs: Optional[int],
    img_request: bool,
) -> List[str]:
    remaining = None
    if max_infs is not None and used_infs is not None:
        remaining = max(max_infs - used_infs, 0)

    phases = [
        "1_intake: clarify chief complaint, onset, severity, associated symptoms",
        "2_differential: compare 2-4 plausible disease families",
        "3_targeted_tests: request only tests that discriminate among candidates",
        "4_closure: if final turn or enough evidence, output DIAGNOSIS READY: <single diagnosis>",
    ]
    return [
        "role_boundary:",
        "- You are the doctor, not the patient, measurement reader, or evaluator.",
        "- You cannot see the hidden correct diagnosis.",
        "- You may ask one concise question OR request one test per turn.",
        "- Do not repeat a test that appears in ALREADY_REQUESTED_TESTS_DO_NOT_REPEAT.",
        "- Before the final inference round, do not output DIAGNOSIS READY.",
        "- DIAGNOSIS READY must be followed by one concrete disease name, not a plan or uncertainty statement.",
        "- You must not fabricate test results.",
        "phase_order:",
        *_prefix(phases),
        "current_budget:",
        f"- max_questions: {max_infs if max_infs is not None else 'unknown'}",
        f"- used_questions: {used_infs if used_infs is not None else 'unknown'}",
        f"- remaining_questions: {remaining if remaining is not None else 'unknown'}",
        "model_specific_control:",
        f"- {strategy.doctor_mode}",
        f"- {strategy.final_mode}",
        "allowed_outputs:",
        "- normal dialogue: 1-3 sentences",
        "- test request: REQUEST TEST: <test name>",
        "- final answer on final round only: DIAGNOSIS READY: <single concrete diagnosis>",
        "- image request: REQUEST IMAGES" if img_request else "- image request: not available",
        "case_map_for_doctor:",
        _indent(_stringify(presentation)),
    ]


def _patient_map(strategy: ModelStrategy, patient_info: Any) -> List[str]:
    return [
        "role_boundary:",
        "- You are the patient only.",
        "- Do not reveal or name the diagnosis unless the patient facts explicitly say the patient believes it.",
        "- Do not provide test results or clinician reasoning.",
        "phase_order:",
        "- listen to the doctor's latest question",
        "- answer from known patient facts",
        "- if unknown, answer uncertainty naturally",
        "model_specific_control:",
        f"- {strategy.patient_mode}",
        "allowed_outputs:",
        "- 1-3 dialogue sentences in patient voice",
        "patient_fact_map:",
        _indent(_stringify(patient_info)),
    ]


def _measurement_map(strategy: ModelStrategy, exam_info: Any) -> List[str]:
    return [
        "role_boundary:",
        "- You are the measurement reader only.",
        "- You cannot interview the patient or diagnose.",
        "- You cannot create abnormal findings that are absent from the exam map.",
        "lookup_order:",
        "- parse requested test name",
        "- match exact or semantically close available test/finding",
        "- return recorded result if available",
        "- otherwise return RESULTS: UNAVAILABLE / NOT RECORDED",
        "model_specific_control:",
        f"- {strategy.measurement_mode}",
        "allowed_outputs:",
        "- RESULTS: <available result>",
        "- RESULTS: UNAVAILABLE / NOT RECORDED",
        "exam_result_map:",
        _indent(_stringify(exam_info)),
    ]


def _council_map(strategy: ModelStrategy, presentation: Any) -> List[str]:
    return [
        "role_boundary:",
        "- You are a diagnostic council using only the dialogue, tests, and doctor discussion.",
        "- Do not use hidden labels or evaluate against the answer key.",
        "decision_order:",
        "- list decisive positive findings",
        "- list decisive negative/normal findings",
        "- rank candidates",
        "- choose one final diagnosis",
        "model_specific_control:",
        f"- {strategy.final_mode}",
        "allowed_outputs:",
        "- DIAGNOSIS READY: <single diagnosis>",
        "case_objective_map:",
        _indent(_stringify(presentation)),
    ]


def _stringify(value: Any) -> str:
    if isinstance(value, Mapping):
        return "\n".join(f"{key}: {_stringify(val)}" for key, val in value.items())
    if isinstance(value, list):
        return "; ".join(_stringify(item) for item in value)
    return str(value)


def _indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in str(text).splitlines())


def _prefix(lines: Iterable[str]) -> List[str]:
    return [f"- {line}" for line in lines]
