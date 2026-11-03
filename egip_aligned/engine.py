"""Shared legacy EGIP policy. Host baseline code is never patched or executed here."""
import copy
import re
from dataclasses import asdict, dataclass
from .context_engineering import infer_model_strategy, render_role_map, render_turn_prompt
from .planner import plan, _menu_priority
from .protocol import action_catalog, candidate_labels, stringify
from .actions import extract_action, execute_action, reject_text
from .finalizer import finalize


@dataclass(frozen=True)
class Config:
    max_turns: int = 20
    strategy: str = 'small'
    candidate_policy: str = 'legacy'
    evidence_gate: bool = True
    flat_priority: bool = False

    def __post_init__(self):
        if self.max_turns < 1:
            raise ValueError('max_turns must be positive')
        if self.candidate_policy not in ('legacy', 'none'):
            raise ValueError('candidate_policy must be legacy or none')
        infer_model_strategy('', self.strategy)


def run_episode(scenario, model_name, query, config=Config(), on_event=None, terminal=None):
    """query(stage, prompt, system) uses the host transport without changing decoding.

    Does not read diagnosis_information(). Evaluation is a separate host operation.
    """
    strategy = infer_model_strategy(model_name, config.strategy)
    objective = copy.deepcopy(scenario.examiner_information())
    patient_facts = copy.deepcopy(scenario.patient_information())
    exam = copy.deepcopy(scenario.exam_information())
    actions = [a.public for a in action_catalog(exam)]
    candidates = candidate_labels(scenario) if config.candidate_policy == 'legacy' else []
    events, requested, patient_history, decisions = [], [], '', []
    def emit(role, text, **metadata):
        e = dict(metadata, event_id=f'e{len(events)+1:04d}', speaker=role, text=text)
        events.append(e)
        if on_event:
            on_event(copy.deepcopy(events))
        return e
    def history():
        return '\n'.join(f"[{e['event_id']}] {e['speaker']}: {e['text']}" for e in events)
    for turn in range(config.max_turns - 1):
        remaining = [a for a in actions if a['action_id'] not in set(requested)]
        system = render_role_map('doctor', strategy, objective,
                                 max_infs=config.max_turns, used_infs=turn)
        system += ('\nDIAGNOSTIC_TOOL_MAP_RESULT_BLINDED:\n' +
                   plan(actions, requested, 12, config.flat_priority))
        if strategy.name == 'small':
            menu = sorted(remaining, key=lambda a: _menu_priority(a['name']))[:36]
            system += '\nAVAILABLE_TEST_MENU_NAMES_ONLY:\n' + '\n'.join(
                '- '+a['action_id']+' | '+a['name'] for a in menu)
        system += ('\nPrefer one high-priority, non-repeated measurement when it separates diagnoses. '
                   'Use REQUEST TEST: <action ID>. Do not simulate patient or test responses.')
        if candidates:
            system += '\nUNORDERED_DIAGNOSIS_CANDIDATE_MAP:\n' + stringify(candidates)
        planning = history() + '\nALREADY_REQUESTED_TESTS_DO_NOT_REPEAT: ' + ', '.join(requested[-8:])
        planning += '\n' + plan(actions, requested, 6 if strategy.name == 'small' else 8,
                                 config.flat_priority)
        prompt = render_turn_prompt('doctor', planning, '',
                                    phase_hint='choose the next highest-yield question or one test request')
        raw = query(f'doctor_{turn:02d}', prompt, system)
        if not isinstance(raw, str) or not raw.strip() or raw.startswith('Error:'):
            raise RuntimeError('Doctor transport failed; do not synthesize an action')
        effective, reason = raw.strip(), None
        if re.search(r'DIAGNOSIS\s+READY', raw, re.I):
            effective = ('Can you clarify the most important symptom pattern, including onset, '
                         'triggers, progression, and relieving factors?')
            reason = 'premature_finalization'
        if re.search(r'REQUEST\s+IMAGES', effective, re.I):
            action = {'kind': 'invalid', 'reason': 'image_adapter_not_configured'}
        else:
            action = extract_action(effective, actions)
        if action['kind'] == 'test' and action['action_id'] in requested:
            effective = ('Can you clarify the main symptom pattern, including onset, location, '
                         'triggers, progression, and relieving factors?')
            reason = 'repeated_action'
            action = {'kind': 'question'}
        if action['kind'] == 'test':
            effective = action['command']
        emit('Doctor', effective, turn=turn, raw_output=raw, guard_reason=reason)
        decisions.append({'turn': turn, 'raw_output': raw, 'effective_output': effective,
                          'guard_reason': reason, 'action': action})
        if action['kind'] == 'invalid':
            emit('Interface', reject_text(action['reason']), turn=turn,
                 status='action_rejected', reason=action['reason'])
            continue
        if action['kind'] == 'test':
            record = execute_action(action, exam)
            requested.append(action['action_id'])
            text = record.pop('text')
            emit('Measurement', text, turn=turn, question_context=effective, **record)
        else:
            prompt = render_turn_prompt('patient', patient_history, effective,
                                        phase_hint="answer the doctor's latest question without revealing the diagnosis")
            response = query(f'patient_{turn:02d}', prompt,
                             render_role_map('patient', strategy, patient_facts))
            if not isinstance(response, str) or not response.strip() or response.startswith('Error:'):
                raise RuntimeError('Patient transport failed; do not save as a clinical observation')
            patient_history += f'Doctor: {effective}\nPatient: {response}\n'
            emit('Patient', response, turn=turn, question_context=effective, status='patient_report')
    evidence = {'initial_objective': objective, 'candidate_labels': candidates,
                'observations': [copy.deepcopy(e) for e in events if e['speaker'] in ('Patient', 'Measurement')]}
    answer, record = (terminal or finalize)(evidence, strategy, query, config.evidence_gate)
    emit('Doctor', answer, turn=config.max_turns - 1, stage='final')
    return {'prediction': answer, 'events': events, 'finalization': record,
            'action_decisions': decisions, 'config': asdict(config),
            'requested_action_ids': requested, 'clinical_entailment_verified': False}
