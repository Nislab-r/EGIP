"""Explicit interventions; baseline behavior is unchanged.

Acquisition patches are scoped to one serial episode and restored on exit.
Use separate processes for parallel jobs, never threads sharing this module.
"""
import copy
import hashlib
import re
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
from . import engine
from .protocol import resolve_measurement


def remove_terminal_input(evidence, arm):
    result = copy.deepcopy(evidence)
    if arm == 'no_observations':
        result['observations'] = []
    elif arm == 'no_candidate_labels':
        if not result.get('candidate_labels'):
            raise ValueError('No candidate labels to remove; this is not an intervention.')
        result['candidate_labels'] = []
    elif arm != 'baseline':
        raise ValueError('Unknown terminal-input arm')
    return result


@contextmanager
def acquisition_intervention(arm, host, scenario, query):
    with ExitStack() as stack:
        effective_query = query
        if arm == 'no_action_menu':
            original_extract = engine.extract_action
            def extract(raw, actions):
                result = original_extract(raw, actions)
                matches = list(re.finditer(r'REQUEST\s+TEST\s*:', str(raw), re.I))
                if len(matches) != 1 or result['kind'] != 'invalid':
                    return result
                tail = str(raw)[matches[0].end():].strip()
                if not tail:
                    return result
                request = tail.splitlines()[0].strip()
                return dict(kind='test', action_id='unmatched_' + hashlib.sha256(
                    request.casefold().encode()).hexdigest()[:12], name=request,
                    command='REQUEST TEST: '+request, raw=str(raw))
            stack.enter_context(patch.object(engine, 'plan', lambda *a, **k: ''))
            stack.enter_context(patch.object(engine, 'extract_action', extract))
            stack.enter_context(patch.object(engine, 'execute_action',
                lambda action, exam: resolve_measurement(action['command'], exam)))
            stack.enter_context(patch.object(engine, 'reject_text', lambda reason:
                'ACTION_REJECTED: '+reason+'. Request one test by its natural-language name, or ask a patient question.'))
            def effective_query(stage, prompt, system):
                if stage.startswith('doctor_'):
                    system = re.sub(r'\nDIAGNOSTIC_TOOL_MAP_RESULT_BLINDED:.*?(?=\nUNORDERED_DIAGNOSIS_CANDIDATE_MAP:|\Z)', '', system, flags=re.S)
                    system += '\nRequest one test using REQUEST TEST: <test name>, or ask one patient question. Do not simulate patient or test responses.'
                return query(stage, prompt, system)
        elif arm == 'native_measurement':
            reader = host.agents.MeasurementAgent.__new__(host.agents.MeasurementAgent)
            reader.backend = host.model
            class Pipe:
                def query_model(self, prompt, system_prompt, **kwargs):
                    return query('native_measurement', prompt, system_prompt)
            reader.pipe = Pipe()
            reader.update_scenario(scenario)
            def execute(action, exam):
                text = reader.inference_measurement('REQUEST TEST: '+action['name'])
                return dict(text=text, status='native_generated', action_id=action['action_id'],
                            source_paths=[], resolver='native_measurement_agent')
            stack.enter_context(patch.object(engine, 'execute_action', execute))
        elif arm != 'none':
            raise ValueError('Unknown acquisition arm')
        yield effective_query
