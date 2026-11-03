"""Normalize explicit actions without searching private case values."""
import re
from types import SimpleNamespace
from .protocol import resolve_action, resolve_measurement

COMMAND = re.compile(r'REQUEST\s+TEST\s*:', re.I)
ID = re.compile(r'\bA_[0-9a-f]{12}\b', re.I)


def extract_action(raw, public_actions):
    matches = list(COMMAND.finditer(str(raw)))
    if not matches:
        return {'kind': 'question', 'raw': str(raw)}
    if len(matches) != 1:
        return {'kind': 'invalid', 'reason': 'multiple_commands', 'raw': str(raw)}
    raw = str(raw)
    tail = raw[matches[0].end():].strip()
    # An unambiguous ID is a machine command; surrounding prose is never evidence.
    ids = ID.findall(raw)
    if len(ids) > 1:
        return {'kind': 'invalid', 'reason': 'multiple_action_ids', 'raw': raw}
    if ids and not ID.search(tail):
        return {'kind': 'invalid', 'reason': 'id_outside_command', 'raw': raw}
    request = ids[0] if ids else (tail.splitlines()[0].strip() if tail else '')
    action, reason = resolve_action(request, [SimpleNamespace(**a) for a in public_actions])
    if action is None:
        return {'kind': 'invalid', 'reason': reason, 'raw': raw}
    return {'kind': 'test', 'action_id': action.action_id, 'name': action.name,
            'command': 'REQUEST TEST: ' + action.action_id, 'raw': raw}


def reject_text(reason):
    return ('ACTION_REJECTED: ' + reason + '. No examination was executed. '
            'This is an interface error, NOT a missing or normal test result. '
            'Return exactly one REQUEST TEST: <available action ID>, or ask one patient question.')


def execute_action(action, exam):
    if action.get('kind') != 'test':
        raise ValueError('Only a validated public action can reach the resolver')
    result = resolve_measurement(action['command'], exam)
    if result['status'] != 'recorded':
        raise RuntimeError('Public catalog and private resolver disagree; do not invent an observation')
    return result
