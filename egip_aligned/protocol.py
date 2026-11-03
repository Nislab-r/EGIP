"""Evidence interfaces for EGIP v2. No network or model dependencies."""
from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 'egip-observed-v2'
UNAVAILABLE = 'RESULTS: UNAVAILABLE / NOT RECORDED'


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()


def stringify(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


def flatten_records(value, prefix=''):
    """Preserve every value. Narrative records without a named action are explicit bundles."""
    if isinstance(value, dict):
        return [(p, v) for k, child in value.items()
                for p, v in flatten_records(child, f'{prefix}.{k}' if prefix else str(k))]
    if isinstance(value, list):
        return [(prefix or 'Recorded examination report', '; '.join(map(str, value)))]
    if isinstance(value, str) and not prefix:
        chunks = re.split(r'(?:^|\s)(?:\d+[.)]|[-*])\s+', value)
        rows = []
        for chunk in chunks:
            chunk = chunk.strip(' \n.;')
            if not chunk:
                continue
            # Discard only a generic introduction, never a finding-bearing prefix.
            if chunk.lower().endswith(('includes:', 'results include:', 'following:')):
                continue
            label, sep, finding = chunk.partition(':')
            if sep and 0 < len(label.split()) <= 12 and finding.strip():
                rows.append((label.strip(), finding.strip()))
            else:
                rows.append((f'Recorded examination finding {len(rows)+1}', chunk))
        return rows or [('Recorded examination report', value)]
    return [(prefix or 'Recorded examination report', str(value))]


_OUTCOME = re.compile(r'\b(within normal limits|normal range|normal|abnormalities|abnormal|positive|negative|elevated|decreased|increased|absent|present|enlarged)\b', re.I)


def blinded_name(path):
    parts = [p.replace('_', ' ').strip() for p in path.split('.') if p.lower() != 'tests']
    name = ' / '.join(_OUTCOME.sub('recorded finding', p) for p in parts)
    return name or 'Recorded examination finding'


@dataclass(frozen=True)
class Action:
    action_id: str
    name: str
    path: str
    value: str

    @property
    def public(self):
        return {'action_id': self.action_id, 'name': self.name}


def action_catalog(exam):
    return [Action('A_' + hashlib.sha256(f'{index}:{path}'.encode()).hexdigest()[:12],
                   blinded_name(path), path, value)
            for index, (path, value) in enumerate(flatten_records(exam))]


def resolve_action(request, actions):
    """Exact ID or exact normalized public label. No value matching, no fuzzy top-k."""
    text = str(request).strip()
    text = re.sub(r'^REQUEST\s+TEST\s*:\s*', '', text, flags=re.I)
    ids = re.findall(r'\bA_[0-9a-f]{12}\b', text, re.I)
    if ids:
        hits = [a for a in actions if a.action_id.lower() in {x.lower() for x in ids}]
        if len(ids) != 1 or len(hits) != 1:
            return None, 'ambiguous_or_unknown_action'
        # Only an optional literal public label may follow the ID.
        rest = re.sub(re.escape(ids[0]), '', text, count=1, flags=re.I).strip(' |:.-')
        if rest and normalize(rest) != normalize(hits[0].name):
            return None, 'multiple_or_malformed_action'
        return hits[0], 'matched'
    hits = [a for a in actions if normalize(a.name) == normalize(text)]
    return (hits[0], 'matched') if len(hits) == 1 else (None, 'ambiguous_action' if hits else 'unknown_action')


def result_availability(value):
    """Conservative whole-value sentinel adapter, not clinical interpretation.

    Explicit source availability metadata takes precedence in the ID finalizer.
    A missing result is not a negative result. Mixed narratives stay intact.
    """
    if value is None:return 'unknown'
    text=re.sub(r'\s+', ' ', str(value).strip()).casefold().rstrip('.!')
    missing={'', 'none', 'null', 'n/a', 'na', 'unknown', 'unavailable', 'pending',
             'not provided', 'not recorded', 'not reported', 'not available',
             'not assessed', 'not performed', 'not mentioned'}
    return 'unknown' if text in missing else 'known'


def resolve_measurement(request, exam):
    action, status = resolve_action(request, action_catalog(exam))
    if action is None:
        return {'status': 'unavailable', 'reason': status, 'action_id': None,
                'source_paths': [], 'text': UNAVAILABLE, 'resolver': 'exact_action_v2',
                'recorded_value': None, 'result_availability': 'unknown'}
    return {'status': 'recorded', 'reason': 'matched', 'action_id': action.action_id,
            'source_paths': [action.path], 'text': f'RESULTS: [{action.action_id}] {action.name}: {action.value}',
            'resolver': 'exact_action_v2', 'recorded_value': action.value,
            'result_availability': result_availability(action.value)}


def candidate_labels(scenario):
    answers = (getattr(scenario, 'scenario_dict', {}) or {}).get('answers', [])
    labels = [a.get('text', '') if isinstance(a, dict) else a for a in answers]
    # Stable alphabetical order, unrelated to the source answer order or correctness.
    return sorted({str(x).strip() for x in labels if x}, key=str.casefold)


def evidence_interface(objective, events, candidates):
    """Only initial objective, acquired observations, and blinded candidate labels."""
    return {'protocol_version': PROTOCOL_VERSION, 'initial_objective': objective,
            'candidate_labels': list(candidates),
            'observations': [dict(e) for e in events if e['speaker'] in {'Patient', 'Measurement'}],
            'doctor_actions_are_not_evidence': True}


def validate_ledger(raw, evidence):
    """Validate literal quote provenance; does not claim to verify clinical entailment."""
    events = {e['event_id']: e for e in evidence['observations']}
    try:
        data = json.loads(raw)
        if not isinstance(data.get('findings'), list):
            raise ValueError('findings must be a list')
        for item in data['findings']:
            event = events[item['event_id']]
            if not isinstance(item.get('quote'), str) or not item['quote'].strip() or item['quote'] not in event['text']:
                raise ValueError('quote absent from source event')
            if event.get('status') == 'unavailable':
                raise ValueError('unavailable is not a clinical finding')
        return {'valid': True, 'findings': data['findings'],
                'check_scope': 'literal observation provenance; no clinical entailment guarantee'}
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return {'valid': False, 'error': str(exc), 'findings': [],
                'fallback': 'use original evidence interface, not unverified summary'}
