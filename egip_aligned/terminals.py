"""Terminal strategies on the same acquired evidence. No private case inputs."""
import copy
import json
from collections import Counter
from .finalizer import finalize as full, diagnosis

SYSTEM = 'You are the diagnostic doctor. Use only the supplied acquired evidence; unknown is not normal.'
PREFIX = 'Use only these observed facts; candidate labels are hypotheses. Return exactly DIAGNOSIS READY: <single diagnosis>.\n'


def payload(evidence):
    return {k: copy.deepcopy(evidence[k]) for k in
            ('initial_objective', 'candidate_labels', 'observations')}


def direct(evidence, strategy, query, use_gate=False):
    ev = payload(evidence)
    prompt = PREFIX + json.dumps(ev, ensure_ascii=False)
    answer = query('direct_0', prompt, SYSTEM)
    if not isinstance(answer, str) or not answer.strip() or answer.startswith('Error:'):
        raise RuntimeError('Direct transport failed')
    return answer, dict(mode='direct_single_call', evidence_interface=ev, answer=answer,
                       gate_enabled=False, mapping=None, clinical_entailment_verified=False)


def vote(evidence, strategy, query, use_gate=False, first_answer=None):
    ev = payload(evidence)
    prompt = PREFIX + json.dumps(ev, ensure_ascii=False)
    members = [] if first_answer is None else [first_answer]
    for i in range(len(members), 5):
        answer = query('direct_' + str(i), prompt, SYSTEM)
        if not isinstance(answer, str) or not answer.strip() or answer.startswith('Error:'):
            raise RuntimeError('Vote transport failed')
        members.append(answer)
    labels = [diagnosis(x) for x in members]
    counts = Counter(x.casefold() for x in labels if x and x.upper() != 'ABSTAIN')
    answer = 'DIAGNOSIS READY: ' + (max(counts, key=counts.get) if counts else 'ABSTAIN')
    return answer, dict(mode='five_sample_vote', evidence_interface=ev, answer=answer,
                       vote_members=members, reused_first_answer=first_answer is not None,
                       gate_enabled=False, clinical_entailment_verified=False)


TERMINALS = {'direct': direct, 'vote': vote, 'full': full}
