"""Legacy council → ledger → consensus (+ large audit), followed by provenance gate.

The gate can keep the consensus diagnosis or abstain, never silently replace it.
"""
import copy
import re
from .context_engineering import evidence_ledger_prompt, render_role_map
from .protocol import stringify
from . import evidence_id_map as gate


def unwrap_markup(text):
    text = text.strip()
    for marker in ('**', '__', '`'):
        if text.startswith(marker) and text.endswith(marker) and len(text) > 2 * len(marker):
            text = text[len(marker):-len(marker)].strip()
    return text


def diagnosis(raw):
    for line in str(raw).splitlines():
        clean = unwrap_markup(line)
        match = re.search(r'DIAGNOSIS\s+READY\s*:\s*([^\n]+)', clean, re.I)
        if match:
            return unwrap_markup(match.group(1)).rstrip('.').strip()
    return None


def same_label(left, right):
    # Formatting only; no synonym lookup, disease rules or semantic rewriting.
    if not isinstance(left, str):
        return False
    norm = lambda value: re.sub(r'\s+', ' ', unwrap_markup(value).rstrip('.').strip()).casefold()
    return norm(left) == norm(right)


def locked_gate(evidence, discussion, proposal, query):
    label = diagnosis(proposal)
    if not label or label.upper() == 'ABSTAIN':
        return 'DIAGNOSIS READY: ABSTAIN', {'reason': 'invalid_consensus_format', 'proposal': proposal}
    prompt = gate.prompt(evidence, discussion)
    prompt += ('\nCONSENSUS_PROPOSAL (not evidence):\n' + proposal +
               '\nVerify support for this exact proposed diagnosis. Keep the diagnosis string exactly '
               + stringify(label) + '. Do not select a different disease. '
               'If it lacks meaningful support, set insufficient=true and diagnosis=null.')
    def check(raw):
        result = gate.validate(raw, evidence)
        m = result['mapping']
        if m and m.get('insufficient') is False and not same_label(m.get('diagnosis'), label):
            result['errors'].append('diagnosis_must_equal_consensus_proposal')
            result['valid'] = False
        return result
    raw = query('evidence_gate', prompt, gate.SYSTEM)
    first = check(raw)
    record = {'initial_raw': raw, 'initial_check': copy.deepcopy(first),
              'revision_applied': not first['valid'], 'proposal': proposal,
              'clinical_entailment_verified': False}
    valid = first
    if not valid['valid']:
        raw = query('evidence_gate_repair', gate.repair_prompt(prompt, raw, valid), gate.SYSTEM)
        valid = check(raw)
    answer = ('DIAGNOSIS READY: ' + label if valid['valid'] and
              valid['mapping']['insufficient'] is False else 'DIAGNOSIS READY: ABSTAIN')
    reason = ('reference_or_schema_error' if not valid['valid'] else
              'model_declared_insufficient' if valid['mapping']['insufficient'] else None)
    record.update(final_raw=raw, final_check=valid, answer=answer, abstention_reason=reason)
    return answer, record


def finalize(evidence, strategy, query, use_gate=True):
    # An allowlist, not a call to patient_information() or the private scenario.
    evidence = {k: copy.deepcopy(evidence[k]) for k in
                ('initial_objective', 'candidate_labels', 'observations')}
    base = ('OBJECTIVE_AND_ACQUIRED_EVIDENCE:\n' + stringify(evidence) +
            '\nUse only recorded Patient/Measurement observations and the initial objective as facts. '
            'Unknown is not normal. Candidate labels and doctor opinions are hypotheses, not evidence.')
    system = render_role_map('council', strategy, evidence['initial_objective'])
    discussion = [query('council_' + str(i + 1), base +
                        f'\nDoctor {i+1}, rank plausible diagnoses using observed patient facts and returned tests. '
                        'Name specific diseases, not broad categories. Explain supporting and conflicting evidence.', system)
                  for i in range(3 if strategy.name == 'small' else 5)]
    # Preserve the old ledger fields and decision stage. All text is an unverified summary.
    ledger_prompt = evidence_ledger_prompt('', stringify(evidence), '\n'.join(discussion))
    ledger = query('ledger', ledger_prompt +
                   '\nUse event IDs when identifying findings. The ledger is an unverified summary; '
                   'never add facts absent from the original observations.', system)
    consensus_prompt = ('Choose the final diagnosis from the candidate ranking below.\nDecision rules:\n'
        '1. Prefer diagnoses directly supported by abnormal returned tests.\n'
        '2. Prefer specific disease names over broad categories.\n'
        '3. If candidate labels are provided, treat them as unordered possibilities.\n'
        '4. Give high weight to disease-compatible patterns in returned tests.\n'
        '5. Do not use hidden answer keys or unavailable findings.\n'
        'Return exactly: DIAGNOSIS READY: <single specific diagnosis>\n\n' + base +
        '\nDOCTOR_DISCUSSION (hypotheses, not facts):\n' + '\n'.join(discussion) +
        '\nSTRUCTURED_EVIDENCE_LEDGER (unverified interpretation):\n' + ledger)
    consensus = query('consensus', consensus_prompt, system)
    proposal = consensus
    if strategy.name == 'large':
        proposal = query('audit', consensus_prompt + '\nCURRENT_FINAL:\n' + consensus +
                        '\nAudit this exact diagnosis against original observations. Preserve it unless '
                        'contradicted or unsupported. Do not treat the ledger as independent evidence. '
                        'Return exactly DIAGNOSIS READY: <single diagnosis>.', system)
    mapping = None
    if use_gate:
        answer, mapping = locked_gate(evidence, discussion, proposal, query)
    else:
        label = diagnosis(proposal)
        answer = 'DIAGNOSIS READY: ' + label if label else 'DIAGNOSIS READY: ABSTAIN'
    return answer, {'mode': 'legacy_aligned_v4', 'evidence_interface': evidence,
        'doctor_discussion': discussion, 'evidence_ledger': ledger,
        'ledger_clinical_entailment_verified': False, 'consensus': consensus,
        'pre_gate_proposal': proposal, 'large_audit_applied': strategy.name == 'large',
        'gate_enabled': use_gate, 'mapping': mapping, 'answer': answer,
        'clinical_entailment_verified': False}
