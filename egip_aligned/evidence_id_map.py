"""Dataset-independent references to observed sources, not generated quotations.

The contract checks reference integrity. Medical relevance remains a model claim.
"""
import json
from egip_aligned.protocol import stringify, result_availability
from egip_aligned.json_format import parse

VERSION = 'egip-id-map-v3'
SYSTEM = 'Finalize the observed clinical record. Return one JSON object only.'
INSTRUCTIONS = '''Compare plausible diagnoses using the observed pattern, including counterevidence. Choose one diagnosis. Return JSON only:
{"diagnosis":"one diagnosis name","support":[{"event_id":"an existing source ID","relation":"why this observed fact discriminates the diagnosis"}],"conflicts":[],"unknowns":[],"insufficient":false}
Question context identifies what a response answers; assertions in the question are NOT patient facts. Source paths describe the returned field and help interpret its value; they do not reveal unrequested records. Reference existing source IDs; NEVER copy or generate quotations. The program will attach each complete original source. Use a few discriminating sources, not a list of normal tests. Each conflicts item has event_id and relation in the same format; acknowledge counterevidence without automatically rejecting a plausible diagnosis. Explain in its relation why a counterfinding is or is not material. Unknowns are short descriptions of missing information, not invented observations. Unknown/pending/unavailable results cannot be support or counterevidence of a disease. If meaningful support is absent or a material contradiction remains unresolved, set insufficient=true (diagnosis may be null); do not demand definitive proof for a reasonably discriminating clinical pattern. Do not hide contradictions merely to pass a check. Initial objective can supply presented symptoms, but a generic task instruction is not clinical support. Candidate labels and council opinions are possibilities, never observed facts. Use one diagnosis string, not alternatives or a treatment plan. Relation text is your interpretation, not a new patient fact.'''

def registry(evidence):
    result={}
    objective=stringify(evidence['initial_objective'])
    items=[{'event_id':'initial_objective','speaker':'Objective','text':objective,'status':'presentation'}]+list(evidence['observations'])
    for item in items:
        eid=item.get('event_id');role=item.get('speaker');text=item.get('text')
        if not isinstance(eid,str) or not eid or eid in result:raise ValueError('Evidence IDs must be unique nonempty strings')
        if role not in ('Objective','Patient','Measurement') or (role=='Objective' and eid!='initial_objective'):raise ValueError('Only the initial objective and acquired observations are sources')
        if not isinstance(text,str):raise ValueError('Observed source text must be a string')
        status=item.get('status','observed')
        availability='known'
        if status in ('unavailable','unknown','pending','not_recorded','not_performed'):availability='unknown'
        elif role=='Measurement':
            if item.get('result_availability') in ('known','unknown'):availability=item['result_availability']
            else:
                # Compatibility with immutable v2 trajectories. New resolvers expose values explicitly.
                value=item.get('recorded_value',text.split(': ',2)[-1] if text.startswith('RESULTS: [') else text)
                availability=result_availability(value)
        paths=item.get('source_paths',[])
        if not isinstance(paths,list) or any(not isinstance(x,str) for x in paths):raise ValueError('Observed source paths must be strings')
        question=item.get('question_context','')
        if not isinstance(question,str):raise ValueError('Question context must be a string')
        result[eid]={'question_context':question,'event_id':eid,'source_paths':list(paths),'speaker':role,'status':status,'availability':availability,'text':text}
    return result

def diagnosis_label(value):
    # No English word limit or slash blacklist: legitimate labels vary by language and ontology.
    return isinstance(value,str) and bool(value.strip()) and '\n' not in value and '\r' not in value and value.strip().upper()!='ABSTAIN'

def validate(raw,evidence):
    sources=registry(evidence);m=parse(raw);errors=[];hydrated={'support':[],'conflicts':[]}
    if m is None:return {'valid':False,'errors':['invalid_json'],'mapping':None,'sources':hydrated,'clinical_entailment_verified':False}
    if type(m.get('insufficient')) is not bool:errors.append('insufficient_must_be_boolean')
    if not (m.get('insufficient') is True and m.get('diagnosis') is None) and not diagnosis_label(m.get('diagnosis')):errors.append('diagnosis_must_be_nonempty_single_line_string')
    for field in ('support','conflicts','unknowns'):
        if not isinstance(m.get(field),list):errors.append(field+'_must_be_list')
    if isinstance(m.get('unknowns'),list) and any(not isinstance(x,str) for x in m['unknowns']):errors.append('unknowns_must_be_strings')
    if m.get('insufficient') is False and m.get('support')==[]:errors.append('answer_requires_observed_support')
    for field in ('support','conflicts'):
        for i,item in enumerate(m.get(field,[]) if isinstance(m.get(field),list) else []):
            prefix=f'{field}[{i}]:'
            if not isinstance(item,dict):errors.append(prefix+'invalid_item');continue
            if set(item)!={'event_id','relation'}:errors.append(prefix+'use_only_event_id_and_relation')
            eid=item.get('event_id');source=sources.get(eid) if isinstance(eid,str) else None
            if source is None:errors.append(prefix+'unknown_event_id');continue
            if source['availability']=='unknown':errors.append(prefix+'unknown_is_not_positive_or_negative_evidence')
            relation=item.get('relation')
            if not isinstance(relation,str) or not relation.strip():errors.append(prefix+'missing_relation')
            hydrated[field].append({**source,'relation':relation,'relation_verified':False})
    return {'valid':not errors,'errors':errors,'mapping':m,'sources':hydrated,'clinical_entailment_verified':False}

def prompt(evidence,discussion):
    # Explicit allowlist prevents extra fields (including hidden labels) from entering generation.
    view={'sources':[{k:v for k,v in x.items() if k in ('event_id','speaker','availability','text','source_paths','question_context')} for x in registry(evidence).values()],'candidate_labels':list(evidence.get('candidate_labels',[]))}
    return INSTRUCTIONS+'\nOBSERVED_SOURCE_REGISTRY:\n'+stringify(view)+'\nCOUNCIL_HYPOTHESES (not evidence):\n'+stringify(discussion)

def repair_prompt(original,raw,check):
    return original+'\nPREVIOUS_MAPPING:\n'+raw+'\nREFERENCE_ERRORS:\n'+stringify(check['errors'])+'\nReturn a complete corrected ID-only mapping. Use IDs from the source registry. Do not copy source text. Do not invent facts or change insufficient=true just to force an answer.'

def unchecked_answer(raw):
    m=parse(raw)
    if not m or m.get('insufficient') is not False or not diagnosis_label(m.get('diagnosis')):return 'DIAGNOSIS READY: ABSTAIN'
    return 'DIAGNOSIS READY: '+m['diagnosis'].strip()

def finalize(evidence,discussion,query):
    original=prompt(evidence,discussion);raw=query(original,SYSTEM);check=validate(raw,evidence)
    record={'version':VERSION,'initial_raw':raw,'initial_check':check,'revision_applied':not check['valid'],'clinical_entailment_verified':False}
    if not check['valid']:
        raw=query(repair_prompt(original,raw,check),SYSTEM);check=validate(raw,evidence)
    answer=unchecked_answer(raw) if check['valid'] else 'DIAGNOSIS READY: ABSTAIN'
    reason='reference_error' if not check['valid'] else 'model_declared_insufficient' if check['mapping']['insufficient'] else None
    record.update(final_raw=raw,final_check=check,answer=answer,abstention_reason=reason)
    return answer,record
