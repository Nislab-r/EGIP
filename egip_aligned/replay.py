"""Run a terminal strategy or input ablation on one saved evidence payload.

No acquisition is performed. Outputs are fresh calls, not inherited scores. Optional native scoring occurs after prediction.
"""
import argparse
import json
from pathlib import Path
from .terminals import TERMINALS, vote
from .ablations import remove_terminal_input
from .context_engineering import infer_model_strategy
from .hosts import Host, BUNDLED_ROOT
from .run import write, ensure_manifest, ENV_KEYS
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='Saved EGIP result.json containing finalization.evidence_interface')
    p.add_argument('--host',default='MedAgentSim',choices=['MedAgentSim'])
    p.add_argument('--host-root',default=str(BUNDLED_ROOT))
    p.add_argument('--model',required=True)
    p.add_argument('--moderator')
    p.add_argument('--terminal',default='direct',choices=list(TERMINALS))
    p.add_argument('--strategy',default='small',choices=['small','balanced','large'])
    p.add_argument('--arm',default='baseline',choices=['baseline','no_observations','no_candidate_labels'])
    p.add_argument('--first-direct',help='Saved Direct result for first Vote sample on exactly the same input/model')
    p.add_argument('--output',required=True)
    p.add_argument('--score',action='store_true',help='Score only after prediction using explicit host case coordinates')
    p.add_argument('--dataset',choices=['MedQA','MedQA_ext','MIMIC-IV','NEJM','NEJM_ext'])
    p.add_argument('--case-id',type=int)
    a=p.parse_args()
    if a.arm != 'baseline' and a.terminal != 'direct':p.error('Input ablations keep Direct fixed.')
    if a.first_direct and a.terminal != 'vote':p.error('--first-direct requires Vote.')
    if a.score and (a.dataset is None or a.case_id is None):
        p.error('--score requires --dataset and --case-id.')
    if not a.score and (a.dataset is not None or a.case_id is not None):
        p.error('--dataset/--case-id apply to --score.')
    source=Path(a.source); original=json.loads(source.read_text())
    ev=remove_terminal_input(original['finalization']['evidence_interface'],a.arm)
    first=None
    if a.first_direct:
        prior=json.loads(Path(a.first_direct).read_text())
        if prior['finalization']['evidence_interface'] != ev or prior.get('model') != a.model:
            p.error('First Direct answer must use the same evidence and model.')
        if prior['finalization'].get('mode') not in ('direct_single_call','direct_doctor'):
            p.error('First sample is not a Direct result.')
        first=prior['prediction']
    dest=Path(a.output)
    manifest={'source':str(source.resolve()),'host':a.host,'model':a.model,
              'moderator':a.moderator or a.model,'terminal':a.terminal,'strategy':a.strategy,'arm':a.arm,
              'first_direct':a.first_direct,
              'score_after_prediction':a.score,'dataset':a.dataset,'case_id':a.case_id,
              'environment':{k:os.environ.get(k) for k in ENV_KEYS}}
    records=None
    if a.score:
        from .hosts import MAS_FILES
        root=Path(a.host_root)
        data=root/'datasets'/MAS_FILES[a.dataset]
        records=[json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        if not 0 <= a.case_id < len(records):p.error('Invalid case ID.')
        if original.get('dataset',a.dataset) != a.dataset or original.get('case_id',a.case_id) != a.case_id:
            p.error('Scoring coordinates differ from the saved source.')
        manifest['data_file']=str(data.resolve())
    ensure_manifest(dest,manifest,False)
    host=Host(a.host,a.host_root,a.model,a.moderator);calls=[]
    def query(stage,prompt,system):
        text=host.query(prompt,system)
        write(dest/'calls'/f'{len(calls):03d}_{stage}.json',dict(stage=stage,prompt=prompt,system=system,text=text))
        calls.append(stage);return text
    strategy=infer_model_strategy(a.model,a.strategy)
    answer,record=(vote(ev,strategy,query,first_answer=first) if a.terminal=='vote'
                   else TERMINALS[a.terminal](ev,strategy,query,True))
    output=dict(prediction=answer,finalization=record,model=a.model,
                inference_calls=len(calls),evaluation=None,arm=a.arm)
    write(dest/'result.json',output)
    if a.score:
        scenario=host.scenario(a.dataset,records[a.case_id])
        gold=scenario.diagnosis_information()
        saved_gold=(original.get('evaluation') or {}).get('gold_for_evaluation_only')
        if saved_gold is not None and gold != saved_gold:
            raise ValueError('Reference differs from the saved source. Prediction retained without a score.')
        output.update(dataset=a.dataset,case_id=a.case_id,
                      evaluation=dict(correct=host.score(answer,gold),gold_for_evaluation_only=gold,scorer='native_host'))
        write(dest/'result.json',output)


if __name__=='__main__':main()
