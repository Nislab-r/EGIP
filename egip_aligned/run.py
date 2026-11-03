"""Run EGIP with MedAgentSim. The bundled reference dataset is MedQA."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import traceback
from . import VERSION
from .engine import Config, run_episode

# Record run settings, never API credentials.
ENV_KEYS = ['EGIP_API_BASE', 'EGIP_TEMPERATURE', 'EGIP_MAX_TOKENS',
            'EGIP_TIMEOUT', 'EGIP_RETRIES', 'EGIP_DISABLE_THINKING',
            'VLLM_CHAT_URL', 'VLLM_MODEL', 'VLLM_MAX_TOKENS', 'VLLM_TEMPERATURE']


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def manifest(args, cfg, data):
    return {'version': VERSION, 'host': 'MedAgentSim',
            'host_root': str(Path(args.host_root).resolve()),
            'terminal': args.terminal, 'acquisition_ablation': args.acquisition_ablation,
            'model': args.model, 'moderator': args.moderator or args.model,
            'dataset': args.dataset, 'data_file': str(data),
            'config': asdict(cfg), 'case_ids': args.case_ids, 'python_seed': args.seed,
            'provider_seed': None,
            'environment': {k: os.environ.get(k) for k in ENV_KEYS}}


def ensure_manifest(dest, value, resume):
    path = dest/'settings.json'
    if path.exists():
        if not resume or json.loads(path.read_text()) != value:
            raise ValueError('Output exists or protocol changed; select a new output directory')
    else:
        if dest.exists() and any(dest.iterdir()):
            raise ValueError('Nonempty output without settings.json')
        write(path, value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='MedAgentSim', choices=['MedAgentSim'])
    from .hosts import BUNDLED_ROOT
    p.add_argument('--host-root', default=str(BUNDLED_ROOT), help='Optional external MedAgentSim checkout')
    p.add_argument('--model', required=True, help='Exact model ID advertised by the model server')
    p.add_argument('--moderator', help='Defaults to generation alias; native scoring implementation retained')
    p.add_argument('--dataset', default='MedQA', choices=['MedQA', 'MedQA_ext', 'MIMIC-IV', 'NEJM', 'NEJM_ext'])
    p.add_argument('--output', required=True)
    p.add_argument('--strategy', choices=['small', 'balanced', 'large'], default='small')
    p.add_argument('--terminal', choices=['direct', 'vote', 'full'], default='direct')
    p.add_argument('--acquisition-ablation', choices=['none', 'no_action_menu', 'native_measurement'], default='none')
    p.add_argument('--max-turns', type=int, default=20)
    p.add_argument('--candidate-policy', choices=['legacy', 'none'], default='legacy')
    p.add_argument('--without-evidence-gate', action='store_true', help='Explicit ablation, never the default')
    p.add_argument('--flat-priority', action='store_true')
    p.add_argument('--case-ids', type=int, nargs='+')
    p.add_argument('--seed', type=int, help='Python RNG only; does not add a seed to native model requests')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.acquisition_ablation != 'none' and (a.host != 'MedAgentSim' or a.terminal != 'direct'):
        p.error('Acquisition ablations require MedAgentSim and Direct.')
    if a.without_evidence_gate and a.terminal != 'full':
        p.error('--without-evidence-gate applies only to Full.')
    cfg = Config(a.max_turns, a.strategy, a.candidate_policy,
                 a.terminal == 'full' and not a.without_evidence_gate, a.flat_priority)
    from .hosts import Host, MAS_FILES
    root = Path(a.host_root).resolve()
    data = root/'datasets'/MAS_FILES[a.dataset]
    if not data.is_file():
        p.error('Only MedQA is bundled. For other datasets use --host-root with a compatible MedAgentSim checkout.')
    records = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    ids = a.case_ids if a.case_ids is not None else list(range(len(records)))
    if len(set(ids)) != len(ids) or any(i < 0 or i >= len(records) for i in ids):
        raise ValueError('Invalid or duplicate case IDs')
    a.case_ids = ids
    dest = Path(a.output).resolve()
    value = manifest(a, cfg, data)
    ensure_manifest(dest, value, a.resume)
    if a.seed is not None:
        random.seed(a.seed)
    host = Host(a.host, root, a.model, a.moderator)
    failed = 0
    for sid in ids:
        print(f'[{ids.index(sid)+1}/{len(ids)}] case {sid}', flush=True)
        folder = dest/f'case_{sid:04d}'
        if (folder/'result.json').exists():
            continue
        if folder.exists() and any(folder.iterdir()):
            # Preserve interrupted attempts and do not silently mix request histories.
            raise RuntimeError(f'Incomplete attempt at {folder}; archive it before retrying')
        folder.mkdir(parents=True)
        if a.seed is not None:
            random.seed(a.seed + sid)
        counter = 0
        def query(stage, prompt, system):
            nonlocal counter
            n = counter;counter += 1
            start = now()
            text = host.query(prompt, system)
            write(folder/'calls'/f'{n:03d}_{stage}.json',
                  {'stage': stage, 'prompt': prompt, 'system': system, 'text': text,
                   'started_utc': start, 'finished_utc': now(), 'transport': 'native_host'})
            return text
        try:
            scenario = host.scenario(a.dataset, records[sid])
            from .terminals import TERMINALS
            from .ablations import acquisition_intervention
            with acquisition_intervention(a.acquisition_ablation, host, scenario, query) as effective_query:
                result = run_episode(scenario, a.model, effective_query, cfg,
                                     on_event=lambda ev: write(folder/'dialogue_history.json', ev),
                                     terminal=TERMINALS[a.terminal])
            write(folder/'finalization.json', result['finalization'])
            gold = scenario.diagnosis_information()  # Passed only to native evaluator, after prediction.
            result['evaluation'] = {'correct': host.score(result['prediction'], gold),
                                    'gold_for_evaluation_only': gold, 'scorer': 'native_host'}
            result.update(case_id=sid, host=a.host, dataset=a.dataset, model=a.model,
                          policy=VERSION, inference_calls=counter, finished_utc=now())
            write(folder/'result.json', result)
        except Exception as exc:
            failed += 1
            write(folder/'failure.json', {'case_id': sid, 'error': str(exc),
                                         'traceback': traceback.format_exc(), 'failed_utc': now()})
    write(dest/'completion.json', {'finished_utc': now(), 'requested_cases': len(ids),
                                  'failures_this_invocation': failed})
    from .summarize import summarize
    summary = summarize(dest)
    write(dest/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
