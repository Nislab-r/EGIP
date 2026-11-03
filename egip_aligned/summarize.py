"""Summarize one run with the scheduled cases as denominator (no bootstrap)."""
import argparse
import json
from pathlib import Path


def summarize(root):
    root=Path(root);manifest=json.loads((root/'settings.json').read_text())
    ids=manifest['case_ids'];correct=scored=present=failed=0
    for i in ids:
        folder=root/f'case_{i:04d}';file=folder/'result.json'
        if not file.exists():
            failed+=int((folder/'failure.json').exists());continue
        present+=1;r=json.loads(file.read_text());value=(r.get('evaluation') or {}).get('correct')
        if isinstance(value,bool):scored+=1;correct+=int(value)
    n=len(ids)
    return dict(scheduled=n,present=present,scored=scored,unscored_present=present-scored,
                failed=failed,missing_or_incomplete=n-present-failed,correct=correct,
                accuracy_pct=100*correct/n if n else None,
                note='Missing/unscored/non-answers stay in denominator. This is one run, not an across-repeat SD or paired CI.')


def aggregate(roots):
    import statistics
    if not roots:
        raise ValueError('At least one run is required')
    roots = [Path(x) for x in roots]
    if len({x.resolve() for x in roots}) != len(roots):
        raise ValueError('Duplicate run directory')
    settings = [json.loads((root/'settings.json').read_text()) for root in roots]
    keys = ('host','model','moderator','dataset','case_ids','config','terminal','acquisition_ablation','environment')
    for setting in settings[1:]:
        if any(setting.get(key) != settings[0].get(key) for key in keys):
            raise ValueError('Runs have different experiment settings')
    runs = [summarize(root) for root in roots]
    if any(r['accuracy_pct'] is None for r in runs):
        raise ValueError('Cannot summarize an empty case selection')
    values = [r['accuracy_pct'] for r in runs]
    return dict(repetitions=len(runs), runs=runs,
                mean_accuracy_pct=statistics.mean(values),
                sample_sd_pct=statistics.stdev(values) if len(values)>1 else None,
                complete=all(r['present']==r['scheduled'] and r['scored']==r['scheduled'] for r in runs),
                note='Accuracy uses all scheduled cases. SD is across repetitions, not a confidence interval.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+')
    a=p.parse_args()
    result=summarize(a.runs[0]) if len(a.runs)==1 else aggregate(a.runs)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
