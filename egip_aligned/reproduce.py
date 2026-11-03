"""Run the bundled 107-case MedQA reference experiment for three repetitions."""
import argparse
from pathlib import Path
import subprocess
import sys
from .run import write
from .summarize import aggregate

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3-8B')
    p.add_argument('--moderator', help='Defaults to the generation model')
    p.add_argument('--output', default='results/medqa_direct')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--case-ids', type=int, nargs='+', help='Optional quick trial; omit for all 107 cases')
    p.add_argument('--terminal', choices=['direct','vote','full'], default='direct')
    p.add_argument('--acquisition-ablation', choices=['none','no_action_menu','native_measurement'], default='none')
    a = p.parse_args()
    if a.repetitions < 1:
        p.error('--repetitions must be positive')
    if a.acquisition_ablation != 'none' and a.terminal != 'direct':
        p.error('Acquisition ablations keep Direct fixed')
    dest = Path(a.output).resolve()
    if dest.exists() and any(dest.iterdir()):
        p.error('Select an empty output directory to keep repetitions separate')
    runs = []
    for repeat in range(1, a.repetitions + 1):
        run = dest/f'repeat_{repeat}'
        cmd = [sys.executable, '-m', 'egip_aligned.run', '--model', a.model,
               '--output', str(run), '--terminal', a.terminal,
               '--acquisition-ablation', a.acquisition_ablation]
        if a.moderator:
            cmd += ['--moderator', a.moderator]
        if a.case_ids is not None:
            cmd += ['--case-ids'] + list(map(str, a.case_ids))
        print(f'Repetition {repeat}/{a.repetitions}', flush=True)
        process = subprocess.run(cmd)
        if process.returncode:
            raise SystemExit(process.returncode)
        runs.append(run)
    summary = aggregate(runs)
    write(dest/'summary.json', summary)
    print(f'Mean accuracy: {summary["mean_accuracy_pct"]:.2f}%')
    if summary['sample_sd_pct'] is not None:
        print(f'Sample SD: {summary["sample_sd_pct"]:.2f} percentage points')
    print(f'Results: {dest}')

if __name__ == '__main__':
    main()
