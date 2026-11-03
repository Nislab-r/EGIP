"""Delegate a native baseline to an external complete MedAgentSim checkout."""
import argparse
from pathlib import Path
import subprocess
import sys

def command(host, root, python, args):
    if host != 'MedAgentSim':
        raise ValueError('This release supports MedAgentSim only')
    root = Path(root).resolve()
    if not (root/'medsim/main.py').is_file():
        raise FileNotFoundError('Native baseline requires a complete external MedAgentSim checkout')
    return [python, '-m', 'medsim.main'] + list(args), root

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='MedAgentSim', choices=['MedAgentSim'])
    p.add_argument('--root', required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('native_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = a.native_args[1:] if a.native_args[:1] == ['--'] else a.native_args
    cmd, cwd = command(a.host, a.root, a.python, args)
    raise SystemExit(subprocess.call(cmd, cwd=cwd))

if __name__ == '__main__':
    main()
