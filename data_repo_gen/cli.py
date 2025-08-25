
import argparse
from .generator import generate

def main():
    ap = argparse.ArgumentParser(prog='data-repo-gen',
                                 description='Generate an intentionally messy teaching repo from a CSV dataset.')
    ap.add_argument('--name', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--target', default=None)
    ap.add_argument('--task', choices=['regression','classification'], default='regression')
    ap.add_argument('--difficulty', choices=['light','medium','hard'], default='medium')
    ap.add_argument('--out-dir', default='.')
    args = ap.parse_args()

    repo_path = generate(args.name, args.csv, target=args.target, task=args.task,
                         difficulty=args.difficulty, out_dir=args.out_dir)
    print(f'Created messy repo at: {repo_path}')
