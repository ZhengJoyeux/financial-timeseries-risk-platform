"""Create train/validation model-ready NPZ from S1 generated Excel; never open test."""
import argparse
from pathlib import Path
from LSTM.src.data import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--generated', default='VAE/outputs/generated/structured_s1_medianfix_n100')
    ap.add_argument('--output', default='LSTM/data/s1_medianfix_seed2027')
    args = ap.parse_args()
    prepare(args.project_root, args.generated, args.output)

if __name__ == '__main__':
    main()
