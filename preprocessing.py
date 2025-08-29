# -*- coding: utf-8 -*-
"""
Minimal CLI to preprocess ArEEG_Words into PKLs for training.
"""

import argparse
from data_imagine import prepare_areeg_to_pkl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="areeg_words",
                    help="fixed to areeg_words for this pipeline")
    ap.add_argument("--root", type=str, default="data/AREEG_Words",
                    help="dataset root folder")
    ap.add_argument("--overlap", type=float, default=0.0,
                    help="overlap fraction for 250 ms windows, 0.0 means disjoint")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--split", type=str, default="0.7,0.15,0.15",
                    help="train,val,test ratios")
    args = ap.parse_args()

    if args.dataset != "areeg_words":
        raise SystemExit("Unknown dataset. Use --dataset areeg_words")

    tr, va, te = [float(x) for x in args.split.split(",")]
    stats = prepare_areeg_to_pkl(root=args.root,
                                 split_ratio=(tr, va, te),
                                 seed=args.seed,
                                 overlap=args.overlap)
    print("Saved PKLs:", stats)

if __name__ == "__main__":
    main()
