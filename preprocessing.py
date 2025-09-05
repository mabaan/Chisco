#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Data Preparation Command Line Tool - The Easy Button

This is your simple command-line tool to prepare EEG data for machine learning.
Just run one command and watch as your raw CSV files get transformed into 
clean training data!

What it does:
- Takes a folder full of messy EEG CSV files
- Calls the data_imagine.py "kitchen" to cook them into clean data
- Lets you choose how much overlap between time windows you want
- Splits your data into train/validation/test sets
- Does all the heavy lifting so you don't have to

Usage example:
python preprocessing.py --root data/AREEG_Words --overlap 0.75

Think of this as the "easy button" that does all the boring data prep work
so you can focus on training cool AI models!
"""
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
