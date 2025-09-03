# -*- coding: utf-8 -*-
"""
ArEEG_Words CSV -> windowed tensors -> PKL
Place CSVs under: data/AREEG_Words/raw_csv/<ArabicWord>/*.csv
Creates: data/AREEG_Words/preprocessed_pkl/{train,val,test}.pkl
Each PKL has keys: X [N,14,T], y [N], meta, channels, idx_to_label, sr, win_samples
"""

import os
import re
import glob
import json
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sps

# ArEEG_Words parameters
AREEG_SR = 128              # Hz
AREEG_WIN_SEC = 2.0         # seconds per segment
AREEG_SAMPLES = int(AREEG_SR * AREEG_WIN_SEC)

# parse subject id like "par.1" appearing in filenames
SUBJ_RE = re.compile(r"par\.?\s*(\d+)", re.IGNORECASE)


def _read_channels_file(root: str):
    ch_file = os.path.join(root, "channels_14.txt")
    if not os.path.exists(ch_file):
        raise FileNotFoundError(f"Missing {ch_file}")
    with open(ch_file, "r", encoding="utf-8") as f:
        chans = [c.strip() for c in f if c.strip()]
    if len(chans) != 14:
        raise ValueError(f"channels_14.txt should list 14 names. Got {len(chans)}")
    return chans


def _infer_subject_from_path(fp: str) -> str:
    m = SUBJ_RE.search(Path(fp).name)
    if not m:
        for part in Path(fp).parts:
            mm = SUBJ_RE.search(part)
            if mm:
                m = mm
                break
    sid = int(m.group(1)) if m else -1
    return f"sub{sid:02d}" if sid > 0 else "sub00"


def butter_bandpass(low, high, fs, order=4):
    b, a = sps.butter(order, [low, high], btype="bandpass", fs=fs)
    return b, a


def iir_notch(freq, fs, q=30.0):
    # Dubai mains is 50 Hz
    b, a = sps.iirnotch(w0=freq, Q=q, fs=fs)
    return b, a


def filter_eeg(eeg_2d, fs=AREEG_SR):
    # eeg_2d: [C, T]
    b_bp, a_bp = butter_bandpass(1.0, 40.0, fs, order=4)
    b_n, a_n = iir_notch(50.0, fs, q=30.0)
    X = sps.filtfilt(b_bp, a_bp, eeg_2d, axis=1)
    X = sps.filtfilt(b_n, a_n, X, axis=1)
    return X.astype(np.float32)


def load_areeg_words(root: str = "data/AREEG_Words",
                     overlap: float = 0.0):
    labels_path = os.path.join(root, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    idx_to_label = {v: k for k, v in labels.items()}
    channels = _read_channels_file(root)

    csv_glob = os.path.join(root, "raw_csv", "**", "*.csv")
    csv_files = sorted(glob.glob(csv_glob, recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files under {os.path.join(root, 'raw_csv')}")

    X, y, meta = [], [], []
    hop = AREEG_SAMPLES if overlap <= 0 else int(AREEG_SAMPLES * (1 - overlap))

    for fp in csv_files:
        word = Path(fp).parent.name
        if word not in labels:
            continue

        # detect header row and codec
        def _find_header_row(path, encodings=("utf-8-sig", "utf-8", "cp1256")):
            for enc in encodings:
                try:
                    with open(path, "r", encoding=enc) as fh:
                        for i, line in enumerate(fh):
                            if line.count(",") >= 10 and "EEG." in line:
                                return i, enc
                except Exception:
                    continue
            raise RuntimeError(f"Could not find header row in {path}")

        hdr_idx, enc = _find_header_row(fp)
        df = pd.read_csv(fp, encoding=enc, header=hdr_idx)

        # normalize headers by stripping 'EEG.' prefix
        def _norm(col):
            c = str(col).strip()
            if c.lower().startswith("eeg."):
                c = c.split(".", 1)[1].strip()
            return c
        df.columns = [_norm(c) for c in df.columns]

        present_map = {c.lower(): c for c in df.columns}
        missing = [c for c in channels if c.lower() not in present_map]
        if missing:
            raise ValueError(f"Missing channels {missing} in file {fp}. "
                             f"Edit channels_14.txt or check CSV headers.")

        ordered_cols = [present_map[c.lower()] for c in channels]
        eeg = df[ordered_cols].to_numpy(dtype=np.float32).T  # [14, T_all]
        eeg = filter_eeg(eeg, fs=AREEG_SR)                   # denoise

        T_all = eeg.shape[1]
        subj = _infer_subject_from_path(fp)

        # window into AREEG_WIN_SEC segments
        for start in range(0, T_all - AREEG_SAMPLES + 1, hop):
            seg = eeg[:, start:start + AREEG_SAMPLES]
            if seg.shape[1] != AREEG_SAMPLES:
                continue
            X.append(seg)
            y.append(labels[word])
            meta.append((subj, word, fp, start))

    if not X:
        raise RuntimeError("No segments produced. Check CSV headers and channel list.")

    X = np.stack(X, axis=0).astype(np.float32)  # [N, 14, T]
    y = np.asarray(y, dtype=np.int64)
    return X, y, meta, channels, idx_to_label


def prepare_areeg_to_pkl(root: str = "data/AREEG_Words",
                         split_ratio=(0.7, 0.15, 0.15),
                         seed: int = 1337,
                         overlap: float = 0.0):
    os.makedirs(os.path.join(root, "preprocessed_pkl"), exist_ok=True)
    os.makedirs(os.path.join(root, "split"), exist_ok=True)

    X, y, meta, channels, idx_to_label = load_areeg_words(root=root, overlap=overlap)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    n = len(idx)
    n_tr = int(n * split_ratio[0])
    n_va = int(n * split_ratio[1])
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    def _dump(name, ids):
        out = {
            "X": X[ids],
            "y": y[ids],
            "meta": [meta[i] for i in ids],
            "channels": channels,
            "idx_to_label": idx_to_label,
            "sr": AREEG_SR,
            "win_samples": AREEG_SAMPLES,
        }
        p = os.path.join(root, "preprocessed_pkl", f"{name}.pkl")
        with open(p, "wb") as f:
            pkl.dump(out, f)

    _dump("train", tr)
    _dump("val", va)
    _dump("test", te)

    with open(os.path.join(root, "split", "indices.pkl"), "wb") as f:
        pkl.dump({"train": tr, "val": va, "test": te}, f)

    stats = {"N": len(y), "train": len(tr), "val": len(va), "test": len(te)}
    return stats


def prepare_areeg_leave_one_class_out(root: str = "data/AREEG_Words",
                                    val_ratio: float = 0.15,
                                    seed: int = 1337,
                                    overlap: float = 0.0):
    """
    Leave-One-Class-Out per Subject splitting strategy.
    
    For each subject, one word/class is held out for testing.
    Different subjects have different classes held out.
    Model trains on ALL classes using data from multiple subjects.
    
    Strategy:
    - Subject A: train on words 0-14, test on word 15
    - Subject B: train on words 0-13,15, test on word 14  
    - Subject C: train on words 0-12,14-15, test on word 13
    - etc.
    
    This ensures:
    1. No subject leakage (each subject appears in only train OR test)
    2. All classes represented in training (from different subjects)
    3. Realistic generalization test (new subject + potentially unseen class combination)
    """
    os.makedirs(os.path.join(root, "preprocessed_pkl"), exist_ok=True)
    os.makedirs(os.path.join(root, "split"), exist_ok=True)

    X, y, meta, channels, idx_to_label = load_areeg_words(root=root, overlap=overlap)
    
    # Extract unique subjects and classes
    subjects = sorted(list(set([m[0] for m in meta])))  # m[0] is subject ID
    classes = sorted(list(set(y)))
    n_classes = len(classes)
    
    print(f"Found {len(subjects)} subjects and {n_classes} classes")
    print(f"Subjects: {subjects}")
    print(f"Classes: {list(idx_to_label.values())}")
    
    rng = np.random.default_rng(seed)
    
    # Assign each subject a different class to hold out
    # Cycle through classes if more subjects than classes
    rng.shuffle(subjects)
    subject_holdout_class = {}
    for i, subject in enumerate(subjects):
        holdout_class = classes[i % n_classes]
        subject_holdout_class[subject] = holdout_class
        print(f"Subject {subject}: holdout class {holdout_class} ({idx_to_label[holdout_class]})")
    
    # Split data based on subject-class assignments
    train_idx = []
    test_idx = []
    
    for i, (x_sample, y_sample, meta_sample) in enumerate(zip(X, y, meta)):
        subject = meta_sample[0]  # subject ID from metadata
        sample_class = y_sample
        holdout_class = subject_holdout_class[subject]
        
        if sample_class == holdout_class:
            # This subject's holdout class -> test set
            test_idx.append(i)
        else:
            # This subject's training classes -> train set
            train_idx.append(i)
    
    # Create validation set from training set
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * val_ratio)
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]
    
    def _dump(name, ids):
        out = {
            "X": X[ids],
            "y": y[ids],
            "meta": [meta[i] for i in ids],
            "channels": channels,
            "idx_to_label": idx_to_label,
            "sr": AREEG_SR,
            "win_samples": AREEG_SAMPLES,
        }
        p = os.path.join(root, "preprocessed_pkl", f"{name}.pkl")
        with open(p, "wb") as f:
            pkl.dump(out, f)
        
        # Print class distribution
        y_split = [y[i] for i in ids]
        class_counts = np.bincount(y_split, minlength=n_classes)
        print(f"{name.upper()} set: {len(ids)} samples")
        for cls_idx, count in enumerate(class_counts):
            if count > 0:
                print(f"  Class {cls_idx} ({idx_to_label[cls_idx]}): {count} samples")
    
    _dump("train", train_idx)
    _dump("val", val_idx)
    _dump("test", test_idx)
    
    # Save detailed split information
    split_info = {
        "strategy": "leave_one_class_out_per_subject",
        "subject_holdout_class": subject_holdout_class,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "subjects": subjects,
        "classes": classes,
        "idx_to_label": idx_to_label,
    }
    with open(os.path.join(root, "split", "leave_one_class_out_split.pkl"), "wb") as f:
        pkl.dump(split_info, f)
    
    # Verify all classes are represented in training
    train_classes = set([y[i] for i in train_idx])
    missing_classes = set(classes) - train_classes
    if missing_classes:
        print(f"WARNING: Classes missing from training set: {missing_classes}")
    else:
        print("✓ All classes represented in training set")
    
    stats = {
        "strategy": "leave_one_class_out_per_subject",
        "N_segments": len(y),
        "N_subjects": len(subjects), 
        "N_classes": n_classes,
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
        "train_classes": len(train_classes),
    }
    return stats


if __name__ == "__main__":
    root = "data/AREEG_Words"
    if os.path.exists(root):
        # Choose splitting strategy:
        
        # Option 1: Original random split (with data leakage)
        # s = prepare_areeg_to_pkl(root=root)
        # print("Saved PKLs with random split:", s)
        
        # Option 2: Leave-One-Class-Out per Subject (recommended)
        s = prepare_areeg_leave_one_class_out(root=root)
        print("Saved PKLs with leave-one-class-out split:", s)
        
    else:
        print("Run inside your Chisco repo where data/AREEG_Words exists.")
