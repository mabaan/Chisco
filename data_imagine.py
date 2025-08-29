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


if __name__ == "__main__":
    root = "data/AREEG_Words"
    if os.path.exists(root):
        s = prepare_areeg_to_pkl(root=root)
        print("Saved PKLs:", s)
    else:
        print("Run inside your Chisco repo where data/AREEG_Words exists.")
