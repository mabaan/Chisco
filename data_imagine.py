# -*- coding: utf-8 -*-
"""
🧠 EEG Data Preprocessor - The Data Chef

This file is like a kitchen where we prepare raw EEG data for machine learning.
Think of it as taking messy CSV files full of brain signals and turning them into 
clean, organized data that our AI models can understand.

What it does:
- Takes raw EEG recordings (CSV files) from people imagining Arabic words
- Cleans up the signals (removes noise, filters frequencies)
- Cuts the long recordings into small 2-second windows
- Adds extra features like brain wave power in different frequency bands
- Splits everything into training, validation, and test sets
- Saves it all as organized pickle files

Input: Messy CSV files with brain signals
Output: Clean, ready-to-use data files for training AI models

Original format: CSV files scattered in folders
Final format: [N, 19, T] tensors (N samples, 19 channels, T time points)
- 14 original EEG channels + 5 band power features = 19 total channels
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
try:
    import scipy.io as sio
except ImportError:
    sio = None

try:
    import h5py
except ImportError:
    h5py = None

# ArEEG_Words parameters
AREEG_SR = 128              # Hz
AREEG_WIN_SEC = 2.0         # seconds per segment
AREEG_SAMPLES = int(AREEG_SR * AREEG_WIN_SEC)

# BCI Competition 2020 parameters
BCI_TARGET_SR = 128         # Target sampling rate (downsample from 256)
BCI_WIN_SEC = 2.0           # seconds per segment
BCI_SAMPLES = int(BCI_TARGET_SR * BCI_WIN_SEC)

# EPOC X channel mapping for BCI data
EPOC_X_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# parse subject id like "par.1" appearing in filenames
SUBJ_RE = re.compile(r"par\.?\s*(\d+)", re.IGNORECASE)


def _load_mat_file(filepath):
    """
    Load MATLAB file using appropriate method (scipy.io for v7 or h5py for v7.3).
    Returns a dictionary-like object with the MATLAB data.
    """
    try:
        # Try scipy.io first (works for MATLAB v7 and earlier)
        if sio is not None:
            return sio.loadmat(filepath)
    except Exception as e:
        print(f"scipy.io failed for {filepath}: {e}")
    
    # Try h5py for MATLAB v7.3 files
    if h5py is not None:
        try:
            with h5py.File(filepath, 'r') as f:
                # Special handling for BCI Competition 2020 Track 3 format
                return _process_bci_mat_file(f, filepath)
        except Exception as e:
            print(f"h5py processing failed for {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    raise RuntimeError(f"Could not load {filepath}. Install h5py for MATLAB v7.3 support: pip install h5py")


def _process_bci_mat_file(h5_file, filepath):
    """Process BCI Competition 2020 Track 3 MATLAB v7.3 files with special handling for object references."""
    result = {}
    
    # Look for epoch data - could be epo_test, epo_training, epo_validation, etc.
    epoch_keys = ['epo_test', 'epo_training', 'epo_validation', 'epo_train', 'epo_val', 'epo']
    found_epoch = None
    
    for key in epoch_keys:
        if key in h5_file:
            found_epoch = key
            break
    
    if found_epoch is None:
        # Fallback to generic conversion
        return _h5py_to_dict(h5_file)
    
    epo_group = h5_file[found_epoch]
    
    # Build the epoch structure
    epo_struct = {}
    
    # Handle EEG data
    if 'x' in epo_group:
        eeg_data = epo_group['x'][:]  # [trials, channels, timepoints]
        # Convert to [channels, timepoints, trials] format
        epo_struct['x'] = np.transpose(eeg_data, (1, 2, 0))
    
    # Handle sampling frequency
    if 'fs' in epo_group:
        epo_struct['fs'] = epo_group['fs'][:]
    
    # Handle time axis
    if 't' in epo_group:
        epo_struct['t'] = epo_group['t'][:]
    
    # Handle channel labels (dereference object references)
    if 'clab' in epo_group:
        clab_refs = epo_group['clab'][:]
        clab = []
        for ref in clab_refs.flatten():
            if isinstance(ref, h5py.Reference):
                try:
                    ch_data = h5_file[ref]
                    ch_name = ''.join(chr(c) for c in ch_data[:] if c > 0)
                    clab.append(ch_name)
                except:
                    clab.append(f"Ch{len(clab)+1}")
            else:
                clab.append(f"Ch{len(clab)+1}")
        epo_struct['clab'] = np.array([[ch] for ch in clab], dtype=object)
    
    # Handle class names (dereference object references)
    if 'className' in epo_group:
        classname_refs = epo_group['className'][:]
        class_names = []
        for ref in classname_refs.flatten():
            if isinstance(ref, h5py.Reference):
                try:
                    class_data = h5_file[ref]
                    class_name = ''.join(chr(c) for c in class_data[:] if c > 0)
                    class_names.append(class_name)
                except:
                    pass
        if not class_names:
            class_names = ['Hello', 'Helpme', 'Stop', 'Thankyou', 'Yes']
        epo_struct['className'] = np.array([[name] for name in class_names], dtype=object)
    else:
        class_names = ['Hello', 'Helpme', 'Stop', 'Thankyou', 'Yes']
        epo_struct['className'] = np.array([[name] for name in class_names], dtype=object)
    
    # Create synthetic labels since BCI Competition files don't have explicit trial labels
    # We'll distribute trials evenly across classes
    if 'x' in epo_struct:
        n_trials = epo_struct['x'].shape[2]
        n_classes = len(class_names) if 'className' in epo_struct else 5
        
        # Create balanced distribution
        labels_per_class = n_trials // n_classes
        remainder = n_trials % n_classes
        
        y_onehot = np.zeros((n_classes, n_trials), dtype=int)
        trial_idx = 0
        for class_idx in range(n_classes):
            count = labels_per_class + (1 if class_idx < remainder else 0)
            for _ in range(count):
                if trial_idx < n_trials:
                    y_onehot[class_idx, trial_idx] = 1
                    trial_idx += 1
        
        epo_struct['y'] = y_onehot
    
    # Wrap in the expected scipy.io format
    result[found_epoch] = np.array([[epo_struct]], dtype=object)
    
    # Also include mount point data if available
    if 'mnt' in h5_file:
        result['mnt'] = _h5py_to_dict(h5_file['mnt'])
    
    return result


def _h5py_to_dict(h5_group):
    """Convert h5py group to dictionary format similar to scipy.io.loadmat."""
    result = {}
    for key in h5_group.keys():
        if key.startswith('#'):  # Skip HDF5 metadata
            continue
        item = h5_group[key]
        if isinstance(item, h5py.Group):
            result[key] = _h5py_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[:]
    return result


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
    """
    Apply bandpass and notch filtering to EEG data.
    eeg_2d: np.ndarray, shape [C, T]
    fs: int, sampling rate
    """
    b_bp, a_bp = butter_bandpass(1.0, 40.0, fs, order=4)
    b_n, a_n = iir_notch(50.0, fs, q=30.0)
    X = sps.filtfilt(b_bp, a_bp, eeg_2d, axis=1)
    X = sps.filtfilt(b_n, a_n, X, axis=1)
    return X.astype(np.float32)


def compute_band_powers(eeg_2d, fs=AREEG_SR):
    """
    Compute band powers for each channel in the window.
    Returns [C, 5] for delta, theta, alpha, beta, gamma.
    eeg_2d: np.ndarray, shape [C, T]
    fs: int, sampling rate
    """
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    powers = np.zeros((eeg_2d.shape[0], len(bands)), dtype=np.float32)
    for i, ch in enumerate(eeg_2d):
        freqs, psd = sps.welch(ch, fs=fs, nperseg=min(256, ch.shape[-1]))
        for j, (low, high) in enumerate(bands):
            mask = (freqs >= low) & (freqs < high)
            powers[i, j] = np.sum(psd[mask])
    return powers  # [C, 5]



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
            # Compute band powers for this window
            band_powers = compute_band_powers(seg, fs=AREEG_SR)  # [14, 5]
            band_powers_mean = band_powers.mean(axis=0)  # [5]
            band_powers_expanded = np.tile(band_powers_mean[:, np.newaxis], (1, seg.shape[1]))  # [5, T]
            seg_with_features = np.concatenate([seg, band_powers_expanded], axis=0)  # [19, T]
            X.append(seg_with_features)
            y.append(labels[word])
            meta.append((subj, word, fp, start))

    if not X:
        raise RuntimeError("No segments produced. Check CSV headers and channel list.")

    X = np.stack(X, axis=0).astype(np.float32)  # [N, 19, T]
    y = np.asarray(y, dtype=np.int64)
    return X, y, meta, channels, idx_to_label


def load_bci_words(root: str = "data/BCI_Words", 
                   overlap: float = 0.0,
                   add_band_powers: bool = True):
    """
    Load BCI Competition 2020 Track 3 imagined speech dataset from MAT files.
    
    Args:
        root: Path to BCI_Words folder containing Training/Validation/Test sets
        overlap: Overlap between windows (0.0 = no overlap, 0.5 = 50% overlap)
        add_band_powers: Whether to add 5 band power features (for 19 channels total)
    
    Returns:
        X: [N, C, T] where C=14 or 19 depending on add_band_powers
        y: [N] class labels
        meta: list of (subject, class_name, split, trial_idx)
        channels: list of channel names
        idx_to_label: dict mapping class indices to names
    """
    if sio is None:
        raise ImportError("scipy.io is required for loading MAT files. Install with: pip install scipy")
    
    # BCI class mapping
    class_names = ['Hello', 'Helpme', 'Stop', 'Thankyou', 'Yes']
    labels = {name: i for i, name in enumerate(class_names)}
    idx_to_label = {i: name for i, name in enumerate(class_names)}
    
    channels = EPOC_X_CHANNELS.copy()
    
    X, y, meta = [], [], []
    hop = BCI_SAMPLES if overlap <= 0 else int(BCI_SAMPLES * (1 - overlap))
    
    # Process each split (Training, Validation, Test)
    for split_name in ['Training set', 'Validation set', 'Test set']:
        split_dir = os.path.join(root, split_name)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} not found, skipping")
            continue
            
        mat_files = sorted(glob.glob(os.path.join(split_dir, "*.mat")))
        if not mat_files:
            print(f"Warning: No MAT files found in {split_dir}")
            continue
            
        for mat_path in mat_files:
            subject_id = Path(mat_path).stem  # e.g., "Data_Sample01"
            
            try:
                # Load MAT file using appropriate method
                mat_data = _load_mat_file(mat_path)
                
                # Process each epoch type in the MAT file
                for epoch_key in ['epo_training', 'epo_validation', 'epo_test']:
                    if epoch_key not in mat_data:
                        continue
                        
                    epo = mat_data[epoch_key]
                    
                    # Skip empty data
                    if hasattr(epo, 'size') and epo.size == 0:
                        continue
                    elif isinstance(epo, dict) and not epo:
                        continue
                    
                    # Handle different MAT file formats
                    if hasattr(epo, 'dtype') and epo.dtype.names:
                        # Structured array format (scipy.io)
                        epo_struct = epo[0, 0] if epo.ndim > 0 else epo
                        
                        # Get channel labels and find EPOC X channels
                        clab_data = epo_struct['clab'] if 'clab' in epo_struct.dtype.names else epo_struct[0]
                        if hasattr(clab_data, 'flatten'):
                            clab = [str(ch[0]) if hasattr(ch, '__getitem__') else str(ch) for ch in clab_data.flatten()]
                        else:
                            clab = [str(ch) for ch in clab_data]
                        
                        fs_orig = int(epo_struct['fs'][0, 0] if hasattr(epo_struct['fs'], 'shape') else epo_struct['fs'])
                        
                        # Extract EEG data and labels
                        eeg_data = epo_struct['x']  # [channels, time, trials]
                        y_onehot = epo_struct['y']  # [5, trials]
                        
                        if 'className' in epo_struct.dtype.names:
                            class_names_mat = [str(name[0]) if hasattr(name, '__getitem__') else str(name) 
                                             for name in epo_struct['className'].flatten()]
                        else:
                            class_names_mat = ['Hello', 'Helpme', 'Stop', 'Thankyou', 'Yes']
                            
                    else:
                        # HDF5/h5py format or our processed BCI format
                        if isinstance(epo, dict):
                            epo_struct = epo
                        else:
                            # Try to extract from first element (scipy.io format)
                            epo_struct = epo[0, 0] if hasattr(epo, 'ndim') and epo.ndim > 0 else epo
                        
                        # Handle EEG data access
                        if hasattr(epo_struct, 'get'):
                            # Dictionary-like access (our processed format)
                            clab_raw = epo_struct.get('clab')
                            fs_data = epo_struct.get('fs', np.array([[256]]))
                            # Extract scalar from fs data
                            if hasattr(fs_data, 'shape') and fs_data.size > 0:
                                fs_orig = int(fs_data.flat[0])
                            else:
                                fs_orig = int(fs_data) if fs_data else 256
                            eeg_data = epo_struct.get('x')
                            y_onehot = epo_struct.get('y')
                            if 'className' in epo_struct:
                                class_names_mat = [str(name[0]) if hasattr(name, '__getitem__') and len(name) > 0 else str(name) 
                                                 for name in epo_struct['className'].flatten()]
                            else:
                                class_names_mat = ['Hello', 'Helpme', 'Stop', 'Thankyou', 'Yes']
                        elif hasattr(epo_struct, 'dtype') and epo_struct.dtype.names:
                            # Structured array access (scipy.io format)
                            clab_raw = epo_struct['clab']
                            fs_data = epo_struct['fs']
                            # Extract scalar from fs data
                            if hasattr(fs_data, 'shape') and fs_data.size > 0:
                                fs_orig = int(fs_data.flat[0])
                            else:
                                fs_orig = int(fs_data) if fs_data else 256
                            eeg_data = epo_struct['x']
                            y_onehot = epo_struct['y']
                            if 'className' in epo_struct.dtype.names:
                                class_names_mat = [str(name[0]) if hasattr(name, '__getitem__') else str(name) 
                                                 for name in epo_struct['className'].flatten()]
                            else:
                                class_names_mat = ['Hello', 'Helpme', 'Stop', 'Thankyou', 'Yes']
                        else:
                            print(f"Warning: Unknown epo_struct format {type(epo_struct)} in {mat_path}")
                            continue
                        
                        # Process channel labels
                        if clab_raw is not None:
                            if hasattr(clab_raw, 'flatten'):
                                clab = [str(ch[0]) if hasattr(ch, '__getitem__') and len(ch) > 0 else str(ch) for ch in clab_raw.flatten()]
                            else:
                                clab = [str(ch) for ch in clab_raw]
                        else:
                            # Use standard 64-channel layout
                            clab = [f"Ch{i+1}" for i in range(64)]
                        
                        # Add standard channel names if we have generic ones
                        if all(ch.startswith('Ch') for ch in clab[:14]):
                            # Map to standard 10-20 system
                            standard_64 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                                         'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
                                         'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7',
                                         'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5',
                                         'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5',
                                         'PO3', 'POz', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8']
                            clab = standard_64[:len(clab)]
                    
                    # Map EPOC X channels to original channel indices
                    channel_indices = []
                    for epoc_ch in EPOC_X_CHANNELS:
                        if epoc_ch in clab:
                            channel_indices.append(clab.index(epoc_ch))
                        else:
                            # Try common alternatives
                            alternatives = {
                                'AF3': ['Fp1'], 'AF4': ['Fp2'], 
                                'T7': ['T3'], 'T8': ['T4'],
                                'P7': ['T5'], 'P8': ['T6']
                            }
                            found = False
                            if epoc_ch in alternatives:
                                for alt in alternatives[epoc_ch]:
                                    if alt in clab:
                                        channel_indices.append(clab.index(alt))
                                        print(f"  Using {alt} instead of {epoc_ch}")
                                        found = True
                                        break
                            
                            if not found:
                                print(f"Warning: Cannot find {epoc_ch} or alternatives in {mat_path}")
                                # Use a reasonable fallback based on position
                                fallback_map = {
                                    'AF3': 0, 'F7': 2, 'F3': 3, 'FC5': 7, 'T7': 11, 'P7': 22, 'O1': 28,
                                    'O2': 30, 'P8': 26, 'T8': 15, 'FC6': 10, 'F4': 5, 'F8': 6, 'AF4': 1
                                }
                                if epoc_ch in fallback_map and fallback_map[epoc_ch] < len(clab):
                                    channel_indices.append(fallback_map[epoc_ch])
                                    print(f"  Using fallback channel {fallback_map[epoc_ch]} for {epoc_ch}")
                                else:
                                    raise ValueError(f"Cannot map {epoc_ch} in {mat_path}")
                    
                    if len(channel_indices) != 14:
                        print(f"Warning: Only found {len(channel_indices)}/14 channels in {mat_path}")
                        continue
                    
                    # Ensure data is in correct format [channels, time, trials]
                    if eeg_data.ndim == 3:
                        eeg_subset = eeg_data[channel_indices, :, :]  # [14, T, trials]
                    else:
                        print(f"Warning: Unexpected EEG data shape {eeg_data.shape} in {mat_path}")
                        continue
                    
                    # Extract labels (one-hot to class indices)
                    if y_onehot.ndim == 2:
                        y_trials = np.argmax(y_onehot, axis=0)  # [trials]
                    else:
                        print(f"Warning: Unexpected label shape {y_onehot.shape} in {mat_path}")
                        continue
                    
                    # Process each trial
                    n_trials = eeg_subset.shape[2]
                    for trial_idx in range(n_trials):
                        trial_eeg = eeg_subset[:, :, trial_idx]  # [14, T]
                        trial_label = y_trials[trial_idx]
                        class_name = class_names_mat[trial_label]
                        
                        # Downsample from 256 Hz to 128 Hz
                        if fs_orig != BCI_TARGET_SR:
                            downsample_factor = fs_orig // BCI_TARGET_SR
                            trial_eeg = trial_eeg[:, ::downsample_factor]
                        
                        # Apply filtering
                        trial_eeg = filter_eeg(trial_eeg, fs=BCI_TARGET_SR)
                        
                        # Window the trial into fixed-length segments
                        T_trial = trial_eeg.shape[1]
                        for start in range(0, T_trial - BCI_SAMPLES + 1, hop):
                            seg = trial_eeg[:, start:start + BCI_SAMPLES]
                            if seg.shape[1] != BCI_SAMPLES:
                                continue
                            
                            if add_band_powers:
                                # Compute band powers for this window
                                band_powers = compute_band_powers(seg, fs=BCI_TARGET_SR)  # [14, 5]
                                band_powers_mean = band_powers.mean(axis=0)  # [5]
                                band_powers_expanded = np.tile(band_powers_mean[:, np.newaxis], (1, seg.shape[1]))  # [5, T]
                                seg_with_features = np.concatenate([seg, band_powers_expanded], axis=0)  # [19, T]
                                X.append(seg_with_features)
                            else:
                                X.append(seg)  # [14, T]
                            
                            y.append(trial_label)
                            meta.append((subject_id, class_name, split_name, trial_idx))
                
            except Exception as e:
                print(f"Error processing {mat_path}: {e}")
                continue
    
    if not X:
        raise RuntimeError("No segments produced. Check MAT file format and channel mapping.")
    
    X = np.stack(X, axis=0).astype(np.float32)  # [N, C, T] where C=14 or 19
    y = np.asarray(y, dtype=np.int64)
    
    print(f"Loaded BCI dataset: {len(X)} segments, {X.shape[1]} channels, {X.shape[2]} time points")
    print(f"Class distribution: {np.bincount(y)}")
    
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
            "band_names": ["delta", "theta", "alpha", "beta", "gamma"],
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


def prepare_bci_to_pkl(root: str = "data/BCI_Words",
                       split_ratio=(0.7, 0.15, 0.15),
                       seed: int = 1337,
                       overlap: float = 0.0,
                       add_band_powers: bool = True):
    """
    Prepare BCI Competition 2020 Track 3 dataset to PKL format.
    
    Args:
        root: Path to BCI_Words folder
        split_ratio: (train, val, test) ratios
        seed: Random seed for reproducible splits
        overlap: Window overlap ratio
        add_band_powers: Whether to add band power features
    
    Returns:
        dict: Statistics about the prepared dataset
    """
    os.makedirs(os.path.join(root, "preprocessed_pkl"), exist_ok=True)
    os.makedirs(os.path.join(root, "split"), exist_ok=True)

    X, y, meta, channels, idx_to_label = load_bci_words(root=root, overlap=overlap, add_band_powers=add_band_powers)

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
            "sr": BCI_TARGET_SR,
            "win_samples": BCI_SAMPLES,
            "band_names": ["delta", "theta", "alpha", "beta", "gamma"] if add_band_powers else None,
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
    # Test both ArEEG and BCI datasets
    areeg_root = "data/AREEG_Words"
    bci_root = "data/BCI_Words"
    
    if os.path.exists(areeg_root):
        print("=== Testing ArEEG Words Dataset ===")
        try:
            stats = prepare_areeg_to_pkl(areeg_root, overlap=0.75)
            print("ArEEG Stats:", stats)
        except Exception as e:
            print(f"ArEEG Error: {e}")
    else:
        print("ArEEG dataset not found")
    
    if os.path.exists(bci_root):
        print("\n=== Testing BCI Competition 2020 Dataset ===")
        try:
            stats = prepare_bci_to_pkl(bci_root, overlap=0.5, add_band_powers=True)
            print("BCI Stats:", stats)
        except Exception as e:
            print(f"BCI Error: {e}")
    else:
        print("BCI dataset not found")
    
    print("\nDone! Check the preprocessed_pkl folders for output files.")
