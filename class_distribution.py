import os
import pickle as pkl
import numpy as np

def print_class_distribution(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pkl.load(f)
    y = obj["y"]
    idx_to_label = obj["idx_to_label"]
    counts = np.bincount(y, minlength=len(idx_to_label))
    print(f"Class distribution for {pkl_path}:")
    for i, count in enumerate(counts):
        print(f"  Class {i} ({idx_to_label[i]}): {count} samples")
    print()

if __name__ == "__main__":
    base = "data/AREEG_Words/preprocessed_pkl"
    for split in ["train", "val", "test"]:
        pkl_path = os.path.join(base, f"{split}.pkl")
        print_class_distribution(pkl_path)
