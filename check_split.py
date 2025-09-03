#!/usr/bin/env python3
import pickle

# Load split information
with open('data/AREEG_Words/split/leave_one_class_out_split.pkl', 'rb') as f:
    split_info = pickle.load(f)

print("Split strategy:", split_info['strategy'])
print(f"Total subjects: {len(split_info['subjects'])}")
print(f"Total classes: {len(split_info['classes'])}")
print()

print("Subject-Class assignments:")
for subj, cls in split_info['subject_holdout_class'].items():
    word = split_info['idx_to_label'][cls]
    print(f"  {subj}: holdout class {cls} ({word})")

print()
print("Verification - All classes in training:")
# Load training data
with open('data/AREEG_Words/preprocessed_pkl/train.pkl', 'rb') as f:
    train_data = pickle.load(f)

train_classes = set(train_data['y'])
all_classes = set(range(len(split_info['classes'])))
missing = all_classes - train_classes

if missing:
    print(f"❌ Missing classes in training: {missing}")
else:
    print("✅ All classes present in training set")

print(f"Training set has {len(train_classes)} out of {len(all_classes)} classes")
