import os
import re
from pathlib import Path

RAW_CSV = 'data/AREEG_Words/raw_csv'
SUBJ_RE = re.compile(r"par\.?\s*(\d+)", re.IGNORECASE)

def get_all_subjects():
    subjects = set()
    subject_files = {}
    for word in os.listdir(RAW_CSV):
        word_path = os.path.join(RAW_CSV, word)
        if not os.path.isdir(word_path):
            continue
        for fname in os.listdir(word_path):
            m = SUBJ_RE.search(fname)
            if m:
                sid = f"sub{int(m.group(1)):02d}"
                subjects.add(sid)
                subject_files.setdefault(sid, []).append((word, fname))
            else:
                subject_files.setdefault('NO_MATCH', []).append((word, fname))
    return subjects, subject_files

def main():
    subjects, subject_files = get_all_subjects()
    print(f"Total unique subjects: {len(subjects)}")
    print(f"Subject IDs: {sorted(subjects)}\n")
    for sid in sorted(subject_files):
        print(f"{sid}: {len(subject_files[sid])} files")
        for word, fname in subject_files[sid]:
            print(f"  {word}/{fname}")
        print()
    if 'NO_MATCH' in subject_files:
        print("Files with no subject match:")
        for word, fname in subject_files['NO_MATCH']:
            print(f"  {word}/{fname}")

if __name__ == "__main__":
    main()
