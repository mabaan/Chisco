import os
import re
import shutil
from pathlib import Path

RAW_CSV = 'data/AREEG_Words/raw_csv'
SUBJ_RE = re.compile(r"par\.?\s*(\d+)", re.IGNORECASE)

# List of all expected words/classes
EXPECTED_WORDS = [
    'اختر', 'اسفل', 'اعلى', 'انذار', 'ايقاف تشغيل', 'تشغيل', 'جوع', 'حذف', 'حمام', 'دواء', 'عطش', 'لا', 'مسافة', 'نعم', 'يسار', 'يمين'
]

def get_subject_files():
    subject_files = {}
    for word in EXPECTED_WORDS:
        word_path = os.path.join(RAW_CSV, word)
        if not os.path.isdir(word_path):
            continue
        for fname in os.listdir(word_path):
            m = SUBJ_RE.search(fname)
            if m:
                sid = f"sub{int(m.group(1)):02d}"
                subject_files.setdefault(sid, {})[word] = fname
    return subject_files

def clean_and_filter_subjects():
    subject_files = get_subject_files()
    # Find subjects missing any class
    incomplete_subjects = [sid for sid, files in subject_files.items() if len(files) < len(EXPECTED_WORDS)]
    complete_subjects = [sid for sid, files in subject_files.items() if len(files) == len(EXPECTED_WORDS)]
    print(f"Dropping {len(incomplete_subjects)} incomplete subjects: {incomplete_subjects}")
    print(f"Keeping {len(complete_subjects)} complete subjects: {complete_subjects}")
    # Remove files for incomplete subjects
    for sid in incomplete_subjects:
        files = subject_files[sid]
        for word, fname in files.items():
            fpath = os.path.join(RAW_CSV, word, fname)
            print(f"Deleting {fpath}")
            os.remove(fpath)
    # Optionally, clean up filenames for complete subjects
    for sid in complete_subjects:
        for word, fname in subject_files[sid].items():
            # Standardize filename: par.<num> <word>_EPOCX_...
            m = SUBJ_RE.search(fname)
            num = m.group(1)
            ext = fname.split('.')[-1]
            new_fname = f"par.{num} {word}_EPOCX_{'_'.join(fname.split('_')[2:])}"
            old_path = os.path.join(RAW_CSV, word, fname)
            new_path = os.path.join(RAW_CSV, word, new_fname)
            if fname != new_fname:
                print(f"Renaming {old_path} -> {new_path}")
                shutil.move(old_path, new_path)

if __name__ == "__main__":
    clean_and_filter_subjects()
