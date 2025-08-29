# ArEEG Words: Imagined Speech EEG Classification

Reference dataset: ArrEEG An EEG based BCI dataset for decoding of imagined speech
https://arxiv.org/pdf/2411.18888

This repository contains a compact pipeline to preprocess the ArEEG Words data and train EEGNet style models to classify imagined Arabic words from 14 channel EEG at 128 Hz. It includes data preparation, windowing, denoising, train or val or test splits, model definitions, training with simple augmentations, and evaluation.

Includes: 
- Preprocessing CLI to build train or val or test PKL files from raw CSVs
- Three model heads on top of a compact EEGNet backbone
- Class reweighting and label smoothing for stable training
- Warmup plus cosine learning rate schedule
- Reproducible splits and a saved checkpoint file


## Quick Start

1. Install dependencies
   - Create a fresh environment and install
     - pip install -r requirements.txt

2. Prepare the dataset layout
   - Expected root: data/AREEG_Words
   - Place CSV files under
     - data/AREEG_Words/raw_csv/<ArabicWord>/*.csv
   - Provide two small text files in the dataset root
     - data/AREEG_Words/labels.json mapping each word to a numeric class id
     - data/AREEG_Words/channels_14.txt listing the 14 EEG channel names one per line

3. Build windowed PKLs
   - python preprocessing.py --root data/AREEG_Words --overlap 0.0 --split 0.7,0.15,0.15
   - Output files appear in
     - data/AREEG_Words/preprocessed_pkl/train.pkl
     - data/AREEG_Words/preprocessed_pkl/val.pkl
     - data/AREEG_Words/preprocessed_pkl/test.pkl

4. Train a model
   - Example with attention head
     - python EEGclassify.py --root data/AREEG_Words --head attn --epochs 200 --batch 256
   - The script prints val accuracy and saves the best checkpoint to best_areeg_eegnet.pt by default

5. Evaluate the best checkpoint
   - By default training reloads the best checkpoint and reports test accuracy at the end


## Data and Preprocessing

Files and structure
- Raw CSVs: one CSV per trial or segment under folders named by the Arabic word label
- Labels: labels.json is a dict like {"WORD": class_index}
- Channels list: channels_14.txt contains the canonical channel order used by the model

Signal assumptions
- Sampling rate: 128 Hz
- Channels: 14 EEG channels
- Segment duration: 2.0 seconds per window so 256 samples

CSV parsing and channel alignment
- The loader searches for a header row that contains channel names. It is tolerant to encodings utf 8 sig, utf 8, and cp1256
- Column names like EEG.Fp1 are normalized by dropping the EEG. prefix
- The code enforces an exact match to the names listed in channels_14.txt and will raise a helpful error if any channel is missing

Denoising filters
- 1 to 40 Hz fourth order Butterworth bandpass
- 50 Hz notch filter for mains interference
- Implemented in scipy.signal with zero phase filtfilt

Windowing and overlap
- Each recording is split into fixed windows of 2.0 seconds
- Optional overlap can be set with --overlap in [0.0, 1.0). For example 0.5 gives a 50 percent overlap so a hop size of half a window

Standardization at training time
- Each window is z scored per channel using the window mean and standard deviation

Splits and reproducibility
- Random shuffling with a fixed seed controls the split
- Ratios default to 70 or 15 or 15 for train or val or test
- Split indices are saved to data/AREEG_Words/split/indices.pkl

Implementation references
- data_imagine.py: core loading, filtering, windowing, and PKL writing
- preprocessing.py: small CLI wrapper around prepare_areeg_to_pkl

See the following for details
- data_imagine.py:1
- preprocessing.py:1


## Model Architecture

Backbone: EEGNet style CNN
- Temporal convolution on each channel then a depthwise spatial convolution across channels
- Depthwise temporal convolution followed by pointwise convolution to mix features
- Average pooling and dropout after each block
- Output is a feature map with shape [batch, F2, T_out]

Heads
- Average pooling head: global average pooling over time then a linear classifier
- GRU head: bidirectional GRU across the temporal axis then a linear classifier
- Attention head: 1D attention over time with softmax weights then a linear classifier

Default hyperparameters
- Channels = 14, F1 = 16, D = 2, F2 = 64
- Kernel sizes: k1 = 15, k2 = 7
- Pool sizes: P1 = 2, P2 = 2
- Dropout rate = 0.5

Implementation references
- eegcnn.py:1
- EEGclassify.py:1


## Training Procedure

Optimisation
- Optimizer: AdamW
- Learning rate: 3e-4 by default
- Weight decay: 1e-2 by default
- Learning rate schedule: linear warmup for the first warmup epochs then cosine decay to zero

Loss and regularisation
- Cross entropy with class weights inversely proportional to class counts in the training set
- Label smoothing of 0.05

Data augmentation
- Time shift: random circular shift in time of up to 8 samples which is about 62 ms at 128 Hz
- Mixup: convex combination of pairs in a batch with Beta alpha equal to 0.2

Metrics and logging
- Top 1 and Top 3 accuracy on validation and test sets
- Best validation model is saved to disk and reloaded for final test evaluation

Reproducibility
- NumPy and PyTorch seeds are set from the command line argument seed

Practical notes
- Adjust batch size based on GPU memory. 256 works on common 8 GB or 12 GB GPUs. Reduce if you see out of memory errors
- If your data has more than 14 channels or a different order, update channels_14.txt and rebuild the PKLs


## Expected Inputs and Outputs

Inputs to training
- Preprocessed PKLs with keys X, y, meta, channels, idx_to_label, sr, win_samples
- Shapes: X is [N, 14, T], y is [N]

Model I or O
- Input tensor: float32 EEG windows of shape [B, 14, T]
- Output tensor: logits of shape [B, n_classes]

Saved artifacts
- Model checkpoint file best_areeg_eegnet.pt
- Split indices at data/AREEG_Words/split/indices.pkl


## Results

Preprocessing
- Command
  
  ```
  python preprocessing.py --dataset areeg_words --root data/AREEG_Words --overlap 0.75
  ```

- Output
  
  ```
  Saved PKLs: {'N': 6520, 'train': 4564, 'val': 978, 'test': 978}
  ```

- Note: 0.75 overlap increases window count to 6520 and yields the split sizes above with the default 70 or 15 or 15 ratio.

Training
- Command
  
  ```
  python EEGclassify.py --root data/AREEG_Words --epochs 2000 --batch 256 --lr 3e-4 --weight_decay 1e-2 --warmup 30 --F1 48 --D 2 --F2 192 --k1 31 --k2 15 --P1 2 --P2 2 --dropout 0.55 --head gru --rnn_hidden 128 --rnn_layers 1
  ```

- Final log
  
  ```
  epoch 2000  lr 1.91e-10  train_loss 0.7872  val_loss 0.4484  val@1 0.8988  val@3 0.9581  best 0.9049
  test_loss 0.4036  test@1 0.9151  test@3 0.9652
  ```

- Summary: GRU head with wider backbone reached about 89.9 percent top 1 and 95.8 percent top 3 on validation. Best seen during training was about 90.5 percent. Final test accuracy was about 91.5 percent top 1 and 96.5 percent top 3.


## Repository Map

- data_imagine.py: CSV to windowed tensors to PKL for ArEEG Words
- preprocessing.py: CLI to prepare PKLs
- eegcnn.py: EEGNet backbone and three classifier heads avg, gru, attn
- EEGclassify.py: training loop, evaluation, and checkpointing
- mapping.ipynb: Generates a stable mapping from word to class id using the folder names found in raw_csv/. This preserves Arabic text. The results are stored in data\AREEG_Words\labels.json . 
- ArEEG.pdf: Dataset Paper
- 
## Acknowledgment

All credit for the ArEEG dataset and data collection belongs to the authors of ArEEG_Words: Dataset for Envisioned Speech Recognition using EEG for Arabic Words.
