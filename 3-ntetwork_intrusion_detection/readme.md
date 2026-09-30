# Network Intrusion Detection

My third machine learning project — a model that classifies network traffic as benign or as a
specific attack type, based on CICIDS2017-style flow statistics.

## Overview

Given a network flow's packet, timing, flag, and byte-level features, the model predicts the
traffic **Label** — `BENIGN` or one of eight attack classes (`DDoS`, `DoS Hulk`, `DoS GoldenEye`,
`DoS slowloris`, `DoS Slowhttptest`, `PortScan`, `FTP-Patator`, `SSH-Patator`). The training
script in `src/train.py` fits a Decision Tree with `GridSearchCV` and saves the best estimator.

## Project structure

```
3-ntetwork_intrusion_detection/
├── data/
│   ├── merged.csv                     # merged + class-balanced sample
│   └── clean_data.csv                 # cleaned dataset used for training
├── models/
│   └── network_intrusion_detection_model.pkl   # saved Decision Tree (GridSearchCV best)
├── notebooks/
│   ├── csvmerge.ipynb                 # merge daily CSVs, filter labels, downsample
│   ├── EDA.ipynb                      # data cleaning & exploratory analysis
│   └── model_train.ipynb              # training, tuning and evaluation
├── src/
│   └── train.py                       # script version of the GridSearchCV pipeline
└── .gitignore
```

## This project uses the CIC-IDS2017 dataset.
## The original dataset is not included in this repository because of its large size.
refer this link --> https://cicresearch.ca/CICDataset/CIC-IDS-2017/
or  https://www.unb.ca/cic/datasets/ids-2017.html

## Dataset

Each row is one network flow, with columns such as:

- **Flow / size**: `Destination Port`, `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`, packet-length stats
- **Rates**: `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`
- **Timing**: IAT (inter-arrival time), idle, and active statistics
- **Flags**: `PSH Flag Count`, `ACK Flag Count`, `SYN Flag Count`, `FIN Flag Count`, and related flag columns
- **Target**: `Label` — `BENIGN` or an attack type

The eight daily CSVs (`Monday.csv` through `Friday-aft2.csv`) are concatenated in
`notebooks/csvmerge.ipynb`. Labels are stripped of stray whitespace, rare classes (Bot, web
attacks, Infiltration, Heartbleed) are dropped, and each of the nine kept classes is sampled to
5,000 rows (`merged.csv`).

In `EDA.ipynb` the leftover index column is dropped, infinities in `Flow Bytes/s` and
`Flow Packets/s` are replaced with NaN and filled with the median, constant columns and the
duplicate `Fwd Header Length.1` column are removed, duplicates are dropped, and near-constant
flag columns (`Fwd URG Flags`, `RST Flag Count`, `CWE Flag Count`, `ECE Flag Count`) are dropped.
The result is saved as `clean_data.csv` (~39,820 rows × 66 columns), which is what the model is
trained on.

## Approach

1. **Merge / sample** (`notebooks/csvmerge.ipynb`) — joined the eight day files, inspected class
   counts (BENIGN dominates the raw data), kept nine labels with enough samples, and downsampled
   to 5,000 rows per class so training isn't skewed toward BENIGN.
2. **EDA** (`notebooks/EDA.ipynb`) — cleaned inf/NaN values, dropped constant and duplicate
   columns, visualized class counts and feature distributions, looked at a correlation heatmap,
   and used flag-vs-label crosstabs to drop flags that almost never fire.
3. **Preprocessing + training** (`notebooks/model_train.ipynb`, mirrored in `src/train.py`) —
   features are already numeric after EDA, so the data is split 80/20 (`random_state=42`) with
   `Label` as the target. A `DecisionTreeClassifier` is tuned with `GridSearchCV` (`cv=5`,
   scoring `f1_micro`) over `criterion`, `max_depth`, `min_samples_split`, and
   `min_samples_leaf`. The notebook also compares a simple tree and `RandomizedSearchCV`; the
   saved model is the GridSearch best estimator (`models/network_intrusion_detection_model.pkl`).
4. **Evaluation** — accuracy, a per-class classification report, a confusion matrix, feature
   importances (destination port, packet rates, and related flow stats rank highest), and a
   shallow tree plot are used to check that all nine classes are distinguished, not just overall
   accuracy.

## Getting started

```bash
# retrain the model from clean_data.csv
python src/train.py
```

## Model Evaluation

- **Accuracy (GridSearchCV, saved model):** 99.70%

- **Model comparison (same test set):**

| Model | Accuracy | Macro F1-Score | Weighted F1-Score |
|---|---:|---:|---:|
| Simple Decision Tree | 99.55% | 99.56% | 99.55% |
| RandomizedSearchCV | 99.69% | 99.70% | 99.69% |
| GridSearchCV | 99.70% | 99.71% | 99.70% |

- **Classification Report (GridSearchCV):**

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| BENIGN | 0.9990 | 0.9879 | 0.9934 | 993 |
| DDoS | 0.9990 | 0.9990 | 0.9990 | 1000 |
| DoS GoldenEye | 0.9970 | 0.9990 | 0.9980 | 1000 |
| DoS Hulk | 0.9987 | 0.9962 | 0.9975 | 796 |
| DoS Slowhttptest | 0.9937 | 0.9979 | 0.9958 | 953 |
| DoS slowloris | 0.9925 | 0.9957 | 0.9941 | 930 |
| FTP-Patator | 0.9974 | 1.0000 | 0.9987 | 766 |
| PortScan | 0.9979 | 0.9990 | 0.9985 | 973 |
| SSH-Patator | 0.9982 | 1.0000 | 0.9991 | 553 |
| **Accuracy** | | | **0.9970** | **7964** |
| **Macro Avg** | **0.9971** | **0.9972** | **0.9971** | **7964** |
| **Weighted Avg** | **0.9970** | **0.9970** | **0.9970** | **7964** |

## Notes / what's next

This is a Decision Tree classifier built to practice a full multi-class workflow on a large
network-flow dataset (merge → downsample → cleaning → EDA → training → tuning → evaluation).
Tuning only moved accuracy a little past the simple tree, so an interesting next step would be
trying a tree ensemble (Random Forest / XGBoost), bringing the rare attack classes back in, or
adding a `predict.py` CLI that scores a single flow the same way projects 1 and 2 do.
