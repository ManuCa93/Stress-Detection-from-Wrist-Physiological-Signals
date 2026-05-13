# Stress Detection from Wrist Physiological Signals

A machine learning pipeline for detecting psychological stress using wrist-worn sensor data from the **WESAD** (Wearable Stress and Affect Detection) dataset. The project covers the full lifecycle — signal preprocessing, feature engineering, model training, evaluation, explainability, and critical discussion.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Summary](#pipeline-summary)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project trains and evaluates four machine learning classifiers (Logistic Regression, SVM with RBF kernel, Gradient Boosting, and MLP) on physiological signals recorded from a wrist-worn Empatica E4 device. Both binary classification (Stress vs. Non-Stress) and 3-class classification (Baseline / Stress / Amusement) are explored.

Models are evaluated using **Leave-One-Subject-Out (LOSO) cross-validation** — the gold standard for subject-independent wearable sensing research — ensuring no data from the test subject is present during training.

---

## Dataset

**WESAD** — Wearable Stress and Affect Detection  
Schmidt et al., ACM ICMI 2018 | UCI ML Repository

- 15 subjects (S2–S17; S1 and S12 excluded due to sensor malfunction)
- Wrist device: **Empatica E4** — ACC (32 Hz), BVP (64 Hz), EDA (4 Hz), TEMP (4 Hz)
- Labels recorded at 700 Hz via a standardized protocol:
  - `1` → Baseline (neutral, reading magazines)
  - `2` → Stress (Trier Social Stress Test: public speaking + mental arithmetic)
  - `3` → Amusement (funny video clips)
- Data files: one `.pkl` file per subject, available from the [WESAD website](https://archive.ics.uci.edu/dataset/465/wesad)

Place the dataset in a folder named `WESAD/` at the project root, with the structure:
```
WESAD/
  S2/S2.pkl
  S3/S3.pkl
  ...
  S17/S17.pkl
```

---

## Project Structure

```
.
├── notebook.ipynb       # Main analysis notebook (all sections)
├── functions.py         # Helper functions (loading, windowing, filtering, features)
├── WESAD/               # Dataset directory (not included in repo)
├── explain.md           # Full in-depth explanation of every method used
└── README.md            # This file
```

---

## Pipeline Summary

### 1. Data Loading
Raw `.pkl` files are loaded per subject. Only wrist signals are retained; chest-strap data is discarded.

### 2. Preprocessing & Windowing
Signals are segmented into **30-second sliding windows** with a **5-second shift**. Each window is labeled by majority vote from the 700 Hz label stream (≥80% label purity required). Quality control removes windows where mean skin temperature < 25°C (sensor not in contact).

Signal filters applied inside each window:
- **BVP:** 4th-order Butterworth bandpass filter (0.5–8.0 Hz) — isolates cardiac frequency band
- **EDA:** 4th-order Butterworth lowpass filter (≤1.0 Hz) — removes high-frequency noise while preserving skin conductance responses

All filters use zero-phase `filtfilt` to avoid temporal distortion.

### 3. Feature Extraction (107 features total)

| Signal | Features |
|--------|----------|
| ACC (x, y, z + magnitude) | 11 statistical + 3 frequency features × 4 = 56 |
| BVP | 11 statistical + 3 frequency = 14 |
| EDA | 11 statistical + 3 frequency + 4 EDA-specific = 18 |
| TEMP | 11 statistical = 11 |
| HRV (from BVP peaks) | MeanHR, SDNN, RMSSD, pNN50, n_beats, LF, HF, LF/HF = 8 |

Statistical features: mean, std, min, max, median, range, skewness, kurtosis, RMS, MAD, SAD  
Frequency features (Welch PSD): total power, dominant frequency, spectral entropy

### 4. Cross-Validation
Leave-One-Subject-Out (LOSO) — 15 folds, each training on 14 subjects and testing on 1. Per-fold imputation (median) and scaling (StandardScaler) are applied strictly to training data to prevent leakage.

### 5. Models
- Logistic Regression (`class_weight='balanced'`)
- SVM with RBF kernel (`C=10`, `gamma='scale'`)
- Gradient Boosting (`n_estimators=100`, `max_depth=5`, `lr=0.1`)
- MLP (`hidden_layers=(128, 64)`, early stopping)

### 6. Stacking Ensemble
A Logistic Regression meta-model is trained on the LOSO probability outputs of all four base models, also evaluated via LOSO.

### 7. Explainability
- Gradient Boosting feature importances
- SHAP TreeExplainer (for GB) and KernelExplainer (for stacking ensemble)
- Subject variability plots for top features

### 8. Advanced Extensions
- **RFE:** Recursive Feature Elimination selects 15 features per LOSO fold (leak-free)
- **Noise Robustness:** Gaussian noise added to test features at σ = 0–0.50
- **Subject Gap:** General LOSO model vs. personalized within-subject model
- **Conceptual LSTM:** TensorFlow/Keras architecture for future deep learning extension

---

## Results

### Binary Classification (Stress vs. Non-Stress)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.9035 | 0.8366 | 0.8426 | 0.8396 |
| SVM (RBF) | 0.8781 | 0.8238 | 0.7549 | 0.7878 |
| Gradient Boosting | 0.8950 | 0.8491 | 0.7900 | 0.8185 |
| **MLP** | **0.9092** | **0.8747** | **0.8137** | **0.8431** |
| Stacking Ensemble | 0.8989 | 0.8003 | 0.8829 | 0.8395 |

### 3-Class Classification (Macro-Averaged)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.6975 | 0.6467 |
| SVM (RBF) | 0.7464 | 0.6710 |
| Gradient Boosting | 0.7598 | 0.6459 |
| MLP | 0.7643 | 0.6736 |

The **Amusement** class is the primary source of error in 3-class mode (~0.34 F1). Both Stress and Amusement produce high sympathetic arousal; distinguishing them reliably requires respiration data (unavailable from the wrist device alone).

### RFE & Noise Results

| Experiment | F1-Score |
|---|---|
| GB with all 107 features | 0.8185 |
| GB with top 15 features (RFE, leak-free) | 0.8289 |
| GB at σ=0.10 noise | 0.8506 |
| GB at σ=0.50 noise | 0.7936 |

### Subject Gap

| Evaluation | F1-Score |
|---|---|
| General model (LOSO) | ~0.82 |
| Personalized model (within-subject 5-fold) | ~0.99 |

Note: the personalized result is inflated by window overlap leakage and should be interpreted with caution.

---

## Requirements

```
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
shap
tensorflow   # optional, for the conceptual LSTM cell only
```

Install with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn shap tensorflow
```

---

## Usage

1. Download the WESAD dataset and place it in the `WESAD/` directory.
2. Install dependencies.
3. Run the notebook from top to bottom:

```bash
jupyter notebook notebook.ipynb
```

The notebook is self-contained and follows a linear execution order. All helper functions are imported from `functions.py`, which must be in the same directory.

---

## Key Design Decisions

**Why wrist-only?** Chest-strap devices (like the Respiban used in WESAD) provide higher signal quality but are impractical for everyday use. Wrist-worn devices like the Empatica E4 are the form factor used in consumer smartwatches, making wrist-only classification more ecologically relevant.

**Why LOSO?** Standard K-fold cross-validation applied to sliding-window physiological data suffers from two problems: adjacent windows overlap by 83%, and within-subject windows are correlated by definition. LOSO is the only validation strategy that produces an unbiased estimate of cross-subject generalization.

**Why binary over 3-class?** The binary formulation (Stress vs. Non-Stress) is more actionable in practice and more reliably solvable with wrist sensors. The Amusement class is physiologically ambiguous without respiration data, making 3-class classification significantly harder.

**Why `class_weight='balanced'`?** The dataset contains ~2.3× more Non-Stress than Stress windows. Without correction, models are incentivized to predict Non-Stress indiscriminately. Balanced class weights rescale the loss function so each class contributes equally during training.

**Why impute inside the fold?** Fitting the imputer on the full dataset before cross-validation would allow test-set statistics (medians) to influence training. Fitting inside each fold ensures the imputation is learned only from training data.

---

## Limitations & Future Work

- **Wrist vs. chest modality gap:** Including respiration rate would dramatically improve 3-class accuracy, particularly for the Amusement class.
- **Homogeneous cohort:** All participants are healthy university students. Generalization to clinical populations, elderly users, or users with cardiovascular conditions is unvalidated.
- **Lab vs. real-world:** TPSS-induced stress differs from naturalistic chronic stress. Ecological validity requires field studies.
- **Window overlap leakage:** The "personalized model" evaluation is confounded by the 83% overlap between adjacent windows. Chronological block splits are needed for a fair personalized assessment.
- **Deep learning:** 1D-CNNs or LSTMs applied to raw signals could bypass hand-crafted features and achieve F1 > 0.90, as demonstrated in recent literature.
- **Domain adaptation:** Techniques like DANN could reduce inter-subject variance and narrow the Subject Gap without requiring per-user calibration.

---

## References

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*. ACM ICMI 2018.

Sarkar, P., & Etemad, A. (2020). *Self-supervised ECG Representation Learning for Emotion Recognition*. IEEE TAFFC.
