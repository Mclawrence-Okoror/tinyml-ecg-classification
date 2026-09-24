# TinyML-Based ECG Abnormality Detection for Resource-Constrained Devices

A lightweight neural-network approach for binary ECG beat classification, with a focus on **model compression, integer inference, and embedded deployment** on resource-constrained systems.

This project investigates how a conventional ECG machine-learning workflow can be reduced to a compact neural network and translated into an integer-based implementation suitable for embedded inference.

> **Task:** classify ECG beats as **Normal (N)** or **Selected Abnormal (A/V)**
> **Dataset:** MIT-BIH Arrhythmia Database
> **Final model:** 90 → 8 → 1 MLP
> **Quantization:** INT8 weights and activations, INT32 accumulators
> **Embedded target:** ESP32-S3 / Wokwi

---

## Overview

Machine-learning models used for biomedical signal analysis can be difficult to deploy on small embedded devices because memory, computation, and storage are limited.

This project explores a different question:

**How small can an ECG classification model become while retaining useful classification performance and remaining practical to implement with integer arithmetic on a microcontroller?**

The work follows the complete path from dataset-level experimentation to embedded implementation:

```text
MIT-BIH Arrhythmia Database
          ↓
Annotated ECG beat extraction
          ↓
Source-aware train/test split
          ↓
Input representation experiments
          ↓
Model comparison and reduction
          ↓
90 → 8 → 1 MLP
          ↓
INT8 quantization
          ↓
Integer-style inference
          ↓
C implementation
          ↓
ESP32-S3 / Wokwi validation
```

The project is an **engineering/research prototype**, not a clinical diagnostic system.

---

## Dataset

The experiments use the **MIT-BIH Arrhythmia Database**.

The selected records contain annotated ECG beats sampled at **360 Hz**. Beat segments were extracted around annotated beats using:

* 90 samples before the annotation
* 90 samples after the annotation

This produced an original representation of **180 samples per beat**.

Across the selected records:

| Category                |  Beats |
| ----------------------- | -----: |
| Total                   | 84,708 |
| Normal (N)              | 75,033 |
| Selected abnormal (A/V) |  9,675 |

For the final compact representation, each beat was represented using **90 raw ECG samples**.

The classification task was intentionally simplified to:

```text
Normal (N)
      vs.
Selected Abnormal (A/V)
```

This should not be interpreted as classification of every possible cardiac arrhythmia.

---

## Experimental Methodology

The project was developed through several stages rather than selecting a neural-network architecture arbitrarily.

### 1. Baseline models

Initial experiments compared conventional approaches including:

* Logistic Regression
* Random Forest

The Random Forest baseline used 100 trees.

### 2. Input reduction

Different ECG input lengths were evaluated:

```text
180 → 90 → 60 → 45 → 30 samples
```

Reducing the representation from 180 to 90 samples reduced the raw input storage requirement by **50%** while maintaining comparable classification performance.

This motivated the use of the 90-sample representation for subsequent compact-model experiments.

### 3. Noise robustness

Training and evaluation experiments were performed with different levels of synthetic Gaussian noise.

The purpose was to investigate whether the compact classifier retained useful behaviour when the input signal was perturbed.

The noise experiments are intended as a controlled robustness experiment rather than a complete model of real ECG acquisition noise.

### 4. Model complexity reduction

Random Forest complexity was varied by changing the number of trees:

```text
100 → 50 → 25 → 10 → 5
```

Smaller neural-network architectures were also evaluated:

```text
90 → 16 → 8 → 1
90 → 8 → 4 → 1
90 → 8 → 1
90 → 4 → 1
90 → 2 → 1
```

The 90 → 8 → 1 architecture was selected for the final embedded implementation based on the experimental results and its compact computational footprint.

---

## Final Model

The selected model is a small multilayer perceptron:

```text
Input
90 ECG samples
     │
     ▼
┌─────────────┐
│  Dense (8)  │
│    ReLU     │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Dense (1)   │
└─────────────┘
     │
     ▼
Binary output
```

### Model characteristics

| Property                    |       Value |
| --------------------------- | ----------: |
| Architecture                |  90 → 8 → 1 |
| Trainable parameters        |         737 |
| MACs / inference            |         728 |
| Floating-point network size | 2,948 bytes |
| INT8 weight storage         |   728 bytes |

On the held-out test set, the floating-point model achieved:

| Metric             | Result |
| ------------------ | -----: |
| Accuracy           | 86.27% |
| Abnormal precision | 40.12% |
| Abnormal recall    | 82.29% |
| Abnormal F1        | 53.94% |

Because the abnormal class is substantially smaller than the normal class, accuracy alone does not adequately describe the classifier's behaviour. Precision, recall, and F1 are therefore reported alongside accuracy.

---

## Quantization

The final model was converted from floating-point inference to an integer-oriented implementation.

The implemented pipeline uses:

* INT8 input
* INT8 weights
* INT32 accumulators
* INT32 biases
* INT8 hidden activation
* ReLU
* Integer-style scaling between stages

Conceptually:

```text
Standardized ECG
       ↓
     INT8
       ↓
INT8 × INT8
       ↓
   INT32 accumulation
       ↓
      Rescale
       ↓
      ReLU
       ↓
     INT8 hidden
       ↓
INT8 × INT8
       ↓
   INT32 accumulation
       ↓
  Binary decision
```

The integer-style implementation achieved:

| Metric    |  Float | Integer |
| --------- | -----: | ------: |
| Accuracy  | 86.27% |  86.28% |
| Precision | 40.12% |  40.14% |
| Recall    | 82.29% |  82.37% |
| F1        | 53.94% |  53.98% |

The difference from the floating-point reference was therefore very small.

### Embedded resource profile

The reported model constants require approximately:

**1,500 bytes (~1.46 KiB)**

consisting of:

* INT8 weights: 728 bytes
* INT32 biases: 36 bytes
* Input scaler: 720 bytes
* Quantization scales: 16 bytes

The reported inference working memory is approximately:

**130 bytes**

consisting of:

* INT8 input buffer: 90 bytes
* INT8 hidden activation: 8 bytes
* accumulator/working storage: 32 bytes

The 1,500-byte figure describes the reported model constants and does **not** represent the total firmware flash footprint.

---

## Python → C Verification

Before embedded deployment, the integer model was exported into C.

The generated C implementation was tested against the Python reference using:

**27,054 test vectors**

Results:

```text
Test vectors:       27,054
Matches:            27,054
Mismatches:         0
Agreement:          100.00%
```

This verifies that the C implementation reproduced the Python integer inference behaviour for the complete test-vector set.

This result represents **implementation agreement**, not 100% ECG classification accuracy.

---

## ESP32-S3 / Wokwi Validation

The integer inference implementation was subsequently integrated for an **ESP32-S3** target and evaluated in **Wokwi**.

A selected set of 100 test vectors was used for embedded prediction verification.

```text
Vectors tested:     100
Matches:            100
Mismatches:         0
Agreement:          100%
```

The embedded validation demonstrates consistency between the reference implementation and the ESP32-S3 implementation.

The ESP32-S3 experiments were performed in **Wokwi simulation**, not on physical hardware. Therefore, the reported simulation timing should not be interpreted as a measurement of physical ESP32-S3 execution time.

---

## Key Findings

The experiments produced several observations relevant to TinyML deployment:

1. **Input reduction can substantially reduce data storage without necessarily causing a proportional performance loss.**
   Reducing the ECG representation from 180 to 90 samples halved the input storage requirement while maintaining comparable performance.

2. **Model architecture has a large effect on embedded resource requirements.**
   The final 90 → 8 → 1 MLP contains only 737 trainable parameters and requires 728 MACs per inference.

3. **Integer quantization preserved the behaviour of the selected model closely.**
   The floating-point and integer implementations produced nearly identical test-set metrics.

4. **The integer implementation was reproducible across software layers.**
   Python and C produced identical predictions for all 27,054 verification vectors.

5. **The compact implementation can be translated to an MCU-oriented inference pipeline.**
   The ESP32-S3/Wokwi implementation reproduced the selected reference predictions without mismatches.

---

## Repository Structure

```text
tinyml-ecg-classification/
│
├── README.md
│
├── src/
│   ├── annotations.py
│   ├── check_patients.py
│   ├── count_labels.py
│   ├── extract_beats.py
│   ├── ecg_baseline.py
│   ├── ecg_input_reduction.py
│   ├── ecg_noise_robustness.py
│   ├── ecg_noise_clean_train.py
│   ├── ecg_experiment4.py
│   ├── ecg_experiment5.py
│   ├── ecg_experiment6.py
│   ├── ecg_experiment7.py
│   ├── ecg_experiment7b.py
│   ├── ecg_experiment8a.py
│   ├── export_mlp8_to_c.py
│   ├── export_verified_beat.py
│   └── make_test_vectors.py
│
├── models/
│   ├── mlp8_int8_weights.npz
│   ├── mlp8_integer_model.npz
│   └── mlp8_quantized.npz
│
├── embedded/
│   ├── mlp8_inference.c
│   ├── mlp8_model.h
│   ├── mlp8_benchmark.c
│   ├── mlp8_timing.c
│   └── mlp8_esp32_test.h
│
├── test_vectors/
│   └── verified_test_beat.h
│
├── results/
│   └── logistic_confusion_matrix.png
│
└── docs/
```

The MIT-BIH dataset itself is **not included** in this repository.

---

## Reproducibility

The repository contains the source code used for the major experiments, exported model artifacts, embedded implementation, and verification resources.

The complete 27,054-vector C test header is intentionally not stored in the repository because of its size. The vector-generation script is included so that the verification data can be regenerated from the appropriate dataset and preprocessing pipeline.

The dataset should be obtained separately from its official PhysioNet distribution.

---

## Limitations

Several limitations should be considered when interpreting the results:

* The task is binary classification of **Normal (N) versus selected Abnormal (A/V) annotations**, rather than multi-class arrhythmia classification.
* The MIT-BIH Arrhythmia Database is a public benchmark dataset and does not establish clinical generalization to unseen populations.
* The same held-out test set was reused across several experiments, so it should not be interpreted as a pristine final holdout for every model-selection decision.
* The train/test split was source-aware but was not designed as a full patient-stratified clinical validation protocol.
* Synthetic Gaussian noise was used as a controlled robustness experiment and does not represent all real-world ECG acquisition artifacts.
* The final ESP32-S3 evaluation was performed in Wokwi simulation rather than on physical hardware.
* The reported 1,500-byte model footprint excludes firmware and other program overhead.
* The 100-vector embedded validation demonstrates implementation agreement, not classifier accuracy or clinical validity.
* No clinical validation was performed.

---

## Project Status

**Completed research prototype**

The current implementation covers:

* [x] ECG beat extraction
* [x] Source-aware dataset splitting
* [x] Input representation experiments
* [x] Baseline model comparison
* [x] Noise robustness experiments
* [x] Model complexity reduction
* [x] Compact MLP selection
* [x] INT8 quantization
* [x] Integer-style inference
* [x] Python/C verification
* [x] ESP32-S3/Wokwi implementation
* [x] Embedded prediction verification

The accompanying research paper is being prepared separately.

---

## Author

**Mclawrence Okoror**

Computer Engineering graduate working at the intersection of:

* Embedded AI
* Edge AI
* TinyML
* Resource-constrained machine learning
* Intelligent embedded systems
* AI–hardware co-design

GitHub: [Mclawrence-Okoror](https://github.com/Mclawrence-Okoror)

---

## Citation

If you use this implementation or build upon the experimental methodology, please cite the associated research work once the paper is published.

Repository:

**https://github.com/Mclawrence-Okoror/tinyml-ecg-classification**
