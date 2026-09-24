import os
import numpy as np
import wfdb

from collections import Counter
from scipy.signal import resample_poly

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "mit-bih-arrhythmia-database-1.0.0"

BEFORE = 90
AFTER = 90

RANDOM_STATE = 42


# ============================================================
# RECORDS
# ============================================================

RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119",
    "121","122","123","124",
    "200","201","202","203","205","207","208","209","210",
    "212","213","214","215","217","219","220","221","222",
    "223","228","230","231","232","233","234"
]


# ============================================================
# LOAD DATA
# ============================================================

X = []
y = []
groups = []


for record_name in RECORDS:

    record_path = os.path.join(
        DATASET_PATH,
        record_name
    )

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(
        record_path,
        "atr"
    )

    signal = record.p_signal[:, 0]

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol
    ):

        if symbol not in ["N", "A", "V"]:
            continue

        start = sample - BEFORE
        end = sample + AFTER

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]

        if len(beat) != 180:
            continue

        beat_90 = resample_poly(
            beat,
            up=1,
            down=2
        )

        if len(beat_90) != 90:
            continue

        X.append(beat_90)

        y.append(
            0 if symbol == "N" else 1
        )

        if record_name in ["201", "202"]:
            groups.append("201_202")
        else:
            groups.append(record_name)


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)

groups = np.array(groups)


print("\nDataset")
print("-------")
print("Shape:", X.shape)
print("Classes:", Counter(y))


# ============================================================
# SOURCE-INDEPENDENT SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups
    )
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]


# ============================================================
# BALANCE TRAINING SET
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

normal_idx = np.where(
    y_train == 0
)[0]

abnormal_idx = np.where(
    y_train == 1
)[0]

n = min(
    len(normal_idx),
    len(abnormal_idx)
)

normal_idx = rng.choice(
    normal_idx,
    size=n,
    replace=False
)

abnormal_idx = rng.choice(
    abnormal_idx,
    size=n,
    replace=False
)

balanced_idx = np.concatenate([
    normal_idx,
    abnormal_idx
])

rng.shuffle(
    balanced_idx
)

X_train_balanced = X_train[
    balanced_idx
]

y_train_balanced = y_train[
    balanced_idx
]


print("\nBalanced training")
print("------------------")
print(
    "Normal:",
    np.sum(y_train_balanced == 0)
)

print(
    "Abnormal:",
    np.sum(y_train_balanced == 1)
)


# ============================================================
# SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_balanced
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# TRAIN MLP-8
# ============================================================

mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    random_state=RANDOM_STATE
)

mlp.fit(
    X_train_scaled,
    y_train_balanced
)


# ============================================================
# SKLEARN REFERENCE
# ============================================================

sklearn_probabilities = mlp.predict_proba(
    X_test_scaled
)[:, 1]

sklearn_predictions = (
    sklearn_probabilities >= 0.5
).astype(int)


# ============================================================
# EXTRACT PARAMETERS
# ============================================================

W1 = mlp.coefs_[0].astype(
    np.float32
)

b1 = mlp.intercepts_[0].astype(
    np.float32
)

W2 = mlp.coefs_[1].astype(
    np.float32
)

b2 = mlp.intercepts_[1].astype(
    np.float32
)


# ============================================================
# MANUAL FORWARD PASS
# ============================================================

def manual_forward(
    X,
    W1,
    b1,
    W2,
    b2
):

    hidden = X @ W1 + b1

    hidden = np.maximum(
        hidden,
        0
    )

    output = (
        hidden @ W2
    ) + b2

    probability = (
        1.0 /
        (
            1.0 +
            np.exp(-output)
        )
    )

    return probability.ravel()


manual_float_probabilities = manual_forward(
    X_test_scaled.astype(np.float32),
    W1,
    b1,
    W2,
    b2
)

manual_float_predictions = (
    manual_float_probabilities >= 0.5
).astype(int)


# ============================================================
# COMPARE SKLEARN VS MANUAL FLOAT
# ============================================================

print("\n================================")
print("SKLEARN VS MANUAL FLOAT")
print("================================")

probability_difference = np.mean(
    np.abs(
        sklearn_probabilities -
        manual_float_probabilities
    )
)

prediction_disagreement = np.mean(
    sklearn_predictions !=
    manual_float_predictions
)


print(
    "Mean probability difference:",
    probability_difference
)

print(
    "Prediction disagreement:",
    prediction_disagreement * 100,
    "%"
)


# ============================================================
# METRICS
# ============================================================

def metrics(
    name,
    predictions
):

    print("\n" + name)
    print("-" * len(name))

    print(
        "Accuracy : {:.2f}%".format(
            accuracy_score(
                y_test,
                predictions
            ) * 100
        )
    )

    print(
        "Precision: {:.2f}%".format(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ) * 100
        )
    )

    print(
        "Recall   : {:.2f}%".format(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ) * 100
        )
    )

    print(
        "F1       : {:.2f}%".format(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ) * 100
        )
    )


metrics(
    "SKLEARN FLOAT32",
    sklearn_predictions
)

metrics(
    "MANUAL FLOAT32",
    manual_float_predictions
)


# ============================================================
# INT8 QUANTIZATION
# ============================================================

def quantize_int8(
    values
):

    max_abs = np.max(
        np.abs(values)
    )

    if max_abs == 0:
        scale = 1.0
    else:
        scale = (
            max_abs / 127.0
        )

    quantized = np.round(
        values / scale
    ).astype(np.int8)

    return (
        quantized,
        scale
    )


W1_q, W1_scale = quantize_int8(
    W1
)

b1_q, b1_scale = quantize_int8(
    b1
)

W2_q, W2_scale = quantize_int8(
    W2
)

b2_q, b2_scale = quantize_int8(
    b2
)


# ============================================================
# DEQUANTIZE
# ============================================================

W1_dq = (
    W1_q.astype(np.float32)
    * W1_scale
)

b1_dq = (
    b1_q.astype(np.float32)
    * b1_scale
)

W2_dq = (
    W2_q.astype(np.float32)
    * W2_scale
)

b2_dq = (
    b2_q.astype(np.float32)
    * b2_scale
)


# ============================================================
# QUANTIZED WEIGHT MODEL
# ============================================================

quantized_probabilities = manual_forward(
    X_test_scaled.astype(np.float32),
    W1_dq,
    b1_dq,
    W2_dq,
    b2_dq
)

quantized_predictions = (
    quantized_probabilities >= 0.5
).astype(int)


metrics(
    "INT8 WEIGHTS + FLOAT32 INPUT",
    quantized_predictions
)


# ============================================================
# QUANTIZED VS MANUAL FLOAT
# ============================================================

print("\n================================")
print("FLOAT VS INT8 WEIGHTS")
print("================================")

probability_difference = np.mean(
    np.abs(
        manual_float_probabilities -
        quantized_probabilities
    )
)

prediction_disagreement = np.mean(
    manual_float_predictions !=
    quantized_predictions
)


print(
    "Mean probability difference:",
    probability_difference
)

print(
    "Prediction disagreement:",
    prediction_disagreement * 100,
    "%"
)


# ============================================================
# PARAMETER STORAGE
# ============================================================

parameter_count = (
    W1.size +
    b1.size +
    W2.size +
    b2.size
)

float32_network_bytes = (
    parameter_count * 4
)

int8_network_bytes = (
    parameter_count
)

scaler_bytes = (
    scaler.mean_.size +
    scaler.scale_.size
) * 4


print("\n================================")
print("RAW STORAGE")
print("================================")

print(
    "Parameters:",
    parameter_count
)

print(
    "Float32 network:",
    float32_network_bytes,
    "bytes"
)

print(
    "Int8 network:",
    int8_network_bytes,
    "bytes"
)

print(
    "Scaler:",
    scaler_bytes,
    "bytes"
)

print(
    "Float32 total:",
    float32_network_bytes +
    scaler_bytes,
    "bytes"
)

print(
    "Int8 + float32 scaler:",
    int8_network_bytes +
    scaler_bytes,
    "bytes"
)


# ============================================================
# SAVE PARAMETERS
# ============================================================

np.savez(
    "mlp8_int8_weights.npz",

    W1=W1_q,
    b1=b1_q,

    W2=W2_q,
    b2=b2_q,

    W1_scale=np.array(
        W1_scale,
        dtype=np.float32
    ),

    b1_scale=np.array(
        b1_scale,
        dtype=np.float32
    ),

    W2_scale=np.array(
        W2_scale,
        dtype=np.float32
    ),

    b2_scale=np.array(
        b2_scale,
        dtype=np.float32
    ),

    scaler_mean=scaler.mean_.astype(
        np.float32
    ),

    scaler_scale=scaler.scale_.astype(
        np.float32
    )
)


print("\nSaved:")
print(
    "mlp8_int8_weights.npz"
)