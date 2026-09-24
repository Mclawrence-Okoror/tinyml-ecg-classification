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
# LOAD ECG DATA
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


X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(y)
groups = np.asarray(groups)


print("\nDataset")
print("-------")
print("Shape:", X.shape)
print("Classes:", Counter(y))


# ============================================================
# SAME SOURCE-INDEPENDENT SPLIT
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
# BALANCE TRAINING DATA
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
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_balanced
).astype(np.float32)

X_test_scaled = scaler.transform(
    X_test
).astype(np.float32)


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
# EXTRACT FLOAT PARAMETERS
# ============================================================

W1 = mlp.coefs_[0].astype(np.float32)
b1 = mlp.intercepts_[0].astype(np.float32)

W2 = mlp.coefs_[1].astype(np.float32)
b2 = mlp.intercepts_[1].astype(np.float32)


# ============================================================
# FLOAT32 REFERENCE
# ============================================================

def float_forward(X):

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


float_probabilities = float_forward(
    X_test_scaled
)

float_predictions = (
    float_probabilities >= 0.5
).astype(np.int32)


# ============================================================
# METRICS
# ============================================================

def show_metrics(
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


show_metrics(
    "FLOAT32 REFERENCE",
    float_predictions
)


# ============================================================
# SYMMETRIC INT8 QUANTIZER
# ============================================================

def calculate_scale(values):

    maximum = np.max(
        np.abs(values)
    )

    if maximum == 0:
        return np.float32(1.0)

    return np.float32(
        maximum / 127.0
    )


def quantize_int8(
    values,
    scale
):

    q = np.round(
        values / scale
    )

    q = np.clip(
        q,
        -127,
        127
    )

    return q.astype(
        np.int8
    )


# ============================================================
# INPUT QUANTIZATION
#
# Scale is determined ONLY from training data.
# ============================================================

input_scale = calculate_scale(
    X_train_scaled
)

X_train_q = quantize_int8(
    X_train_scaled,
    input_scale
)

X_test_q = quantize_int8(
    X_test_scaled,
    input_scale
)


# ============================================================
# FIRST LAYER WEIGHT QUANTIZATION
# ============================================================

W1_scale = calculate_scale(
    W1
)

W1_q = quantize_int8(
    W1,
    W1_scale
)


# ============================================================
# FIRST LAYER BIAS
#
# Bias is represented in the same real-value scale as
# the accumulated W1/input product.
#
# accumulator scale =
#
# input_scale × W1_scale
#
# ============================================================

acc1_scale = (
    input_scale *
    W1_scale
)

b1_int32 = np.round(
    b1 / acc1_scale
).astype(
    np.int32
)


# ============================================================
# FIRST INTEGER LAYER
# ============================================================

def first_layer_integer(
    X_q
):

    # int8 × int8 -> int32
    accumulator = (
        X_q.astype(np.int32)
        @ W1_q.astype(np.int32)
    )

    # Add integer bias
    accumulator = (
        accumulator +
        b1_int32
    )

    # Convert accumulated integer back into
    # approximate real value ONLY for calibration.
    hidden_float = (
        accumulator.astype(np.float32)
        * acc1_scale
    )

    # ReLU
    hidden_float = np.maximum(
        hidden_float,
        0
    )

    return (
        accumulator,
        hidden_float
    )


train_acc1, train_hidden_float = (
    first_layer_integer(
        X_train_q
    )
)

test_acc1, test_hidden_float = (
    first_layer_integer(
        X_test_q
    )
)


# ============================================================
# HIDDEN ACTIVATION QUANTIZATION
#
# This scale is obtained ONLY from the training set.
# ============================================================

hidden_scale = calculate_scale(
    train_hidden_float
)

train_hidden_q = quantize_int8(
    train_hidden_float,
    hidden_scale
)

test_hidden_q = quantize_int8(
    test_hidden_float,
    hidden_scale
)


print("\nQuantization scales")
print("-------------------")

print(
    "Input scale :",
    input_scale
)

print(
    "W1 scale    :",
    W1_scale
)

print(
    "Hidden scale:",
    hidden_scale
)


# ============================================================
# SECOND LAYER
# ============================================================

W2_scale = calculate_scale(
    W2
)

W2_q = quantize_int8(
    W2,
    W2_scale
)


# Hidden × W2 accumulator scale
acc2_scale = (
    hidden_scale *
    W2_scale
)


# ============================================================
# OUTPUT BIAS
# ============================================================

b2_int32 = np.round(
    b2 / acc2_scale
).astype(
    np.int32
)


# ============================================================
# SECOND INTEGER LAYER
# ============================================================

def second_layer_integer(
    hidden_q
):

    accumulator = (
        hidden_q.astype(np.int32)
        @ W2_q.astype(np.int32)
    )

    accumulator = (
        accumulator +
        b2_int32
    )

    output_float = (
        accumulator.astype(np.float32)
        * acc2_scale
    )

    return output_float.ravel()


integer_output = (
    second_layer_integer(
        test_hidden_q
    )
)


# ============================================================
# SIGMOID
#
# The final sigmoid is not the important MCU operation here.
# For binary classification, sigmoid >= 0.5 is equivalent
# to the raw output/logit being >= 0.
# ============================================================

integer_predictions = (
    integer_output >= 0
).astype(
    np.int32
)


show_metrics(
    "INTEGER MLP",
    integer_predictions
)


# ============================================================
# COMPARE FLOAT VS INTEGER
# ============================================================

prediction_disagreement = np.mean(
    float_predictions !=
    integer_predictions
)

# Compare logits rather than probabilities
float_logits = (
    np.log(
        np.clip(
            float_probabilities,
            1e-7,
            1 - 1e-7
        )
        /
        np.clip(
            1 - float_probabilities,
            1e-7,
            1
        )
    )
)

logit_difference = np.mean(
    np.abs(
        float_logits -
        integer_output
    )
)


print("\n================================")
print("FLOAT VS INTEGER")
print("================================")

print(
    "Prediction disagreement:",
    prediction_disagreement * 100,
    "%"
)

print(
    "Mean logit difference:",
    logit_difference
)


# ============================================================
# INTEGER RANGE CHECKS
# ============================================================

print("\n================================")
print("INTEGER RANGE CHECK")
print("================================")

print(
    "Layer 1 accumulator min:",
    np.min(train_acc1)
)

print(
    "Layer 1 accumulator max:",
    np.max(train_acc1)
)

print(
    "Layer 2 accumulator min:",
    np.min(
        second_layer_integer(
            train_hidden_q
        )
    )
)

print(
    "Layer 2 accumulator max:",
    np.max(
        second_layer_integer(
            train_hidden_q
        )
    )
)


# ============================================================
# STORAGE
# ============================================================

parameter_count = (
    W1.size +
    W2.size
)

# INT8 weights
weight_bytes = (
    W1.size +
    W2.size
)

# INT32 biases
bias_bytes = (
    b1_int32.size +
    b2_int32.size
) * 4

# Scaler
scaler_bytes = (
    scaler.mean_.size +
    scaler.scale_.size
) * 4

# Quantization scales:
#
# input
# W1
# hidden
# W2
#
# Four float32 values.
#
scale_bytes = 4 * 4


integer_model_bytes = (
    weight_bytes +
    bias_bytes +
    scaler_bytes +
    scale_bytes
)


print("\n================================")
print("INTEGER MODEL STORAGE")
print("================================")

print(
    "INT8 weights:",
    weight_bytes,
    "bytes"
)

print(
    "INT32 biases:",
    bias_bytes,
    "bytes"
)

print(
    "Float32 scaler:",
    scaler_bytes,
    "bytes"
)

print(
    "Quantization scales:",
    scale_bytes,
    "bytes"
)

print(
    "Total:",
    integer_model_bytes,
    "bytes"
)


# ============================================================
# SAVE PARAMETERS FOR C EXPORT
# ============================================================

np.savez(
    "mlp8_integer_model.npz",

    W1_q=W1_q,
    W2_q=W2_q,

    b1_int32=b1_int32,
    b2_int32=b2_int32,

    scaler_mean=scaler.mean_.astype(
        np.float32
    ),

    scaler_scale=scaler.scale_.astype(
        np.float32
    ),

    input_scale=np.array(
        input_scale,
        dtype=np.float32
    ),

    W1_scale=np.array(
        W1_scale,
        dtype=np.float32
    ),

    hidden_scale=np.array(
        hidden_scale,
        dtype=np.float32
    ),

    W2_scale=np.array(
        W2_scale,
        dtype=np.float32
    )
)


print("\nSaved:")
print(
    "mlp8_integer_model.npz"
)