import numpy as np
import wfdb
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path


DATASET_PATH = "mit-bih-arrhythmia-database-1.0.0"

BEFORE = 90
AFTER = 90


RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234"
]


# ============================================================
# Load exact integer model
# ============================================================

model = np.load("mlp8_integer_model.npz")

W1 = model["W1_q"].astype(np.int8)
b1 = model["b1_int32"].astype(np.int32)

W2 = model["W2_q"].astype(np.int8).reshape(-1)
b2 = model["b2_int32"].astype(np.int32)

scaler_mean = model["scaler_mean"].astype(np.float32)
scaler_scale = model["scaler_scale"].astype(np.float32)

input_scale = float(model["input_scale"])
W1_scale = float(model["W1_scale"])
hidden_scale = float(model["hidden_scale"])
W2_scale = float(model["W2_scale"])

acc1_scale = input_scale * W1_scale
acc2_scale = hidden_scale * W2_scale


# ============================================================
# Integer reference model
# ============================================================

def quantize_int8(x, scale):

    q = np.rint(x / scale)

    q = np.clip(
        q,
        -127,
        127
    )

    return q.astype(np.int8)


def integer_predict(x):

    # Standardize
    standardized = (
        x - scaler_mean
    ) / scaler_scale


    # Input quantization
    x_q = quantize_int8(
        standardized,
        input_scale
    )


    # Layer 1
    acc1 = (
        x_q.astype(np.int32)
        @
        W1.astype(np.int32)
    )

    acc1 += b1


    # Convert accumulator to real-valued activation
    hidden_real = (
        acc1.astype(np.float32)
        * acc1_scale
    )


    # ReLU
    hidden_real = np.maximum(
        hidden_real,
        0.0
    )


    # Hidden quantization
    hidden_q = quantize_int8(
        hidden_real,
        hidden_scale
    )


    # Layer 2
    acc2 = int(
        hidden_q.astype(np.int32)
        @
        W2.astype(np.int32)
    )

    acc2 += int(b2[0])


    # Binary decision
    return 1 if acc2 >= 0 else 0


# ============================================================
# Rebuild exact dataset
# ============================================================

X = []
y = []
groups = []

print("Loading ECG records...")


for record_id in RECORDS:

    record_path = str(
        Path(DATASET_PATH) / record_id
    )

    record = wfdb.rdrecord(
        record_path
    )

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


        # 180 -> 90
        beat_90 = (
            beat
            .reshape(90, 2)
            .mean(axis=1)
        )


        X.append(
            beat_90.astype(np.float32)
        )


        y.append(
            0 if symbol == "N" else 1
        )


        # Prevent 201/202 source leakage
        if record_id in ["201", "202"]:
            groups.append("201_202")
        else:
            groups.append(record_id)


X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int8
)

groups = np.asarray(
    groups
)


print()
print("Full dataset:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# Recreate exact source-independent split
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)


train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_test = X[test_idx]
y_test = y[test_idx]


print()
print("Test set:", X_test.shape)

print(
    "Normal:",
    np.sum(y_test == 0)
)

print(
    "Abnormal:",
    np.sum(y_test == 1)
)


# ============================================================
# Generate predictions for ALL 27,054 test beats
# ============================================================

print()
print("Generating Python reference predictions...")


predictions = []


for i, beat in enumerate(X_test):

    predictions.append(
        integer_predict(beat)
    )

    if (i + 1) % 5000 == 0:
        print(
            f"Processed {i + 1} / {len(X_test)}"
        )


predictions = np.asarray(
    predictions,
    dtype=np.int8
)


print()
print("Generated predictions for:")
print(
    len(predictions),
    "test beats"
)

print(
    "True abnormal:",
    np.sum(y_test == 1)
)

print(
    "Predicted abnormal:",
    np.sum(predictions == 1)
)


# ============================================================
# C float formatting
# ============================================================

def c_float(value):

    s = f"{float(value):.9g}"

    if (
        "." not in s
        and "e" not in s
        and "E" not in s
    ):
        s += ".0"

    return s + "f"


# ============================================================
# Write C header
# ============================================================

header = """\
#ifndef MLP8_TEST_VECTORS_H
#define MLP8_TEST_VECTORS_H

#include <stdint.h>

#define NUM_TEST_VECTORS 27054
#define TEST_INPUT_SIZE 90

"""


# ============================================================
# ECG test inputs
# ============================================================

header += (
    "static const float "
    "test_inputs[NUM_TEST_VECTORS][TEST_INPUT_SIZE] = {\n"
)


print()
print("Writing ECG test vectors...")


for index, beat in enumerate(X_test):

    values = ", ".join(
        c_float(v)
        for v in beat
    )

    header += (
        f"    {{{values}}},\n"
    )

    if (index + 1) % 5000 == 0:
        print(
            f"Wrote {index + 1} / {len(X_test)}"
        )


header += "};\n\n"


# ============================================================
# True labels
# ============================================================

header += (
    "static const int8_t "
    "test_labels[NUM_TEST_VECTORS] = {\n"
)


for label in y_test:

    header += (
        f"    {int(label)},\n"
    )


header += "};\n\n"


# ============================================================
# Python reference predictions
# ============================================================

header += (
    "static const int8_t "
    "expected_predictions[NUM_TEST_VECTORS] = {\n"
)


for pred in predictions:

    header += (
        f"    {int(pred)},\n"
    )


header += "};\n\n"


# ============================================================
# End header
# ============================================================

header += "#endif\n"


Path(
    "mlp8_test_vectors.h"
).write_text(
    header
)


print()
print("Created: mlp8_test_vectors.h")
print("Test vectors:", len(X_test))

# ============================================================
# Save first test beat for ESP32 verification
# ============================================================

with open("verified_test_beat.h", "w") as f:

    f.write("#ifndef VERIFIED_TEST_BEAT_H\n")
    f.write("#define VERIFIED_TEST_BEAT_H\n\n")

    f.write("static const float verified_test_beat[90] = {\n")

    for i, value in enumerate(X_test[0]):
        f.write(f"    {c_float(value)}")
        if i < 89:
            f.write(",")
        f.write("\n")

    f.write("};\n\n")

    f.write(
        f"static const int verified_true_label = "
        f"{int(y_test[0])};\n"
    )

    f.write(
        f"static const int verified_python_prediction = "
        f"{int(predictions[0])};\n"
    )

    f.write("\n#endif\n")

print("Created: verified_test_beat.h")
print("True label:", int(y_test[0]))
print("Python prediction:", int(predictions[0]))

# ============================================================
# Create small ESP32 validation set
# ============================================================

NUM_ESP32_TESTS = 100

with open("mlp8_esp32_test.h", "w") as f:

    f.write("#ifndef MLP8_ESP32_TEST_H\n")
    f.write("#define MLP8_ESP32_TEST_H\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write(f"#define ESP32_TEST_COUNT {NUM_ESP32_TESTS}\n")
    f.write("#define ESP32_INPUT_SIZE 90\n\n")

    f.write(
        "static const float "
        "esp32_test_inputs[ESP32_TEST_COUNT][ESP32_INPUT_SIZE] = {\n"
    )

    for beat in X_test[:NUM_ESP32_TESTS]:

        values = ", ".join(
            c_float(v)
            for v in beat
        )

        f.write(f"    {{{values}}},\n")

    f.write("};\n\n")

    f.write(
        "static const int8_t "
        "esp32_test_labels[ESP32_TEST_COUNT] = {\n"
    )

    for label in y_test[:NUM_ESP32_TESTS]:
        f.write(f"    {int(label)},\n")

    f.write("};\n\n")

    f.write(
        "static const int8_t "
        "esp32_expected_predictions[ESP32_TEST_COUNT] = {\n"
    )

    for pred in predictions[:NUM_ESP32_TESTS]:
        f.write(f"    {int(pred)},\n")

    f.write("};\n\n")

    f.write("#endif\n")

print("Created: mlp8_esp32_test.h")
print("ESP32 test vectors:", NUM_ESP32_TESTS)