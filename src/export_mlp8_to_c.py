import numpy as np
from pathlib import Path

MODEL_FILE = "mlp8_integer_model.npz"
OUTPUT_HEADER = "mlp8_model.h"

model = np.load(MODEL_FILE)

print("Model contents:")
for key in model.files:
    arr = model[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def c_int8_array(name, arr):
    arr = np.asarray(arr, dtype=np.int8).flatten()

    values = ", ".join(str(int(x)) for x in arr)

    return f"""
static const int8_t {name}[{len(arr)}] = {{
    {values}
}};
"""

def c_int32_array(name, arr):
    arr = np.asarray(arr, dtype=np.int32).flatten()

    values = ", ".join(str(int(x)) for x in arr)

    return f"""
static const int32_t {name}[{len(arr)}] = {{
    {values}
}};
"""

def c_float_array(name, arr):
    arr = np.asarray(arr, dtype=np.float32).flatten()

    values = ", ".join(f"{float(x):.9g}f" for x in arr)

    return f"""
static const float {name}[{len(arr)}] = {{
    {values}
}};
"""

# ------------------------------------------------------------
# Load exact model arrays
# ------------------------------------------------------------

W1 = model["W1_q"]
b1 = model["b1_int32"]

W2 = model["W2_q"]
b2 = model["b2_int32"]

scaler_mean = model["scaler_mean"]
scaler_scale = model["scaler_scale"]

input_scale = float(model["input_scale"])
W1_scale = float(model["W1_scale"])
hidden_scale = float(model["hidden_scale"])
W2_scale = float(model["W2_scale"])

acc1_scale = input_scale * W1_scale
acc2_scale = hidden_scale * W2_scale

# ------------------------------------------------------------
# Build header
# ------------------------------------------------------------

header = """\
#ifndef MLP8_MODEL_H
#define MLP8_MODEL_H

#include <stdint.h>

#define MLP_INPUT_SIZE 90
#define MLP_HIDDEN_SIZE 8

"""

header += c_int8_array("W1_q", W1)
header += c_int32_array("b1_int32", b1)
header += c_int8_array("W2_q", W2)
header += c_int32_array("b2_int32", b2)

header += c_float_array("scaler_mean", scaler_mean)
header += c_float_array("scaler_scale", scaler_scale)

header += f"""
static const float input_scale = {input_scale:.9g}f;
static const float W1_scale = {W1_scale:.9g}f;
static const float hidden_scale = {hidden_scale:.9g}f;
static const float W2_scale = {W2_scale:.9g}f;

static const float acc1_scale = {acc1_scale:.9g}f;
static const float acc2_scale = {acc2_scale:.9g}f;

#endif
"""

Path(OUTPUT_HEADER).write_text(header)

print()
print(f"Created: {OUTPUT_HEADER}")
print(f"W1: {W1.shape}")
print(f"b1: {b1.shape}")
print(f"W2: {W2.shape}")
print(f"b2: {b2.shape}")
print(f"Input scale: {input_scale}")
print(f"Hidden scale: {hidden_scale}")
print(f"Accumulator 1 scale: {acc1_scale}")
print(f"Accumulator 2 scale: {acc2_scale}")