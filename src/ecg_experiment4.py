import os
import joblib
import numpy as np
import wfdb

from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
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

# We are using the 90-sample representation from Experiment 2
TARGET_SAMPLES = 90

TREE_COUNTS = [100, 50, 25, 10, 5]

RANDOM_STATE = 42


RECORDS = [
    "100", "101", "102", "103", "104", "105",
    "106", "107", "108", "109", "111", "112",
    "113", "114", "115", "116", "117", "118",
    "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234"
]


# ============================================================
# STEP 1 — LOAD ECG BEATS
# ============================================================

X_list = []
y_list = []
group_list = []

print("Loading ECG records...")

for record_id in RECORDS:

    record_path = os.path.join(DATASET_PATH, record_id)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

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

        if len(beat) != BEFORE + AFTER:
            continue

        X_list.append(beat)
        y_list.append(0 if symbol == "N" else 1)

        # Prevent source leakage between 201 and 202
        if record_id in ["201", "202"]:
            group_list.append("201_202")
        else:
            group_list.append(record_id)


X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)
groups = np.array(group_list)


print()
print("Full dataset:")
print("X shape:", X.shape)
print("Normal:", np.sum(y == 0))
print("Abnormal:", np.sum(y == 1))


# ============================================================
# STEP 2 — SOURCE-INDEPENDENT TRAIN/TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train_full = X[train_idx]
y_train_full = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]


print()
print("Before balancing:")
print("Training:", len(X_train_full))
print("Testing:", len(X_test))


# ============================================================
# STEP 3 — BALANCE TRAINING DATA
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

normal_indices = np.where(y_train_full == 0)[0]
abnormal_indices = np.where(y_train_full == 1)[0]

n_each = min(
    len(normal_indices),
    len(abnormal_indices)
)

selected_normal = rng.choice(
    normal_indices,
    size=n_each,
    replace=False
)

selected_abnormal = rng.choice(
    abnormal_indices,
    size=n_each,
    replace=False
)

balanced_indices = np.concatenate([
    selected_normal,
    selected_abnormal
])

# Shuffle balanced training set
rng.shuffle(balanced_indices)

X_train = X_train_full[balanced_indices]
y_train = y_train_full[balanced_indices]


print()
print("Balanced training:")
print("Total:", len(X_train))
print("Normal:", np.sum(y_train == 0))
print("Abnormal:", np.sum(y_train == 1))


# ============================================================
# STEP 4 — REDUCE 180 SAMPLES → 90 SAMPLES
# ============================================================

if X_train.shape[1] == 180:

    X_train_90 = resample_poly(
        X_train,
        up=1,
        down=2,
        axis=1
    )

    X_test_90 = resample_poly(
        X_test,
        up=1,
        down=2,
        axis=1
    )

else:
    raise ValueError(
        f"Expected 180 samples before reduction, "
        f"got {X_train.shape[1]}"
    )


print()
print("After input reduction:")
print("Training shape:", X_train_90.shape)
print("Testing shape:", X_test_90.shape)


# ============================================================
# STEP 5 — EXPERIMENT 4
# ============================================================

results = []

os.makedirs("experiment4_models", exist_ok=True)


for n_trees in TREE_COUNTS:

    print()
    print("=" * 60)
    print(f"Training Random Forest with {n_trees} trees")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train_90,
        y_train
    )

    predictions = model.predict(X_test_90)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # Total number of decision-tree nodes
    total_nodes = sum(
        tree.tree_.node_count
        for tree in model.estimators_
    )

    # Save model so we can measure serialized size
    model_path = os.path.join(
        "experiment4_models",
        f"rf_{n_trees}_trees.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    model_size_bytes = os.path.getsize(
        model_path
    )

    model_size_kb = model_size_bytes / 1024

    results.append({
        "trees": n_trees,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_nodes": total_nodes,
        "model_size_kb": model_size_kb
    })


# ============================================================
# STEP 6 — PRINT RESULTS
# ============================================================

print()
print()
print("=" * 95)
print("EXPERIMENT 4 — MODEL COMPLEXITY REDUCTION")
print("=" * 95)

print(
    f"{'Trees':>8}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Nodes':>14}"
    f"{'Size (KB)':>14}"
)

print("-" * 95)


for result in results:

    print(
        f"{result['trees']:>8}"
        f"{result['accuracy'] * 100:>11.2f}%"
        f"{result['precision'] * 100:>11.2f}%"
        f"{result['recall'] * 100:>11.2f}%"
        f"{result['f1'] * 100:>11.2f}%"
        f"{result['total_nodes']:>14,}"
        f"{result['model_size_kb']:>14.2f}"
    )


print()
print("Experiment 4 complete.")