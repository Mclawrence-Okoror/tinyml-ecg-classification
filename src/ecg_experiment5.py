import os
import joblib
import numpy as np
import wfdb

from scipy.signal import resample_poly
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "mit-bih-arrhythmia-database-1.0.0"

BEFORE = 90
AFTER = 90

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
# STEP 2 — SOURCE-INDEPENDENT SPLIT
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
# STEP 3 — BALANCE TRAINING SET
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

rng.shuffle(balanced_indices)

X_train = X_train_full[balanced_indices]
y_train = y_train_full[balanced_indices]


print()
print("Balanced training:")
print("Total:", len(X_train))
print("Normal:", np.sum(y_train == 0))
print("Abnormal:", np.sum(y_train == 1))


# ============================================================
# STEP 4 — REDUCE INPUT: 180 → 90
# ============================================================

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


print()
print("90-sample representation:")
print("Training:", X_train_90.shape)
print("Testing:", X_test_90.shape)


# ============================================================
# STEP 5 — DEFINE MODELS
# ============================================================

models = {

    "RF-10": RandomForestClassifier(
        n_estimators=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=RANDOM_STATE
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ]),

    "Small MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True
        ))
    ])
}


# ============================================================
# STEP 6 — TRAIN + EVALUATE
# ============================================================

results = []

os.makedirs(
    "experiment5_models",
    exist_ok=True
)


for name, model in models.items():

    print()
    print("=" * 65)
    print("Training:", name)
    print("=" * 65)

    model.fit(
        X_train_90,
        y_train
    )

    predictions = model.predict(
        X_test_90
    )

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

    # --------------------------------------------------------
    # Model-specific complexity
    # --------------------------------------------------------

    if name == "RF-10":

        total_nodes = sum(
            tree.tree_.node_count
            for tree in model.estimators_
        )

    elif name == "Decision Tree":

        total_nodes = model.tree_.node_count

    elif name == "Logistic Regression":

        total_nodes = 0

    elif name == "Small MLP":

        total_nodes = sum(
            weights.size
            for weights in model.named_steps["model"].coefs_
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    filename = name.lower().replace(" ", "_").replace("-", "_")

    model_path = os.path.join(
        "experiment5_models",
        filename + ".joblib"
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
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "complexity": total_nodes,
        "size_kb": model_size_kb
    })


# ============================================================
# STEP 7 — RESULTS
# ============================================================

print()
print()
print("=" * 100)
print("EXPERIMENT 5 — COMPACT MODEL COMPARISON")
print("=" * 100)

print(
    f"{'Model':<22}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Complexity':>15}"
    f"{'Size (KB)':>15}"
)

print("-" * 100)


for result in results:

    print(
        f"{result['model']:<22}"
        f"{result['accuracy'] * 100:>11.2f}%"
        f"{result['precision'] * 100:>11.2f}%"
        f"{result['recall'] * 100:>11.2f}%"
        f"{result['f1'] * 100:>11.2f}%"
        f"{result['complexity']:>15,}"
        f"{result['size_kb']:>15.2f}"
    )


print()
print("Experiment 5 complete.")