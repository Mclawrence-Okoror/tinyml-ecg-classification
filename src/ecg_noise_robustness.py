import wfdb
import numpy as np

from scipy.signal import resample_poly
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
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

TARGET_SAMPLES = 90

RANDOM_STATE = 42

RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119",
    "121","122","123","124","200","201","202","203","205",
    "207","208","209","210","212","213","214","215","217",
    "219","220","221","222","223","228","230","231","232",
    "233","234"
]


# ============================================================
# LOAD ECG DATA
# ============================================================

print("=" * 60)
print("LOADING MIT-BIH ECG DATASET")
print("=" * 60)

X = []
y = []
groups = []

for record_name in RECORDS:

    print(f"Processing record: {record_name}")

    record_path = f"{DATASET_PATH}/{record_name}"

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

        if len(beat) != 180:
            continue

        X.append(beat)

        if symbol == "N":
            y.append(0)
        else:
            y.append(1)

        # 201 and 202 originate from the same analog tape
        if record_name in ["201", "202"]:
            groups.append("201_202")
        else:
            groups.append(record_name)


X = np.array(X)
y = np.array(y)
groups = np.array(groups)


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total beats: {len(X)}")
print(f"Beat shape: {X.shape}")

print()
print(f"Normal beats: {np.sum(y == 0)}")
print(f"Abnormal beats: {np.sum(y == 1)}")


# ============================================================
# SOURCE-INDEPENDENT SPLIT
# ============================================================

print()
print("=" * 60)
print("SOURCE-INDEPENDENT TRAIN/TEST SPLIT")
print("=" * 60)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print()
print("Training beats:", len(X_train))
print("Testing beats:", len(X_test))


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

print()
print("=" * 60)
print("BALANCING TRAINING DATA")
print("=" * 60)

normal_indices = np.where(y_train == 0)[0]
abnormal_indices = np.where(y_train == 1)[0]

rng = np.random.default_rng(RANDOM_STATE)

normal_selected = rng.choice(
    normal_indices,
    size=len(abnormal_indices),
    replace=False
)

balanced_indices = np.concatenate(
    [normal_selected, abnormal_indices]
)

X_train_balanced = X_train[balanced_indices]
y_train_balanced = y_train[balanced_indices]

print(
    f"Balanced training samples: "
    f"{len(X_train_balanced)}"
)

print(
    f"Normal: {np.sum(y_train_balanced == 0)}"
)

print(
    f"Abnormal: {np.sum(y_train_balanced == 1)}"
)


# ============================================================
# REDUCE 180 → 90 SAMPLES
# ============================================================

print()
print("=" * 60)
print("REDUCING ECG INPUT: 180 → 90 SAMPLES")
print("=" * 60)

X_train_90 = resample_poly(
    X_train_balanced,
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

print("Training shape:", X_train_90.shape)
print("Testing shape:", X_test_90.shape)


# ============================================================
# NOISE FUNCTION
# ============================================================

def add_noise(X, noise_level, seed=42):

    rng = np.random.default_rng(seed)

    signal_std = np.std(X)

    noise = rng.normal(
        0,
        noise_level * signal_std,
        X.shape
    )

    return X + noise


# ============================================================
# EXPERIMENT 3
# ============================================================

print()
print("=" * 60)
print("EXPERIMENT 3 — ECG NOISE ROBUSTNESS")
print("=" * 60)


noise_conditions = {
    "Clean": 0.00,
    "Mild": 0.05,
    "Moderate": 0.10,
    "Strong": 0.20
}


results = []


for condition, noise_level in noise_conditions.items():

    print()
    print("-" * 60)
    print(
        f"TESTING {condition.upper()} NOISE "
        f"({noise_level:.2f} × signal std)"
    )
    print("-" * 60)

    # --------------------------------------------------------
    # Add noise ONLY to the input signals
    # --------------------------------------------------------

    X_train_noisy = add_noise(
        X_train_90,
        noise_level,
        seed=RANDOM_STATE
    )

    X_test_noisy = add_noise(
        X_test_90,
        noise_level,
        seed=RANDOM_STATE + 1
    )

    # --------------------------------------------------------
    # Train fresh model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train_noisy,
        y_train_balanced
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(X_test_noisy)

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

    print(
        f"Accuracy:           {accuracy:.4f}"
    )

    print(
        f"Abnormal precision: {precision:.4f}"
    )

    print(
        f"Abnormal recall:    {recall:.4f}"
    )

    print(
        f"Abnormal F1:        {f1:.4f}"
    )

    results.append(
        (
            condition,
            noise_level,
            accuracy,
            precision,
            recall,
            f1
        )
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 75)
print("EXPERIMENT 3 RESULTS")
print("=" * 75)

print(
    f"{'Condition':<12}"
    f"{'Noise':>10}"
    f"{'Accuracy':>12}"
    f"{'Precision':>14}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print("-" * 75)

for result in results:

    condition, noise, accuracy, precision, recall, f1 = result

    print(
        f"{condition:<12}"
        f"{noise:>10.2f}"
        f"{accuracy * 100:>11.2f}%"
        f"{precision * 100:>13.2f}%"
        f"{recall * 100:>11.2f}%"
        f"{f1 * 100:>11.2f}%"
    )


print()
print("Experiment 3 complete.")