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

RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234"
]

BEFORE = 90
AFTER = 90


# ============================================================
# STEP 1 — LOAD ECG RECORDS AND EXTRACT BEATS
# ============================================================

print("=" * 60)
print("LOADING MIT-BIH ECG DATASET")
print("=" * 60)

all_beats = []
all_labels = []
all_record_ids = []

for record_id in RECORDS:

    print("Processing record:", record_id)

    record_path = DATASET_PATH + "/" + record_id

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    signal = record.p_signal[:, 0]

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol
    ):

        if symbol not in ["N", "A", "V"]:
            continue

        if sample - BEFORE < 0:
            continue

        if sample + AFTER >= len(signal):
            continue

        beat = signal[
            sample - BEFORE :
            sample + AFTER
        ]

        all_beats.append(beat)

        if symbol == "N":
            all_labels.append(0)
        else:
            all_labels.append(1)

        all_record_ids.append(record_id)


X = np.array(all_beats)
y = np.array(all_labels)
record_ids = np.array(all_record_ids)


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Total beats:", len(X))
print("Beat shape:", X.shape)

print()
print("Normal beats:", np.sum(y == 0))
print("Abnormal beats:", np.sum(y == 1))


# ============================================================
# STEP 2 — CREATE SOURCE GROUPS
# ============================================================

print()
print("=" * 60)
print("CREATING SOURCE GROUPS")
print("=" * 60)

groups = []

for record_id in record_ids:

    if record_id in ["201", "202"]:
        groups.append("201_202")

    else:
        groups.append(record_id)


groups = np.array(groups)


# ============================================================
# STEP 3 — SAME SOURCE-INDEPENDENT SPLIT
# ============================================================

print()
print("=" * 60)
print("SOURCE-INDEPENDENT TRAIN/TEST SPLIT")
print("=" * 60)

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


X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]


print()
print("Training records:")
print(np.unique(record_ids[train_idx]))

print()
print("Testing records:")
print(np.unique(record_ids[test_idx]))

print()
print("Training beats:", len(X_train))
print("Testing beats:", len(X_test))

print()
print("Training class distribution:")
print("Normal:", np.sum(y_train == 0))
print("Abnormal:", np.sum(y_train == 1))

print()
print("Testing class distribution:")
print("Normal:", np.sum(y_test == 0))
print("Abnormal:", np.sum(y_test == 1))


# ============================================================
# STEP 4 — BALANCE TRAINING DATA
# ============================================================

print()
print("=" * 60)
print("BALANCING TRAINING DATA")
print("=" * 60)

normal_indices = np.where(y_train == 0)[0]
abnormal_indices = np.where(y_train == 1)[0]

number_to_use = min(
    len(normal_indices),
    len(abnormal_indices)
)

rng = np.random.default_rng(42)

normal_selected = rng.choice(
    normal_indices,
    size=number_to_use,
    replace=False
)

abnormal_selected = rng.choice(
    abnormal_indices,
    size=number_to_use,
    replace=False
)

selected_indices = np.concatenate([
    normal_selected,
    abnormal_selected
])

rng.shuffle(selected_indices)

X_train_balanced = X_train[selected_indices]
y_train_balanced = y_train[selected_indices]


print(
    "Balanced training samples:",
    len(X_train_balanced)
)

print(
    "Normal:",
    np.sum(y_train_balanced == 0)
)

print(
    "Abnormal:",
    np.sum(y_train_balanced == 1)
)


# ============================================================
# STEP 5 — INPUT REDUCTION FUNCTION
# ============================================================

def reduce_samples(X, target_samples):

    original_samples = X.shape[1]

    factor = original_samples // target_samples

    if original_samples % target_samples != 0:
        raise ValueError(
            "Target sample count must divide "
            "180 exactly."
        )

    reduced = resample_poly(
        X,
        up=1,
        down=factor,
        axis=1
    )

    return reduced


# ============================================================
# STEP 6 — RUN INPUT REDUCTION EXPERIMENT
# ============================================================

print()
print("=" * 60)
print("EXPERIMENT 2 — ECG INPUT REDUCTION")
print("=" * 60)

sample_sizes = [
    180,
    90,
    60,
    45,
    30
]

results = []


for samples in sample_sizes:

    print()
    print("-" * 60)
    print(
        "TESTING",
        samples,
        "SAMPLES PER HEARTBEAT"
    )
    print("-" * 60)

    # --------------------------------------------------------
    # Reduce the ECG representation
    # --------------------------------------------------------

    if samples == 180:

        X_train_reduced = X_train_balanced
        X_test_reduced = X_test

    else:

        X_train_reduced = reduce_samples(
            X_train_balanced,
            samples
        )

        X_test_reduced = reduce_samples(
            X_test,
            samples
        )


    print(
        "Training shape:",
        X_train_reduced.shape
    )

    print(
        "Testing shape:",
        X_test_reduced.shape
    )


    # --------------------------------------------------------
    # TRAIN RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_reduced,
        y_train_balanced
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_reduced
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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
    # INPUT MEMORY
    # --------------------------------------------------------

    # float32 = 4 bytes per sample

    input_bytes = samples * 4


    print()
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

    print(
        f"Input size:         {input_bytes} bytes"
    )


    results.append({
        "samples": samples,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "input_bytes": input_bytes
    })


# ============================================================
# STEP 7 — FINAL RESULTS TABLE
# ============================================================

print()
print()
print("=" * 75)
print("EXPERIMENT 2 RESULTS")
print("=" * 75)

print()

print(
    f"{'Samples':>8}"
    f"{'Accuracy':>12}"
    f"{'Precision':>14}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Bytes':>10}"
)

print("-" * 75)


for result in results:

    print(
        f"{result['samples']:>8}"
        f"{result['accuracy'] * 100:>11.2f}%"
        f"{result['precision'] * 100:>13.2f}%"
        f"{result['recall'] * 100:>11.2f}%"
        f"{result['f1'] * 100:>11.2f}%"
        f"{result['input_bytes']:>10}"
    )


print()
print("Experiment 2 complete.")