import wfdb
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
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

        # We only use:
        # N = normal
        # A = abnormal
        # V = abnormal

        if symbol not in ["N", "A", "V"]:
            continue

        # Make sure the full 180-sample window fits
        if sample - BEFORE < 0:
            continue

        if sample + AFTER >= len(signal):
            continue

        beat = signal[
            sample - BEFORE :
            sample + AFTER
        ]

        all_beats.append(beat)

        # Normal = 0
        # Abnormal = 1
        if symbol == "N":
            all_labels.append(0)
        else:
            all_labels.append(1)

        all_record_ids.append(record_id)


# Convert to NumPy arrays

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

    # Records 201 and 202 came from the same analog tape.
    # Therefore they must never be separated between
    # training and testing.

    if record_id in ["201", "202"]:
        groups.append("201_202")

    else:
        # Every other record gets its own group.
        groups.append(record_id)


groups = np.array(groups)


# ============================================================
# STEP 3 — SOURCE-INDEPENDENT TRAIN/TEST SPLIT
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


# Create training data

X_train = X[train_idx]
y_train = y[train_idx]

record_train = record_ids[train_idx]
groups_train = groups[train_idx]


# Create testing data

X_test = X[test_idx]
y_test = y[test_idx]

record_test = record_ids[test_idx]
groups_test = groups[test_idx]


# ============================================================
# SHOW SPLIT
# ============================================================

print()
print("Training records:")
print(np.unique(record_train))

print()
print("Testing records:")
print(np.unique(record_test))

print()
print("Training groups:")
print(np.unique(groups_train))

print()
print("Testing groups:")
print(np.unique(groups_test))

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
# STEP 4 — BALANCE ONLY THE TRAINING DATA
# ============================================================

print()
print("=" * 60)
print("BALANCING TRAINING DATA")
print("=" * 60)

normal_indices = np.where(y_train == 0)[0]
abnormal_indices = np.where(y_train == 1)[0]

# Use the smaller class size

number_to_use = min(
    len(normal_indices),
    len(abnormal_indices)
)

# Randomly select equal numbers

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

# Shuffle the balanced training set

rng.shuffle(selected_indices)

X_train_balanced = X_train[selected_indices]
y_train_balanced = y_train[selected_indices]


print("Balanced training samples:",
      len(X_train_balanced))

print("Normal:",
      np.sum(y_train_balanced == 0))

print("Abnormal:",
      np.sum(y_train_balanced == 1))


# ============================================================
# STEP 5 — LOGISTIC REGRESSION
# ============================================================

print()
print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_balanced,
    y_train_balanced
)

logistic_predictions = logistic_model.predict(
    X_test
)


# Metrics

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    zero_division=0
)


print("Accuracy :", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall   :", logistic_recall)
print("F1 Score :", logistic_f1)

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=[
            "Normal",
            "Abnormal"
        ],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_predictions
    )
)


# ============================================================
# STEP 6 — RANDOM FOREST
# ============================================================

print()
print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(
    X_train_balanced,
    y_train_balanced
)

rf_predictions = random_forest.predict(
    X_test
)


# Metrics

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    zero_division=0
)


print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=[
            "Normal",
            "Abnormal"
        ],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print()

print("Model                 Accuracy    Precision    Recall    F1")
print("-" * 60)

print(
    f"Logistic Regression   "
    f"{logistic_accuracy:.4f}      "
    f"{logistic_precision:.4f}       "
    f"{logistic_recall:.4f}    "
    f"{logistic_f1:.4f}"
)

print(
    f"Random Forest         "
    f"{rf_accuracy:.4f}      "
    f"{rf_precision:.4f}       "
    f"{rf_recall:.4f}    "
    f"{rf_f1:.4f}"
)

print()
print("Experiment complete.")