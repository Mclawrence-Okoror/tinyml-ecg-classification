import re


# --------------------------------------------------
# READ THE EXACT TEST VECTOR FILE
# --------------------------------------------------

with open("mlp8_test_vectors.h", "r") as f:
    text = f.read()


# --------------------------------------------------
# EXTRACT FIRST ECG VECTOR
# --------------------------------------------------

match = re.search(
    r"test_inputs\[.*?\]\[90\]\s*=\s*\{(.*?)\};",
    text,
    re.DOTALL
)

if match is None:
    print("Could not find test_inputs in mlp8_test_vectors.h")
    raise SystemExit


values_text = match.group(1)


values = re.findall(
    r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
    values_text
)


beat = [float(v) for v in values]


if len(beat) != 90:
    print(
        "ERROR: Expected 90 samples, found:",
        len(beat)
    )
    raise SystemExit


# --------------------------------------------------
# EXTRACT FIRST LABEL
# --------------------------------------------------

label_match = re.search(
    r"test_labels\[\d+\]\s*=\s*\{(.*?)\};",
    text,
    re.DOTALL
)

if label_match is None:
    print("Could not find test_labels")
    raise SystemExit


labels = re.findall(
    r"\b[01]\b",
    label_match.group(1)
)


if len(labels) == 0:
    print("Could not extract label")
    raise SystemExit


label = int(labels[0])


# --------------------------------------------------
# WRITE VERIFIED C HEADER
# --------------------------------------------------

with open("verified_test_beat.h", "w") as f:

    f.write("#ifndef VERIFIED_TEST_BEAT_H\n")
    f.write("#define VERIFIED_TEST_BEAT_H\n\n")

    f.write("#define VERIFIED_TEST_BEAT_SIZE 90\n\n")

    f.write(
        "static const float verified_test_beat[90] = {\n"
    )

    for i, value in enumerate(beat):

        if i % 5 == 0:
            f.write("    ")

        f.write(f"{value:.9f}f")

        if i != 89:
            f.write(", ")

        if i % 5 == 4:
            f.write("\n")

    f.write("};\n\n")

    f.write(
        f"#define VERIFIED_TEST_LABEL {label}\n\n"
    )

    f.write("#endif\n")


# --------------------------------------------------
# SHOW RESULT
# --------------------------------------------------

print("========================================")
print("Verified test beat export")
print("========================================")

print()
print("Samples:", len(beat))
print("True label:", label)

if label == 0:
    print("True class: NORMAL")
else:
    print("True class: ABNORMAL")

print()
print("Created: verified_test_beat.h")