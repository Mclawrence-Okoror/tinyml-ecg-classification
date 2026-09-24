import wfdb

record = wfdb.rdrecord('100', pn_dir='mitdb')
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')

print("Number of annotations:", len(annotation.sample))

for i in range(100):
    print(
        "Sample:", annotation.sample[i],
        "Symbol:", annotation.symbol[i]
    )