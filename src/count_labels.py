import wfdb
from collections import Counter

annotations = wfdb.rdann('100', 'atr', pn_dir='mitdb')

counts = Counter(annotations.symbol)

print("Annotation types: ")
print()

for symbol, count in counts.most_common():
    print(symbol, ":", count)