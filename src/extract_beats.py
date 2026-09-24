import wfdb
import numpy as np

# Load the ECG signal
record = wfdb.rdrecord('100', pn_dir='mitdb')
signal = record.p_signal[:, 0]

annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')


before = 90
after = 90

beats = []
labels = []

for sample, symbol in zip(annotation.sample, annotation.symbol):

  
    if symbol == 'N' or symbol == 'A':

        
        if sample - before >= 0 and sample + after < len(signal):

            
            beat = signal[sample - before:sample + after]

            beats.append(beat)

           
            if symbol == 'N':
                labels.append(0)
            else:
                labels.append(1)


beats = np.array(beats)
labels = np.array(labels)

print("Number of beats:", len(beats))
print("Shape of beats:", beats.shape)
print("Shape of labels:", labels.shape)

print()
print("First beat:")
print(beats[0])

print()
print("First label:", labels[0])