import numpy as np

with open('bach_chorales.npy', 'rb') as f:
    bach_chorales = np.load(f, allow_pickle=True)

def listify(chorale):
    retval = []
    for note in chorale:
        while len(retval) < note[0]:
            retval.append(0)
        for i in range(note[2]):
            retval.append((note[1] - 59)/16.0)
    return retval

list_chorales = np.array([listify(chorale) for chorale in bach_chorales])

def stagger(list_chorale, window=16):
    buffer = [0] * window
    X = []
    Y = []
    for note in list_chorale:
        X.append(buffer)
        Y.append(note)
        buffer = buffer[1:]
        buffer.append(note)
    X.append(buffer)
    Y.append(0)
    return np.array(X), np.array(Y)