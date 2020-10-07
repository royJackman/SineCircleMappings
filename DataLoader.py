import numpy as np

with open('bach_chorales.npy', 'rb') as f:
    bach_chorales = np.load(f, allow_pickle=True)
