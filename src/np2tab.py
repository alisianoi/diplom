import numpy as np

def np2tab(fname, data, labels, nan="-1"):
    cls = np.unique(labels)
    [N, D] = data.shape

    with open(fname, 'w') as dst:
        l = " ".join([str(i) for i in np.cumsum(np.binocount(labels))])
        print("{} {}".format(D, N) + " " + l + " " + nan, file=dst)
