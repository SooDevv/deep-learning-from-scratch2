import numpy as np


def load_data(seed=1993):
    np.random.seed(seed)
    N = 100 # number of sample per class
    DIM = 2 # number of feature
    CLS_NUM = 3 # class number

    x = np.zeros((N*CLS_NUM, DIM)) # (300 x 2)
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int) # (300 x 3)

    for j in range(CLS_NUM): # 3
        for i in range(N): # 100
            rate = i / N # 0, 0.01, 0.02
            radius = 1.0*rate # 0, 0.01, 0.02
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()

            t[ix, j] = 1

    return x, t