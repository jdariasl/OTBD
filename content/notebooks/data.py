import numpy as np
import pandas as pd


def load_data(type, index):
    if type == "regression":
        if index == 1:
            df = pd.read_csv("datasets/housing.csv", sep=",")
            M = df.values
            # Erase data >45, as dataset is capped
            M = np.delete(M, M[:, -1] > 45, axis=0)
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[:400, :12]  # To ensure A can be divided by 4 for ADMM!
            b = M[:400, 13]  # The regression objective: the price!
    return A, b
