import numpy as np
import pandas as pd
import scipy as sc

def load_data(type, index):
    if type == "regression":
        if index == 1:
            df = pd.read_csv("local/datasets/housing.csv", sep=",")
            M = df.values
            # Erase data >45, as dataset is capped
            M = np.delete(M, M[:, -1] > 45, axis=0)
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[:400, :12]  # To ensure A can be divided by 4 for ADMM!
            b = M[:400, 13]  # The regression objective: the price!
        elif index == 2:
            df = pd.read_csv("local/datasets/housing.csv", sep=",")
            M = df.values
            # Erase data >45, as dataset is capped
            M = np.delete(M, M[:, -1] > 45, axis=0)
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[:400, 1:13]  # To ensure A can be divided by 4 for ADMM!
            b = M[:400, 13]  # The regression objective: the price!
        elif index == 3:
            df = pd.read_csv("local/datasets/housing.csv", sep=",")
            M = df.values
            # Erase data >45, as dataset is capped
            M = np.delete(M, M[:, -1] > 45, axis=0)
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[:400, 2:14]  # To ensure A can be divided by 4 for ADMM!
            b = M[:400, 13]  # The regression objective: the price!
        elif index == 4:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[:1000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[:1000, 21]  # The regression objective: next_Tmax!
        elif index == 5:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[:1000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[:1000, 22]  # The regression objective: next_Tmin!
        elif index == 6:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[1000:2000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[1000:2000, 21]  # The regression objective: next_Tmax!
        elif index == 7:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[1000:2000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[1000:2000, 22]  # The regression objective: next_Tmax!
        elif index == 8:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[2000:3000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[2000:3000, 21]  # The regression objective: next_Tmax!
        elif index == 9:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[2000:3000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[2000:3000, 22]  # The regression objective: next_Tmax!
        elif index == 10:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[3000:4000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[3000:4000, 21]  # The regression objective: next_Tmax!
        elif index == 11:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[3000:4000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[3000:4000, 22]  # The regression objective: next_Tmax!
        elif index == 12:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[4000:5000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[4000:5000, 21]  # The regression objective: next_Tmax!
        elif index == 13:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[4000:5000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[4000:5000, 22]  # The regression objective: next_Tmax!
        elif index == 14:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[5000:6000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[5000:6000, 21]  # The regression objective: next_Tmax!
        elif index == 15:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[5000:6000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[5000:6000, 22]  # The regression objective: next_Tmax!
        elif index == 16:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[6000:7000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[6000:7000, 21]  # The regression objective: next_Tmax!
        elif index == 17:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[6000:7000, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[6000:7000, 22]  # The regression objective: next_Tmax!
        elif index == 18:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[7000:7500, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[7000:7500, 21]  # The regression objective: next_Tmax!
        elif index == 19:
            mat_contents = sc.io.loadmat('local/datasets/weather.mat')
            M = mat_contents['M']
            M = M[~np.isnan(M).any(axis=1),:] #Delete NaNs!
            # Randomize data
            np.random.seed(0)
            # To make things repeatable
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[7000:7500, :20] # To ensure A can be divided by 4 for ADMM!
            b = M[7000:7500, 22]  # The regression objective: next_Tmax!
        else:
            print('Non valid index')
    elif type == 'classification':
        if index == 1:
            mat_contents = sc.io.loadmat('local/datasets/iris.mat')  
            A = mat_contents['A']
            b = mat_contents['b']
            # Randomize data
            np.random.seed(0)
            idx = np.random.permutation(A.shape[0])
            A = A[idx, :]
            b=b[idx]
            b = b.flatten()
        elif index == 2:
            df = pd.read_csv("local/datasets/banknote.csv", sep=",")
            M = df.values
            # Randomize data
            np.random.seed(0)
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[:500, :4]  # To ensure A can be divided by 4 for ADMM!
            b = M[:500, 4]   # The class
            b[b<=0] = -1     # Original authentic class is 0: change to -1!!
        elif index == 3:
            df = pd.read_csv("local/datasets/banknote.csv", sep=",")
            M = df.values
            # Randomize data
            np.random.seed(0)
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[200:700, :4]  # To ensure A can be divided by 4 for ADMM!
            b = M[200:700, 4]   # The class
            b[b<=0] = -1     # Original authentic class is 0: change to -1!!
        elif index == 4:
            df = pd.read_csv("local/datasets/banknote.csv", sep=",")
            M = df.values
            # Randomize data
            np.random.seed(0)
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[400:900, :4]  # To ensure A can be divided by 4 for ADMM!
            b = M[400:900, 4]   # The class
            b[b<=0] = -1     # Original authentic class is 0: change to -1!!
        elif index == 5:
            df = pd.read_csv("local/datasets/banknote.csv", sep=",")
            M = df.values
            # Randomize data
            np.random.seed(0)
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[600:1100, :4]  # To ensure A can be divided by 4 for ADMM!
            b = M[600:1100, 4]   # The class
            b[b<=0] = -1     # Original authentic class is 0: change to -1!!
        elif index == 6:
            df = pd.read_csv("local/datasets/banknote.csv", sep=",")
            M = df.values
            # Randomize data
            np.random.seed(0)
            M = M[np.random.permutation(M.shape[0]), :]
            A = M[800:1300, :4]  # To ensure A can be divided by 4 for ADMM!
            b = M[800:1300, 4]   # The class
            b[b<=0] = -1     # Original authentic class is 0: change to -1!!
        elif index == 7:
            mat_contents = sc.io.loadmat('local/datasets/cancer.mat')
            M = mat_contents['M']
            A = M[:500, 1:9]  # To ensure A can be divided by 4 for ADMM!
            b = M[:500, 10]   # The class
        elif index == 8:
            mat_contents = sc.io.loadmat('local/datasets/cancer.mat')
            M = mat_contents['M']
            A = M[100:600, 1:9]  # To ensure A can be divided by 4 for ADMM!
            b = M[100:600, 10]   # The class
        elif index == 9:
            mat_contents = sc.io.loadmat('local/datasets/cancer.mat')
            M = mat_contents['M']
            A = M[199:699, 1:9]  # To ensure A can be divided by 4 for ADMM!
            b = M[199:699, 10]   # The class
        elif index == 10:
            mat_contents = sc.io.loadmat('local/datasets/cancer.mat')
            M = mat_contents['M']
            A = M[:500, 2:10]  # To ensure A can be divided by 4 for ADMM!
            b = M[:500, 10]   # The class
        elif index == 11:
            mat_contents = sc.io.loadmat('local/datasets/cancer.mat')
            M = mat_contents['M']
            A = M[100:600, 2:10]  # To ensure A can be divided by 4 for ADMM!
            b = M[100:600, 10]   # The class
        elif index == 12:
            mat_contents = sc.io.loadmat('local/datasets/cancer.mat')
            M = mat_contents['M']
            A = M[199:699, 2:10]  # To ensure A can be divided by 4 for ADMM!
            b = M[199:699, 10]   # The class
        else:
            print('Non valid index')

    return A, b
