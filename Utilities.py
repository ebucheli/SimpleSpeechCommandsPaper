import numpy as np

def make_oh(y):
    N = len(y)
    n_classes = len(np.unique(y))
    
    y_oh = np.zeros((N,n_classes))
    
    for i in range(N):
        col = int(y[i])
        y_oh[i,col] = 1
    
    return y_oh