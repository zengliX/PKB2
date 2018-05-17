"""
file: program to calculate kernel matrices
author: lizeng
"""

from sklearn import metrics
import numpy as np
import sys

"""
get kernel matrices for each group
X: (N1,Ngene)
Y: (N2,Ngene)
inputs: an input_obj
weights: pd.Series of gene weights/ or None (from inputs.weights)
output: shape (N1,N2,Ngroup), array of kernel matrices
"""
def get_kernels(X,Y,inputs,weights):
    M = inputs.Ngroup
    N1 = X.shape[0]
    N2 = Y.shape[0]
    out = np.zeros([N1,N2,M])

    ct = 0
    for value in inputs.pred_sets.values:
        genes = value.split(" ")
        a = X[genes].copy()
        b = Y[genes].copy()
        # need to transform the values when weights provided
        if not weights is None:
            Ws = weights.loc[genes]
            newW = np.sqrt(len(genes)*Ws/Ws.sum())
            a *= newW
            b *= newW
        if inputs.kernel=='rbf':
            out[:,:,ct] = metrics.pairwise.rbf_kernel(X= a,Y =b,gamma= 1/len(genes))
        elif inputs.kernel[:4]=='poly':
            deg = int(inputs.kernel[4])
            out[:,:,ct] = metrics.pairwise.polynomial_kernel(X = a, Y= b, degree = deg, gamma = 1/len(genes))
        else:
            print("wrong kernel option: "+inputs.kernel+'\n')
            sys.exit(-3)
        ct += 1
    return out
