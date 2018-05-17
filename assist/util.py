"""
utility functions
author: li zeng
"""
import numpy as np
import scipy

"""-----------------
FUNCTIONS
--------------------"""


"""
line search function
K_train: kernel matrix (Ntrain, Ntrain, Ngroup)
Z: clinical data, shape (len(sele_loc),Npred_clin)
model: a specific model class
pars: [m, beta, c] from each iteration
sele_loc: used in CV, only a subset of data used
"""
def line_search(K_train,Z,model,pars,max_val = 10):
    [m,beta,gamma] = pars
    # get K
    Km = get_K(K_train,m)
    b = Km.dot(beta)+ Z.dot(gamma)
    # line search function
    if model.problem == 'survival':
        def temp_fun(x):
            return model.loss_fun_train(model.F_train+x*b)
    else:
        def temp_fun(x):
            return model.loss_fun(model.ytrain, model.F_train+x*b)
    out = scipy.optimize.minimize_scalar(temp_fun,tol=0.01)
    if not out.success:
        print("warning: minimization failure")
    return min(out.x,max_val)


"""
stratified sampling
y: dataframe
col: the column used for stratification
"""
def subsamp(y,col,fold=3):
    grouped = y.groupby(col)
    out = [list() for i in range(fold)]
    for i in grouped.groups.keys():
        ind = grouped.get_group(i)
        n = ind.shape[0]
        r = list(range(0,n+1,n//fold))
        r[-1] = n+1
        # permute index
        perm_index = np.random.permutation(ind.index)
        for j in range(fold):
            out[j] += list(perm_index[r[j]:r[j+1]])
    return out

"""
stratified sampling (not considering the outcome distribution)
y: dataframe
"""
def simple_subsamp(y,fold=3):
    out = [list() for i in range(fold)]
    n = y.shape[0]
    r = list(range(0,n+1,n//fold))
    r[-1] = n+1
    # permute index
    perm_index = np.random.permutation(y.index)
    for j in range(fold):
        out[j] += list(perm_index[r[j]:r[j+1]])
    return out

"""
get kernel matrices from K_train
return K[:,:,m][sele_loc,selec_l]
"""
def get_K(K_train,m):
    Km = K_train[:,:,m]
    return Km

"""
test if Inf appears in v, if true, raise an error
v: np.array of values
"""
def testInf(v,ytrain,F_train):
    if np.any(np.isinf(v)):
        loc = np.isinf(v)
        raise Exception("inf in calcu_h: \nytrain {} \nF_train {}".format(ytrain[loc], F_train[loc]))


"""
print title
"---- title ----"
"""
def print_section(s, width = 60):
    print()
    if not s:
        print('-'*width)
    else:
        ct_left = (width - len(s) -2)//2
        ct_right = width - len(s) - 2 - ct_left
        print('-'*ct_left + ' {} '.format(s) + '-'*ct_right)

"""
raise error: undefined
"""
def undefined():
    raise Exception("method undefined")
