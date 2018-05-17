"""
L1 penalty method
author: li zeng
"""

import numpy as np
from assist.util import get_K, undefined
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
from sklearn import linear_model


# function to be paralleled
"""
solve L1 penalized regression for pathway m
K_train: kernel matrix (Ntrain, Ntrain, Ngroup)
model: model class object
m: pathway index
eta: from calcu_eta
eta_tilde: tranformed eta vlue
mid_mat: transforming matrix for eta and Km
e: intermediate value
w: from calcu_w
"""
def paral_fun_L1(K_train,Z,model,m,eta,eta_tilde,mid_mat,e,w,Lambda):
    Nsamp = K_train.shape[0]
    # working Lambda
    new_Lambda= Lambda/2 # due to setting of sklearn.linear_model.Lasso
    # get K
    Km = get_K(K_train,m)
    # transform K for penalized regression
    Km_tilde = mid_mat.dot(Km)
    # get beta
    lasso_fit = linear_model.Lasso(alpha = new_Lambda,fit_intercept = False,\
                selection='random',max_iter=200,tol=10**-4)
    lasso_fit.fit(-Km_tilde,eta_tilde)
    beta = lasso_fit.coef_
    #get gamma
    if model.problem in ('classification','survival'):
        if model.problem == 'survival' and not model.hasClinical:
            gamma = np.array([0.0])
        else:
            gamma = - np.linalg.solve(Z.T.dot(w).dot(Z), Z.T.dot(w)).dot(eta + Km.dot(beta))
    elif model.problem == 'regression':
        gamma = - e.dot(eta + Km.dot(beta))
    # calculate val
    val = np.sum((eta_tilde+Km_tilde.dot(beta))**2)+ Lambda*Nsamp*np.sum(np.abs(beta))
    return [val,[m,beta,gamma]]


"""
find a feasible lambda for L2 problem
K_train: training kernel, shape (Ntrain, Ntrain, Ngroup)
Z: training clinical data, shape (Ntrain, Npred_clin)
model: model class object
"""
def find_Lambda_L1(K_train,Z,model):
    Nsamp = K_train.shape[0]
    Ngroup = K_train.shape[2]
    prod = []
    if model.problem in ('classification','survival'):
        h = model.calcu_h()
        q = model.calcu_q()
        eta = model.calcu_eta(h,q)
        w = model.calcu_w(q)
        w_half = model.calcu_w_half(q)
        if model.problem == 'survival' and not model.hasClinical:
            mid_mat = w_half
        else:
            mid_mat = w_half.dot( np.eye(Nsamp) - Z.dot( np.linalg.solve(Z.T.dot(w).dot(Z), Z.T.dot(w))) )
        eta_tilde = mid_mat.dot(eta)
    elif model.problem == 'regression':
        eta = model.calcu_eta()
        mid_mat = np.eye(Z.shape[0]) - Z.dot( np.linalg.solve(Z.T.dot(Z), Z.T) )
        eta_tilde = mid_mat.dot(eta)
    for m in range(Ngroup):
        Km = K_train[:,:,m]
        Km_tilde = mid_mat.dot(Km)
        prod += list(Km_tilde.dot(eta_tilde))
    return 2*np.percentile(np.abs(prod),85)/Nsamp

"""
perform one iteration of L1-boosting
K_train: kernel matrix (Ntrain, Ntrain, Ngroup)
model: model class object
group_subset: bool, whether randomly choose a subset of pathways
"""
def oneiter_L1(K_train,Z,model,Lambda,\
               parallel=False,group_subset = False):
    Nsamp = K_train.shape[0]
    Ngroup = K_train.shape[2]
    h = model.calcu_h()
    q = model.calcu_q()
    if model.problem in ('classification','survival'):
        eta = model.calcu_eta(h,q)
        w = model.calcu_w(q)
        w_half = model.calcu_w_half(q)
        if model.problem == 'survival' and not model.hasClinical:
            mid_mat = w_half
        else:
            mid_mat = w_half.dot( np.eye(Nsamp) - Z.dot( np.linalg.solve(Z.T.dot(w).dot(Z), Z.T.dot(w))) )
        eta_tilde = mid_mat.dot(eta)
        e = None
    elif model.problem == 'regression':
        eta = model.calcu_eta()
        e = np.linalg.solve(Z.T.dot(Z), Z.T)
        w = None
        mid_mat = np.eye(Nsamp) - Z.dot(e)
        eta_tilde = mid_mat.dot(eta)

    # identify best fit K_m
        # random subset of groups
    mlist = range(Ngroup)
    if group_subset:
        mlist= np.random.choice(mlist,min([Ngroup//3,100]),replace=False)
    if parallel:
        raise Exception("parallel algorithm currently not supported.")
    else:
        out = []
        for m in mlist:
            out.append(paral_fun_L1(K_train,Z,model,m,eta,eta_tilde,mid_mat,e,w,Lambda))
    return out[np.argmin([x[0] for x in out])][1]
