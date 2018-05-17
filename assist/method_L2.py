"""
L2 penalty method
author: li zeng
"""

import numpy as np
from assist.util import get_K, undefined
import scipy


# function to paralleled
"""
solve L2 penalized regression for pathway m
K_train: Kernel matrix of training data (Ntrain, Ntrain, Ngroup)
model: model class object
m: pathway index
eta: from calcu_eta
eta_tilde: tranformed eta vlue
mid_mat: transforming matrix for eta and Km
e: intermediate value
w: from calcu_w
"""
def paral_fun_L2(K_train,Z,model,m,eta,eta_tilde,mid_mat,e,w,Lambda):
    Nsamp = K_train.shape[0]
    # working Lambda
    new_Lambda= Nsamp*Lambda
    # get K
    Km = get_K(K_train,m)
    # transform Km
    Km_tilde = mid_mat.dot(Km)
    # L2 solution
    beta = - np.linalg.solve( Km_tilde.T.dot(Km_tilde) + np.eye(Nsamp)*new_Lambda, Km_tilde.T.dot(eta_tilde))
    #get gamma
    if model.problem in ('classification','survival'):
        if model.problem == 'survival' and not model.hasClinical:
            gamma = np.array([0.0])
        else:
            gamma = - np.linalg.solve(Z.T.dot(w).dot(Z), Z.T.dot(w)).dot(eta + Km.dot(beta))
    elif model.problem == 'regression':
        gamma = - e.dot(eta + Km.dot(beta))
    val = np.sum((eta_tilde+Km_tilde.dot(beta))**2) + new_Lambda*np.sum(beta**2)
    return [val,[m,beta,gamma]]

"""
find a feasible lambda for L2 problem
K_train: training kernel, shape (Ntrain, Ntrain, Ngroup)
Z: training clinical data, shape (Ntrain, Npred_clin)
model: model class object
C: control |K*b| <= C*sqrt(Ntrain)
"""
def find_Lambda_L2(K_train,Z,model,C = 0.1):
    Nsamp = K_train.shape[0]
    Ngroup = K_train.shape[2]
    C = C*np.sqrt(Nsamp)
    l_list = [] # list of lambdas from each group
    if model.problem in ('classification','survival'):
        h = model.calcu_h()
        q = model.calcu_q()
        eta = model.calcu_eta(h,q)
        w = model.calcu_w(q)
        w_half = model.calcu_w_half(q)
        if model.problem == 'survival' and not model.hasClinical:
            mid_mat = w_half
        else:
            mid_mat = w_half.dot( np.eye(Nsamp) - Z.dot(np.linalg.solve(Z.T.dot(w).dot(Z), Z.T.dot(w))) )
        eta_tilde = mid_mat.dot(eta)
    elif model.problem == 'regression':
        eta = model.calcu_eta()
        mid_mat = np.eye(Z.shape[0]) - Z.dot( np.linalg.solve(Z.T.dot(Z), Z.T) )
        eta_tilde = mid_mat.dot(eta)
    for m in range(Ngroup):
        Km = K_train[:,:,m]
        Km_tilde = mid_mat.dot(Km)
        try:
            d = np.linalg.svd(Km_tilde)[1][0]
        except:
            continue
        l = d*(np.sqrt(np.sum(eta_tilde**2)) - C)/(C*Nsamp)
        l = l if l>0 else 0.01
        l_list.append(l)
    return np.percentile(l_list,85)


"""
perform one iteration of L2-boosting
K_train: kernel matrix
model: model class object
group_subset: bool, whether randomly choose a subset of pathways
"""
def oneiter_L2(K_train,Z,model,Lambda,\
               parallel=False,group_subset = False):
    Nsamp = K_train.shape[0]
    Ngroup = K_train.shape[2]
    # calculate derivatives h,q
    h = model.calcu_h()
    q = model.calcu_q()
    # calculate eta, eta_tilde, and intermediate values
    if model.problem in ('classification','survival'):
        eta = model.calcu_eta(h,q)
        w = model.calcu_w(q)
        w_half = model.calcu_w_half(q)
        if model.problem == 'survival' and not model.hasClinical:
            mid_mat = w_half
        else:
            mid_mat = w_half.dot( np.eye(Nsamp) - Z.dot(np.linalg.solve(Z.T.dot(w).dot(Z), Z.T.dot(w))) )
        eta_tilde = w_half.dot(mid_mat).dot(eta)
        e = None
    elif model.problem == 'regression':
        e = np.linalg.solve(Z.T.dot(Z), Z.T)
        eta = model.calcu_eta()
        mid_mat = np.eye(Nsamp) - Z.dot(e)
        eta_tilde = mid_mat.dot(eta)
        w = None
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
            out.append(paral_fun_L2(K_train,Z,model,m,eta,eta_tilde,mid_mat,e,w,Lambda))
    return out[np.argmin([x[0] for x in out])][1]
