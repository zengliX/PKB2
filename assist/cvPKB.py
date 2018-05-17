"""
CV to determine optimal number of iterations
author: li zeng
"""

import assist
from assist.util import line_search, simple_subsamp, subsamp, print_section, undefined
import numpy as np
import pandas as pd
from assist.method_L1 import oneiter_L1
from assist.method_L2 import oneiter_L2
from matplotlib import pyplot as plt
import time

"""
CVinputs class
- used to initiate Model class in cross-validation procedure
"""
class CVinputs:
    def __init__(self,inputs,ytrain,ytest):
        self.Ntrain = ytrain.shape[0]
        self.Ntest = ytest.shape[0]
        self.Ngroup = inputs.Ngroup
        self.hasTest = True
        self.hasClinical = inputs.hasClinical
        self.Npred_clin = inputs.Npred_clin
        self.inputs = inputs

"""
Cross-Validation function
K_train: shape (Nsamp, Nsamp, Ngroup) kernel matrices
Lambda: penalty parameter
nfold: number of CV folds
ESTOP: early stop steps
parallel: use parallel algorithm (not supported in current version)
gr_sub: use random group selection or not
"""
def CV_PKB(inputs,K_train,Lambda,nfold=3,ESTOP=50,parallel=False,gr_sub=False,plot=False):
    ########## split data ###############
    temp = pd.Series(range(inputs.Ntrain),index= inputs.train_response.index)
    if inputs.problem == "classification":
        test_inds = subsamp(inputs.train_response,inputs.train_response.columns[0],nfold)
    elif inputs.problem == 'survival':
        test_inds = subsamp(inputs.train_response,inputs.train_response.columns[1],nfold)
    elif inputs.problem == "regression":
        test_inds = simple_subsamp(inputs.train_response,nfold)
    folds = []
    for i in range(nfold):
        folds.append([ temp[test_inds[i]].values, np.setdiff1d(temp.values,temp[test_inds[i]].values)])

    ########## initiate model for each fold ###############
    Ztrain_ls = [inputs.train_clinical.values[folds[i][1],:] for i in range(nfold)]
    Ztest_ls = [inputs.train_clinical.values[folds[i][0],:] for i in range(nfold)]
    K_train_ls = [K_train[np.ix_(folds[i][1],folds[i][1])] for i in range(nfold)]
    K_test_ls = [K_train[np.ix_(folds[i][1],folds[i][0])] for i in range(nfold)]

    if inputs.problem == "classification":
        ytrain_ls = [np.squeeze(inputs.train_response.iloc[folds[i][1]].values) for i in range(nfold)]
        ytest_ls = [np.squeeze(inputs.train_response.iloc[folds[i][0]].values) for i in range(nfold)]
        inputs_class = [CVinputs(inputs, ytrain_ls[i], ytest_ls[i]) for i in range(nfold)]
        models = [assist.Classification.PKB_Classification(inputs_class[i], ytrain_ls[i], ytest_ls[i]) for i in range(nfold)]
    elif inputs.problem == 'survival':
        ytrain_ls = [inputs.train_response.iloc[folds[i][1],].values for i in range(nfold)]
        ytest_ls = [inputs.train_response.iloc[folds[i][0],].values for i in range(nfold)]
        inputs_class = [CVinputs(inputs, ytrain_ls[i], ytest_ls[i]) for i in range(nfold)]
        models = [assist.Survival.PKB_Survival(inputs_class[i], ytrain_ls[i], ytest_ls[i]) for i in range(nfold)]
    elif inputs.problem == "regression":
        ytrain_ls = [np.squeeze(inputs.train_response.iloc[folds[i][1]].values) for i in range(nfold)]
        ytest_ls = [np.squeeze(inputs.train_response.iloc[folds[i][0]].values) for i in range(nfold)]
        inputs_class = [CVinputs(inputs, ytrain_ls[i], ytest_ls[i]) for i in range(nfold)]
        models = [assist.Regression.PKB_Regression(inputs_class[i], ytrain_ls[i], ytest_ls[i]) for i in range(nfold)]


    for x in models:
        x.init_F()

    ########## boosting for each fold ###############
    opt_iter = 0
    tmp_list = [x.test_loss[0] for x in models]
    min_loss = prev_loss =  np.mean( [x.test_loss[0] for x in models] )
    ave_loss = [prev_loss]

    print_section('Cross-Validation')
    print("{:>9}\t{:>14}\t{:>24}".format("iteration","Mean test loss","time (if no E-stop)"))

    time0 = time.time()
    for t in range(1,inputs.maxiter+1):
        # one iteration
        for k in range(nfold):
            if inputs.method == 'L2':
                [m,beta,gamma] = oneiter_L2(K_train_ls[k],Ztrain_ls[k],models[k],\
                                Lambda=Lambda,parallel = parallel,group_subset = gr_sub)
            if inputs.method == 'L1':
                [m,beta,gamma] = oneiter_L1(K_train_ls[k],Ztrain_ls[k],models[k],\
                                Lambda=Lambda,parallel = parallel,group_subset = gr_sub)
            # line search
            x = line_search(K_train_ls[k],Ztrain_ls[k],models[k],[m,beta,gamma])
            beta *= x
            gamma *= x

            # update model
            models[k].update([m,beta,gamma],K_train_ls[k][:,:,m],K_test_ls[k][:,:,m],Ztrain_ls[k],Ztest_ls[k],inputs.nu)

        # save iteration
        cur_loss = np.mean([x.test_loss[-1] for x in models])
            #update best loss
        if cur_loss < min_loss:
            min_loss = cur_loss
            opt_iter = t
        ave_loss.append(cur_loss)

        # print report
        if t%10 == 0:
            iter_persec = t/(time.time() - time0) # time of one iteration
            rem_time = (inputs.maxiter-t)/iter_persec # remaining time
            print("{:9.0f}\t{:14.4f}\t{:24.4f}".format(t,cur_loss,rem_time/60))

        # detect early stop
        if t-opt_iter >= ESTOP:
            print('Early stop criterion satisfied: break CV.')
            break

    # print the number of iterations used
    print('using iteration number:',opt_iter)

    # visualization
    if plot:
        folder = inputs.output_folder
        f=plt.figure()
        plt.plot(ave_loss)
        plt.xlabel("iterations")
        plt.ylabel("CV loss")
        f.savefig(folder+'/CV_loss.pdf')
    return opt_iter
