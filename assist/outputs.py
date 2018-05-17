# -*- coding: utf-8 -*-
"""
The outputs class
author: li zeng
"""

import numpy as np
from matplotlib import pyplot as plt

def weight_calc(mat):
    weights = []
    for j in range(mat.shape[1]):
        weights.append( np.sqrt((mat[:,j]**2).sum()) )
    return weights


"""
output object:
- with model as a member
- has visualization function to interact with model
"""
class output_obj:
    # initialization
    """
    coef_mat: beta coefficients for pathways, shape (Ntrain, Ngroup)
    coef_clinical: alpha coefficients for clinical variables and intercept, shape (Npred_clin+1,)
    """
    def __init__(self,model,inputs):
        self.inputs = inputs
        self.model = model
        self.coef_mat =  np.zeros([inputs.Ntrain,inputs.Ngroup])
        self.coef_clinical = None
        return

    """
    CLASSIFICATION ONLY
    show the trace of classification error
    """
    def show_err(self):
        f = plt.figure()
        plt.plot(self.model.train_err,'b')
        plt.text(len(self.model.train_err),self.model.train_err[-1], "training error")
        if self.model.hasTest:
            plt.plot(self.model.test_err,'r')
            plt.text(len(self.model.test_err),self.model.test_err[-1], "testing error")
        plt.xlabel("iterations")
        plt.ylabel("classification error")
        plt.title("Classifiction errors in each iteration")
        return f

    """
    SURVIVAL ONLY
    show the trace of survival prediction Cindex
    """
    def show_Cind(self):
        f = plt.figure()
        plt.plot(self.model.train_Cind,'b')
        plt.text(len(self.model.train_Cind),self.model.train_Cind[-1], "training C-index")
        if self.model.hasTest:
            plt.plot(self.model.test_Cind,'r')
            plt.text(len(self.model.test_Cind),self.model.test_Cind[-1], "testing C-index")
        plt.xlabel("iterations")
        plt.ylabel("C-index")
        plt.title("Survival prediction C-index in each iteration")
        return f

    """
    show the trace plot of loss function
    """
    def show_loss(self):
        f = plt.figure()
        plt.plot(self.model.train_loss,'b')
        plt.text(len(self.model.train_loss),self.model.train_loss[-1], "training loss")
        if self.model.hasTest:
            plt.plot(self.model.test_loss,'r')
            plt.text(len(self.model.test_loss),self.model.test_loss[-1], "testing loss")
        plt.xlabel("iterations")
        plt.ylabel("loss function")
        plt.title("Loss function at each iteration")
        return f

    """
    return group,clinical weights at iteration t
    """
    def weights_timeT(self,t):
        self.coef_mat.fill(0)
        self.coef_clinical = self.model.trace[0][2].copy()

        # calculate coefficient matrix at step t
        for i in range(1,t+1):
            [m,beta,gamma] = self.model.trace[i]
            self.coef_mat[:,m] += beta*self.inputs.nu
            self.coef_clinical += gamma*self.inputs.nu

        # calculate pathway weights
        weights = weight_calc(self.coef_mat)
        return [weights,self.coef_clinical]

    """
    plot group,clinical weights at iteration t
    """

    def plot_group_weights(self,weights):
        f=plt.figure()
        plt.bar(range(1,self.inputs.Ngroup+1),weights)
        plt.xlabel("groups")
        plt.ylabel("group weights")
        return f

    def plot_clinical_weights(self,weights):
        f=plt.figure()
        if self.model.problem == 'survival':
            labels = self.inputs.train_clinical.columns
        else:
            labels = self.inputs.train_clinical.columns[:-1]
            weights = weights[:-1]
        x = range(len(labels))
        plt.bar(x,weights)
        plt.xticks(x, labels)
        plt.xlabel("clinical variables")
        plt.ylabel("clinical coefficients")
        return f

    """
    show the path of weights for each group
    """
    def weights_path(self):
        self.coef_mat.fill(0)
        self.coef_clinical = self.model.trace[0][2].copy()
        # calculate coefficient matrix at step t
        weight_mat = np.zeros([len(self.model.train_loss),self.inputs.Ngroup])
        weight_mat_clin = np.zeros([len(self.model.train_loss),self.inputs.Npred_clin])

        # calculate weights for each iteration
        for i in range(1,len(self.model.train_loss)):
            [m,beta,gamma] = self.model.trace[i]
            self.coef_mat[:,m] += beta*self.inputs.nu
            self.coef_clinical += gamma*self.inputs.nu
            # update group weights
            weight_mat[i,:] = weight_mat[i-1,:]
            weight_mat[i,m] =  np.sqrt((self.coef_mat[:,m]**2).sum())
            # update clinical weights
            if self.inputs.problem == 'survival' and self.inputs.hasClinical:
                weight_mat_clin[i,:] = self.coef_clinical
            else:
                weight_mat_clin[i,:] = self.coef_clinical[:-1]

        # weights in last iteration
        first5 = weight_mat[-1,:].argsort()[-5:][::-1]
        first5_clin = weight_mat_clin[-1,:].argsort()[-5:][::-1]

        # visualization
        f1=plt.figure()
        for m in range(weight_mat.shape[1]):
            if m in first5:
                plt.plot(weight_mat[:,m],label=str(self.inputs.group_names[m]))
            else:
                plt.plot(weight_mat[:,m])
        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel("weights")
        plt.title("pathway weights trace")

        if self.inputs.hasClinical:
            Cnames = self.inputs.train_clinical.columns[:self.inputs.Npred_clin]
            f2=plt.figure()
            for m in range(weight_mat_clin.shape[1]):
                if m in first5_clin:
                    plt.plot(weight_mat_clin[:,m],label=str(Cnames[m]))
                else:
                    plt.plot(weight_mat_clin[:,m])
            plt.legend()
            plt.xlabel("iterations")
            plt.ylabel("weights")
            plt.title("clinical variable weights trace")
        else:
            f2 = None
        return [f1,f2]

    """
    clean up big data information before being pickled
    """
    def clean_up(self):
        self.inputs.test_predictors =None
        self.inputs.train_predictors = None
        self.inputs.test_response=None
        self.inputs.train_response=None
