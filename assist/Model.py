"""
Base model class
author: li zeng
"""

import numpy as np

"""
Model class is used to
- initialize parameters
- keep track of each iteration in boosting
- at the end of program, will be used for visualization
- only the outcome (ytrain, ytest) and predicting function (Ftrain, Ftest) will be stored
- other parameters kernel matrix, Z matrix ... will be passed in as function arguments
"""
class BaseModel:
    """
    model initialization
    trace: increment parameters in each boosting iteration
    train_loss: training loss of each iteration
    test_loss: testing loss of each iteration
    hasTest: bool, indicating existence of test data
    """
    def __init__(self,inputs,ytrain,ytest):
        self.ytrain = ytrain
        self.ytest = ytest
        self.Ntrain = inputs.Ntrain
        self.Ntest = inputs.Ntest
        self.hasTest = inputs.hasTest
        self.hasClinical = inputs.hasClinical
        self.Npred_clin = inputs.Npred_clin
        self.F0 = None

        # tracking of performance
        self.trace = []  # keep track of each iteration
        self.train_loss = [] # loss function at each iteration
        self.test_loss = []

        # time consumption
        #self.TIME1 = 0
        #self.TIME2 = 0
        #self.TIME3 = 0

    """
    initialize F_train, ytrain, F_test, ytest
    """
    def init_F(self):
        self.F_train = []
        self.F_test = []

    """
    update [F_train, F_test, trian_loss, test_loss] after
    calculation of [m,beta,c] in each iteration
    pars: [m, beta, c]
    K: training kernel matrix
    K1: testing kernel matrix
    Z: training clinical matrix
    Z1: testing clinical matrix
    rate: learning rate parameter
    """
    def update(self,pars,K,K1,Z,Z1,rate):
        m,beta,gamma = pars
        self.trace.append([m,beta,gamma])
        self.F_train += ( K.dot(beta) + Z.dot(gamma) )*rate
        self.train_loss.append(self.loss_fun(self.ytrain,self.F_train))
        if self.hasTest:
            self.F_test += (K1.T.dot(beta)+ Z1.dot(gamma) )*rate
            new_loss = self.loss_fun(self.ytest,self.F_test)
            self.test_loss.append(new_loss)

    """
    make predictions (of F) using new data
    beta: estimated beta (Ntrain, Ngroup)
    gamma: estimated gamma (Ntrain,)
    Knew: (Nnew, Ntrain, Ngroup)-shape new kernel matrix
    Znew: (Nnew, Npred_clin)-shape new clinical matrix
    """
    def predict(self,beta,gamma,Knew,Znew):
        out = Znew.dot(gamma)
        for i in range(Knew.shape[2]):
            out += Knew[:,:,i].dot(beta[:,i])
        return out

    """------------------------------------
    other functions to be added in child classes

    #for classification and survival:

    #first order derivative
    def calcu_q(self):
        pass

    #second order derivative
    def calcu_h(self):
        pass

    #calculate eta，W, W^(1/2) from h and q

    def calcu_eta(self,h,q):
        pass

    def calcu_w(self,q):
        pass

    def calcu_w_half(self,q):
        pass

    #loss function
    def loss_fun(self,y,f):
        pass
    ---------------------------------------"""
