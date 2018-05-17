from assist.Model import BaseModel
from assist.util import undefined
import numpy as np

class PKB_Regression(BaseModel):
    def __init__(self, inputs, ytrain, ytest):
        super().__init__(inputs, ytrain, ytest)
        # for regression only
        self.inputs = inputs
        self.problem = 'regression'
        self.F_train = []
        self.F_test = []
        self.train_err = [] # training error
        self.test_err = [] # testing error
    """
    initialize regression model
    """
    def init_F(self):
        # add training loss[0]
        self.F0 = np.mean(self.ytrain)
        self.F_train = np.repeat(self.F0,self.Ntrain)
        self.train_loss.append(self.loss_fun(self.ytrain,self.F_train))
        # add testing loss[0]
        if self.hasTest:
            self.F_test = np.repeat(self.F0,self.Ntest)
            self.test_loss.append(self.loss_fun(self.ytest,self.F_test))
        else:
            self.F_test = None
        # add trace[0]
        self.trace.append([0,np.repeat(0,self.Ntrain),\
        np.append( np.repeat(0,self.Npred_clin),[self.F0] )])

    """
    calculate eta, negative residual
    eta = -r in note
    shape (Ntrain,)
    """
    def calcu_eta(self):
        rt = self.ytrain - self.F_train
        return -rt

    """
    regression loss function, MSE
    y: np.array of shape (N,)
    f: np.array of shape (N,)
    """
    def loss_fun(self,y,f):
        return np.mean( (y-f)**2 )


    """
    calculate first order derivative
    return gradient, shape (Ntrain,)
    """
    def calcu_h(self):
        return None

    """
    calculate second order derivative
    return hessian matrix, shape (Ntrain, Ntrain)
    """
    def calcu_q(self):
        return None
