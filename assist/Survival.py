from assist.Model import BaseModel
from assist.util import undefined
from assist.util import testInf
import numpy as np
import numpy.linalg as npl
from scipy.linalg import sqrtm

# censor = 1 means censor

class PKB_Survival(BaseModel):
    def __init__(self,inputs,ytrain,ytest):
        super().__init__(inputs,ytrain,ytest)
        # for survival only
        self.problem = 'survival'
        self.ytrain_time = self.ytrain[:,0]
        self.ytrain_cen = self.ytrain[:,1]
        self.ytrain_delta = 1 - self.ytrain[:,1]
        self.ytrain_tau = np.zeros((self.Ntrain, self.Ntrain))
        self.train_Cind = []
        for i in range(self.Ntrain):
            self.ytrain_tau[i,:] = self.ytrain_time >= self.ytrain_time[i]
        # if test data exists
        if self.hasTest:
            self.ytest_time = self.ytest[:,0]
            self.ytest_cen = self.ytest[:,1]
            self.ytest_delta = 1 - self.ytest[:,1]
            self.ytest_tau = np.zeros((self.Ntest, self.Ntest))
            self.test_Cind = []
            for i in range(self.Ntest):
                self.ytest_tau[i,:] = self.ytest_time >= self.ytest_time[i]

    """
    initialize survival model
    """
    def init_F(self):
        # initial value
        self.F0 = 0.0
        # update training loss, c-index ...
        self.F_train = np.repeat(self.F0, self.Ntrain)
        self.exp_ftrain = np.exp(self.F_train)
        self.exp_indicate = np.dot(self.ytrain_tau, self.exp_ftrain)
        self.train_loss.append(self.loss_fun_train(self.F_train))
        self.train_Cind.append(self.C_index(self.ytrain, self.F_train))
        # update testing loss, c-index ...
        if self.hasTest:
            self.F_test = np.repeat(self.F0,self.Ntest)
            self.exp_ftest = np.exp(self.F_test)
            self.test_loss.append(self.loss_fun_test())
            self.test_Cind.append(self.C_index(self.ytest, self.F_test))
        else:
            self.F_test = None
            self.exp_ftest = None
        # update trace
        self.trace.append([0,np.repeat(0.0,self.Ntrain),np.repeat(0.0,self.Npred_clin)])
        self.fraction_matrix = self.exp_ftrain*np.dot(self.ytrain_tau.T, (self.ytrain_delta/self.exp_indicate))

    """
    calculate first order derivative
    return gradient, shape (Ntrain,)
    """
    def calcu_h(self):
        return -self.ytrain_delta+self.fraction_matrix

    """
    calculate second order derivative
    return hessian matrix, shape (Ntrain, Ntrain)
    """
    def calcu_q(self):
        temp = self.ytrain_delta/self.exp_indicate*self.ytrain_tau.T
        mat = np.matrix(self.exp_ftrain)
        Q = np.diag(self.fraction_matrix) - np.multiply(np.dot(mat.T, mat),np.dot(temp,temp.T))
        return np.array(Q)

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
        # update training info
        self.F_train += ( K.dot(beta) + Z.dot(gamma) )*rate
        self.exp_ftrain = np.exp(self.F_train)
        self.exp_indicate = np.dot(self.ytrain_tau, self.exp_ftrain)
        self.fraction_matrix = self.exp_ftrain*np.dot(self.ytrain_tau.T, (self.ytrain_delta/self.exp_indicate))
        self.train_loss.append(self.loss_fun_train(self.F_train))
        self.train_Cind.append(self.C_index(self.ytrain, self.F_train))
        # update testing info
        if self.hasTest:
            self.F_test += (K1.T.dot(beta)+ Z1.dot(gamma) )*rate
            self.exp_ftest = np.exp(self.F_test)
            self.test_loss.append(self.loss_fun_test())
            self.test_Cind.append(self.C_index(self.ytest, self.F_test))

    """
    survival loss function (only on training data), negative log-likelihood
    will use precomputed values; make sure F_train-related values are updated
    f: np.array of shape (Ntrain,)
    """
    def loss_fun_train(self,f):
        if id(f) == id(self.F_train):
            expf = self.exp_ftrain
        else:
            expf = np.exp(f)
        res = -self.ytrain_delta*(f - np.log(self.ytrain_tau.dot(expf)))
        return np.mean(res)

    """
    survival loss function (only on testing data), negative log-likelihood
    will use precomputed values; make sure F_test-related values are updated
    f: np.array of shape (Ntest,)
    """
    def loss_fun_test(self):
        res = -self.ytest_delta*(self.F_test - np.log(self.ytest_tau.dot(self.exp_ftest)))
        return np.mean(res)

    """
    C-index (concordance index) function
    y: np.array of shape (Ntrain,2)
    f: np.array of shape (Ntrain,)
    """
    def C_index(self,y,f):
        ct_pairs = 0.0 # count concordant pairs
        ct=  0 # count total pairs
        for i in range(len(f)-1):
            for j in range(i+1,len(f)):
                val =  self.concordant(y[i,0],y[j,0],y[i,1],y[j,1],f[i],f[j])
                if val is None: continue
                ct_pairs += val
                ct += 1
        return ct_pairs/ct

    def concordant(self,y1,y2,d1,d2,f1,f2):
        """
        y1,y2: survival time
        d1,d2: censoring
        f1,f2: predicted risk
        return None for non-permissible
        """
        # non-permissible
        if (y1<y2 and d1 == 1) or (y2<y1 and d2 == 1):
            return None
        if y1 == y2 and d1==d2==1:
            return None
        # permissible
        if y1 != y2:
            if f1 == f2: return 0.5
            return (1 if (y1>y2) == (f1<f2) else 0)
        else:
            # y1 == y2
            if d1==d2==0:
                return (1 if f1==f2 else 0.5)
            else:
                return (0.5 if f1==f2 else 1 if (d1>d2)==(f2>f1) else 0)

    """
    calculate eta，W, W^(1/2) from h and q
    h: gradient, shape (Ntrain,)
    q: Hessian, shape (Ntrain, Ntrain)
    """
    def calcu_eta(self,h,q):
        u, s, vh = npl.svd(q)
        abss = np.abs(s)
        med = np.median(abss)
        s = s[abss>=0.01*med]
        u = u[:,abss>=0.01*med]
        vh = vh[abss>=0.01*med,:]
        S = np.diag(1/s)
        return np.dot(np.dot(u, np.dot(S, vh)), h)

    def calcu_w(self,q):
        return q/2

    def calcu_w_half(self,q):
        u, s, vh = npl.svd(q/2)
        S = np.diag(np.sqrt(s))
        return np.dot(u, np.dot(S, vh))
