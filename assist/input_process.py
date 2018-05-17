"""
source file for input parameter processing
author: li zeng
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from assist.util import print_section

# test existence of data
def have_file(myfile):
    if not os.path.exists(myfile):
            print("file:",myfile,"does not exist")
            sys.exit(-1)
    else:
        print("reading file:",myfile)

"""
input object:
- parse command line arguments
- import data
"""
class input_obj:
    "class object for input data"
    # data status
    loaded = False

    # type of problem
    problem = None
    hasTest = False # whether has test data
    hasClinical = False # whether has clincal data

    #folder/files
    input_folder = None
    output_folder= None
    group_file = None
    train_predictor_file = None
    train_response_file = None
    clinical_file = None

    # input, output pars
    train_predictors=None
    train_response=None
    pred_sets=None
    train_clinical = None
    train_index = None

    # optional
    test_file = None
    test_predictors = None
    test_response = None
    test_clinical = None
    weights = None
    hasWeights = False

    # input summary
    Ngroup = 0
    Ntrain = 0
    Ntest = 0
    Npred = 0 # number of predictor
    Npred_clin = 0 # number of clinical predictors
    group_names = None
    clin_names = None

    # model pars
    nu = 0.05
    maxiter = 800
    Lambda = None
    kernel = None
    method = None
    pen = 1

    """
    initialize input class with command line arguments
    """
    def __init__(self,args):
        # assign to class objects
        self.problem = args.problem
        self.input_folder = args.input
        self.output_folder = args.output
        self.train_predictor_file = args.predictor
        self.train_response_file = args.response
        self.group_file = args.predictor_set
        self.kernel = args.kernel
        self.method = args.method
        if not args.maxiter is None: self.maxiter= int(args.maxiter)
        if not args.rate is None: self.nu = float(args.rate)
        if not args.Lambda is None: self.Lambda = float(args.Lambda)
        # test file
        if not args.test is None:
            self.test_file = args.test
            self.hasTest = True
        if not args.pen is None: self.pen = float(args.pen)
        # clinical file
        if not args.clinical is None:
            self.clinical_file = args.clinical
            self.hasClinical = True
        # handle weight file
        if not args.weights is None:
            self.hasWeights = True
            self.weights_file = args.weights

    # ██████  ██████   ██████   ██████     ██ ███    ██ ██████  ██    ██ ████████
    # ██   ██ ██   ██ ██    ██ ██          ██ ████   ██ ██   ██ ██    ██    ██
    # ██████  ██████  ██    ██ ██          ██ ██ ██  ██ ██████  ██    ██    ██
    # ██      ██   ██ ██    ██ ██          ██ ██  ██ ██ ██      ██    ██    ██
    # ██      ██   ██  ██████   ██████     ██ ██   ████ ██       ██████     ██

    """
    process input parameters
    - load data
    """
    def proc_input(self):
        """
        load corresponding data
        """
        print_section('LOAD DATA')
        # make output folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # training data
        thisfile = self.input_folder +"/"+ self.group_file
        have_file(thisfile)
        self.pred_sets = pd.Series.from_csv(thisfile)

        thisfile = self.input_folder + "/"+ self.train_predictor_file
        have_file(thisfile)
        self.train_predictors = pd.DataFrame.from_csv(thisfile)

        thisfile = self.input_folder + "/"+ self.train_response_file
        have_file(thisfile)
        self.train_response = pd.DataFrame.from_csv(thisfile)

        # clinical file
        if self.hasClinical:
            thisfile = self.input_folder + "/"+ self.clinical_file
            have_file(thisfile)
            self.train_clinical = pd.DataFrame.from_csv(thisfile)
            self.clin_names = self.train_clinical.columns
        # weights data
        if self.hasWeights:
            thisfile = self.input_folder + "/"+ self.weights_file
            have_file(thisfile)
            raw_weights = pd.read_csv(thisfile,header=None,index_col=0, squeeze=True)
            self.proc_weight(raw_weights)

        # data summary
        self.Ntrain = self.train_predictors.shape[0]
        self.Ngroup = self.pred_sets.shape[0]
        self.Npred = self.train_predictors.shape[1]
        if self.hasClinical:
            self.Npred_clin = self.train_clinical.shape[1]
        self.group_names = self.pred_sets.index

        # change loaded indicator
        self.loaded = True
        return

    """
    process weight file
    raw_weights: pd.Series of (genes, weights); may not be the same genes as in expression data
    output:
        add self.weights = pd.Series of (genes, weights)
    """
    def proc_weight(self,raw_weights):
        new_weights = pd.Series(index = self.train_predictors.columns)
        shared_g = np.intersect1d(new_weights.index, raw_weights.index)
        if len(shared_g) < 10:
            raise Exception("Too few genes from input data are provided weights.")
        else:
            new_weights.loc[shared_g] = raw_weights.loc[shared_g]
            # assign min value in raw_weights for genes not contained in raw_weights
            m = np.min(raw_weights)
            if m <=0:
                raise Exception("need all weights > 0")
            new_weights.loc[new_weights.isnull()] = m
            self.weights = new_weights


    """
    data processing
    - center gene predictor
    - or normalize gene predictors
    - add intercept to clinical data
        (if no clinical data, then self.train_clinical is just one column for intercept)
    """
    def data_preprocessing(self,center = False,norm=False):
        print_section('PROCESS DATA')
        if not self.loaded:
            print("No data loaded. Can not preprocess.")
            return

        # center genomic data
        if center:
            print('Centering data.')
            scale(self.train_predictors,copy=False,with_std=False)

        # normalize data
        if norm:
            print("Normalizing data.")
            scale(self.train_predictors,copy=False,with_mean=False)

        # check groups
        print("Checking groups.")
        to_drop =[]
        for i in range(len(self.pred_sets)):
            genes = self.pred_sets.values[i].split(" ")
            shared = np.intersect1d(self.train_predictors.columns.values,genes)
            if len(shared)==0:
                print("Drop group:",self.pred_sets.index[i])
                to_drop.append(i)
            else:
                self.pred_sets.values[i] = ' '.join(shared)
        if len(to_drop)>0:
            self.pred_sets = self.pred_sets.drop(self.pred_sets.index[to_drop])
            self.group_names = self.pred_sets.index

        # add intercept column to clinical data
        intercept_col = pd.DataFrame( {'intercept':np.ones(self.Ntrain)} , index=self.train_predictors.index)
        if self.hasClinical:
            if self.problem != "survival":
                self.train_clinical = pd.concat([self.train_clinical, intercept_col],axis=1)
        else:
            self.train_clinical = intercept_col

        # calculate summary
        self.Ngroup = len(self.pred_sets)
        return

    """
    split the data into test and train when test_file is present
    """
    def data_split(self):
        if not self.hasTest: return
        print_section('SPLIT DATA')
        print("Using test label: ",self.test_file)
        # load test file
        thisfile = self.input_folder+'/'+self.test_file
        f  = open(thisfile,'r')
        test_ind = [x.strip() for x in f]
        f.close()
        # split data
        self.test_predictors = self.train_predictors.loc[test_ind]
        self.test_response = self.train_response.loc[test_ind]
        self.test_clinical = self.train_clinical.loc[test_ind]
        train_ind = np.setdiff1d(self.train_predictors.index.values,np.array(test_ind))
        self.train_index = train_ind
        self.train_predictors = self.train_predictors.loc[train_ind]
        self.train_response = self.train_response.loc[train_ind]
        self.train_clinical = self.train_clinical.loc[train_ind]

        # update summary
        self.Ntest = len(self.test_response)
        self.Ntrain = len(self.train_response)

    # ██████  ██████  ██ ███    ██ ████████
    # ██   ██ ██   ██ ██ ████   ██    ██
    # ██████  ██████  ██ ██ ██  ██    ██
    # ██      ██   ██ ██ ██  ██ ██    ██
    # ██      ██   ██ ██ ██   ████    ██

    def input_summary(self):
        print_section('SUMMARY')
        print("Analysis type:",self.problem)
        print("input folder:", self.input_folder)
        print("output folder:",self.output_folder)
        print("number of training samples:",self.Ntrain)
        print("number of testing samples:",self.Ntest)
        print("number of pathways:", self.Ngroup)
        print("number of gene predictors:", self.Npred)
        print("number of clinical predictors:", self.Npred_clin)
        return

    def model_param(self):
        print_section('PARAMETERS')
        print("learning rate:",self.nu)
        print("Lambda:",self.Lambda)
        print("maximum iteration:", self.maxiter)
        print("kernel function: ",self.kernel)
        print("method: ",self.method)
        return
