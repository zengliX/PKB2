"""
make prediction for new data
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("pckl", help="the .pckl file saved from fitting of PKB model")
parser.add_argument("new_genomic", help="file for gene expression for new data")
parser.add_argument("-new_clinical", help="file for clinical features for new data")
args = parser.parse_args()

import pickle
import pandas as pd
import numpy as np
import assist
from sklearn.preprocessing import scale

"""
load data
"""
# model file
with open(args.pckl,'rb') as f:
    saved = pickle.load(f)

# old genomic file
old_predictor_file = '/'.join( [saved.inputs.input_folder, saved.inputs.train_predictor_file] )
print(old_predictor_file)
old_genom = pd.DataFrame.from_csv(old_predictor_file)
if saved.model.hasTest:
        old_genom = old_genom.loc[saved.inputs.train_index]

# new genomic file
new_genom = pd.DataFrame.from_csv(args.new_genomic)
ids = new_genom.index # index for patients

# new clinical file
intercept_col = pd.DataFrame( {'intercept':np.ones(new_genom.shape[0])} , index=ids)
if args.new_clinical:
    new_clin = pd.DataFrame.from_csv(args.new_clinical)
    new_clin = pd.concat([new_clin, intercept_col],axis=1)
else:
    new_clin = intercept_col

"""
center data and calculate kernel
"""
# center data
old_mean = old_genom.mean()
scale(old_genom,copy=False,with_std=False)
new_genom = new_genom - old_mean

# calculate new kernel matrix
Knew = assist.kernel_calc.get_kernels(new_genom,old_genom,saved.inputs)

"""
make predictions
"""
[f1, f2] = saved.weights_path()
beta, gamma = saved.coef_mat, saved.coef_clinical
pred = saved.model.predict(beta,gamma,Knew,new_clin)
print("predicted values:\n")
print(pred)
