# -*- coding: utf-8 -*-
"""
main program
author: li zeng
"""

#  █████  ██████   ██████  ███████
# ██   ██ ██   ██ ██       ██
# ███████ ██████  ██   ███ ███████
# ██   ██ ██   ██ ██    ██      ██
# ██   ██ ██   ██  ██████  ███████

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("problem", help="type of analysis (classification/regression/survival)")
parser.add_argument("input", help="Input folder")
parser.add_argument("output", help="Output folder")
parser.add_argument("predictor", help="predictor file")
parser.add_argument("predictor_set",help="file that specifies predictor group structure")
parser.add_argument("response",help="outcome data file")
parser.add_argument("kernel",help="kernel function (rbf/poly3)")
parser.add_argument("method",help="regularization (L1/L2)")
parser.add_argument("-clinical",help="file of clinical predictors")
parser.add_argument("-maxiter",help="maximum number of iteration (default 800)")
parser.add_argument("-rate",help="learning rate parameter (default 0.05)")
parser.add_argument("-Lambda",help="penalty parameter")
parser.add_argument("-test",help="file containing test data index")
parser.add_argument("-pen",help="penalty multiplier")
parser.add_argument("-weights",help="file with gene weights")
args = parser.parse_args()

# ██ ███    ███ ██████   ██████  ██████  ████████
# ██ ████  ████ ██   ██ ██    ██ ██   ██    ██
# ██ ██ ████ ██ ██████  ██    ██ ██████     ██
# ██ ██  ██  ██ ██      ██    ██ ██   ██    ██
# ██ ██      ██ ██       ██████  ██   ██    ██
import numpy as np
import matplotlib
matplotlib.use('Agg')
import assist
from assist.cvPKB import CV_PKB
from assist.method_L1 import oneiter_L1, find_Lambda_L1
from assist.method_L2 import oneiter_L2, find_Lambda_L2
import time


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

if __name__ == "__main__":
    inputs = assist.input_process.input_obj(args)
    inputs.proc_input()
    # process data
    inputs.data_preprocessing(center=True)
    # split to test, train
    inputs.data_split()
    # report
    inputs.input_summary()
    inputs.model_param()

    # ██████  ██████  ███████ ██████
    # ██   ██ ██   ██ ██      ██   ██
    # ██████  ██████  █████   ██████
    # ██      ██   ██ ██      ██
    # ██      ██   ██ ███████ ██

    """---------------------------
    CALCULATE KERNEL; GET CLINICAL MATRIX
    ----------------------------"""
    #importlib.reload(assist.kernel_calc)
    K_train = assist.kernel_calc.get_kernels(inputs.train_predictors,inputs.train_predictors,inputs, inputs.weights)
    Z_train = inputs.train_clinical.values
    if inputs.hasTest:
        K_test= assist.kernel_calc.get_kernels(inputs.train_predictors,inputs.test_predictors,inputs, inputs.weights)
        Z_test = inputs.test_clinical.values

    """---------------------------
    initialize model object
    ----------------------------"""
    if inputs.problem == "classification":
        ytrain = np.squeeze(inputs.train_response.values)
        ytest = np.squeeze(inputs.test_response.values) if inputs.hasTest else None
        model = assist.Classification.PKB_Classification(inputs,ytrain,ytest)
        model.init_F()
    elif inputs.problem == "regression":
        ytrain = np.squeeze(inputs.train_response.values)
        ytest = np.squeeze(inputs.test_response.values) if inputs.hasTest else None
        model = assist.Regression.PKB_Regression(inputs,ytrain,ytest)
        model.init_F()
    elif inputs.problem == "survival":
        ytrain = np.squeeze(inputs.train_response.values)
        ytest = np.squeeze(inputs.test_response.values) if inputs.hasTest else None
        model = assist.Survival.PKB_Survival(inputs,ytrain,ytest)
        model.init_F()
    else:
        print("Analysis ",inputs.problem," not supported"); exit(-1)

    # ██████   ██████   ██████  ███████ ████████ ██ ███    ██  ██████
    # ██   ██ ██    ██ ██    ██ ██         ██    ██ ████   ██ ██
    # ██████  ██    ██ ██    ██ ███████    ██    ██ ██ ██  ██ ██   ███
    # ██   ██ ██    ██ ██    ██      ██    ██    ██ ██  ██ ██ ██    ██
    # ██████   ██████   ██████  ███████    ██    ██ ██   ████  ██████

    """---------------------------
    BOOSTING PARAMETERS
    ----------------------------"""
    # use seed for reproducibility
    np.random.seed(1)
    Lambda = inputs.Lambda

    # automatic selection of Lambda
    if inputs.method == 'L1' and Lambda is None:
        Lambda = find_Lambda_L1(K_train,Z_train,model)
        Lambda *= inputs.pen
        print("L1 method: use Lambda",Lambda)
    if inputs.method == 'L2' and Lambda is None:
        Lambda = find_Lambda_L2(K_train,Z_train,model)
        Lambda *= inputs.pen
        print("L2 method: use Lambda",Lambda)

    # use random groups when sample size or group number are large
    if (inputs.Ntrain > 500 or inputs.Ngroup > 40):
        parallel = False
        gr_sub = True
        print("Algorithm: random groups selected in each iteration")
    else:
        parallel = False
        gr_sub = False

    ESTOP = 50 # early stop if test_loss have no increase


    """---------------------------
    CV FOR NUMBER OF ITERATIONS
    ----------------------------"""

    opt_iter = CV_PKB(inputs,K_train,Lambda,nfold=3,ESTOP=ESTOP,\
                      parallel=parallel,gr_sub=gr_sub,plot=True)

    """---------------------------
    BOOSTING ITERATIONS
    ----------------------------"""

    time0 = time.time()
    assist.util.print_section("BOOSTING")
    print("iteration\ttrain loss\ttest loss\t    time")
    for t in range(1,opt_iter+1):
        # one iteration
        if inputs.method == 'L2':
            [m,beta,gamma] = oneiter_L2(K_train,Z_train,model,Lambda=Lambda,\
                        parallel = parallel,group_subset = gr_sub)
        if inputs.method == 'L1':
            [m,beta,gamma] = oneiter_L1(K_train,Z_train,model,\
                    Lambda=Lambda,parallel = parallel,group_subset = gr_sub)

        # line search
        x = assist.util.line_search(K_train,Z_train,model,[m,beta,gamma])
        beta *= x
        gamma *= x

        # update model parameters

        if model.hasTest:
            model.update([m,beta,gamma],K_train[:,:,m],K_test[:,:,m],Z_train,Z_test,inputs.nu)
        else:
            model.update([m,beta,gamma],K_train[:,:,m],None,Z_train,None,inputs.nu)
        # print time report
        if t%10 == 0:
            iter_persec = t/(time.time() - time0) # time of one iteration
            rem_time = (opt_iter-t)/iter_persec # remaining time
            if model.hasTest:
                print("%9.0f\t%10.4f\t%9.4f\t%8.4f" % \
                  (t,model.train_loss[t],model.test_loss[t],rem_time/60))
            else:
                print("%9.0f\t%10.4f\t---------\t%8.4f" % \
                  (t,model.train_loss[t],rem_time/60))
    print()

    # ██████  ███████ ███████ ██    ██ ██   ████████ ███████
    # ██   ██ ██      ██      ██    ██ ██      ██    ██
    # ██████  █████   ███████ ██    ██ ██      ██    ███████
    # ██   ██ ██           ██ ██    ██ ██      ██         ██
    # ██   ██ ███████ ███████  ██████  ███████ ██    ███████

    outputs = assist.outputs.output_obj(model,inputs)
    ## show results
    # trace
    if inputs.problem == "classification":
        f = outputs.show_err()
        f.savefig(inputs.output_folder + "/err.pdf")
    if inputs.problem == "survival":
        f = outputs.show_Cind()
        f.savefig(inputs.output_folder + "/Cindex.pdf")
    f = outputs.show_loss()
    f.savefig(inputs.output_folder + "/loss.pdf")

    # final weights
    [G_weights, C_weights] = outputs.weights_timeT(opt_iter-1)
    f0 = outputs.plot_group_weights(G_weights)
    f0.savefig(inputs.output_folder + "/group_weights.pdf")
    if inputs.hasClinical:
        f1 = outputs.plot_clinical_weights(C_weights)
        f1.savefig(inputs.output_folder + "/clinical_weights.pdf")

    # weights paths
    [f2,f3] = outputs.weights_path()
    f2.savefig(inputs.output_folder + "/group_weights_path.pdf")
    if inputs.hasClinical:
        f3.savefig(inputs.output_folder + "/clinical_weights_path.pdf")


    # ███████  █████  ██    ██ ███████
    # ██      ██   ██ ██    ██ ██
    # ███████ ███████ ██    ██ █████
    #      ██ ██   ██  ██  ██  ██
    # ███████ ██   ██   ████   ███████

    # save outputs to files
    import pickle
    out_file = inputs.output_folder + "/results.pckl"
    f = open(out_file,'wb')
    outputs.clean_up()
    pickle.dump(outputs,f)
    f.close()
    print("results saved to:",out_file)
