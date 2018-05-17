#PKB instructions

This article provides an instruction for using the **PKB** (Pathway-based Kernel Boosting) model to analyze cancer genomic datasets. PKB is designed to perform classification, regression, or survival analysis on datasets from typical cancer genomic studies. It utilizes cancer patients' **clinical features** (e.g. age, gender, tumor stage, etc), and **gene expression** profile, to predict the outcome variable, which can be **categorical** (e.g. metastasis status), **continuous** (e.g. drug response, tumor size), or **survival** (e.g. overall survival, disease free survival). It also incorporates gene pathway information to improve prediction accuracy, and provides more interpretable results.

This repository is a generalized version of [PKB](https://github.com/zengliX/PKB). The earlier version can only perform classification analysis and cannot use clinical features as predictors. 

### Reference
Zeng, L., Yu, Z., Zhang, Y. and Zhao, H. (2018) A general kernel boosting framework to integrate pathways for cancer genomic data analysis – _Working paper_


### Page Contents

- [About PKB](#pkb)
- [Software requirement](#software)
- [Input Data Preparation](#data)
- [Using PKB](#run)
- [Results Interpretation](#results)

## <a name=pkb></a> About PKB
PKB is a boosting-based method for utilizing pathway information to better predict clinical outcomes. It constructs base learners from each pathway using kernel functions. In each boosting iteration, it identifies the optimal base learner and adds it to the prediction function.

The algorithm has two parts. The first part involves determining an optimal number of iterations using cross validation (CV). In this part, we split the training data into 3-folds, fit the boosting model, and monitor the loss values (averaged over the 3 test data) at each iteration. The iteration (T) with minimum CV loss is used as iteration numbers.

In part two, we use the whole training data to fit the boosting model to T iterations. Figures and tables will be generated to present relevant clinical predictors/pathways, and the boosting procedure.

Please refer to the original paper for more technical details.

## <a name=software></a> Software requirement   
The program is written in **Python3**. Please make sre the following packages are properly installed before running the program: 

- pandas
- numpy
- scipy
- matplotlib
- pickle
- argparse
- sklearn


## <a name=data></a> Input data preparation
PKB requires the input datasets to be formatted in certain ways. 

### Outcome variable file
Please refer to `example/*/response.txt` for examples of response file for different problems. It needs to be a comma-separated file with two columns (for regression and classification) or three columns (for survival). 

- Example for classification:

  sample | response 
  ------- | --------- 
  sample1 | 1 
  sample2 | -1
  sample3 | -1   
  ...     | ... 

- Example for regression:

  sample | response 
  ------- | --------- 
  sample1 | 2.33
  sample2 | 1.85
  sample3 | 0.51 
  ...     | ... 
  
- Example for survival analysis:

  sample | survival | censor
  ------- | ------- | ----------- 
  sample1 | 25.5  | 1
  sample2 | 50.3  | 0
  sample3 | 3.8   | 0
  ...     | ...   | ...

In the survival file, the `survival` column represent survival times of the patients (using months as unit recommended). The `cencor` column equals to `0` is an endpoint event is observe, otherwise `1`. 

### Gene expression input
Please refer to `example/*/expression.txt` for an example. It is also a comma-separated file. The first column is sample ID, and the other columns are genes. The first row will be used as header, and each other row represents one sample.

Example:

| sample  | gene1 | gene2 | gene3 | gene4 | ... |
|---------|-------|-------|-------|-------|-----|
| sample1 | 1.2   | 3.3   | 4.5   | 0.1   | ... |
| sample2 | 0.5   | 2.6   | 2.3   | 1.2   | ... |
| sample3 | 0.1   | 1.4   | 0.1   | 2.2   | ... |
| sample4 | 0.8   | 0.2   | 8.6   | 1.8   | ... |
| ...     | ...   | ...   | ...   | ...   | ... |

### Clinical predictor input
The format for clinical predictors is the same as in gene expression input. Please check out `example/*/clinical.txt` for examples.

Example:

| sample  | var0 | var1 | var2 | var3 | ... |
|---------|-------|-------|-------|-------|-----|
| sample1 | 1   | 0   | 35   | 168   | ... |
| sample2 | 0   | 0   | 58   | 190   | ... |
| sample3 | 1   | 0   | 77   | 177   | ... |
| sample4 | 1   | 1   | 63   | 175   | ... |
| ...     | ...   | ...   | ...   | ...   | ... |

### Pathway input
You can either provide your own pathway file, or use the built-in files, including  **KEGG, Biocarta, GO Biological Process pathways**. 

To use the built-in pathways, just use the corresponding files in `./pathways` folder when running PKB.

If you would like to use customized pathway file, please refer to `example/*/pathways.txt` for an example. It should be a comma-separated file with no header. The first column are the names of pathways, and the second column are the lists of individual pathway members. Each list is a string of genes separated by spaces.

Example:

  pathway| genes (do not include this row in your file)
  ------- | --------- 
  pathway1 | gene11 gene12 gene13 gene14 
  pathway2 | gene21 gene22
  pathway3 | gene31 gene32 gene33
  pathway4 | gene41 gene42
  ...     | ... 

## <a name=run></a> Running PKB
Follow the steps below in order to run PKB on your own computer (we use our toy dataset as example):

1. clone this git repository :

	```bash
	git clone https://github.com/zengliX/PKB2 PKB2
	cd PKB2
	```
2. prepare datasets and configuration files following the format given in the previous section

3. run PKB: 

	```python
	# python PKB.py path/to/your_config_file.txt
	python PKB.py ./example/config_file.txt
	```

The outputs will be saved in the `output_folder` as you specified in the configuration file.

## <a name=results></a> Results interpretation

### Figures
1. `CV_err.png, CV_loss.png`:    
present classifcation error and loss function value at each iteration of the cross validation process
![](example/example_output/CV_err.png?raw=true)
![](example/example_output/CV_loss.png?raw=true)


2. `opt_weights.png`:    
shows the estimated pathways weights fitted using our boosting model
![](example/example_output/opt_weights.png?raw=true)


3. `weights_path.png`:    
shows the changes of pathways' weights as iteration number increases.
![](example/example_output/weights_path.png?raw=true)


### Tables
1. `opt_weights.txt`:    
a table showing the optimal weights of all pahtways. It is sorted in descending order. The first column are pathways, and the second column are correponding weights.

2. `test_prediction.txt`:   
the predicted outcome values, if `test_file` is provided in the configuration file.

### Pickle file
1. `results.pckl`:   
contains information of the whole boosting process. You can recover the prediction function at every step from this file.


## Contact 
Please feel free to contact <li.zeng@yale.edu> if you have any questions.
