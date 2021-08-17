########################################################################################
# Project 1 README file
########################################################################################

This project will look at performances that can be reached with typical deep architectures 
as well as the influence of some key strategies for such a classification task. 
In particular, the experiments are designed to assess performance improvements achieved 
through weight sharing and auxiliary losses. Additionally, one architecture (namely LeNet5) 
is then further analysed to assess performance improvements obtained with more specific 
techniques such as batch normalization and dropout. 

This file is intended to clarify the structure by which the files are organised 
in the project folder.

|----- models.py:
|    This script contains the 4 models on which we are going to test techniques as
|    auxiliary losses, weight sharing, batch normalization or dropout. 
|
|----- train_test_functions.py:
|    It contains the training and testing functions used for miniproject 1, 
|    as well as functions for loading and shuffling data.
|
|----- setting_learning_rate.py:
|    File used to determine the best learning rate for each architecture. 
|    It runs with auxiliary loss, weight sharing and batch normalization enabled. 
|
|    At the end, it logs the results' tensor and stores it on stored_results/lr_results.pt.
|
|    This file takes a lot to run as it needs to execute so many different configurations.
|
|----- test.py:
|    This file contains the main demo of our project. On it, we first load normalized data, 
|    and then we set the required parameters before training and testing the model.
|
|    For this file, we only take into account LeNet5 model, and we test it with bacth norm enabled,
|    comparing the performance between disabling/enabling auxiliary loss and weight sharing.
|
|    We run each possible configuration 10 times, each of them having a different seed, so we can
|    ensure that data order in training and weight initialization are not biasing our results. Using
|    our own predefined seeds also ensures reproducibility, so each time this file is executed,
|    the same results are observed. 
|
|    During the execution, the mean error on test target for each configuration is logged.
|
|----- grid_search_models.py:
|    We run a loop performing a grid search over all possible configurations of models and
|    auxiliary losses and weight sharing enabled/disabled. Batch normalization is enabled
|    always.
|
|    For each configuration we print its results, and at the end we save two files
|    (one for executions with auxiliary losses, other for the ones without them) on 
|    stored_results/test.pt and stored_results/test_no_aux.pt.
|
|    This file takes a lot to run as it needs to execute so many different configurations.
|
|----- lenet_depth_analysis.py:
|    File to run a deep analysis over LeNet5. 
|    We only make this analysis with auxiliary loss enabled, as it always gives the best results 
|    by far. We try all possible combinations of batch normalization, weight sharing and dropout
|    enabled/disabled. 
|
|    As usual, we run each configuration for 10 different seeds and we take the mean.
|
|    We log the intermediate results and we save a tensor with the final ones on 
|    stored_results/lenet_deep_analysis_results.pt. 
|
|    This file takes a lot to run as it needs to execute so many different configurations.
|
|----- plotting.ipynb:
|    This file is used to, given the results in lr_results.pt, plot them to choose the best 
|    learning rate for each model. It stores the resulting plot in stored_results/lr_plot.png
|
|----- stored_results/
        |----- lr_results.txt:
        |        Console log of executing setting_learning_rate.py
        |
        |----- lr_results.pt:
        |        This file is a dataframe file. It contains the results of the execution of 
        |        setting_learning_rate.py. It has the following shape:
        |
        |        [4, 6, 100]
        |
        |        4 is the number of models, ["Net", "LeNet5", "LeNet5_FullyConv", "ResNet"] 
        |        6 is the number of etas being compared, [3e-1, 5e-2, 3e-2, 5e-3, 3e-3, 5e-4]
        |        100 is the number of epochs that we run for each possible configuration
        |
        |----- lr_plot.png:
        |        Plot representing what is stored on lr_results.pt
        |
        |----- grid_search_results.txt:
        |        Console log of executing grid_search_models.py
        |
        |----- test.pt:
        |        It contains the results of the configurations using auxiliary loss when executing 
        |        grid_search_models.py. Used in Table I on the report. It has the following shape:
        |
        |        [2, 1, 4, 2, 8]
        |
        |        2 represents weight sharing disabled/enabled
        |        1 represents batch normalization enabled
        |        4 is the number of models, ["Net", "LeNet5", "LeNet5_FullyConv", "ResNet"] 
        |        2 is the mean and the std dev
        |        8 are all the metrics we are storing for each possible configuration. (see comments
        |        on code to understand these metrics better)
        |
        |----- test_no_aux.pt:
        |        It contains the results of the configurations using auxiliary loss when executing 
        |        grid_search_models.py. Used in Table I on the report. It has the following shape:
        |
        |        [2, 1, 4, 2, 4]
        |
        |        2 represents weight sharing disabled/enabled
        |        1 represents batch normalization enabled
        |        4 is the number of models, ["Net", "LeNet5", "LeNet5_FullyConv", "ResNet"] 
        |        2 are the mean and the standard deviation
        |        4 are all the metrics we are storing for each possible configuration. (see comments
        |        on code to understand these metrics better)
        |
        |----- lenet_deep_analysis_results.txt:
        |        Console log of executing lenet_depth_analysis.py
        |
        |----- lenet_deep_analysis_results.pt:
        |        Dataframe containing the results of the execution of lenet_depth_analysis.py. 
        |        Used in Table II on the report. Shape:
        |
        |        [2, 2, 2, 2, 8]
        |
        |        2 represents weight sharing disabled/enabled
        |        2 represents batch normalization disabled/enabled
        |        2 represents dropout disabled/enabled
        |        2 are the mean and the standard deviation
        |        8 are all the metrics we are storing for each possible configuration. (see comments
        |        on code to understand these metrics better)


We follow the guidelines of not using scikit-learn or numpy. In plotting.ipynb we use matplotlib and seaborn
and in test.py we import time library to plot execution time. 