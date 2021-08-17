########################################################################################
# Project 2 README file
########################################################################################

This project aims at developing a simple version of a deep learning framework using 
python. More precisely, the challenges of implementing a Multi-Layer Perceptron (MLP) 
model including linear layers, Relu and Tanh activation functions as well as mean square 
error loss are tackled. The aim is to create an automatic mechanism that performs 
backpropagation as well a stochastic gradient descent.

This file is intended to clarify the structure by which the files are organised 
in the project folder.

|----- model.py:
|    File containing our own deep learning framework. We employ a hierarchical structure,
|    where all the classes (ReLU, Tanh, MSELoss, Linear and Sequential) derive from Module.
|
|----- functions.py:
|    Functions that are used to train and test our framework.
|    Some of them have a version with minibatches and other without them.
|
|----- test.py:
|    Testing file for Project 2. We first generate a dataset as stated on the 
|    project description, then we run the model nb_iterations times to get the average 
|    performance.
|
|    During the process, we log the loss every 20 epochs for each of the iterations.
|    We also log the error % for each iteration, and at the end we print the mean error %
|    for both train and test sets.
|
|    On this file we don't use fixed seeds, as we are not allowed to import pytorch.manual_seed(). 
|    Therefore, each time we run the file the results are different, but as we do the mean
|    over all the iterations the results shouldn't change too much.
|
|----- comparison_with_pytorch.py:
|    File where a comparison between our framework and Pytorch is done.
|    We fix the seeds so each time we run it we get the same results.
|
|----- stored_results/
        |----- test_tanh_def_results.txt:
        |    It contains the console log of executing test.py. Its name contains tanh
        |    as it's the used activation function, and def, as we use by default Pytorch
        |    linear layer weight initialization. Used for Table I in the report.
        |
        |----- test_tanh_xavier_results.txt:
        |    It contains the console log of executing test.py. Its name contains tanh
        |    as it's the used activation function, and xavier, as we use Xavier initialization
        |    on linear layers. Used for Table I in the report.
        |
        |----- test_relu_def_results.txt:
        |    It contains the console log of executing test.py. Its name contains relu
        |    as it's the used activation function, and def, as we use by default Pytorch
        |    linear layer weight initialization. Used for Table I in the report.
        |
        |----- test_relu_xavier_results.txt:
        |    It contains the console log of executing test.py. Its name contains relu
        |    as it's the used activation function, and xavier, as we use Xavier initialization
        |    on linear layers. Used for Table I in the report.
        |
        |----- comparison_results.txt:
        |    Here we log the console results of executing comparison_with_pytorch.py.
        |    Used for Table II in the report.
    

In all the files we follow the guidelines of only importing the allowed libraries and
disabling autograd. But in comparison_with_pytorch.py, we import torch.nn and set 
autograd to enabled, as we want to compare Pytorch performance with our framework. 
But both torch.nn and autograd are only used on the Pytorch model, not in our framework.