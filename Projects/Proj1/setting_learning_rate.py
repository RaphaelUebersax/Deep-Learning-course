"""
    File used to determine the best learning rate for each architecture. 
    It runs with auxiliary loss, weight sharing and batch normalization enabled. 

    At the end, it logs the results' tensor and stores it on stored_results/lr_results.pt.
    To understand the structure of the tensor, see readme.txt.
"""

import torch
from torch import nn
from train_test_functions import *
from models import *

print("--------------- Loading data ---------------")

init_train_input, init_train_target, init_train_classes, test_input, test_target, \
    test_classes = load_data()


print("--------------- Data loaded ---------------")
print("")

# possible etas that we are going to test
etas = [3e-1, 5e-2, 3e-2, 5e-3, 3e-3, 5e-4]

mini_batch_size = 100
epochs = 100
criterion = nn.CrossEntropyLoss()
models = ["Net", "LeNet5", "LeNet5_FullyConv", "ResNet"] 

# in this file we only run with 3 seeds instead of 10, 
# if not the execution time would be too high.
seeds = [1,2,3]

models_results = torch.zeros(len(models),len(etas),epochs)

for m, model_s in enumerate(models):
    print("Model: ", model_s)
    etas_results = torch.zeros(len(etas),epochs)
    for e, eta in enumerate(etas):

        seeds_results = torch.zeros(len(seeds),epochs)
        for i , seed in enumerate(seeds):

            print("Seed n.: ", str(i))

            #shuffle the dataset 
            torch.manual_seed(seed)
            train_input,train_target,train_classes = shuffle_data(init_train_input, init_train_target, init_train_classes)

            # weight sharing, auxiliary loss and batch norm enabled
            model = eval(model_s + "(" + str(True) + "," + str(True) + ")")

            losses = train_auxilary_loss(model, train_input, train_target, train_classes, mini_batch_size, eta, criterion, epochs)

            seeds_results[i] = losses
        
        etas_results[e] = seeds_results.mean(dim=0)

    models_results[m] = etas_results

print(models_results.shape)
print(models_results)

torch.save(models_results, 'stored_results/lr_results.pt')