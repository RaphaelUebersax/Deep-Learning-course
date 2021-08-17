"""
    Testing file for Project 1. We first load normalized data, and then we set the required 
    parameters before training and testing the model.

    For this file, we only take into account LeNet5 model, and we test it with bacth norm enabled,
    comparing the performance between disabling/enabling auxiliary loss and weight sharing.

    We run each possible configuration 10 times, each of them having a different seed, so we can
    ensure that data order in training and weight initialization are not biasing our results. Using
    our own predefined seeds also ensures reproducibility, so each time this file is executed,
    the same results are observed. 

    During the execution, the mean error on test target for each configuration is logged.
"""

import torch
from torch import nn
# import time # importing time to analyse running time
from models import *
from train_test_functions import *



print("--------------- Loading data ---------------")

init_train_input, init_train_target, init_train_classes, test_input, test_target, \
    test_classes = load_data()


print("--------------- Data loaded ---------------")
print("")



criterion = nn.CrossEntropyLoss()

model_s = "LeNet5"

# weigh sharing and auxiliary loss disabled/enabled
weight_sharing_list = [False, True]
auxiliary_list = [False, True]
batch_norm = True

seeds_list = [1,2,3,4,5,6,7,8,9,10]
nb_epochs = 30
mini_batch_size = 100
eta = 5e-3

results_seed = torch.zeros(2,len(seeds_list))
# times = torch.zeros(len(seeds_list))

# init_time = time.monotonic()

print("")
print("**********************************************************")
print("*********** Performance comparison for LeNet5 ************")
print("**********************************************************")
print("")

for a, auxiliary in enumerate(auxiliary_list):
    for w, weight_sharing in enumerate(weight_sharing_list):

        print("- Weight sharing: " + str(weight_sharing) + "\t - Auxiliary loss: " + str(auxiliary))

        for i, seed in enumerate(seeds_list):

            # shuffle the dataset, fixing a seed ensures reproducibility.
            torch.manual_seed(seed)
            train_input,train_target,train_classes = shuffle_data(init_train_input, init_train_target, init_train_classes)

            model = eval(model_s + "(" + str(batch_norm) + "," + str(weight_sharing) + ")")

            # start = time.monotonic()

            # checks if we are training with/without auxiliary losses.
            if (auxiliary == True):
                train_auxilary_loss(model, train_input, train_target, train_classes, mini_batch_size, eta, criterion, nb_epochs)

            else:
                train(model, train_input, train_target, train_classes, mini_batch_size, eta, criterion, nb_epochs)

            # times[i] = time.monotonic() - start

            nb_errors_target_training, _ = test_target_fn(model, train_input, train_target, mini_batch_size)
            nb_errors_target, _ = test_target_fn(model, test_input, test_target, mini_batch_size)
            results_seed[0,i] = 100 * nb_errors_target_training / train_input.size(0)
            results_seed[1,i] = 100 * nb_errors_target / test_input.size(0)
        
        std = torch.std(results_seed, dim=1)
        mean = torch.mean(results_seed, dim=1)

        # times_mean = times.mean()
        # times_std = times.std()

        print("The mean error on training target is (%):", round(mean[0].item(),2) ,"with a std dev of:", round(std[0].item(),2))
        print("The mean error on testing target is (%):", round(mean[1].item(),2) ,"with a std dev of:", round(std[1].item(),2))
        # print("Mean training between all seeds:", round(times_mean.item(),3), "sec. Std dev:", round(times_std.item(),3), "sec.")
        print("")


# print("Total execution time of the script:", time.monotonic() - init_time, "sec.") 