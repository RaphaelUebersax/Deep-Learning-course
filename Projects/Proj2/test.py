"""
    Testing file for Project 2. We first generate a dataset as stated on the 
    project description, then we run the model nb_iterations times to get the average 
    performance.

    During the process, we log the loss every 20 epochs for each of the iterations.
    We also log the error % for each iteration, and at the end we print the mean error %
    for both train and test sets.

    On this file we don't use fixed seeds, as we don't want to import pytorch.manual_seed(). 
    Therefore, each time we run the file the results are different, but as we do the mean
    over all the iterations the results shouldn't change too much.
"""

from math import sqrt
from torch import empty as empty
from torch import set_grad_enabled
from functions import *
from model import *

set_grad_enabled(False)


# Generating train and test samples. Adding one hot label encoding for target values
train_input,train_target = generate_disc_set(1000)
test_input,test_target = generate_disc_set(1000)

train_target_hot_label = empty((train_target.shape[0],2)).zero_()
train_target_hot_label[:,0] = 1-train_target
train_target_hot_label[:,1] = train_target

test_target_hot_label = empty((test_target.shape[0],2)).zero_()
test_target_hot_label[:,0] = 1-test_target
test_target_hot_label[:,1] = test_target


# Setting parameters
nb_train_samples = train_input.size(0)
nb_epochs = 250
eta = 0.15
minibatch_size = 100
nb_iterations = 10

# list used to compute the errors' mean
nb_errors_list = empty(nb_iterations,2)

# Looping nb_iterations and getting the mean error %
for i in range(nb_iterations):

    # Two input units, three hidden layers of 25 units and two output units.
    # Using by default Pytorch weight initialization and tanh activation function.
    mod = Sequential([Linear(2,25),Tanh(),Linear(25,25),Tanh(),Linear(25,25),Tanh(),Linear(25,2),Tanh()])
    criterion = MSELoss()

    print("")
    print("ITERATION ", i)  
    train_minibatch(mod, criterion, train_input, train_target_hot_label, nb_epochs, eta, minibatch_size)
    
    print("")
    nb_errors_train, _ = compute_nb_errors_minibatch(mod, train_input, train_target_hot_label, minibatch_size)
    print("Missclassified on training: ", nb_errors_train/10, "%")

    nb_errors_test, predicted_target = compute_nb_errors_minibatch(mod, test_input, test_target_hot_label, minibatch_size)
    print("Missclassified on testing: ", nb_errors_test/10, "%")

    nb_errors_list[i,0] = nb_errors_train/10
    nb_errors_list[i,1] = nb_errors_test/10


# We compute the mean and std dev over all the iterations in the for loop
nb_errors_mean = nb_errors_list.mean(dim=0)
nb_errors_std = nb_errors_list.std(dim=0)


print("")
print("")
print("Training data mean error (%): ", round(nb_errors_mean[0].item(),3))
print("Training data error (%) std dev: ", round(nb_errors_std[0].item(),3))

print("")
print("Test data mean error (%): ", round(nb_errors_mean[1].item(),3))
print("Test data error (%) std dev: ", round(nb_errors_std[1].item(),3))