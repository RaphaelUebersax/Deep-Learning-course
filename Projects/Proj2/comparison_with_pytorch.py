"""
    File where a comparison between our framework and Pytorch is done.

    We fix seeds so each time we run it we get the same results.
"""

import torch
from torch import nn
from functions import *
from model import *

# We ENABLE autograd only on this file, so we can compare our model performance with Pytorch.
# Autograd is only used in Pytorch model, not in our framework.
torch.set_grad_enabled(True)


################################################################################
# Training function compatible with Pytorch methods. 
# It works dividing the training data in minibatches, therefore improving the performance.
################################################################################
def train_pytorch_minibatch(mod, criterion, train_input, train_target_hot_label, nb_epochs, eta, minibatch_size):
    nb_train_samples = train_input.shape[0]

    for e in range(nb_epochs):

        loss_sum = 0

        for b in range(0,nb_train_samples, minibatch_size):
            
            pred = mod(train_input.narrow(0, b, minibatch_size))
            loss = criterion(pred,train_target_hot_label.narrow(0, b, minibatch_size))
            
            loss_sum += loss.item()
            
            mod.zero_grad()
            loss.backward()

            with torch.no_grad():
                for p in mod.parameters():
                    p -= eta * p.grad

        if e % 20 == 0:
            print("Loss at epoch ", e , ": " , loss_sum)



################################################################################
# This function computes the number of missclassified values. 
# Compatible with Pytorch methods. Works with minibatches only.
################################################################################
def compute_nb_errors_pytorch_minibatch(mod, input, target, minibatch_size):
    nb_errors = 0

    predicted_target = torch.empty(target.shape[0])

    for b in range(0, input.size(0), minibatch_size):
        output = mod(input.narrow(0, b, minibatch_size))
        _, predicted_classes = output.max(1)
        for k in range(minibatch_size):
            if target[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1
        
        predicted_target[b:b+minibatch_size] = predicted_classes

    return nb_errors, predicted_target



# Setting seed to get the same data always. This ensures results reproducibility.
torch.manual_seed(1)

# Generating data points and one hot label encoding for targets.
train_input,train_target = generate_disc_set(1000)
test_input,test_target = generate_disc_set(1000)

train_target_hot_label = torch.empty((train_target.shape[0],2)).zero_()
train_target_hot_label[:,0] = 1-train_target
train_target_hot_label[:,1] = train_target

test_target_hot_label = torch.empty((test_target.shape[0],2)).zero_()
test_target_hot_label[:,0] = 1-test_target
test_target_hot_label[:,1] = test_target

# Executing nb_of_iterations times, and getting the average. 
nb_of_iterations = 10
nb_errors_list = torch.empty(nb_of_iterations, 4)

for i in range(nb_of_iterations):

    #Setting seed to get the same weight initialization always, ensuring reproducibility.
    torch.manual_seed(i)

    # Pytorch model & criterion
    model_pytorch = nn.Sequential(
            nn.Linear(2,25),
            nn.Tanh(),
            nn.Linear(25,25),
            nn.Tanh(),
            nn.Linear(25,25),
            nn.Tanh(),
            nn.Linear(25,2),
            nn.Tanh()
        )

    criterion_pytorch = nn.MSELoss()

    #Same weight initialization as in model_pytorch
    torch.manual_seed(i)

    # Our framework model & criterion
    model = Sequential([
            Linear(2,25),
            Tanh(),
            Linear(25,25),
            Tanh(),
            Linear(25,25),
            Tanh(),
            Linear(25,2),
            Tanh()
        ])

    criterion = MSELoss()

    nb_train_samples = train_input.size(0)
    nb_epochs = 250
    eta = 0.15
    minibatch_size = 100


    print("ITERATION ", i)
    print("Training with Pytorch:")

    train_pytorch_minibatch(model_pytorch, criterion_pytorch, train_input, train_target_hot_label, nb_epochs, eta, minibatch_size)
    nb_errors_train_pytorch, _ = compute_nb_errors_pytorch_minibatch(model_pytorch, train_input, train_target_hot_label, minibatch_size)
    nb_errors_pytorch, _ = compute_nb_errors_pytorch_minibatch(model_pytorch, test_input, test_target_hot_label, minibatch_size)

    print("")
    print("Training with our model: ")

    train_minibatch(model, criterion, train_input, train_target_hot_label, nb_epochs, eta, minibatch_size)
    nb_errors_train, _ = compute_nb_errors_minibatch(model, train_input, train_target_hot_label, minibatch_size)
    nb_errors, _ = compute_nb_errors_minibatch(model, test_input, test_target_hot_label, minibatch_size)

    print("")
    print("Number of errors with Pytorch on training set (%): ", nb_errors_train_pytorch/10)
    print("Number of errors with our model on training set (%): ", nb_errors_train/10)
    print("Number of errors with Pytorch on test set (%): ", nb_errors_pytorch/10)
    print("Number of errors with our model on test set (%): ", nb_errors/10)
    print("")
    print("")

    
    nb_errors_list[i,0] = nb_errors_train_pytorch/10
    nb_errors_list[i,1] = nb_errors_train/10
    nb_errors_list[i,2] = nb_errors_pytorch/10
    nb_errors_list[i,3] = nb_errors/10


# Mean error and std dev on testing set. 
nb_errors_mean = nb_errors_list.mean(dim=0)
nb_errors_std = nb_errors_list.std(dim=0)


print("Mean number of errors on train set with Pytorch: ", round(nb_errors_mean[0].item(),3))
print("Standard deviation of the number of errors on train set with Pytorch: ", round(nb_errors_std[0].item(),3))

print("Mean number of errors on train set with our model: ", round(nb_errors_mean[1].item(),3))
print("Standard deviation of the number of errors on train set with our model: ", round(nb_errors_std[1].item(),3))
print("")

print("Mean number of errors on test set with Pytorch: ", round(nb_errors_mean[2].item(),3))
print("Standard deviation of the number of errors on test set with Pytorch: ", round(nb_errors_std[2].item(),3))

print("Mean number of errors on test set with our model: ", round(nb_errors_mean[3].item(),3))
print("Standard deviation of the number of errors on test set with our model: ", round(nb_errors_std[3].item(),3))
