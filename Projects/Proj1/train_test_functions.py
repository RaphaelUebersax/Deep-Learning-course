"""
    This script contains the training and testing functions used for miniproject 1, 
    as well as functions for loading and shuffling data.
"""

import torch
from torch import optim
import dlc_practical_prologue as prologue



###############################################################################
# Given a model as input, this function trains it using auxiliary losses.
# This means that to the main loss (the one taking into account all the layers)
# we sum the loss of previous layers, specifically those that are responsible 
# for determining the number represented in the input image.
# It uses adam optimizer and returns an array of losses.
###############################################################################
def train_auxilary_loss(model, train_input, train_target, train_classes,
                        mini_batch_size, eta, criterion, nb_epochs):

    optimizer = optim.Adam(model.parameters(), lr = eta)

    acc_losses = torch.zeros(nb_epochs)

    for e in range(nb_epochs):

        acc_loss = 0
        
        
        for b in range(0, train_input.size(0), mini_batch_size):
            
            out1, out2, output = model(train_input.narrow(0, b, mini_batch_size))
            loss1 = criterion(out1, train_classes.narrow(0, b, mini_batch_size)[:,0])
            loss2 = criterion(out2, train_classes.narrow(0, b, mini_batch_size)[:,1])
            loss3 = criterion(output, train_target.narrow(0, b, mini_batch_size))
            loss = loss1 + loss2 + loss3
            acc_loss = acc_loss + loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step() 

        acc_losses[e] = acc_loss
    
    return acc_losses



###############################################################################
# Given a model as input, this function trains it. It only trains using a
# main loss, the one over the last layer of the model.
# It uses adam optimizer and returns an array of losses.
###############################################################################  
def train (model, train_input, train_target, train_classes, mini_batch_size,
           eta, criterion, nb_epochs):
    nb_epochs = 25
    optimizer = optim.Adam(model.parameters(), lr = eta)

    acc_losses = torch.zeros(nb_epochs)

    for e in range(nb_epochs):
        
        acc_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            _, _, output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        acc_losses[e] = acc_loss

    return acc_losses



###############################################################################  
# Returns the number of classification errors for the digit comparison task. 
# It also returns the classification errors performed exclusively by the last
# layers of the model in charge of the comparison task.
############################################################################### 
def test_target_fn(model, test_input, test_target, mini_batch_size):
    errors = 0
    last_layer_error = 0 

    for b in range(0, test_input.size(0),mini_batch_size):
        output_img_1, output_img_2, output = model(test_input.narrow(0,b,mini_batch_size))
        _, predicted_classes = output.max(1)
        _, predicted_classes_img_1 = output_img_1.max(1)
        _, predicted_classes_img_2 = output_img_2.max(1)

        for k in range(mini_batch_size):

            optimal_predicted_final_classification = 0
            if predicted_classes_img_1[k] <= predicted_classes_img_2[k]:
                optimal_predicted_final_classification = 1

            if optimal_predicted_final_classification != predicted_classes[k]:
                last_layer_error = last_layer_error + 1 

            if test_target[b+k] != predicted_classes[k]:
                errors = errors + 1


    return errors, last_layer_error



###############################################################################
# Returns the number of classification errors for the digit recognition task.
# As the dataset has two input images per sample, we return two errors values, 
# one per image. 
############################################################################### 
def test_classes_fn(model, test_input, test_classes, mini_batch_size):

    #there are two images to test
    errors_img_1 = 0
    errors_img_2 = 0
    


    for b in range(0, test_input.size(0),mini_batch_size):
        output_img_1, output_img_2 , _ = model(test_input.narrow(0,b,mini_batch_size))

        _, predicted_classes_img_1 = output_img_1.max(1)
        _, predicted_classes_img_2 = output_img_2.max(1)
        
        for k in range(mini_batch_size):

            if test_classes[b+k,0] != predicted_classes_img_1[k]:
                errors_img_1 = errors_img_1 + 1
            if test_classes[b+k, 1] != predicted_classes_img_2[k]:
                errors_img_2 = errors_img_2 + 1

       
    return errors_img_1, errors_img_2



###############################################################################
# Shuffles data to ensure randomness in our experiments. 
###############################################################################
def shuffle_data(train_set, train_target, train_classes):

    idx = torch.randperm(train_set.size()[0])

    shuffled_set = train_set[idx,:,:,:]
    shuffled_target = train_target[idx]
    shuffled_classes = train_classes[idx,:]


    return shuffled_set, shuffled_target,shuffled_classes



###############################################################################
# Loads a dataset of 1000 MNIST samples using the function provided by the teacher.
# If necessary, it can normalize data.
###############################################################################
def load_data(normalize_data=True):

    init_train_input, init_train_target, init_train_classes, test_input, test_target, \
    test_classes = prologue.generate_pair_sets(1000)
   
    if normalize_data:
        init_train_input.sub_(init_train_input.mean()).div_(init_train_input.std())
        test_input.sub(init_train_input.mean()).div_(init_train_input.std())

    return init_train_input, init_train_target, init_train_classes, test_input, \
    test_target, test_classes
    