"""
    Functions that are used to train and test our framework.
    Some of them have a version with minibatches and other without them.
"""
from torch import empty as empty
from torch import set_grad_enabled
from torch import int64, float32 # necessary to generate data
from math import sqrt, pi


set_grad_enabled(False)


################################################################################
# Generating training and testing set of nb 2D points. They are sampled 
# uniformly in (0,1), and the labels are marked with 1 if the points are inside
# the disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi). If not, are marked as 0
################################################################################
def generate_disc_set(nb): 

    input = empty(nb,2, dtype = float32).uniform_(0.0,1.0)
    input_norm = (input - 0.5).pow(2).sum(dim=1).sqrt()
    output = empty(nb, dtype = int64)
    output[:] = 0
    output[input_norm < 1/sqrt(2*pi)] = 1
    
    return input, output



################################################################################
# This function computes the number of missclassified values. 
# Compatible with our framework methods. It doesn't work when using minibatches.
################################################################################
def compute_nb_errors(model, input, target):

    nb_data_errors = 0

    predicted_target = empty(target.shape)

    for i, element in enumerate(input):
        output = model.forward(element)
        if (output[0] > output[1]):
            predicted_classes = 0
        else:
            predicted_classes = 1

        if target[i,predicted_classes] == 0:
            nb_data_errors = nb_data_errors + 1
        
        predicted_target[i] = predicted_classes

    return nb_data_errors, predicted_target


################################################################################
# This function computes the number of missclassified values. 
# Compatible with our framework methods. It works with minibatches.
################################################################################
def compute_nb_errors_minibatch(model, input, target, minibatch_size):
    nb_errors = 0

    predicted_target = empty(target.shape[0])

    for b in range(0, input.size(0), minibatch_size):
        output = model.forward(input.narrow(0, b, minibatch_size))
        _, predicted_classes = output.max(1)
        for k in range(minibatch_size):
            if target[b + k, predicted_classes[k]] <= 0:
                nb_errors = nb_errors + 1
        
        predicted_target[b:b+minibatch_size] = predicted_classes

    return nb_errors, predicted_target



################################################################################
# Training function compatible with our framework. 
# It works without minibatches, applying each data point
# independently to the model. 
# Logs the error every 20 epochs and return a tensor with all the losses.
################################################################################
def train(mod, criterion, train_input, train_target_hot_label, nb_epochs, eta):

    nb_train_samples = train_input.shape[0]
    
    losses = empty(nb_epochs)

    for e in range(nb_epochs):

        loss_sum = 0

        for i in range(0,nb_train_samples):
            
            pred = mod.forward(train_input[i])
            loss = criterion.forward(pred,train_target_hot_label[i])
            
            loss_sum += loss.item()

            grad_loss = criterion.backward(pred,train_target_hot_label[i])
            mod.backward(grad_loss)
            for p in mod.param():
                p.p -= eta * p.gradient

        if e % 20 == 0:
            print("Loss at epoch ", e , ": " , loss_sum)

        losses[e] = loss_sum
    
    return losses



################################################################################
# Training function compatible with our framework. 
# It works dividing the training data in minibatches, therefore improving the performance.
# Logs the error every 20 epochs and return a tensor with all the losses.
################################################################################
def train_minibatch(mod, criterion, train_input, train_target_hot_label, nb_epochs, eta, minibatch_size):
    nb_train_samples = train_input.shape[0]

    losses = empty(nb_epochs)

    for e in range(nb_epochs):

        loss_sum = 0

        for b in range(0,nb_train_samples, minibatch_size):
            
            pred = mod.forward(train_input.narrow(0, b, minibatch_size))
            loss = criterion.forward(pred,train_target_hot_label.narrow(0, b, minibatch_size))
            
            loss_sum += loss.item()

            grad_loss = criterion.backward(pred,train_target_hot_label.narrow(0, b, minibatch_size))
            mod.backward(grad_loss)
            for p in mod.param():
                p.p -= eta * p.gradient

        if e % 20 == 0:
            print("Loss at epoch ", e , ": " , loss_sum)

        losses[e] = loss_sum
    
    return losses